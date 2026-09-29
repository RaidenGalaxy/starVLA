"""Run causal selection, evidence diagnostics, and optional late-query VLM evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, TextIO

from .frames import FEATURE_NAMES, FrameObservation, FrameRef, iter_trial_frames
from .metrics import aggregate_results, refs_in_evidence, selection_metrics
from .predictors import EvidenceOraclePredictor, QwenMultiImagePredictor
from .protocol import StreamTrial, atomic_json, atomic_jsonl, file_sha256, load_trials
from .selection import METHODS, OnlineMemorySelector, SelectionResult
from .utility import UtilityModel


def _stable_seed(seed: int, *parts: object) -> int:
    digest = hashlib.sha256(":".join([str(seed), *(str(item) for item in parts)]).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _overlaps_evidence(ref: FrameRef, trial: StreamTrial) -> bool:
    return any(span.start_sec - 1e-6 <= ref.timestamp_sec <= span.end_sec + 1e-6 for span in trial.evidence_spans)


def _evidence_available(trial: StreamTrial, refs: list[FrameRef] | tuple[FrameRef, ...]) -> bool | None:
    if not trial.evidence_spans:
        return None
    evidence = refs_in_evidence(refs, trial.evidence_spans)
    if any(
        not any(span.start_sec - 1e-6 <= ref.timestamp_sec <= span.end_sec + 1e-6 for ref in refs)
        for span in trial.evidence_spans
    ):
        return False
    minimum = trial.metadata.get("min_required_frames")
    return minimum is None or len(evidence) >= int(minimum)


class _UniformCapCollector:
    def __init__(self, capacity: int, seed: int) -> None:
        import random

        self.capacity = capacity
        self._rng = random.Random(seed)
        self.items: list[FrameObservation] = []
        self.seen = 0

    def add(self, item: FrameObservation) -> None:
        self.seen += 1
        if len(self.items) < self.capacity:
            self.items.append(item)
            return
        index = self._rng.randrange(self.seen)
        if index < self.capacity:
            self.items[index] = item


def _serialize_result(
    trial: StreamTrial,
    result: SelectionResult,
    observed_refs: list[FrameRef],
    predictor: EvidenceOraclePredictor | QwenMultiImagePredictor | None,
    *,
    oracle_stages: dict[str, list[FrameObservation]],
    evaluate_oracle_stages: bool,
) -> dict[str, Any]:
    metrics = selection_metrics(trial, result, observed_refs)
    prediction: dict[str, Any] = {}
    query_latency_sec = None
    if predictor is not None:
        available = bool(metrics["evidence_available"]) if metrics["evidence_available"] is not None else True
        query_started = time.monotonic()
        prediction = predictor.predict(trial.question, result.final_observations, evidence_available=available).to_dict()
        query_latency_sec = time.monotonic() - query_started
    ladder: dict[str, Any] = {
        "dense_offline_cap": _evidence_available(trial, [item.ref for item in oracle_stages["dense_offline_cap"]]),
        "critical_only": _evidence_available(trial, [item.ref for item in oracle_stages["critical_only"]]),
        "selected_capture": _evidence_available(trial, result.captured),
        "final_memory": _evidence_available(trial, result.final_refs),
        "oracle_retrieval": _evidence_available(
            trial,
            [item.ref for item in result.final_observations if _overlaps_evidence(item.ref, trial)],
        ),
    }
    stage_predictions: dict[str, Any] = {}
    if predictor is not None and evaluate_oracle_stages:
        stages = dict(oracle_stages)
        stages["oracle_retrieval"] = [
            item for item in result.final_observations if _overlaps_evidence(item.ref, trial)
        ]
        for name, observations in stages.items():
            available = _evidence_available(trial, [item.ref for item in observations])
            stage_predictions[name] = predictor.predict(
                trial.question,
                observations,
                evidence_available=bool(available) if available is not None else True,
            ).to_dict()
    return {
        "format_version": 1,
        "protocol_version": trial.protocol_version,
        "trial_id": trial.trial_id,
        "sample_id": trial.sample_id,
        "dataset": trial.dataset,
        "split": trial.split,
        "source_group": trial.source_group,
        "method": result.method,
        "memory_frames": result.memory_frames,
        "encode_fps": result.encode_fps,
        "window_sec": result.window_sec,
        "query_time_sec": trial.query_time_sec,
        "query_delay_sec": trial.query_delay_sec,
        "condition": trial.condition,
        "question": {
            "answer_type": trial.question.answer_type,
            "answer": trial.question.answer,
            "answer_text": trial.question.answer_text,
        },
        "metrics": metrics,
        "prediction": prediction,
        "query_latency_sec": query_latency_sec,
        "oracle_ladder": ladder,
        "stage_predictions": stage_predictions,
        "captured_frames": [item.to_dict() for item in result.captured],
        "retained_frames": [item.to_dict() for item in result.final_refs],
        "evicted_frames": [item.to_dict() for item in result.evicted],
    }


def _write_feature(handle: TextIO, trial: StreamTrial, observation: FrameObservation) -> None:
    row = {
        "format_version": 1,
        "trial_id": trial.trial_id,
        "sample_id": trial.sample_id,
        "source_group": trial.source_group,
        "dataset": trial.dataset,
        "split": trial.split,
        "frame_id": observation.ref.frame_id,
        "timestamp_sec": observation.ref.timestamp_sec,
        "role": observation.ref.role,
        "feature_names": FEATURE_NAMES,
        "features": observation.features.tolist(),
        "evidence_overlap": _overlaps_evidence(observation.ref, trial),
    }
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _report_markdown(summary: dict[str, Any], *, predictor_name: str) -> str:
    lines = [
        "# 瞬态证据流式 Pilot 报告",
        "",
        (
            f"预测器：`{predictor_name}`。所有方法在问题出现前运行，"
            "且最终保留帧数不超过固定预算。"
        ),
        "",
        (
            "| 方法 | Memory | 编码 FPS | Accuracy | 捕获 span recall | 保留 span recall | "
            "Required-frame coverage | 每分钟退化 |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def show(value: object) -> str:
        return "—" if value is None else f"{float(value):.4f}"

    for item in summary["configurations"]:
        lines.append(
            (
                "| {method} | {memory_frames} | {encode_fps:g} | {accuracy} | {capture} | "
                "{retention} | {coverage} | {slope} |"
            ).format(
                method=item["method"],
                memory_frames=item["memory_frames"],
                encode_fps=item["encode_fps"],
                accuracy=show(item["accuracy"]),
                capture=show(item["capture_span_recall"]),
                retention=show(item["retention_span_recall"]),
                coverage=show(item["required_frame_coverage"]),
                slope=show(item["degradation_per_delay_minute"]),
            )
        )
    gates = summary["research_gates"]
    lines.extend(
        [
            "",
            "## Research gates",
            "",
            f"- Oracle gap：{show(gates['oracle_gap_pp'])} pp；10 pp gate：{gates['failure_exists_gate_10pp']}。",
            (
                f"- Future-utility 最佳等预算增益：{show(gates['best_matched_future_utility_gain_pp'])} pp；"
                f"5 pp gate：{gates['method_gain_gate_5pp']}。"
            ),
            f"- 全部预算合规：{gates['all_budgets_respected']}。",
            "",
            (
                "说明：`evidence_oracle` 的 accuracy 只检验必要证据是否在查询时仍可用，"
                "不代表真实 VLM 推理质量。"
            ),
            (
                "Moment-Video 没有官方 temporal certificate，因此其证据召回列保持为空，"
                "只报告端到端模型结果。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS[:-1]))
    parser.add_argument("--memory-frames", type=int, nargs="+", default=[16, 64])
    parser.add_argument("--encode-fps", type=float, nargs="+", default=[1.0, 4.0])
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--directory-fps", type=float, default=30.0)
    parser.add_argument("--utility-model", type=Path)
    parser.add_argument("--predictor", choices=["none", "evidence_oracle", "qwen"], default="evidence_oracle")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--evaluate-oracle-stages", action="store_true")
    parser.add_argument("--oracle-max-frames", type=int, default=256)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--split", nargs="+")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export-features", action="store_true")
    parser.add_argument("--skip-missing-media", action="store_true")
    args = parser.parse_args()
    if "future_utility" in args.methods and args.utility_model is None:
        parser.error("future_utility requires --utility-model")
    if args.predictor == "qwen" and args.model_path is None:
        parser.error("qwen predictor requires --model-path")
    if any(value <= 0 for value in [*args.memory_frames, *args.encode_fps, args.window_sec, args.oracle_max_frames]):
        parser.error("all budgets must be positive")

    trials = load_trials(args.manifest)
    if args.split:
        trials = [trial for trial in trials if trial.split in args.split]
    if args.max_samples is not None:
        trials = trials[: args.max_samples]
    if not trials:
        raise ValueError("no trials selected")
    utility_model = UtilityModel.load(args.utility_model) if args.utility_model else None
    if utility_model is not None and utility_model.feature_names != FEATURE_NAMES:
        raise ValueError("utility model feature schema does not match the runtime scout")
    if args.predictor == "qwen":
        predictor = QwenMultiImagePredictor(args.model_path, device=args.device, max_new_tokens=args.max_new_tokens)
    elif args.predictor == "evidence_oracle":
        predictor = EvidenceOraclePredictor()
    else:
        predictor = None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_handle: TextIO | None = None
    feature_temporary: Path | None = None
    if args.export_features:
        opened = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=args.output_dir, delete=False)
        feature_handle = opened
        feature_temporary = Path(opened.name)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.monotonic()
    try:
        for trial_index, trial in enumerate(trials):
            selectors = [
                OnlineMemorySelector(
                    method,
                    memory_frames=memory_frames,
                    encode_fps=encode_fps,
                    window_sec=args.window_sec,
                    seed=_stable_seed(args.seed, trial.trial_id, method, memory_frames, encode_fps),
                    utility_model=utility_model,
                )
                for method in args.methods
                for memory_frames in args.memory_frames
                for encode_fps in args.encode_fps
            ]
            observed_refs: list[FrameRef] = []
            dense = _UniformCapCollector(args.oracle_max_frames, _stable_seed(args.seed, trial.trial_id, "dense"))
            critical: list[FrameObservation] = []
            current_window: list[FrameObservation] = []
            current_window_index: int | None = None

            def flush() -> None:
                nonlocal current_window
                if not current_window:
                    return
                for selector in selectors:
                    selector.ingest_window(current_window)
                current_window = []

            try:
                for observation in iter_trial_frames(trial, directory_fps=args.directory_fps):
                    observed_refs.append(observation.ref)
                    dense.add(observation)
                    if _overlaps_evidence(observation.ref, trial) and len(critical) < args.oracle_max_frames:
                        critical.append(observation)
                    if feature_handle is not None:
                        _write_feature(feature_handle, trial, observation)
                    window_index = math.floor(observation.ref.timestamp_sec / args.window_sec + 1e-9)
                    if current_window_index is None:
                        current_window_index = window_index
                    if window_index != current_window_index:
                        flush()
                        current_window_index = window_index
                    current_window.append(observation)
                flush()
                oracle_stages = {
                    "dense_offline_cap": sorted(dense.items, key=lambda item: item.ref.timestamp_sec),
                    "critical_only": critical,
                }
                for selector in selectors:
                    rows.append(
                        _serialize_result(
                            trial,
                            selector.result(),
                            observed_refs,
                            predictor,
                            oracle_stages=oracle_stages,
                            evaluate_oracle_stages=args.evaluate_oracle_stages,
                        )
                    )
            except (FileNotFoundError, ValueError) as error:
                if not args.skip_missing_media:
                    raise
                failures.append({"trial_id": trial.trial_id, "error": f"{type(error).__name__}: {error}"})
            print(f"[{trial_index + 1}/{len(trials)}] {trial.trial_id}", flush=True)
    finally:
        if feature_handle is not None:
            feature_handle.close()
    if feature_temporary is not None:
        os.replace(feature_temporary, args.output_dir / "frame_features.jsonl")
    if not rows:
        raise RuntimeError("no benchmark results were produced")
    atomic_jsonl(args.output_dir / "results.jsonl", rows)
    atomic_jsonl(args.output_dir / "failures.jsonl", failures)
    summary = aggregate_results(rows)
    summary["runtime_seconds"] = time.monotonic() - started
    summary["trials_requested"] = len(trials)
    summary["trials_completed"] = len({row["trial_id"] for row in rows})
    summary["failures"] = len(failures)
    atomic_json(args.output_dir / "summary.json", summary)
    atomic_json(
        args.output_dir / "run_config.json",
        {
            "protocol_version": 1,
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": file_sha256(args.manifest),
            "methods": args.methods,
            "memory_frames": args.memory_frames,
            "encode_fps": args.encode_fps,
            "window_sec": args.window_sec,
            "predictor": args.predictor,
            "model_path": str(args.model_path.resolve()) if args.model_path else None,
            "utility_model": str(args.utility_model.resolve()) if args.utility_model else None,
            "utility_model_sha256": file_sha256(args.utility_model) if args.utility_model else None,
            "seed": args.seed,
            "runtime_query_visible_to_selector": False,
            "raw_video_replay_at_query": False,
        },
    )
    (args.output_dir / "report_zh.md").write_text(
        _report_markdown(summary, predictor_name=args.predictor), encoding="utf-8"
    )
    print(f"wrote {len(rows)} result rows to {args.output_dir}")


if __name__ == "__main__":
    main()
