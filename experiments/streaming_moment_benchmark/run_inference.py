"""Run resumable late-query prediction from frozen frames only."""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import time
from pathlib import Path
from typing import Any

from .artifacts import (
    FrameStore,
    artifact_observations,
    artifact_stored_frames,
    artifact_trial,
    inference_cache_key,
    load_artifact,
    path_fingerprint,
)
from .metrics import aggregate_results, refs_in_evidence
from .predictors import EvidenceOraclePredictor, QwenMultiImagePredictor
from .protocol import StreamTrial, atomic_json, atomic_jsonl, file_sha256


STAGES = ("final_memory", "critical_only", "dense_offline_cap", "oracle_retrieval")


class QueryTimeoutError(TimeoutError):
    pass


def _timeout_handler(_signum: int, _frame: object) -> None:
    raise QueryTimeoutError("query exceeded wall-time limit")


def _evidence_available(trial: StreamTrial, observations: list[Any]) -> bool | None:
    return _evidence_available_refs(trial, [item.ref for item in observations])


def _evidence_available_refs(trial: StreamTrial, refs: list[Any]) -> bool | None:
    if not trial.evidence_spans:
        return None
    evidence = refs_in_evidence(refs, trial.evidence_spans)
    spans_present = all(
        any(span.start_sec - 1e-6 <= ref.timestamp_sec <= span.end_sec + 1e-6 for ref in refs)
        for span in trial.evidence_spans
    )
    minimum = trial.metadata.get("min_required_frames")
    return bool(spans_present and (minimum is None or len(evidence) >= int(minimum)))


def _load_index(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"selection index is empty: {path}")
    return rows


def _existing_prediction(path: Path, key: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            row = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return row if row.get("inference_cache_key") == key and row.get("status") == "ok" else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-index", type=Path, required=True)
    parser.add_argument("--frame-store", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=["final_memory"])
    parser.add_argument("--predictor", choices=["evidence_oracle", "qwen"], default="qwen")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--min-pixels", type=int)
    parser.add_argument("--max-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--query-timeout-sec", type=int, default=600)
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--memory-frames", type=int, nargs="+")
    parser.add_argument("--encode-fps", type=float, nargs="+")
    parser.add_argument("--split", nargs="+")
    parser.add_argument("--profiles", nargs="+")
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--verify-frames", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if args.predictor == "qwen" and args.model_path is None:
        parser.error("qwen predictor requires --model-path")
    if min(args.max_new_tokens, args.cpu_threads, args.interop_threads, args.query_timeout_sec) <= 0:
        parser.error("generation and CPU limits must be positive")
    if args.max_pixels is not None and args.max_pixels <= 0:
        parser.error("max-pixels must be positive")

    os.environ.setdefault("OMP_NUM_THREADS", str(args.cpu_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.cpu_threads))
    index_rows = _load_index(args.selection_index)
    selected: list[tuple[dict[str, Any], Path, dict[str, Any]]] = []
    for index in index_rows:
        if args.methods and index["method"] not in args.methods:
            continue
        if args.memory_frames and int(index["memory_frames"]) not in args.memory_frames:
            continue
        if args.encode_fps and float(index["encode_fps"]) not in args.encode_fps:
            continue
        artifact_path = Path(index["path"])
        artifact = load_artifact(artifact_path)
        trial = artifact_trial(artifact)
        if args.split and trial.split not in args.split:
            continue
        if args.profiles and trial.condition.get("distractor_profile") not in args.profiles:
            continue
        selected.append((index, artifact_path, artifact))
    if not selected:
        raise ValueError("no frozen selections match the requested filters")

    model_fingerprint = path_fingerprint(args.model_path) if args.model_path else None
    prompt_version = (
        QwenMultiImagePredictor.PROMPT_VERSION if args.predictor == "qwen" else "evidence-oracle-v1"
    )
    jobs: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, artifact_path, artifact in selected:
        for stage in args.stages:
            key = inference_cache_key(
                selection_artifact_sha256=artifact["artifact_sha256"],
                stage=stage,
                predictor=args.predictor,
                model_fingerprint=model_fingerprint,
                prompt_version=prompt_version,
                min_pixels=args.min_pixels,
                max_pixels=args.max_pixels,
                max_new_tokens=args.max_new_tokens,
            )
            result_path = args.output_dir / "predictions" / key[:2] / f"{key}.json"
            cached = None if args.no_resume else _existing_prediction(result_path, key)
            if cached is not None:
                cached["cached"] = True
                results.append(cached)
            else:
                jobs.append(
                    {
                        "index": index,
                        "artifact_path": artifact_path,
                        "artifact": artifact,
                        "stage": stage,
                        "key": key,
                        "result_path": result_path,
                    }
                )
    if args.max_jobs is not None:
        jobs = jobs[: args.max_jobs]

    predictor: EvidenceOraclePredictor | QwenMultiImagePredictor | None = None
    if jobs:
        if args.predictor == "qwen":
            predictor = QwenMultiImagePredictor(
                args.model_path,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
                min_pixels=args.min_pixels,
                max_pixels=args.max_pixels,
                cpu_threads=args.cpu_threads,
                interop_threads=args.interop_threads,
            )
        else:
            predictor = EvidenceOraclePredictor()
    store = FrameStore(args.frame_store)
    failures: list[dict[str, Any]] = []
    started = time.monotonic()
    previous_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    try:
        for job_index, job in enumerate(jobs):
            artifact = job["artifact"]
            trial = artifact_trial(artifact)
            query_started = time.monotonic()
            try:
                observations = artifact_observations(
                    artifact, store, job["stage"], verify=args.verify_frames
                ) if args.predictor == "qwen" else []
                if args.predictor == "qwen":
                    available = _evidence_available(trial, observations)
                    frames_queried = len(observations)
                else:
                    stored_frames = artifact_stored_frames(artifact, job["stage"])
                    available = _evidence_available_refs(trial, [item.ref for item in stored_frames])
                    frames_queried = len(stored_frames)
                signal.setitimer(signal.ITIMER_REAL, args.query_timeout_sec)
                assert predictor is not None
                prediction = predictor.predict(
                    trial.question,
                    observations,
                    evidence_available=bool(available) if available is not None else True,
                ).to_dict()
                signal.setitimer(signal.ITIMER_REAL, 0)
                row = {
                    "format_version": 1,
                    "status": "ok",
                    "inference_cache_key": job["key"],
                    "selection_cache_key": artifact["cache_key"],
                    "selection_artifact_sha256": artifact["artifact_sha256"],
                    "selection_artifact": str(job["artifact_path"].resolve()),
                    "trial_id": trial.trial_id,
                    "sample_id": trial.sample_id,
                    "dataset": trial.dataset,
                    "split": trial.split,
                    "source_group": trial.source_group,
                    "condition": trial.condition,
                    "question": {
                        "answer_type": trial.question.answer_type,
                        "answer": trial.question.answer,
                        "answer_text": trial.question.answer_text,
                    },
                    "method": artifact["selection_config"]["method"],
                    "memory_frames": artifact["selection_config"]["memory_frames"],
                    "encode_fps": artifact["selection_config"]["encode_fps"],
                    "window_sec": artifact["selection_config"]["window_sec"],
                    "stage": job["stage"],
                    "frames_queried": frames_queried,
                    "metrics": artifact["metrics"],
                    "prediction": prediction,
                    "evidence_available_at_stage": available,
                    "query_latency_sec": time.monotonic() - query_started,
                    "model_fingerprint": model_fingerprint,
                    "prompt_version": prompt_version,
                    "min_pixels": args.min_pixels,
                    "max_pixels": args.max_pixels,
                    "max_new_tokens": args.max_new_tokens,
                    "raw_video_replay_at_query": False,
                    "cached": False,
                }
                atomic_json(job["result_path"], row)
                results.append(row)
            except (OSError, ValueError, KeyError, QueryTimeoutError, RuntimeError) as error:
                signal.setitimer(signal.ITIMER_REAL, 0)
                failure = {
                    "inference_cache_key": job["key"],
                    "trial_id": trial.trial_id,
                    "stage": job["stage"],
                    "error": f"{type(error).__name__}: {error}",
                    "runtime_sec": time.monotonic() - query_started,
                }
                failures.append(failure)
            elapsed = time.monotonic() - started
            rate = (job_index + 1) / elapsed if elapsed else 0.0
            eta = (len(jobs) - job_index - 1) / rate if rate else math.nan
            should_print = (
                args.predictor == "qwen"
                or len(jobs) <= 50
                or job_index == 0
                or job_index + 1 == len(jobs)
                or (job_index + 1) % 100 == 0
            )
            if should_print:
                print(
                    f"[{job_index + 1}/{len(jobs)}] {trial.trial_id} stage={job['stage']} "
                    f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)

    results.sort(key=lambda row: (row["trial_id"], row["stage"], row["method"], row["memory_frames"], row["encode_fps"]))
    atomic_jsonl(args.output_dir / "results.jsonl", results)
    atomic_jsonl(args.output_dir / "failures.jsonl", failures)
    final_rows = [row for row in results if row["stage"] == "final_memory"]
    summary = aggregate_results(final_rows) if final_rows else {"format_version": 1, "configurations": []}
    summary.update(
        {
            "results": len(results),
            "new_jobs_requested": len(jobs),
            "failures": len(failures),
            "runtime_seconds": time.monotonic() - started,
        }
    )
    atomic_json(args.output_dir / "summary.json", summary)
    atomic_json(
        args.output_dir / "inference_run.json",
        {
            "format_version": 1,
            "selection_index": str(args.selection_index.resolve()),
            "selection_index_sha256": file_sha256(args.selection_index),
            "frame_store": str(store.root),
            "predictor": args.predictor,
            "model_fingerprint": model_fingerprint,
            "prompt_version": prompt_version,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
            "max_new_tokens": args.max_new_tokens,
            "cpu_threads": args.cpu_threads,
            "interop_threads": args.interop_threads,
            "query_timeout_sec": args.query_timeout_sec,
            "raw_video_replay_at_query": False,
        },
    )
    print(f"wrote {len(results)} predictions and {len(failures)} failures to {args.output_dir}")


if __name__ == "__main__":
    main()
