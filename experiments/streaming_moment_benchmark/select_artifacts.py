"""Decode streams once and freeze causal memory selections for later inference."""
from __future__ import annotations

import argparse
import hashlib
import math
import time
from pathlib import Path
from typing import Any

from .artifacts import (
    ARTIFACT_FORMAT_VERSION,
    FRAME_CODEC,
    FrameStore,
    canonical_sha256,
    load_artifact,
    load_oracle_stages,
    media_fingerprints,
    selection_cache_key,
    write_artifact,
    write_oracle_stages,
)
from .frames import FEATURE_NAMES, FrameObservation, FrameRef, iter_trial_frames
from .metrics import selection_metrics
from .protocol import StreamTrial, atomic_json, atomic_jsonl, file_sha256, load_trials
from .selection import METHODS, OnlineMemorySelector
from .utility import UtilityModel


def _stable_seed(seed: int, *parts: object) -> int:
    digest = hashlib.sha256(":".join([str(seed), *(str(item) for item in parts)]).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _overlaps_evidence(ref: FrameRef, trial: StreamTrial) -> bool:
    return any(span.start_sec - 1e-6 <= ref.timestamp_sec <= span.end_sec + 1e-6 for span in trial.evidence_spans)


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


def _artifact_path(root: Path, key: str) -> Path:
    return root / "selections" / key[:2] / f"{key}.json"


def _valid_cached(path: Path, key: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        artifact = load_artifact(path)
    except (OSError, ValueError, KeyError):
        return None
    return artifact if artifact.get("cache_key") == key else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-store", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS[:-1]))
    parser.add_argument("--memory-frames", type=int, nargs="+", default=[16, 64, 256])
    parser.add_argument("--encode-fps", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--directory-fps", type=float, default=30.0)
    parser.add_argument("--oracle-max-frames", type=int, default=256)
    parser.add_argument("--utility-model", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--split", nargs="+")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-missing-media", action="store_true")
    parser.add_argument("--export-features", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if "future_utility" in args.methods and args.utility_model is None:
        parser.error("future_utility requires --utility-model")
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
        raise ValueError("utility model feature schema does not match runtime features")
    utility_sha = file_sha256(args.utility_model) if args.utility_model else None
    store = FrameStore(args.frame_store)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    media_cache: dict[str, dict[str, Any]] = {}
    index_rows: list[dict[str, Any]] = []
    feature_index_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.monotonic()
    rows_produced = 0

    for trial_index, trial in enumerate(trials):
        try:
            media = media_fingerprints(trial, media_cache)
        except (FileNotFoundError, OSError) as error:
            if not args.skip_missing_media:
                raise
            failures.append({"trial_id": trial.trial_id, "stage": "fingerprint", "error": str(error)})
            continue
        feature_key = canonical_sha256(
            {
                "format_version": 1,
                "feature_names": FEATURE_NAMES,
                "trial": trial.to_dict(),
                "media": media,
                "directory_fps": args.directory_fps,
            }
        )
        feature_path = args.output_dir / "features" / feature_key[:2] / f"{feature_key}.jsonl"
        need_features = bool(args.export_features and (args.no_resume or not feature_path.exists()))
        if args.export_features and not need_features:
            feature_index_rows.append(
                {"trial_id": trial.trial_id, "feature_key": feature_key, "path": str(feature_path.resolve()), "cached": True}
            )
        configs: list[dict[str, Any]] = []
        for method in args.methods:
            for memory_frames in args.memory_frames:
                for encode_fps in args.encode_fps:
                    selector_seed = _stable_seed(args.seed, trial.trial_id, method, memory_frames, encode_fps)
                    key = selection_cache_key(
                        trial,
                        media=media,
                        method=method,
                        memory_frames=memory_frames,
                        encode_fps=encode_fps,
                        window_sec=args.window_sec,
                        seed=selector_seed,
                        oracle_max_frames=args.oracle_max_frames,
                        directory_fps=args.directory_fps,
                        utility_model_sha256=utility_sha,
                    )
                    path = _artifact_path(args.output_dir, key)
                    cached = None if args.no_resume else _valid_cached(path, key)
                    if cached is not None:
                        index_rows.append(
                            {
                                "cache_key": key,
                                "artifact_sha256": cached["artifact_sha256"],
                                "path": str(path.resolve()),
                                "trial_id": trial.trial_id,
                                "method": method,
                                "memory_frames": memory_frames,
                                "encode_fps": encode_fps,
                                "cached": True,
                            }
                        )
                    else:
                        configs.append(
                            {
                                "method": method,
                                "memory_frames": memory_frames,
                                "encode_fps": encode_fps,
                                "selector_seed": selector_seed,
                                "cache_key": key,
                                "path": path,
                            }
                        )
        if configs or need_features:
            selectors = [
                OnlineMemorySelector(
                    item["method"],
                    memory_frames=item["memory_frames"],
                    encode_fps=item["encode_fps"],
                    window_sec=args.window_sec,
                    seed=item["selector_seed"],
                    utility_model=utility_model,
                )
                for item in configs
            ]
            observed_refs: list[FrameRef] = []
            dense = _UniformCapCollector(args.oracle_max_frames, _stable_seed(args.seed, trial.trial_id, "dense"))
            critical = _UniformCapCollector(
                args.oracle_max_frames,
                _stable_seed(args.seed, trial.trial_id, "critical-reservoir-v1"),
            )
            current_window: list[FrameObservation] = []
            current_window_index: int | None = None
            feature_rows: list[dict[str, Any]] = []

            def flush() -> None:
                nonlocal current_window
                if not current_window:
                    return
                for selector in selectors:
                    selector.ingest_window(current_window)
                current_window = []

            trial_started = time.monotonic()
            try:
                for observation in iter_trial_frames(trial, directory_fps=args.directory_fps):
                    observed_refs.append(observation.ref)
                    dense.add(observation)
                    if _overlaps_evidence(observation.ref, trial):
                        critical.add(observation)
                    if need_features:
                        feature_rows.append(
                            {
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
                        )
                    window_index = math.floor(observation.ref.timestamp_sec / args.window_sec + 1e-9)
                    if current_window_index is None:
                        current_window_index = window_index
                    if window_index != current_window_index:
                        flush()
                        current_window_index = window_index
                    current_window.append(observation)
                flush()
                if need_features:
                    atomic_jsonl(feature_path, feature_rows)
                    feature_index_rows.append(
                        {
                            "trial_id": trial.trial_id,
                            "feature_key": feature_key,
                            "path": str(feature_path.resolve()),
                            "cached": False,
                        }
                    )
                dense_frames = (
                    [store.put(item).to_dict() for item in sorted(dense.items, key=lambda x: x.ref.timestamp_sec)]
                    if configs
                    else []
                )
                critical_frames = (
                    [
                        store.put(item).to_dict()
                        for item in sorted(critical.items, key=lambda x: x.ref.timestamp_sec)
                    ]
                    if configs
                    else []
                )
                oracle_reference = None
                if configs:
                    oracle_key = canonical_sha256(
                        {
                            "format_version": ARTIFACT_FORMAT_VERSION,
                            "frame_codec": FRAME_CODEC,
                            "trial": trial.to_dict(),
                            "media": media,
                            "oracle_max_frames": args.oracle_max_frames,
                            "critical_sampling": "reservoir-v1",
                        }
                    )
                    oracle_path = args.output_dir / "oracle_stages" / oracle_key[:2] / f"{oracle_key}.json"
                    try:
                        oracle_artifact = load_oracle_stages(oracle_path) if oracle_path.exists() else None
                    except (OSError, ValueError, KeyError):
                        oracle_artifact = None
                    if oracle_artifact is None:
                        oracle_artifact = write_oracle_stages(
                            oracle_path,
                            {
                                "cache_key": oracle_key,
                                "trial_id": trial.trial_id,
                                "source_media": media,
                                "stages": {
                                    "dense_offline_cap": dense_frames,
                                    "critical_only": critical_frames,
                                },
                                "critical_sampling": "reservoir-v1",
                            },
                        )
                    oracle_reference = {
                        "path": str(oracle_path.resolve()),
                        "artifact_sha256": oracle_artifact["artifact_sha256"],
                    }
                for config, selector in zip(configs, selectors):
                    result = selector.result()
                    final_frames = [store.put(item).to_dict() for item in result.final_observations]
                    retrieval = [
                        stored
                        for stored, observation in zip(final_frames, result.final_observations)
                        if _overlaps_evidence(observation.ref, trial)
                    ]
                    artifact = {
                        "format_version": ARTIFACT_FORMAT_VERSION,
                        "artifact_type": "frozen_selection",
                        "cache_key": config["cache_key"],
                        "frame_codec": FRAME_CODEC,
                        "trial": trial.to_dict(),
                        "source_media": media,
                        "selection_config": {
                            "method": result.method,
                            "memory_frames": result.memory_frames,
                            "encode_fps": result.encode_fps,
                            "window_sec": result.window_sec,
                            "seed": config["selector_seed"],
                            "utility_model_sha256": utility_sha,
                        },
                        "selection": {
                            "frames_seen": result.frames_seen,
                            "frames_admitted": result.frames_admitted,
                            "captured_frames": [item.to_dict() for item in result.captured],
                            "evicted_frames": [item.to_dict() for item in result.evicted],
                            "final_frames": final_frames,
                        },
                        "oracle_stages": {"oracle_retrieval": retrieval},
                        "oracle_stages_ref": oracle_reference,
                        "metrics": selection_metrics(trial, result, observed_refs),
                        "selection_latency_sec": time.monotonic() - trial_started,
                        "runtime_query_visible_to_selector": False,
                        "raw_video_replay_at_query": False,
                    }
                    write_artifact(config["path"], artifact)
                    loaded = load_artifact(config["path"])
                    index_rows.append(
                        {
                            "cache_key": config["cache_key"],
                            "artifact_sha256": loaded["artifact_sha256"],
                            "path": str(config["path"].resolve()),
                            "trial_id": trial.trial_id,
                            "method": result.method,
                            "memory_frames": result.memory_frames,
                            "encode_fps": result.encode_fps,
                            "cached": False,
                        }
                    )
                    rows_produced += 1
            except (FileNotFoundError, OSError, ValueError) as error:
                if not args.skip_missing_media:
                    raise
                failures.append({"trial_id": trial.trial_id, "stage": "selection", "error": str(error)})
        elapsed = time.monotonic() - started
        rate = (trial_index + 1) / elapsed if elapsed else 0.0
        eta = (len(trials) - trial_index - 1) / rate if rate else 0.0
        print(
            f"[{trial_index + 1}/{len(trials)}] {trial.trial_id} "
            f"new={len(configs)} elapsed={elapsed:.1f}s eta={eta:.1f}s",
            flush=True,
        )

    index_rows.sort(key=lambda row: (row["trial_id"], row["method"], row["memory_frames"], row["encode_fps"]))
    atomic_jsonl(args.output_dir / "selection_index.jsonl", index_rows)
    if args.export_features:
        feature_index_rows.sort(key=lambda row: row["trial_id"])
        atomic_jsonl(args.output_dir / "feature_index.jsonl", feature_index_rows)
    atomic_jsonl(args.output_dir / "failures.jsonl", failures)
    atomic_json(
        args.output_dir / "selection_run.json",
        {
            "format_version": ARTIFACT_FORMAT_VERSION,
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": file_sha256(args.manifest),
            "frame_store": str(store.root),
            "methods": args.methods,
            "memory_frames": args.memory_frames,
            "encode_fps": args.encode_fps,
            "window_sec": args.window_sec,
            "utility_model_sha256": utility_sha,
            "trials_requested": len(trials),
            "artifacts_indexed": len(index_rows),
            "artifacts_new": rows_produced,
            "feature_artifacts": len(feature_index_rows),
            "failures": len(failures),
            "runtime_seconds": time.monotonic() - started,
            "runtime_query_visible_to_selector": False,
            "raw_video_replay_at_query": False,
        },
    )
    print(f"indexed {len(index_rows)} frozen selections in {args.output_dir}")


if __name__ == "__main__":
    main()
