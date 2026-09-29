"""Label query-agnostic scout frames by drop-one-window correct-answer log-probability."""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import (
    FrameStore,
    artifact_observations,
    artifact_trial,
    canonical_sha256,
    load_artifact,
    path_fingerprint,
)
from .predictors import QwenMultiImagePredictor
from .protocol import atomic_json, atomic_jsonl, file_sha256


class QueryTimeoutError(TimeoutError):
    pass


def _timeout(_signum: int, _frame: object) -> None:
    raise QueryTimeoutError("counterfactual model query timed out")


def _load_unique_artifacts(index_path: Path, splits: set[str]) -> list[tuple[Path, dict[str, Any]]]:
    candidates: dict[str, tuple[Path, dict[str, Any]]] = {}
    with index_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            path = Path(row["path"])
            artifact = load_artifact(path)
            trial = artifact_trial(artifact)
            if trial.split not in splits or trial.question.answer_type != "multiple_choice":
                continue
            current = candidates.get(trial.trial_id)
            order = (
                int(artifact["selection_config"]["memory_frames"]),
                float(artifact["selection_config"]["encode_fps"]),
                str(artifact["selection_config"]["method"]),
            )
            if current is None:
                candidates[trial.trial_id] = (path, artifact)
            else:
                old = current[1]
                old_order = (
                    int(old["selection_config"]["memory_frames"]),
                    float(old["selection_config"]["encode_fps"]),
                    str(old["selection_config"]["method"]),
                )
                if order < old_order:
                    candidates[trial.trial_id] = (path, artifact)
    return [candidates[key] for key in sorted(candidates)]


def _cached_rows(path: Path, key: str) -> list[dict[str, Any]] | None:
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    except (OSError, json.JSONDecodeError):
        return None
    return rows if rows and all(row.get("label_cache_key") == key for row in rows) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-index", type=Path, required=True)
    parser.add_argument("--frame-store", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", nargs="+", default=["dev"])
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--max-context-frames", type=int, default=32)
    parser.add_argument("--min-pixels", type=int)
    parser.add_argument("--max-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--query-timeout-sec", type=int, default=600)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if min(args.window_sec, args.max_context_frames, args.cpu_threads, args.interop_threads, args.query_timeout_sec) <= 0:
        parser.error("window and resource limits must be positive")
    os.environ.setdefault("OMP_NUM_THREADS", str(args.cpu_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.cpu_threads))
    artifacts = _load_unique_artifacts(args.selection_index, set(args.split))
    if args.max_samples is not None:
        artifacts = artifacts[: args.max_samples]
    if not artifacts:
        raise ValueError("no closed dev artifacts found")
    model_fingerprint = path_fingerprint(args.model_path)
    jobs = []
    combined: list[dict[str, Any]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path, artifact in artifacts:
        key = canonical_sha256(
            {
                "format_version": 1,
                "artifact_sha256": artifact["artifact_sha256"],
                "stage": "dense_offline_cap",
                "model": model_fingerprint,
                "prompt_version": QwenMultiImagePredictor.PROMPT_VERSION,
                "window_sec": args.window_sec,
                "max_context_frames": args.max_context_frames,
                "min_pixels": args.min_pixels,
                "max_pixels": args.max_pixels,
            }
        )
        destination = args.output_dir / "labels" / key[:2] / f"{key}.jsonl"
        cached = None if args.no_resume else _cached_rows(destination, key)
        if cached is None:
            jobs.append((path, artifact, key, destination))
        else:
            combined.extend(cached)

    predictor = QwenMultiImagePredictor(
        args.model_path,
        device="cpu",
        max_new_tokens=4,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        cpu_threads=args.cpu_threads,
        interop_threads=args.interop_threads,
    ) if jobs else None
    store = FrameStore(args.frame_store)
    failures = []
    started = time.monotonic()
    previous = signal.signal(signal.SIGALRM, _timeout)
    try:
        for job_index, (path, artifact, key, destination) in enumerate(jobs):
            trial = artifact_trial(artifact)
            try:
                frames = artifact_observations(artifact, store, "dense_offline_cap")
                if len(frames) > args.max_context_frames:
                    indices = np.linspace(0, len(frames) - 1, args.max_context_frames, dtype=int)
                    frames = [frames[int(index)] for index in indices]
                by_window: dict[int, list[Any]] = defaultdict(list)
                for frame in frames:
                    by_window[math.floor(frame.ref.timestamp_sec / args.window_sec + 1e-9)].append(frame)
                windows = sorted(by_window)
                if args.max_windows is not None:
                    windows = windows[: args.max_windows]
                assert predictor is not None
                signal.setitimer(signal.ITIMER_REAL, args.query_timeout_sec)
                full = predictor.correct_answer_logprob(trial.question, frames)
                signal.setitimer(signal.ITIMER_REAL, 0)
                rows = []
                for window_index, window in enumerate(windows):
                    retained = [
                        frame
                        for candidate_window, window_frames in by_window.items()
                        if candidate_window != window
                        for frame in window_frames
                    ]
                    drop_started = time.monotonic()
                    signal.setitimer(signal.ITIMER_REAL, args.query_timeout_sec)
                    dropped = predictor.correct_answer_logprob(trial.question, retained)
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    drop_latency = time.monotonic() - drop_started
                    utility = max(0.0, full - dropped)
                    for frame in by_window[window]:
                        rows.append(
                            {
                                "format_version": 1,
                                "label_cache_key": key,
                                "trial_id": trial.trial_id,
                                "sample_id": trial.sample_id,
                                "source_group": trial.source_group,
                                "frame_id": frame.ref.frame_id,
                                "window_index": window,
                                "window_sec": args.window_sec,
                                "full_correct_logprob": full,
                                "dropped_correct_logprob": dropped,
                                "utility": utility,
                                "drop_query_latency_sec": drop_latency,
                                "selection_artifact": str(path.resolve()),
                            }
                        )
                    print(
                        f"  {trial.trial_id} window {window_index + 1}/{len(windows)} "
                        f"latency={drop_latency:.1f}s utility={utility:.6f}",
                        flush=True,
                    )
                atomic_jsonl(destination, rows)
                combined.extend(rows)
            except (OSError, ValueError, RuntimeError, QueryTimeoutError, KeyError) as error:
                signal.setitimer(signal.ITIMER_REAL, 0)
                failures.append(
                    {"trial_id": trial.trial_id, "error": f"{type(error).__name__}: {error}", "retryable": True}
                )
            elapsed = time.monotonic() - started
            rate = (job_index + 1) / elapsed if elapsed else 0.0
            eta = (len(jobs) - job_index - 1) / rate if rate else 0.0
            print(f"[{job_index + 1}/{len(jobs)}] {trial.trial_id} elapsed={elapsed:.1f}s eta={eta:.1f}s", flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    combined.sort(key=lambda row: (row["trial_id"], row["window_index"], row["frame_id"]))
    atomic_jsonl(args.output_dir / "counterfactual_labels.jsonl", combined)
    atomic_jsonl(args.output_dir / "failures.jsonl", failures)
    atomic_json(
        args.output_dir / "label_run.json",
        {
            "format_version": 1,
            "selection_index": str(args.selection_index.resolve()),
            "selection_index_sha256": file_sha256(args.selection_index),
            "model_fingerprint": model_fingerprint,
            "window_sec": args.window_sec,
            "max_context_frames": args.max_context_frames,
            "samples": len(artifacts),
            "new_jobs": len(jobs),
            "labels": len(combined),
            "failures": len(failures),
            "runtime_seconds": time.monotonic() - started,
            "runtime_query_visible_to_selector": False,
        },
    )
    print(f"wrote {len(combined)} counterfactual frame labels to {args.output_dir}")


if __name__ == "__main__":
    main()
