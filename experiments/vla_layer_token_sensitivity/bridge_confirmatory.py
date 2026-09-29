"""Build a held-out, task-balanced Bridge horizon-confirmation manifest."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from experiments.vla_layer_token_sensitivity.bridge_subset import (
    CAMERA_KEY,
    DEFAULT_REPO_ID,
    _atomic_jsonl,
    _decode_nearest_frames,
    _download_subset,
    _episode_table,
    _selected_data,
    _task_mapping,
)
from experiments.vla_layer_token_sensitivity.qwen_oft_horizon_data import (
    load_bridge_statistics,
    normalize_bridge_actions,
    normalize_bridge_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--frame-index", type=int, default=4)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--max-retries", type=int, default=50)
    return parser.parse_args()


def excluded_episode_ids(path: Path) -> set[int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {int(row["episode_index"]) for row in rows}


def select_balanced_episodes(
    episodes: pd.DataFrame,
    *,
    count: int,
    seed: int,
    excluded: set[int],
    minimum_length: int,
) -> pd.DataFrame:
    """Select episodes round-robin across task text with deterministic shuffling."""

    if count <= 0:
        raise ValueError("episode count must be positive")
    candidates = episodes.copy()
    candidates["instruction"] = candidates["tasks"].map(
        lambda tasks: str(tasks[0] if isinstance(tasks, (list, tuple, np.ndarray)) else tasks).strip()
    )
    candidates = candidates[
        (~candidates["episode_index"].isin(excluded))
        & (candidates["length"] >= minimum_length)
        & (candidates["instruction"] != "")
    ]
    if len(candidates) < count:
        raise ValueError(f"only {len(candidates)} eligible held-out episodes are available")

    rng = np.random.default_rng(seed)
    tasks = sorted(candidates["instruction"].unique().tolist())
    pools: dict[str, list[int]] = {}
    for task in tasks:
        indices = candidates.index[candidates["instruction"] == task].to_numpy(dtype=np.int64).copy()
        rng.shuffle(indices)
        pools[task] = indices.tolist()

    selected: list[int] = []
    while len(selected) < count:
        progressed = False
        for task in tasks:
            if pools[task] and len(selected) < count:
                selected.append(pools[task].pop())
                progressed = True
        if not progressed:
            break
    if len(selected) != count:
        raise RuntimeError("task-balanced selection could not satisfy requested count")
    return candidates.loc[selected].sort_values("episode_index").reset_index(drop=True)


def build_confirmatory_subset(
    *,
    repo_id: str,
    output_dir: Path,
    statistics_path: Path,
    exclude_manifest: Path,
    episode_count: int,
    frame_index: int,
    action_horizon: int,
    seed: int,
    max_retries: int = 50,
) -> Path:
    if frame_index < 0 or action_horizon <= 0:
        raise ValueError("frame-index must be non-negative and action-horizon positive")
    paths = _download_subset(repo_id, max_retries=max_retries)
    minimum_length = frame_index + action_horizon
    episodes = select_balanced_episodes(
        _episode_table(paths["meta/episodes/chunk-000/file-000.parquet"]),
        count=episode_count,
        seed=seed,
        excluded=excluded_episode_ids(exclude_manifest),
        minimum_length=minimum_length,
    )
    episode_ids = [int(value) for value in episodes["episode_index"]]
    frames = _selected_data(paths["data/chunk-000/file-000.parquet"], episode_ids)
    tasks = _task_mapping(pq.read_table(paths["meta/tasks.parquet"]))
    statistics = load_bridge_statistics(statistics_path)
    video_prefix = f"videos/{CAMERA_KEY}"

    specs: list[dict] = []
    for episode in episodes.to_dict(orient="records"):
        episode_id = int(episode["episode_index"])
        episode_frames = frames[frames["episode_index"] == episode_id].set_index("frame_index")
        required_indices = list(range(frame_index, frame_index + action_horizon))
        missing = [index for index in required_indices if index not in episode_frames.index]
        if missing:
            raise RuntimeError(f"episode {episode_id} is missing future frames {missing}")
        current = episode_frames.loc[frame_index]
        if isinstance(current, pd.DataFrame):
            raise RuntimeError(f"episode {episode_id} has duplicate frame {frame_index}")
        future = episode_frames.loc[required_indices]
        raw_state = np.asarray(current["observation.state"], dtype=np.float32)
        raw_actions = np.stack(
            [np.asarray(value, dtype=np.float32) for value in future["action"].tolist()]
        )
        normalized_state = normalize_bridge_state(raw_state, statistics)
        normalized_actions = normalize_bridge_actions(raw_actions, statistics)
        task_index = int(current["task_index"])
        instruction = tasks.get(task_index, str(episode["instruction"]))
        target_time = float(episode[f"{video_prefix}/from_timestamp"]) + float(current["timestamp"])
        specs.append(
            {
                "sample_id": f"confirm_ep{episode_id:05d}_f{frame_index:04d}",
                "split": "confirmatory",
                "episode_index": episode_id,
                "task_index": task_index,
                "frame_index": frame_index,
                "trajectory_fraction": float(frame_index / (int(episode["length"]) - 1)),
                "instruction": instruction,
                "raw_state": raw_state.tolist(),
                "normalized_state": normalized_state.tolist(),
                "raw_action_chunk": raw_actions.tolist(),
                "normalized_action_chunk": normalized_actions.tolist(),
                "action_frame_indices": required_indices,
                "target_video_timestamp": target_time,
            }
        )

    by_time = sorted(specs, key=lambda row: row["target_video_timestamp"])
    decoded = _decode_nearest_frames(
        paths[f"videos/{CAMERA_KEY}/chunk-000/file-000.mp4"],
        [float(row["target_video_timestamp"]) for row in by_time],
    )
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for row, (image, decoded_time) in zip(by_time, decoded):
        image_path = (frame_dir / f"{row['sample_id']}.jpg").resolve()
        image.save(image_path, format="JPEG", quality=95, subsampling=0)
        row["image_path"] = str(image_path)
        row["decoded_video_timestamp"] = float(decoded_time)
        row["video_timestamp_error"] = abs(float(decoded_time) - row["target_video_timestamp"])

    specs.sort(key=lambda row: row["episode_index"])
    manifest = output_dir / "manifest.jsonl"
    _atomic_jsonl(manifest, specs)
    summary = {
        "schema_version": 2,
        "split": "confirmatory",
        "seed": seed,
        "sample_count": len(specs),
        "episode_count": len({row["episode_index"] for row in specs}),
        "episode_indices": [row["episode_index"] for row in specs],
        "excluded_episode_indices": sorted(excluded_episode_ids(exclude_manifest)),
        "task_counts": {
            task: sum(row["instruction"] == task for row in specs)
            for task in sorted({row["instruction"] for row in specs})
        },
        "frame_index": frame_index,
        "action_horizon": action_horizon,
        "statistics": str(statistics_path.resolve()),
        "source_files": {key: str(value.resolve()) for key, value in paths.items()},
        "manifest": str(manifest.resolve()),
        "max_video_timestamp_error": max(row["video_timestamp_error"] for row in specs),
    }
    summary_path = output_dir / "summary.json"
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(summary_path)
    return manifest


def main() -> None:
    args = parse_args()
    manifest = build_confirmatory_subset(
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        statistics_path=args.statistics,
        exclude_manifest=args.exclude_manifest,
        episode_count=args.episodes,
        frame_index=args.frame_index,
        action_horizon=args.action_horizon,
        seed=args.seed,
        max_retries=args.max_retries,
    )
    print(manifest.resolve())


if __name__ == "__main__":
    main()
