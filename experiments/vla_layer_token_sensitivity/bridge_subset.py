"""Download and extract a deterministic, tiny BridgeData2 calibration set.

Only the first metadata/data/video shard is requested from Hugging Face.  The
result is a JSONL manifest plus 32 JPEG observations (eight episodes at four
trajectory fractions), not a local copy of the full dataset.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Sequence
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


DEFAULT_REPO_ID = "nvidia/bridge_lerobot_v3"
CAMERA_KEY = "observation.images.image_0"
SUBSET_FILES = (
    "data/chunk-000/file-000.parquet",
    "meta/episodes/chunk-000/file-000.parquet",
    "meta/tasks.parquet",
    f"videos/{CAMERA_KEY}/chunk-000/file-000.mp4",
)
DEFAULT_FRACTIONS = (0.2, 0.4, 0.6, 0.8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/vla_layer_token_sensitivity/bridge_subset"),
    )
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--fractions", default="0.2,0.4,0.6,0.8")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-retries", type=int, default=50)
    return parser.parse_args()


def fraction_frame_indices(length: int, fractions: Sequence[float]) -> list[int]:
    """Map trajectory fractions to distinct, zero-based frame indices."""

    if length <= 0:
        raise ValueError("episode length must be positive")
    parsed = [float(value) for value in fractions]
    if not parsed or any(not 0.0 <= value <= 1.0 for value in parsed):
        raise ValueError("fractions must be non-empty and within [0, 1]")
    indices = [int(round(value * (length - 1))) for value in parsed]
    if len(set(indices)) != len(indices):
        raise ValueError(f"episode length {length} cannot provide distinct requested frames")
    return indices


def _instruction(tasks: object) -> str:
    if isinstance(tasks, np.ndarray):
        tasks = tasks.tolist()
    if isinstance(tasks, (list, tuple)) and tasks:
        value = str(tasks[0]).strip()
    else:
        value = str(tasks).strip()
    if not value:
        raise ValueError("episode has an empty task instruction")
    return value


def select_episodes(episodes: pd.DataFrame, *, count: int, seed: int) -> pd.DataFrame:
    """Select one episode per task when possible, deterministically."""

    if count <= 0:
        raise ValueError("episode count must be positive")
    required = {"episode_index", "tasks", "length"}
    missing = required - set(episodes.columns)
    if missing:
        raise ValueError(f"episode metadata is missing columns: {sorted(missing)}")

    candidates = episodes.copy()
    candidates["instruction"] = candidates["tasks"].map(_instruction)
    candidates = candidates[candidates["length"] >= 8]
    if len(candidates) < count:
        raise ValueError(f"only {len(candidates)} usable episodes are available")

    rng = np.random.default_rng(seed)
    task_names = np.array(sorted(candidates["instruction"].unique()), dtype=object)
    rng.shuffle(task_names)
    selected_indices: list[int] = []
    for task in task_names:
        group = candidates[candidates["instruction"] == task]
        choices = np.array(sorted(group.index.tolist()), dtype=np.int64)
        selected_indices.append(int(rng.choice(choices)))
        if len(selected_indices) == count:
            break

    if len(selected_indices) < count:
        remaining = candidates.drop(index=selected_indices)
        choices = np.array(sorted(remaining.index.tolist()), dtype=np.int64)
        extra = rng.choice(choices, size=count - len(selected_indices), replace=False)
        selected_indices.extend(int(value) for value in extra)

    return candidates.loc[selected_indices].sort_values("episode_index").reset_index(drop=True)


def _download_subset(repo_id: str, *, max_retries: int) -> dict[str, Path]:
    if max_retries <= 0:
        raise ValueError("max_retries must be positive")
    paths: dict[str, Path] = {}
    for filename in SUBSET_FILES:
        for attempt in range(1, max_retries + 1):
            try:
                print(f"download {filename} (attempt {attempt})", flush=True)
                paths[filename] = Path(
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        repo_type="dataset",
                    )
                )
                break
            except KeyboardInterrupt:
                raise
            except Exception:
                if attempt == max_retries:
                    raise
                time.sleep(5)
    return paths


def _episode_table(path: Path) -> pd.DataFrame:
    video_prefix = f"videos/{CAMERA_KEY}"
    columns = [
        "episode_index",
        "tasks",
        "length",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
        "dataset_to_index",
        f"{video_prefix}/chunk_index",
        f"{video_prefix}/file_index",
        f"{video_prefix}/from_timestamp",
        f"{video_prefix}/to_timestamp",
    ]
    table = pq.read_table(path, columns=columns).to_pandas()
    mask = (
        (table["data/chunk_index"] == 0)
        & (table["data/file_index"] == 0)
        & (table[f"{video_prefix}/chunk_index"] == 0)
        & (table[f"{video_prefix}/file_index"] == 0)
    )
    selected = table[mask].copy()
    if selected.empty:
        raise RuntimeError("the requested data/video shards share no episodes")
    return selected


def _selected_data(path: Path, episode_ids: Sequence[int]) -> pd.DataFrame:
    columns = [
        "observation.state",
        "action",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    ]
    table = pq.read_table(path, columns=columns)
    mask = pc.is_in(table["episode_index"], value_set=pa.array(list(episode_ids), type=pa.int64()))
    selected = table.filter(mask).to_pandas()
    if selected.empty:
        raise RuntimeError("selected episode IDs have no frame records")
    return selected


def _task_mapping(table: pa.Table) -> dict[int, str]:
    """Read task text without allowing pandas index metadata to hide it.

    The published tasks parquet marks ``task`` as its pandas index.  Calling
    ``to_pandas()`` therefore removes it from the DataFrame columns even though
    it remains a real Arrow column.  Reading the two Arrow arrays directly is
    stable for both indexed and ordinary parquet metadata.
    """

    required = {"task_index", "task"}
    missing = required - set(table.column_names)
    if missing:
        raise ValueError(f"task metadata is missing columns: {sorted(missing)}")
    mapping: dict[int, str] = {}
    for task_index, task in zip(table["task_index"].to_pylist(), table["task"].to_pylist()):
        if task is None:
            continue
        instruction = str(task).strip()
        if not instruction:
            # Some published rows are intentionally blank.  Omitting them
            # lets the sample builder fall back to the non-empty instruction
            # stored in the selected episode metadata.
            continue
        mapping[int(task_index)] = instruction
    return mapping


def _frame_time(frame: av.VideoFrame, stream: av.video.stream.VideoStream) -> float:
    if frame.time is not None:
        return float(frame.time)
    if frame.pts is None:
        raise RuntimeError("decoded video frame has neither time nor PTS")
    return float(frame.pts * stream.time_base)


def _decode_nearest_frames(video_path: Path, targets: Sequence[float]) -> list[tuple[object, float]]:
    """Seek and decode the frame nearest each absolute video timestamp."""

    decoded: list[tuple[object, float]] = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for target in targets:
            seek_time = max(0.0, float(target) - 2.0)
            offset = int(seek_time / float(stream.time_base))
            container.seek(offset, stream=stream, any_frame=False, backward=True)
            best_frame = None
            best_time = math.nan
            best_delta = math.inf
            for frame in container.decode(stream):
                timestamp = _frame_time(frame, stream)
                delta = abs(timestamp - target)
                if delta < best_delta:
                    best_frame = frame
                    best_time = timestamp
                    best_delta = delta
                if timestamp >= target and best_frame is not None:
                    break
                if timestamp > target + 2.0:
                    break
            if best_frame is None or best_delta > 0.25:
                raise RuntimeError(
                    f"could not decode frame near t={target:.3f}s; nearest delta={best_delta:.3f}s"
                )
            decoded.append((best_frame.to_image().convert("RGB"), best_time))
    return decoded


def _atomic_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def build_subset(
    *,
    repo_id: str,
    output_dir: Path,
    episode_count: int,
    fractions: Sequence[float],
    seed: int,
    max_retries: int = 50,
) -> Path:
    paths = _download_subset(repo_id, max_retries=max_retries)
    episode_rows = select_episodes(
        _episode_table(paths["meta/episodes/chunk-000/file-000.parquet"]),
        count=episode_count,
        seed=seed,
    )
    episode_ids = [int(value) for value in episode_rows["episode_index"]]
    frame_rows = _selected_data(paths["data/chunk-000/file-000.parquet"], episode_ids)
    task_by_index = _task_mapping(pq.read_table(paths["meta/tasks.parquet"]))

    video_prefix = f"videos/{CAMERA_KEY}"
    specs: list[dict] = []
    for episode in episode_rows.to_dict(orient="records"):
        episode_index = int(episode["episode_index"])
        length = int(episode["length"])
        indices = fraction_frame_indices(length, fractions)
        episode_frames = frame_rows[frame_rows["episode_index"] == episode_index].set_index("frame_index")
        from_timestamp = float(episode[f"{video_prefix}/from_timestamp"])
        episode_instruction = str(episode["instruction"])
        for fraction, frame_index in zip(fractions, indices):
            if frame_index not in episode_frames.index:
                raise RuntimeError(f"episode {episode_index} is missing frame {frame_index}")
            frame = episode_frames.loc[frame_index]
            if isinstance(frame, pd.DataFrame):
                raise RuntimeError(f"episode {episode_index} has duplicate frame {frame_index}")
            task_index = int(frame["task_index"])
            instruction = task_by_index.get(task_index, episode_instruction)
            specs.append(
                {
                    "episode_index": episode_index,
                    "frame_index": frame_index,
                    "trajectory_fraction": float(fraction),
                    "instruction": instruction,
                    "episode_instruction": episode_instruction,
                    "task_index": task_index,
                    "action": np.asarray(frame["action"], dtype=np.float32).tolist(),
                    "state": np.asarray(frame["observation.state"], dtype=np.float32).tolist(),
                    "data_index": int(frame["index"]),
                    "target_video_timestamp": from_timestamp + float(frame["timestamp"]),
                }
            )

    specs.sort(key=lambda row: row["target_video_timestamp"])
    decoded = _decode_nearest_frames(
        paths[f"videos/{CAMERA_KEY}/chunk-000/file-000.mp4"],
        [float(row["target_video_timestamp"]) for row in specs],
    )
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for row, (image, decoded_timestamp) in zip(specs, decoded):
        sample_id = f"ep{row['episode_index']:05d}_f{row['frame_index']:04d}"
        image_path = (frame_dir / f"{sample_id}.jpg").resolve()
        image.save(image_path, format="JPEG", quality=95, subsampling=0)
        row["sample_id"] = sample_id
        row["image_path"] = str(image_path)
        row["decoded_video_timestamp"] = float(decoded_timestamp)
        row["video_timestamp_error"] = float(abs(decoded_timestamp - row["target_video_timestamp"]))

    specs.sort(key=lambda row: (row["episode_index"], row["frame_index"]))
    manifest_path = output_dir / "manifest.jsonl"
    _atomic_jsonl(manifest_path, specs)
    summary = {
        "repo_id": repo_id,
        "seed": seed,
        "episode_count": episode_count,
        "sample_count": len(specs),
        "fractions": [float(value) for value in fractions],
        "episode_indices": sorted(set(row["episode_index"] for row in specs)),
        "instructions": sorted(set(row["instruction"] for row in specs)),
        "source_files": {key: str(value.resolve()) for key, value in paths.items()},
        "manifest": str(manifest_path.resolve()),
    }
    summary_path = output_dir / "summary.json"
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(summary_path)
    return manifest_path


def main() -> None:
    args = parse_args()
    fractions = tuple(float(value) for value in args.fractions.split(",") if value.strip())
    manifest = build_subset(
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        episode_count=args.episodes,
        fractions=fractions,
        seed=args.seed,
        max_retries=args.max_retries,
    )
    print(manifest.resolve())


if __name__ == "__main__":
    main()
