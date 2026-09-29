"""Causal frame decoding and inexpensive temporal features."""
from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from PIL import Image

from .protocol import StreamTrial


GRID_SIZE = 4
HISTOGRAM_BINS = 8
FEATURE_NAMES = tuple(
    [f"grid_{row}_{column}_{channel}" for row in range(GRID_SIZE) for column in range(GRID_SIZE) for channel in "rgb"]
    + [f"mean_{channel}" for channel in "rgb"]
    + [f"std_{channel}" for channel in "rgb"]
    + [f"gray_hist_{index}" for index in range(HISTOGRAM_BINS)]
    + ["edge_density", "motion_short", "motion_long"]
)


@dataclasses.dataclass(frozen=True)
class FrameRef:
    frame_id: str
    frame_index: int
    timestamp_sec: float
    source_timestamp_sec: float
    segment_index: int
    role: str

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class FrameObservation:
    ref: FrameRef
    features: np.ndarray
    image: np.ndarray


class CheapTemporalFeatureExtractor:
    """Small CPU scout; it never sees the future query or future frames."""

    def __init__(self) -> None:
        self._previous: np.ndarray | None = None
        self._long_ema: np.ndarray | None = None

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("expected an RGB HxWx3 frame")
        thumbnail = np.asarray(
            Image.fromarray(image).resize((32, 32), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        cells = thumbnail.reshape(GRID_SIZE, 8, GRID_SIZE, 8, 3).mean(axis=(1, 3)).reshape(-1)
        means = thumbnail.mean(axis=(0, 1))
        standard_deviations = thumbnail.std(axis=(0, 1))
        gray = thumbnail.mean(axis=2)
        histogram, _ = np.histogram(gray, bins=HISTOGRAM_BINS, range=(0.0, 1.0))
        histogram = histogram.astype(np.float32) / gray.size
        edge_density = float(
            (np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean()) / 2
        )
        motion_short = 0.0 if self._previous is None else float(np.abs(thumbnail - self._previous).mean())
        motion_long = 0.0 if self._long_ema is None else float(np.abs(thumbnail - self._long_ema).mean())
        self._previous = thumbnail
        self._long_ema = thumbnail if self._long_ema is None else 0.95 * self._long_ema + 0.05 * thumbnail
        values = np.concatenate(
            [
                cells,
                means,
                standard_deviations,
                histogram,
                np.asarray([edge_density, motion_short, motion_long], dtype=np.float32),
            ]
        ).astype(np.float32)
        if values.shape != (len(FEATURE_NAMES),) or not np.isfinite(values).all():
            raise ValueError("invalid scout feature vector")
        return values


def _directory_frames(
    path: Path,
    start_sec: float,
    end_sec: float,
    fps: float,
) -> Iterator[tuple[int, float, np.ndarray]]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted(item for item in path.iterdir() if item.suffix.lower() in extensions)
    start_index = max(0, math.ceil(start_sec * fps - 1e-9))
    end_index = min(len(files), math.ceil(end_sec * fps - 1e-9))
    for frame_index in range(start_index, end_index):
        with Image.open(files[frame_index]) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        yield frame_index, frame_index / fps, rgb


def _video_frames(path: Path, start_sec: float, end_sec: float) -> Iterator[tuple[int, float, np.ndarray]]:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("OpenCV is required to decode video files") from error
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise FileNotFoundError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError(f"video reports invalid FPS: {path}")
    start_index = max(0, math.ceil(start_sec * fps - 1e-9))
    end_index = math.ceil(end_sec * fps - 1e-9)
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_index)
    try:
        for frame_index in range(start_index, end_index):
            ok, frame = capture.read()
            if not ok:
                break
            yield frame_index, frame_index / fps, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def iter_trial_frames(trial: StreamTrial, *, directory_fps: float = 30.0) -> Iterator[FrameObservation]:
    extractor = CheapTemporalFeatureExtractor()
    stream_offset = 0.0
    global_index = 0
    for segment_index, segment in enumerate(trial.segments):
        path = Path(segment.path)
        if not path.exists():
            raise FileNotFoundError(f"missing media: {path}")
        if path.is_dir():
            fps = trial.native_fps or directory_fps
            iterator = _directory_frames(path, segment.start_sec, segment.end_sec, fps)
        else:
            iterator = _video_frames(path, segment.start_sec, segment.end_sec)
        observed = 0
        for source_index, source_timestamp, image in iterator:
            relative_timestamp = min(
                stream_offset + max(0.0, source_timestamp - segment.start_sec),
                stream_offset + segment.duration_sec,
            )
            ref = FrameRef(
                frame_id=f"s{segment_index}:f{source_index}",
                frame_index=global_index,
                timestamp_sec=relative_timestamp,
                source_timestamp_sec=source_timestamp,
                segment_index=segment_index,
                role=segment.role,
            )
            yield FrameObservation(ref=ref, features=extractor(image), image=image)
            global_index += 1
            observed += 1
        if observed == 0:
            raise ValueError(f"segment produced no frames: {segment.path} [{segment.start_sec}, {segment.end_sec})")
        stream_offset += segment.duration_sec
