"""Frozen selection artifacts and a content-addressed frame store.

The inference stage deliberately depends on this module rather than video
decoders.  A completed selection artifact is therefore an auditable boundary:
late-query inference cannot replay the source video.
"""
from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .frames import FrameObservation, FrameRef
from .protocol import Question, StreamTrial, atomic_json, file_sha256


ARTIFACT_FORMAT_VERSION = 1
FRAME_CODEC = "jpeg-q90-v1"


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def path_fingerprint(path: Path) -> dict[str, Any]:
    """Return a content fingerprint suitable for experiment cache keys."""
    resolved = path.expanduser().resolve()
    if resolved.is_file():
        return {
            "path": str(resolved),
            "size": resolved.stat().st_size,
            "sha256": file_sha256(resolved),
        }
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    digest = hashlib.sha256()
    files = sorted(item for item in resolved.rglob("*") if item.is_file())
    total = 0
    for item in files:
        relative = item.relative_to(resolved).as_posix()
        size = item.stat().st_size
        total += size
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(str(size).encode("ascii") + b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
    return {
        "path": str(resolved),
        "files": len(files),
        "size": total,
        "sha256": digest.hexdigest(),
    }


def media_fingerprints(trial: StreamTrial, cache: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    cache = cache if cache is not None else {}
    result = []
    for segment in trial.segments:
        path = str(Path(segment.path).resolve())
        if path not in cache:
            cache[path] = path_fingerprint(Path(path))
        result.append(
            {
                **cache[path],
                "start_sec": segment.start_sec,
                "end_sec": segment.end_sec,
                "role": segment.role,
                "source_group": segment.source_group,
            }
        )
    return result


@dataclasses.dataclass(frozen=True)
class StoredFrame:
    ref: FrameRef
    object_sha256: str
    features: tuple[float, ...]
    width: int
    height: int
    encoded_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref.to_dict(),
            "object_sha256": self.object_sha256,
            "features": list(self.features),
            "width": self.width,
            "height": self.height,
            "encoded_bytes": self.encoded_bytes,
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "StoredFrame":
        ref = FrameRef(**row["ref"])
        return cls(
            ref=ref,
            object_sha256=str(row["object_sha256"]),
            features=tuple(float(item) for item in row["features"]),
            width=int(row["width"]),
            height=int(row["height"]),
            encoded_bytes=int(row["encoded_bytes"]),
        )


class FrameStore:
    """Content-addressed JPEG objects with atomic, de-duplicated writes."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.objects = self.root / "objects"

    def object_path(self, digest: str) -> Path:
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("invalid SHA256 object id")
        return self.objects / digest[:2] / f"{digest}.jpg"

    @staticmethod
    def _encode(image: np.ndarray) -> bytes:
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("frame store expects uint8 RGB images")
        buffer = io.BytesIO()
        Image.fromarray(image, "RGB").save(
            buffer,
            format="JPEG",
            quality=90,
            optimize=False,
            progressive=False,
            subsampling=0,
        )
        return buffer.getvalue()

    def put(self, observation: FrameObservation) -> StoredFrame:
        payload = self._encode(observation.image)
        digest = hashlib.sha256(payload).hexdigest()
        destination = self.object_path(digest)
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile("wb", dir=destination.parent, delete=False) as handle:
                handle.write(payload)
                temporary = Path(handle.name)
            try:
                os.replace(temporary, destination)
            finally:
                if temporary.exists():
                    temporary.unlink()
        height, width = observation.image.shape[:2]
        return StoredFrame(
            ref=observation.ref,
            object_sha256=digest,
            features=tuple(float(item) for item in observation.features),
            width=width,
            height=height,
            encoded_bytes=len(payload),
        )

    def get(self, frame: StoredFrame, *, verify: bool = False) -> FrameObservation:
        path = self.object_path(frame.object_sha256)
        payload = path.read_bytes()
        if verify and hashlib.sha256(payload).hexdigest() != frame.object_sha256:
            raise ValueError(f"corrupt frame object: {path}")
        with Image.open(io.BytesIO(payload)) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        if array.shape[:2] != (frame.height, frame.width):
            raise ValueError(f"frame shape mismatch: {path}")
        return FrameObservation(frame.ref, np.asarray(frame.features, dtype=np.float32), array)

    def get_many(self, frames: Iterable[StoredFrame], *, verify: bool = False) -> list[FrameObservation]:
        return [self.get(item, verify=verify) for item in frames]


def selection_cache_key(
    trial: StreamTrial,
    *,
    media: list[dict[str, Any]],
    method: str,
    memory_frames: int,
    encode_fps: float,
    window_sec: float,
    seed: int,
    oracle_max_frames: int,
    directory_fps: float,
    utility_model_sha256: str | None,
) -> str:
    return canonical_sha256(
        {
            "artifact_format_version": ARTIFACT_FORMAT_VERSION,
            "frame_codec": FRAME_CODEC,
            "trial": trial.to_dict(),
            "media": media,
            "selector": {
                "method": method,
                "memory_frames": memory_frames,
                "encode_fps": encode_fps,
                "window_sec": window_sec,
                "seed": seed,
            },
            "oracle_max_frames": oracle_max_frames,
            "directory_fps": directory_fps,
            "utility_model_sha256": utility_model_sha256,
        }
    )


def artifact_stored_frames(artifact: dict[str, Any], stage: str) -> list[StoredFrame]:
    if stage == "final_memory":
        rows = artifact["selection"]["final_frames"]
    elif stage in {"critical_only", "dense_offline_cap"} and artifact.get("oracle_stages_ref"):
        reference = artifact["oracle_stages_ref"]
        oracle = load_oracle_stages(Path(reference["path"]))
        if oracle["artifact_sha256"] != reference["artifact_sha256"]:
            raise ValueError("referenced oracle-stage artifact checksum changed")
        rows = oracle["stages"][stage]
    else:
        rows = artifact["oracle_stages"][stage]
    return [StoredFrame.from_dict(row) for row in rows]


def artifact_observations(
    artifact: dict[str, Any],
    store: FrameStore,
    stage: str,
    *,
    verify: bool = False,
) -> list[FrameObservation]:
    return store.get_many(artifact_stored_frames(artifact, stage), verify=verify)


def artifact_trial(artifact: dict[str, Any]) -> StreamTrial:
    return StreamTrial.from_dict(dict(artifact["trial"]))


def write_artifact(path: Path, artifact: dict[str, Any]) -> None:
    if artifact.get("format_version") != ARTIFACT_FORMAT_VERSION:
        raise ValueError("unsupported selection artifact format")
    if artifact.get("artifact_type") != "frozen_selection":
        raise ValueError("not a frozen selection artifact")
    expected = str(artifact["cache_key"])
    payload = dict(artifact)
    payload.pop("artifact_sha256", None)
    artifact["artifact_sha256"] = canonical_sha256(payload)
    if expected != artifact["cache_key"]:
        raise AssertionError("cache key changed while writing")
    atomic_json(path, artifact)


def load_artifact(path: Path, *, verify: bool = True) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        artifact = json.load(handle)
    if artifact.get("format_version") != ARTIFACT_FORMAT_VERSION:
        raise ValueError(f"unsupported artifact format: {path}")
    if artifact.get("artifact_type") != "frozen_selection":
        raise ValueError(f"not a frozen selection artifact: {path}")
    if verify:
        expected = artifact.get("artifact_sha256")
        payload = dict(artifact)
        payload.pop("artifact_sha256", None)
        if canonical_sha256(payload) != expected:
            raise ValueError(f"selection artifact checksum mismatch: {path}")
    return artifact


def write_oracle_stages(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "artifact_type": "frozen_oracle_stages",
        **value,
    }
    payload["artifact_sha256"] = canonical_sha256(payload)
    atomic_json(path, payload)
    return payload


def load_oracle_stages(path: Path, *, verify: bool = True) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        artifact = json.load(handle)
    if artifact.get("format_version") != ARTIFACT_FORMAT_VERSION or artifact.get("artifact_type") != "frozen_oracle_stages":
        raise ValueError(f"not a supported oracle-stage artifact: {path}")
    if verify:
        expected = artifact.get("artifact_sha256")
        payload = dict(artifact)
        payload.pop("artifact_sha256", None)
        if canonical_sha256(payload) != expected:
            raise ValueError(f"oracle-stage artifact checksum mismatch: {path}")
    return artifact


def inference_cache_key(
    *,
    selection_artifact_sha256: str,
    stage: str,
    predictor: str,
    model_fingerprint: dict[str, Any] | None,
    prompt_version: str,
    min_pixels: int | None,
    max_pixels: int | None,
    max_new_tokens: int,
) -> str:
    return canonical_sha256(
        {
            "selection_artifact_sha256": selection_artifact_sha256,
            "stage": stage,
            "predictor": predictor,
            "model_fingerprint": model_fingerprint,
            "prompt_version": prompt_version,
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
            "max_new_tokens": max_new_tokens,
        }
    )
