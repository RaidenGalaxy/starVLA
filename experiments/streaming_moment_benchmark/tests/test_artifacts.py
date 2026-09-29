from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.streaming_moment_benchmark.artifacts import (
    FrameStore,
    artifact_observations,
    inference_cache_key,
    load_artifact,
    selection_cache_key,
    write_artifact,
    write_oracle_stages,
)
from experiments.streaming_moment_benchmark.frames import FrameObservation, FrameRef
from experiments.streaming_moment_benchmark.protocol import Question, Segment, StreamTrial


def _observation(frame_id: str = "f0") -> FrameObservation:
    return FrameObservation(
        FrameRef(frame_id, 0, 0.0, 0.0, 0, "source"),
        np.asarray([0.25, 0.5], dtype=np.float32),
        np.full((12, 16, 3), 127, dtype=np.uint8),
    )


def _trial(media: Path) -> StreamTrial:
    return StreamTrial(
        "trial", "sample", "test", "eval", "source",
        Question("Which?", "A", "multiple_choice", "one", {"A": "one", "B": "two"}),
        (Segment(str(media), 0.0, 1.0),),
        query_time_sec=1.0,
    )


def test_frame_store_deduplicates_and_roundtrips(tmp_path: Path) -> None:
    store = FrameStore(tmp_path / "store")
    first = store.put(_observation("f0"))
    second = store.put(_observation("f1"))
    assert first.object_sha256 == second.object_sha256
    assert len(list((tmp_path / "store" / "objects").rglob("*.jpg"))) == 1
    loaded = store.get(second, verify=True)
    assert loaded.ref.frame_id == "f1"
    assert loaded.image.shape == (12, 16, 3)


def test_selection_key_invalidates_on_media_and_selector(tmp_path: Path) -> None:
    media_path = tmp_path / "source.bin"
    media_path.write_bytes(b"a")
    trial = _trial(media_path)
    common = dict(
        trial=trial,
        method="reservoir",
        memory_frames=16,
        encode_fps=1.0,
        window_sec=1.0,
        seed=42,
        oracle_max_frames=64,
        directory_fps=30.0,
        utility_model_sha256=None,
    )
    media_a = [{"path": str(media_path), "sha256": "a" * 64}]
    media_b = [{"path": str(media_path), "sha256": "b" * 64}]
    key = selection_cache_key(media=media_a, **common)
    assert key != selection_cache_key(media=media_b, **common)
    assert key != selection_cache_key(media=media_a, **{**common, "memory_frames": 64})


def test_inference_uses_frozen_objects_without_source_media(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"not read by inference")
    trial = _trial(source)
    store = FrameStore(tmp_path / "store")
    frozen = store.put(_observation()).to_dict()
    artifact = {
        "format_version": 1,
        "artifact_type": "frozen_selection",
        "cache_key": "k",
        "trial": trial.to_dict(),
        "selection": {"final_frames": [frozen]},
        "oracle_stages": {"critical_only": [frozen]},
    }
    path = tmp_path / "artifact.json"
    write_artifact(path, artifact)
    source.unlink()
    loaded = load_artifact(path)
    observations = artifact_observations(loaded, store, "final_memory", verify=True)
    assert len(observations) == 1
    assert observations[0].image.shape == (12, 16, 3)


def test_inference_key_includes_pixels_prompt_and_model() -> None:
    common = dict(
        selection_artifact_sha256="s",
        stage="final_memory",
        predictor="qwen",
        model_fingerprint={"sha256": "m"},
        prompt_version="v1",
        min_pixels=None,
        max_pixels=100,
        max_new_tokens=4,
    )
    key = inference_cache_key(**common)
    assert key != inference_cache_key(**{**common, "max_pixels": 200})
    assert key != inference_cache_key(**{**common, "prompt_version": "v2"})
    assert key != inference_cache_key(**{**common, "model_fingerprint": {"sha256": "n"}})


def test_shared_oracle_stage_artifact_is_resolved(tmp_path: Path) -> None:
    store = FrameStore(tmp_path / "store")
    frozen = store.put(_observation()).to_dict()
    oracle_path = tmp_path / "oracle.json"
    oracle = write_oracle_stages(
        oracle_path,
        {"cache_key": "o", "trial_id": "trial", "source_media": [], "stages": {"critical_only": [frozen]}},
    )
    artifact = {
        "selection": {"final_frames": []},
        "oracle_stages": {},
        "oracle_stages_ref": {"path": str(oracle_path), "artifact_sha256": oracle["artifact_sha256"]},
    }
    loaded = artifact_observations(artifact, store, "critical_only", verify=True)
    assert len(loaded) == 1
