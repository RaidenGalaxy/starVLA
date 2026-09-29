"""Create a tiny 30-FPS blink-and-miss stream without external downloads."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from .protocol import EvidenceSpan, Question, Segment, StreamTrial, atomic_json, atomic_jsonl, file_sha256


def _make_frames(path: Path, *, seed: int, event_color: tuple[int, int, int]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    background = np.asarray([35 + seed % 20, 45, 55], dtype=np.uint8)
    for index in range(120):
        frame = np.broadcast_to(background, (64, 64, 3)).copy()
        frame[8:24, 8:24] = (40, 40, 40)
        if 30 <= index <= 32:
            frame[8:24, 8:24] = event_color
        if index >= 36:
            # High-motion but answer-irrelevant distractors compete for novelty.
            block = rng.integers(0, 256, size=(4, 4, 3), dtype=np.uint8)
            frame = np.repeat(np.repeat(block, 16, axis=0), 16, axis=1)
        Image.fromarray(frame).save(path / f"{index:04d}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    frames_root = args.output_dir / "frames"
    specifications = [
        ("train_0", "train", 11),
        ("train_1", "train", 17),
        ("validation_0", "validation", 23),
        ("test_0", "test", 31),
        ("test_1", "test", 37),
    ]
    trials = []
    for sample_id, split, seed in specifications:
        frame_path = frames_root / sample_id
        _make_frames(frame_path, seed=seed, event_color=(220, 25, 25))
        trials.append(
            StreamTrial(
                trial_id=f"smoke_{sample_id}__delay2p9__novelty",
                sample_id=f"smoke_{sample_id}",
                dataset="synthetic_moment_smoke",
                split=split,
                source_group=f"synthetic:{sample_id}",
                question=Question(
                    text="What color did the small indicator briefly turn?",
                    answer="A",
                    answer_type="multiple_choice",
                    answer_text="red",
                    choices={"A": "red", "B": "green", "C": "blue", "D": "white"},
                ),
                segments=(Segment(str(frame_path.resolve()), 0.0, 4.0, "source", f"synthetic:{sample_id}"),),
                evidence_spans=(EvidenceSpan(1.0, 32 / 30, "synthetic_exact"),),
                query_time_sec=4.0,
                min_fps=30.0,
                native_fps=30.0,
                condition={"delay_sec": 4.0 - 32 / 30, "distractor_profile": "novelty"},
                metadata={
                    "task_type": "blink_and_miss",
                    "visual_domain": "synthetic_gui",
                    "min_required_frames": 2,
                    "synthetic": True,
                },
            )
        )
    manifest = args.output_dir / "smoke_trials.jsonl"
    atomic_jsonl(manifest, (trial.to_dict(relative_to=manifest.parent.resolve()) for trial in trials))
    atomic_json(
        args.output_dir / "smoke_protocol.json",
        {
            "protocol_version": 1,
            "purpose": "integration test only; not a scientific result",
            "samples": len(trials),
            "manifest_sha256": file_sha256(manifest),
        },
    )
    print(f"wrote {len(trials)} synthetic streams to {args.output_dir}")


if __name__ == "__main__":
    main()
