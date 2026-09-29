"""Prepare protocol-v1 JSONL manifests from upstream annotations."""
from __future__ import annotations

import argparse
from pathlib import Path

from .adapters import (
    add_delayed_distractors,
    fpsbench_trials,
    load_distractor_bank,
    moment_video_trials,
    probe_media_lengths,
)
from .protocol import atomic_json, atomic_jsonl, file_sha256, load_trials


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    fps = subparsers.add_parser("fpsbench")
    fps.add_argument("--annotations", type=Path, required=True)
    fps.add_argument("--video-root", type=Path, required=True)
    fps.add_argument("--path-template", default="{id}.mp4")
    fps.add_argument("--output", type=Path, required=True)

    moment = subparsers.add_parser("moment-video")
    moment.add_argument("--annotations", type=Path, required=True)
    moment.add_argument("--video-root", type=Path, required=True)
    moment.add_argument("--probe-media", action="store_true")
    moment.add_argument("--output", type=Path, required=True)

    delay = subparsers.add_parser("add-delays")
    delay.add_argument("--input", type=Path, required=True)
    delay.add_argument("--distractor-bank", type=Path, required=True)
    delay.add_argument("--delays", type=float, nargs="+", default=[0.0, 30.0, 120.0, 600.0])
    delay.add_argument("--profiles", nargs="+", default=["clean", "redundancy", "novelty"])
    delay.add_argument("--seed", type=int, default=42)
    delay.add_argument("--output", type=Path, required=True)

    combine = subparsers.add_parser("combine")
    combine.add_argument("--inputs", type=Path, nargs="+", required=True)
    combine.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "fpsbench":
        trials = fpsbench_trials(args.annotations, args.video_root, path_template=args.path_template)
        sources = {"annotations": str(args.annotations.resolve()), "annotations_sha256": file_sha256(args.annotations)}
    elif args.command == "moment-video":
        trials = moment_video_trials(args.annotations, args.video_root)
        if args.probe_media:
            trials = probe_media_lengths(trials)
        sources = {"annotations": str(args.annotations.resolve()), "annotations_sha256": file_sha256(args.annotations)}
    elif args.command == "add-delays":
        trials = add_delayed_distractors(
            load_trials(args.input),
            load_distractor_bank(args.distractor_bank),
            delays_sec=args.delays,
            profiles=args.profiles,
            seed=args.seed,
        )
        sources = {
            "base_manifest": str(args.input.resolve()),
            "base_manifest_sha256": file_sha256(args.input),
            "distractor_bank": str(args.distractor_bank.resolve()),
            "distractor_bank_sha256": file_sha256(args.distractor_bank),
        }
    else:
        trials = []
        for path in args.inputs:
            trials.extend(load_trials(path))
        sources = {
            "input_manifests": [
                {"path": str(path.resolve()), "sha256": file_sha256(path)} for path in args.inputs
            ]
        }
    atomic_jsonl(args.output, (trial.to_dict(relative_to=args.output.parent.resolve()) for trial in trials))
    atomic_json(
        args.output.with_suffix(".protocol.json"),
        {"protocol_version": 1, "samples": len(trials), "sources": sources, "manifest_sha256": file_sha256(args.output)},
    )
    print(f"wrote {len(trials)} trials to {args.output}")


if __name__ == "__main__":
    main()
