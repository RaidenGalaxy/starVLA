"""Freeze a deterministic subset whose referenced media are locally available."""
from __future__ import annotations

import argparse
from pathlib import Path

from .protocol import atomic_json, atomic_jsonl, file_sha256, load_trials


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--max-source-groups",
        type=int,
        help="Keep every trial for the first N source groups after filtering.",
    )
    parser.add_argument("--split", nargs="+")
    parser.add_argument("--profiles", nargs="+", help="Keep these distractor_profile values.")
    parser.add_argument(
        "--appended-delays",
        type=float,
        nargs="+",
        help="Keep these appended_delay_sec values.",
    )
    args = parser.parse_args()
    trials = load_trials(args.input)
    selected = [
        trial
        for trial in trials
        if (not args.split or trial.split in args.split)
        and (not args.profiles or trial.condition.get("distractor_profile") in args.profiles)
        and (
            not args.appended_delays
            or float(trial.condition.get("appended_delay_sec", 0.0) or 0.0) in args.appended_delays
        )
        and all(Path(segment.path).is_file() or Path(segment.path).is_dir() for segment in trial.segments)
    ]
    if args.max_source_groups is not None:
        if args.max_source_groups <= 0:
            raise ValueError("--max-source-groups must be positive")
        source_groups: list[str] = []
        for trial in selected:
            if trial.source_group not in source_groups:
                source_groups.append(trial.source_group)
            if len(source_groups) == args.max_source_groups:
                break
        keep = set(source_groups)
        selected = [trial for trial in selected if trial.source_group in keep]
    if args.max_samples is not None:
        selected = selected[: args.max_samples]
    if not selected:
        raise ValueError("no trials have all media available")
    atomic_jsonl(args.output, (trial.to_dict(relative_to=args.output.parent.resolve()) for trial in selected))
    atomic_json(
        args.output.with_suffix(".report.json"),
        {
            "format_version": 1,
            "source_manifest": str(args.input.resolve()),
            "source_manifest_sha256": file_sha256(args.input),
            "source_trials": len(trials),
            "available_trials": len(selected),
            "unique_source_groups": len({trial.source_group for trial in selected}),
            "output_sha256": file_sha256(args.output),
        },
    )
    print(f"wrote {len(selected)} available trials to {args.output}")


if __name__ == "__main__":
    main()
