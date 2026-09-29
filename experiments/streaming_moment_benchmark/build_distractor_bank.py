"""Measure clip motion and build low-motion redundant / high-motion novel banks."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .adapters import probe_media_lengths
from .frames import iter_trial_frames
from .protocol import StreamTrial, atomic_json, atomic_jsonl, file_sha256, load_trials


def _domain(trial: StreamTrial) -> str:
    if trial.metadata.get("visual_domain"):
        return str(trial.metadata["visual_domain"])
    category = str(trial.metadata.get("category", ""))
    subclass = str(trial.metadata.get("subclass", "")).lower()
    if category == "games":
        return "Hobbies & Gaming"
    if category == "GUI":
        return "Hobbies & Gaming"
    if category == "human" and any(token in subclass for token in ("sport", "soccer", "ball")):
        return "Sports & Fitness"
    if category in {"human", "animal"}:
        return "Media & Entertainment"
    return "Miscellaneous"


def _round_robin_lowest(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["semantic_domain"]].append(row)
    for values in groups.values():
        values.sort(key=lambda row: (row["mean_motion"], row["source_group"]))
    keys = sorted(groups)
    result = []
    while len(result) < count:
        progressed = False
        for key in keys:
            if groups[key] and len(result) < count:
                result.append(groups[key].pop(0))
                progressed = True
        if not progressed:
            break
    if len(result) < count:
        raise ValueError(f"only {len(result)} clips available for {count} redundancy entries")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--redundancy-count", type=int, default=12)
    parser.add_argument("--novelty-count", type=int, default=12)
    parser.add_argument("--max-segment-sec", type=float, default=30.0)
    args = parser.parse_args()
    if min(args.redundancy_count, args.novelty_count) < 0 or args.max_segment_sec <= 0:
        parser.error("bank counts must be non-negative and segment duration positive")
    trials = load_trials(args.manifest, require_files=True)
    if any(trial.metadata.get("needs_media_probe") for trial in trials):
        trials = probe_media_lengths(trials)
    rows = []
    for index, trial in enumerate(trials):
        motions = [float(item.features[-2]) for item in iter_trial_frames(trial)]
        if not motions:
            raise ValueError(f"no frames for {trial.trial_id}")
        segment = trial.segments[0]
        rows.append(
            {
                "path": segment.path,
                "start_sec": segment.start_sec,
                "end_sec": min(segment.end_sec, segment.start_sec + args.max_segment_sec),
                "source_group": trial.source_group,
                "semantic_domain": _domain(trial),
                "mean_motion": float(np.mean(motions)),
                "p90_motion": float(np.quantile(motions, 0.9)),
            }
        )
        print(f"[{index + 1}/{len(trials)}] {trial.trial_id}", flush=True)
    redundant = _round_robin_lowest(rows, args.redundancy_count)
    used = {row["source_group"] for row in redundant}
    remaining = [row for row in rows if row["source_group"] not in used]
    novel = sorted(remaining, key=lambda row: (row["p90_motion"], row["mean_motion"]), reverse=True)[
        : args.novelty_count
    ]
    if len(novel) < args.novelty_count:
        raise ValueError("not enough source-disjoint clips for novelty bank")
    output_rows = [dict(row, profile="redundancy") for row in redundant]
    output_rows.extend(dict(row, profile="novelty") for row in novel)
    atomic_jsonl(args.output, output_rows)
    atomic_json(
        args.output.with_suffix(".report.json"),
        {
            "format_version": 1,
            "source_manifest": str(args.manifest.resolve()),
            "source_manifest_sha256": file_sha256(args.manifest),
            "clips_measured": len(rows),
            "redundancy": len(redundant),
            "novelty": len(novel),
            "source_overlap": sorted(used & {row["source_group"] for row in novel}),
        },
    )
    print(f"wrote {len(output_rows)} distractor segments to {args.output}")


if __name__ == "__main__":
    main()
