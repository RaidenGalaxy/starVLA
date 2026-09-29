"""Filter a frozen-selection index without decoding source media again."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .artifacts import load_artifact
from .protocol import atomic_json, atomic_jsonl, file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-ids", nargs="+", required=True)
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--profiles", nargs="+")
    args = parser.parse_args()

    requested = list(dict.fromkeys(args.sample_ids))
    requested_set = set(requested)
    rows: list[dict[str, Any]] = []
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if args.methods and row["method"] not in args.methods:
                continue
            artifact = load_artifact(Path(row["path"]))
            trial = artifact["trial"]
            if trial["sample_id"] not in requested_set:
                continue
            if args.profiles and trial.get("condition", {}).get("distractor_profile") not in args.profiles:
                continue
            rows.append(row)
    found = {load_artifact(Path(row["path"]))["trial"]["sample_id"] for row in rows}
    missing = [sample_id for sample_id in requested if sample_id not in found]
    if missing:
        raise ValueError(f"sample_ids absent after filtering: {missing}")
    rows.sort(key=lambda row: (row["trial_id"], row["method"], row["memory_frames"], row["encode_fps"]))
    atomic_jsonl(args.output, rows)
    atomic_json(
        args.output.with_suffix(".report.json"),
        {
            "format_version": 1,
            "source_index": str(args.input.resolve()),
            "source_index_sha256": file_sha256(args.input),
            "requested_sample_ids": requested,
            "rows": len(rows),
            "sample_ids": len(found),
            "output_sha256": file_sha256(args.output),
        },
    )
    print(f"wrote {len(rows)} rows for {len(found)} sample_ids to {args.output}")


if __name__ == "__main__":
    main()
