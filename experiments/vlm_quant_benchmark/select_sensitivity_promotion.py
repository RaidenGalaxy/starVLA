"""Freeze the formal Top-2 and write the disjoint 64x4 promotion manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_manifest, write_manifest
from .promotion import rank_promotion_variants
from .reporting import atomic_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select formal W4 Top-2 variants")
    parser.add_argument("--ablation-jsonl", type=Path, required=True)
    parser.add_argument("--ablation-plan", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        json.loads(line) for line in args.ablation_jsonl.read_text().splitlines()
        if line.strip()
    ]
    plan = json.loads(args.ablation_plan.read_text())
    ranked = rank_promotion_variants(
        rows, bootstrap_samples=args.bootstrap_samples, seed=args.seed,
    )
    selected = [row for row in ranked if row["selected"]]
    if len(selected) != 2:
        raise ValueError(f"exactly two promotion variants are required, found {len(selected)}")
    ids = set(plan["promotion_ids"])
    adaptation = set(plan["adaptation_ids"])
    nll = set(plan["nll_ids"])
    if ids & adaptation or ids & nll:
        raise ValueError("promotion manifest overlaps adaptation or NLL samples")
    evaluation = load_manifest(args.evaluation_manifest)
    samples = [sample for sample in evaluation if sample.id in ids]
    if len(samples) != len(ids):
        raise ValueError("promotion IDs are missing from evaluation manifest")
    configs = {
        item["variant"]: item.get("config") for item in plan["variants"]
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "promotion_manifest.jsonl"
    write_manifest(manifest_path, samples)
    for row in selected:
        config = configs.get(row["variant"])
        if config is None:
            raise ValueError(f"selected variant has no executable config: {row['variant']}")
        atomic_json(args.output_dir / f"{row['variant']}_config.json", config)
    oracle_config = configs.get("mbq_target_oracle")
    if oracle_config is not None:
        atomic_json(args.output_dir / "mbq_target_oracle_config.json", oracle_config)
    atomic_json(args.output_dir / "top2_selection.json", {
        "format_version": 1,
        "selection_rule": "dataset-balanced paired NLL_current_minus_variant",
        "excluded": [
            "bf16", "mbq_current", "squeezellm_current",
            "mbq_target_oracle", "mbq_block_reconstruction",
        ],
        "ranked": ranked,
        "selected": [row["variant"] for row in selected],
        "promotion_manifest": str(manifest_path.resolve()),
        "promotion_samples": len(samples),
        "sets_are_disjoint": True,
        "seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
    })


if __name__ == "__main__":
    main()
