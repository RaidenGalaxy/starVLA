from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .core import atomic_json


POINT_MACRO_DROP_LIMIT = 0.01
UPPER95_MACRO_DROP_LIMIT = 0.03
DATASET_DROP_LIMIT = 0.05


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def scored_rows(root: Path, datasets: list[str]) -> dict[str, dict[str, float]]:
    result = {}
    for dataset in datasets:
        path = root / dataset / "official_samples.jsonl"
        if not path.is_file():
            raise ValueError(f"missing official samples: {path}")
        rows = read_jsonl(path)
        mapped = {}
        for row in rows:
            identifier = str(row["id"])
            score = float(row["official_score"])
            if not math.isfinite(score) or score < 0.0 or score > 1.0:
                raise ValueError(f"invalid official score: {identifier}={score}")
            if identifier in mapped:
                raise ValueError(f"duplicate official sample ID: {identifier}")
            mapped[identifier] = score
        result[dataset] = mapped
    return result


def analyze_comparison(
    manifest: list[dict], baseline_root: Path, frozen_root: Path, *,
    bootstrap_samples: int, seed: int,
) -> dict:
    datasets = sorted({str(row["dataset"]) for row in manifest})
    baseline = scored_rows(baseline_root, datasets)
    frozen = scored_rows(frozen_root, datasets)
    paired: dict[str, np.ndarray] = {}
    dataset_rows = {}
    for dataset in datasets:
        ids = [str(row["id"]) for row in manifest if row["dataset"] == dataset]
        missing_baseline = set(ids) - set(baseline[dataset])
        missing_frozen = set(ids) - set(frozen[dataset])
        if missing_baseline or missing_frozen:
            raise ValueError(
                f"missing comparison rows for {dataset}: "
                f"baseline={len(missing_baseline)}, frozen={len(missing_frozen)}"
            )
        base = np.asarray([baseline[dataset][identifier] for identifier in ids])
        candidate = np.asarray([frozen[dataset][identifier] for identifier in ids])
        loss = base - candidate
        paired[dataset] = loss
        dataset_rows[dataset] = {
            "samples": len(ids),
            "baseline_score": float(base.mean()),
            "frozen_score": float(candidate.mean()),
            "score_drop": float(loss.mean()),
        }
    point_drop = float(np.mean([values.mean() for values in paired.values()]))
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(bootstrap_samples):
        dataset_means = []
        for dataset in datasets:
            values = paired[dataset]
            draw = rng.integers(0, len(values), size=len(values))
            dataset_means.append(float(values[draw].mean()))
        bootstrap.append(float(np.mean(dataset_means)))
    interval = [float(value) for value in np.quantile(bootstrap, [0.025, 0.975])]
    max_dataset_drop = max(row["score_drop"] for row in dataset_rows.values())
    safe = bool(
        point_drop <= POINT_MACRO_DROP_LIMIT
        and interval[1] <= UPPER95_MACRO_DROP_LIMIT
        and max_dataset_drop <= DATASET_DROP_LIMIT
    )
    return {
        "baseline_root": str(baseline_root.resolve()),
        "frozen_root": str(frozen_root.resolve()),
        "samples": sum(row["samples"] for row in dataset_rows.values()),
        "datasets": dataset_rows,
        "macro_score_drop": point_drop,
        "macro_score_drop_ci95": interval,
        "max_dataset_score_drop": max_dataset_drop,
        "safe": safe,
    }


def render_report(result: dict) -> str:
    lines = [
        "# Q-ShortV 官方生成安全预算报告",
        "",
        (
            "安全门限：宏平均分数下降 ≤1pp；配对分层 bootstrap 的 95% 上界 "
            "≤3pp；任一数据集分数下降 ≤5pp。"
        ),
        "",
        "| 策略 | 样本 | 宏平均下降 | 95% CI | 最大数据集下降 | 安全 |",
        "|---|---:|---:|---|---:|---|",
    ]
    for name, row in result["comparisons"].items():
        lines.append(
            f"| {name} | {row['samples']} | {100 * row['macro_score_drop']:.3f}pp "
            f"| [{100 * row['macro_score_drop_ci95'][0]:.3f}, "
            f"{100 * row['macro_score_drop_ci95'][1]:.3f}]pp "
            f"| {100 * row['max_dataset_score_drop']:.3f}pp | {row['safe']} |"
        )
        lines.extend(["", f"## {name}", "", "| 数据集 | baseline | frozen | 下降 |", "|---|---:|---:|---:|"])
        for dataset, values in row["datasets"].items():
            lines.append(
                f"| {dataset} | {100 * values['baseline_score']:.3f} "
                f"| {100 * values['frozen_score']:.3f} "
                f"| {100 * values['score_drop']:.3f}pp |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze paired Q-ShortV generation safety")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--comparison", nargs=3, action="append", metavar=("NAME", "BASELINE", "FROZEN"),
        required=True,
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = read_jsonl(args.manifest)
    comparisons = {}
    for name, baseline, frozen in args.comparison:
        if name in comparisons:
            raise ValueError(f"duplicate comparison name: {name}")
        comparisons[name] = analyze_comparison(
            manifest, Path(baseline), Path(frozen),
            bootstrap_samples=args.bootstrap_samples, seed=args.seed,
        )
    result = {
        "schema_name": "qshortv_generation_gate_v1",
        "schema_version": 1,
        "manifest": str(args.manifest.resolve()),
        "comparisons": comparisons,
        "all_safe": all(row["safe"] for row in comparisons.values()),
        "thresholds": {
            "point_macro_score_drop": POINT_MACRO_DROP_LIMIT,
            "upper95_macro_score_drop": UPPER95_MACRO_DROP_LIMIT,
            "max_dataset_score_drop": DATASET_DROP_LIMIT,
        },
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
    }
    atomic_json(args.output_json, result)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_report.with_suffix(args.output_report.suffix + ".tmp")
    temporary.write_text(render_report(result))
    temporary.replace(args.output_report)


if __name__ == "__main__":
    main()
