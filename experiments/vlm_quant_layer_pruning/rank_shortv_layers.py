from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .core import atomic_json
from .shortv_probe import SCHEMA_NAME


def read_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with path.open() as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    if not rows:
        raise ValueError("Q-ShortV ranking input is empty")
    if any(row.get("schema_name") != SCHEMA_NAME for row in rows):
        raise ValueError("ranking inputs do not use the Q-ShortV schema")
    return rows


def _bootstrap_indices(datasets: list[str], rng: np.random.Generator) -> np.ndarray:
    buckets: dict[str, list[int]] = defaultdict(list)
    for index, dataset in enumerate(datasets):
        buckets[dataset].append(index)
    drawn = []
    for dataset in sorted(buckets):
        values = np.asarray(buckets[dataset], dtype=np.int64)
        drawn.extend(rng.choice(values, size=len(values), replace=True).tolist())
    return np.asarray(drawn, dtype=np.int64)


def build_ranking(
    rows: list[dict], *, bootstrap_samples: int = 2000, seed: int = 42,
) -> dict:
    indexed = {}
    sample_dataset = {}
    for row in rows:
        if (
            row.get("quant_scheme") != "fp"
            or row.get("operation") != "freeze_visual"
            or len(row.get("layer_indices", [])) != 1
        ):
            continue
        sample = str(row["sample_id"])
        layer = int(row["layer_indices"][0])
        key = (sample, layer)
        if key in indexed:
            raise ValueError(f"duplicate FP single-layer row: {key}")
        indexed[key] = row
        sample_dataset.setdefault(sample, str(row["dataset"]))
    samples = sorted(sample_dataset)
    layers = sorted({layer for _, layer in indexed})
    if layers != list(range(36)):
        missing = sorted(set(range(36)) - set(layers))
        extra = sorted(set(layers) - set(range(36)))
        raise ValueError(f"FP ranking requires layers 0..35; missing={missing}, extra={extra}")
    expected = {(sample, layer) for sample in samples for layer in layers}
    missing = expected - set(indexed)
    if missing:
        raise ValueError(f"FP ranking is missing {len(missing)} sample/layer rows")
    lc = np.asarray([
        [indexed[(sample, layer)]["metrics"]["answer_logit_kl"] for layer in layers]
        for sample in samples
    ], dtype=np.float64)
    if not np.isfinite(lc).all() or (lc < -1e-8).any():
        raise ValueError("FP layer-contribution matrix is invalid")
    lc = np.maximum(lc, 0.0)
    datasets = [sample_dataset[sample] for sample in samples]
    point = np.median(lc, axis=0)
    rng = np.random.default_rng(seed)
    boot = np.empty((bootstrap_samples, len(layers)), dtype=np.float64)
    for iteration in range(bootstrap_samples):
        indices = _bootstrap_indices(datasets, rng)
        boot[iteration] = np.median(lc[indices], axis=0)
    per_layer = []
    for position, layer in enumerate(layers):
        ci = np.quantile(boot[:, position], [0.025, 0.975])
        per_layer.append({
            "layer": layer,
            "median_lc": float(point[position]),
            "ci95": [float(ci[0]), float(ci[1])],
            "selection_risk": float(ci[1]),
        })
    ranking = [
        row["layer"]
        for row in sorted(
            per_layer,
            key=lambda row: (row["selection_risk"], row["median_lc"], row["layer"]),
        )
    ]
    return {
        "schema_name": "qshortv_fp_ranking_v1",
        "schema_version": 1,
        "samples": len(samples),
        "datasets": dict(sorted({name: datasets.count(name) for name in set(datasets)}.items())),
        "layers": layers,
        "ranking_rule": "ascending stratified-bootstrap upper95 median FP answer-logit KL",
        "ranking": ranking,
        "per_layer": per_layer,
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
    }


def render_report(result: dict) -> str:
    lines = [
        "# Q-ShortV FP 视觉冻结层排序",
        "",
        f"样本：{result['samples']}；bootstrap：{result['bootstrap_samples']}。",
        "",
        "排序依据：FP answer-logit KL 中位数的分层 bootstrap 95% 上界；越靠前越安全。",
        "",
        f"固定排序：`{','.join(map(str, result['ranking']))}`",
        "",
        "| 层 | median LC | 95% CI | selection risk |",
        "|---:|---:|---|---:|",
    ]
    for row in sorted(result["per_layer"], key=lambda value: result["ranking"].index(value["layer"])):
        lines.append(
            f"| {row['layer']} | {row['median_lc']:.8f} | {row['ci95']} "
            f"| {row['selection_risk']:.8f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the fixed FP Q-ShortV layer ranking")
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = build_ranking(
        read_rows(args.input), bootstrap_samples=args.bootstrap_samples, seed=args.seed,
    )
    atomic_json(args.output_json, result)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_report.with_suffix(args.output_report.suffix + ".tmp")
    temporary.write_text(render_report(result))
    temporary.replace(args.output_report)


if __name__ == "__main__":
    main()
