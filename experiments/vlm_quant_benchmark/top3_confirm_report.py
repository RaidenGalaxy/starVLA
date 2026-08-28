"""Aggregate the Top3 4x256 confirmation protocol with paired statistics."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

from .dataset_protocol import ACTIVE_DATASET_NAMES
from .reporting import atomic_json


PROTOCOL_VERSION = 4
ACTIVE_METHODS = {"mbq", "omniquant", "squeezellm"}
OCR_DATASETS = ("DocVQA_VAL", "TextVQA_VAL")
REASONING_DATASETS = ("MMMU_DEV_VAL", "ScienceQA_TEST")
BINARY_DATASETS = set(REASONING_DATASETS)
LEGACY_CORPUS_ALIASES = {
    "omniquant_w4a16_g128_asym": "local_omniquant_w4a16_g128_asym",
    "squeezellm_w4a16_g128_asym": "local_squeezellm_w4a16_g128_asym",
    "rtn_w4a16_g128_asym": "naive_w4a16_g128_asym",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Top3 4x256 Chinese report")
    parser.add_argument("--confirm-root", type=Path, required=True)
    parser.add_argument("--screen64-root", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--legacy-corpus-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-dataset", type=int, default=256)
    return parser.parse_args()


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_ci(
    values: list[float], *, seed: int = 42, repeats: int = 10000,
) -> list[float]:
    rng = random.Random(seed)
    means = [
        100.0 * sum(rng.choice(values) for _ in values) / len(values)
        for _ in range(repeats)
    ]
    return [percentile(means, 0.025), percentile(means, 0.975)]


def paired_bootstrap(
    candidate: list[float],
    reference: list[float],
    *,
    seed: int = 42,
    repeats: int = 10000,
) -> dict:
    if len(candidate) != len(reference) or not candidate:
        raise ValueError("paired bootstrap requires equal non-empty samples")
    differences = [left - right for left, right in zip(candidate, reference)]
    rng = random.Random(seed)
    means = [
        100.0 * sum(rng.choice(differences) for _ in differences) / len(differences)
        for _ in range(repeats)
    ]
    interval = [percentile(means, 0.025), percentile(means, 0.975)]
    delta = 100.0 * sum(differences) / len(differences)
    conclusion = (
        "improved" if interval[0] > 0 else
        "degraded" if interval[1] < 0 else
        "indistinguishable"
    )
    return {
        "delta_pp": delta,
        "bootstrap_95_ci": interval,
        "conclusion": conclusion,
    }


def wilson_ci(values: list[float], z: float = 1.959963984540054) -> list[float]:
    count = len(values)
    probability = sum(values) / count
    denominator = 1 + z * z / count
    center = (probability + z * z / (2 * count)) / denominator
    margin = z * math.sqrt(
        probability * (1 - probability) / count + z * z / (4 * count * count)
    ) / denominator
    return [100 * (center - margin), 100 * (center + margin)]


def score_ci(dataset: str, values: list[float]) -> dict:
    if dataset in BINARY_DATASETS and set(values) <= {0.0, 1.0}:
        return {"type": "wilson_95", "interval": wilson_ci(values)}
    return {"type": "bootstrap_95", "interval": bootstrap_ci(values)}


def mcnemar_exact(candidate: list[float], reference: list[float]) -> dict:
    if len(candidate) != len(reference):
        raise ValueError("McNemar requires paired samples")
    ref_only = sum(
        right == 1.0 and left == 0.0
        for left, right in zip(candidate, reference)
    )
    candidate_only = sum(
        left == 1.0 and right == 0.0
        for left, right in zip(candidate, reference)
    )
    discordant = ref_only + candidate_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, index)
            for index in range(min(ref_only, candidate_only) + 1)
        ) / (2 ** discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "reference_only_correct": ref_only,
        "candidate_only_correct": candidate_only,
        "discordant": discordant,
        "exact_two_sided_p": p_value,
    }


def load_result(path: Path, expected: int | None = None) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("status") != "complete":
        raise ValueError(f"incomplete result: {path}")
    if set(payload.get("datasets", {})) != set(ACTIVE_DATASET_NAMES):
        raise ValueError(f"dataset mismatch: {path}")
    samples = {name: {} for name in ACTIVE_DATASET_NAMES}
    for row in payload.get("samples", []):
        dataset = row.get("dataset")
        if dataset not in samples:
            continue
        index = str(row["index"])
        if index in samples[dataset]:
            raise ValueError(f"duplicate sample: {path}:{dataset}:{index}")
        score = float(row["official_score"])
        if not math.isfinite(score):
            raise ValueError(f"non-finite score: {path}:{dataset}:{index}")
        samples[dataset][index] = score
    if expected is not None:
        bad = {
            name: len(values) for name, values in samples.items()
            if len(values) != expected
        }
        if bad:
            raise ValueError(f"sample count mismatch at {path}: {bad}")
    return {
        "path": str(path),
        "method": payload["method"],
        "scheme_id": payload["scheme_id"],
        "scope": payload["scope"],
        "weight_bits": payload.get("weight_bits"),
        "activation_bits": payload.get("activation_bits"),
        "logical_bits_per_weight": float(payload["logical_bits_per_weight"]),
        "samples": samples,
    }


def load_ppl(root: Path | None, scheme: str) -> dict | None:
    if root is None:
        return None
    directory = "bf16" if scheme == "bf16_value_fp32_compute" else scheme
    for candidate in (directory, LEGACY_CORPUS_ALIASES.get(directory)):
        if candidate is None:
            continue
        path = root / candidate / "corpus_metrics.json"
        if path.is_file():
            return json.loads(path.read_text())
    return None


def ppl_inflation(
    scheme: str,
    reference_scheme: str,
    corpus_root: Path,
    legacy_root: Path | None,
) -> float:
    candidate = load_ppl(corpus_root, scheme) or load_ppl(legacy_root, scheme)
    reference = (
        load_ppl(corpus_root, reference_scheme)
        or load_ppl(legacy_root, reference_scheme)
    )
    if candidate is None or reference is None:
        return math.inf
    ratios = [
        float(candidate[name]["ppl"]) / float(reference[name]["ppl"])
        for name in sorted(set(candidate) & set(reference))
    ]
    return sum(ratios) / len(ratios) if ratios else math.inf


def values_for(run: dict, dataset: str, indices: list[str]) -> list[float]:
    try:
        return [run["samples"][dataset][index] for index in indices]
    except KeyError as error:
        raise ValueError(
            f"paired index missing from {run['scheme_id']}:{dataset}: {error}"
        ) from error


def scores_for_slice(run: dict, slice_indices: dict[str, list[str]]) -> dict[str, float]:
    return {
        dataset: 100.0 * sum(values_for(run, dataset, indices)) / len(indices)
        for dataset, indices in slice_indices.items()
    }


def classify_quality(
    scores: dict[str, float], reference_scores: dict[str, float],
) -> dict:
    retentions = [
        scores[name] / reference_scores[name] * 100.0
        for name in ACTIVE_DATASET_NAMES
    ]
    drops = [
        max(0.0, reference_scores[name] - scores[name])
        for name in ACTIVE_DATASET_NAMES
    ]
    mean_retention = sum(retentions) / len(retentions)
    worst_retention = min(retentions)
    mean_drop = sum(drops) / len(drops)
    max_drop = max(drops)
    if (
        mean_retention >= 97 and worst_retention >= 95
        and mean_drop <= 2 and max_drop <= 5
    ):
        tier = "high_precision"
    elif (
        mean_retention >= 90 and worst_retention >= 85
        and mean_drop <= 5 and max_drop <= 10
    ):
        tier = "usable"
    else:
        tier = "degraded"
    return {
        "tier": tier,
        "mean_retention": mean_retention,
        "worst_retention": worst_retention,
        "mean_drop_pp": mean_drop,
        "max_drop_pp": max_drop,
        "retentions": {
            name: value for name, value in zip(ACTIVE_DATASET_NAMES, retentions)
        },
    }


def rank_key(row: dict) -> tuple:
    return (
        -row["quality"]["worst_retention"],
        -row["quality"]["mean_retention"],
        row["ppl_inflation"],
        row["logical_bits_per_weight"],
        row["method"],
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table(rows: list[dict]) -> list[str]:
    lines = [
        "| method | format | DocVQA | MMMU | ScienceQA | TextVQA | "
        "worst-ret | mean-ret | mean-drop | max-drop | tier | PPL× | BPW |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in rows:
        scores = row["slices"]["combined256"]["scores"]
        quality = row["quality"]
        ppl = row["ppl_inflation"]
        ppl_text = "-" if not math.isfinite(ppl) else f"{ppl:.3f}"
        lines.append(
            f"| {row['method']} | {row['format']} | "
            f"{scores['DocVQA_VAL']:.2f} | {scores['MMMU_DEV_VAL']:.2f} | "
            f"{scores['ScienceQA_TEST']:.2f} | {scores['TextVQA_VAL']:.2f} | "
            f"{quality['worst_retention']:.2f}% | "
            f"{quality['mean_retention']:.2f}% | "
            f"{quality['mean_drop_pp']:.2f} | {quality['max_drop_pp']:.2f} | "
            f"{quality['tier']} | {ppl_text} | "
            f"{row['logical_bits_per_weight']:.3f} |"
        )
    return lines


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    confirm_runs = [
        load_result(path, args.samples_per_dataset)
        for path in sorted(args.confirm_root.glob("*/official_result.json"))
    ]
    expected_methods = {"bf16_value_fp32_compute", *ACTIVE_METHODS}
    actual_methods = {run["method"] for run in confirm_runs}
    if actual_methods != expected_methods or len(confirm_runs) != 7:
        raise ValueError(
            f"expected seven BF16/Top3 runs, got {len(confirm_runs)}: {actual_methods}"
        )
    if any(run["method"] == "rtn" for run in confirm_runs):
        raise ValueError("RTN must not enter the 256-sample active matrix")

    historical_runs = [
        load_result(path, 64)
        for path in sorted(args.screen64_root.glob("*/official_result.json"))
    ]
    historical_bf = [
        run for run in historical_runs
        if run["method"] == "bf16_value_fp32_compute"
    ]
    if len(historical_bf) != 1:
        raise ValueError("historical screen must contain exactly one BF16 run")
    historical_rtn = [run for run in historical_runs if run["method"] == "rtn"]

    reference = next(
        run for run in confirm_runs
        if run["method"] == "bf16_value_fp32_compute"
    )
    screen64_indices = {
        name: list(historical_bf[0]["samples"][name])
        for name in ACTIVE_DATASET_NAMES
    }
    combined_indices = {
        name: list(reference["samples"][name])
        for name in ACTIVE_DATASET_NAMES
    }
    replication_indices = {}
    for name in ACTIVE_DATASET_NAMES:
        old = set(screen64_indices[name])
        combined = set(combined_indices[name])
        if not old <= combined:
            raise ValueError(f"confirm manifest does not contain old64: {name}")
        replication_indices[name] = [
            index for index in combined_indices[name] if index not in old
        ]
        if len(replication_indices[name]) != args.samples_per_dataset - 64:
            raise ValueError(f"replication slice is not 192 rows: {name}")

    slices = {
        "screen64": screen64_indices,
        "replication192": replication_indices,
        "combined256": combined_indices,
    }
    reference_scores = scores_for_slice(reference, combined_indices)
    reference_slice_scores = {
        name: scores_for_slice(reference, indices)
        for name, indices in slices.items()
    }
    rows = []
    for run in confirm_runs:
        slice_payload = {}
        for slice_name, indices in slices.items():
            slice_payload[slice_name] = {
                "sample_counts": {
                    name: len(values) for name, values in indices.items()
                },
                "scores": scores_for_slice(run, indices),
            }
        combined_scores = slice_payload["combined256"]["scores"]
        if run is reference:
            quality = {
                "tier": "reference",
                "mean_retention": 100.0,
                "worst_retention": 100.0,
                "mean_drop_pp": 0.0,
                "max_drop_pp": 0.0,
                "retentions": {name: 100.0 for name in ACTIVE_DATASET_NAMES},
            }
        else:
            quality = classify_quality(combined_scores, reference_scores)
        confidence = {
            name: score_ci(
                name, values_for(run, name, combined_indices[name]),
            )
            for name in ACTIVE_DATASET_NAMES
        }
        paired = {}
        if run is not reference:
            for name in ACTIVE_DATASET_NAMES:
                candidate_values = values_for(run, name, combined_indices[name])
                reference_values = values_for(reference, name, combined_indices[name])
                paired[name] = paired_bootstrap(candidate_values, reference_values)
                if name in BINARY_DATASETS:
                    paired[name]["mcnemar"] = mcnemar_exact(
                        candidate_values, reference_values,
                    )
        deltas64 = {
            name: (
                slice_payload["screen64"]["scores"][name]
                - reference_slice_scores["screen64"][name]
            )
            for name in ACTIVE_DATASET_NAMES
        }
        deltas192 = {
            name: (
                slice_payload["replication192"]["scores"][name]
                - reference_slice_scores["replication192"][name]
            )
            for name in ACTIVE_DATASET_NAMES
        }
        sign_flips = [
            name for name in ACTIVE_DATASET_NAMES
            if deltas64[name] * deltas192[name] < 0
        ] if run is not reference else []
        bits = run["weight_bits"]
        rows.append({
            "method": run["method"],
            "scheme_id": run["scheme_id"],
            "scope": run["scope"],
            "weight_bits": bits,
            "activation_bits": run["activation_bits"],
            "format": "BF16" if bits is None else f"W{bits}A{run['activation_bits'] or 16}",
            "logical_bits_per_weight": run["logical_bits_per_weight"],
            "ppl_inflation": ppl_inflation(
                run["scheme_id"], reference["scheme_id"],
                args.corpus_root, args.legacy_corpus_root,
            ),
            "slices": slice_payload,
            "confidence_intervals": confidence,
            "paired_vs_bf16": paired,
            "quality": quality,
            "stability": {
                "delta_screen64_pp": deltas64,
                "delta_replication192_pp": deltas192,
                "sign_flips": sign_flips,
                "stable": not sign_flips,
            },
        })

    winners = {}
    pairwise_winner_runner_up = {}
    for bits in (4, 3):
        candidates = sorted(
            [row for row in rows if row["weight_bits"] == bits],
            key=rank_key,
        )
        if not candidates:
            continue
        key = f"w{bits}a16"
        winners[key] = candidates[0]
        if len(candidates) > 1:
            left = next(
                run for run in confirm_runs
                if run["scheme_id"] == candidates[0]["scheme_id"]
            )
            right = next(
                run for run in confirm_runs
                if run["scheme_id"] == candidates[1]["scheme_id"]
            )
            left_values, right_values = [], []
            for name in ACTIVE_DATASET_NAMES:
                left_values.extend(values_for(left, name, combined_indices[name]))
                right_values.extend(values_for(right, name, combined_indices[name]))
            comparison = paired_bootstrap(left_values, right_values)
            comparison["winner"] = candidates[0]["scheme_id"]
            comparison["runner_up"] = candidates[1]["scheme_id"]
            comparison["statistically_tied"] = (
                comparison["conclusion"] == "indistinguishable"
            )
            pairwise_winner_runner_up[key] = comparison

    historical_rtn_payload = [{
        "method": run["method"],
        "scheme_id": run["scheme_id"],
        "format": f"W{run['weight_bits']}A{run['activation_bits'] or 16}",
        "samples_per_dataset": 64,
        "scores": scores_for_slice(run, screen64_indices),
        "status": "reference_baseline_64_only",
        "participates_in_256_ranking": False,
    } for run in historical_rtn]

    def serializable(row: dict) -> dict:
        return row | {
            "ppl_inflation": (
                None if not math.isfinite(row["ppl_inflation"])
                else row["ppl_inflation"]
            ),
        }

    atomic_json(args.output_dir / "summary.json", {
        "status": "complete",
        "protocol_version": PROTOCOL_VERSION,
        "datasets": list(ACTIVE_DATASET_NAMES),
        "samples_per_dataset": args.samples_per_dataset,
        "slices": {"screen64": 64, "replication192": 192, "combined256": 256},
        "runs": [serializable(row) for row in rows],
        "historical_rtn": historical_rtn_payload,
        "pairwise_winner_runner_up": pairwise_winner_runner_up,
        "engineering_thresholds_are_paper_standards": False,
    })
    atomic_json(args.output_dir / "selection.json", {
        "ranking": (
            "worst_retention_desc,mean_retention_desc,"
            "ppl_inflation_asc,logical_bpw_asc"
        ),
        "winners": {key: serializable(value) for key, value in winners.items()},
        "pairwise_winner_runner_up": pairwise_winner_runner_up,
    })

    flat_rows = []
    paired_rows = []
    for row in rows:
        scores = row["slices"]["combined256"]["scores"]
        flat_rows.append({
            "method": row["method"],
            "format": row["format"],
            **scores,
            "worst_retention": row["quality"]["worst_retention"],
            "mean_retention": row["quality"]["mean_retention"],
            "mean_drop_pp": row["quality"]["mean_drop_pp"],
            "max_drop_pp": row["quality"]["max_drop_pp"],
            "tier": row["quality"]["tier"],
            "ppl_inflation": (
                "" if not math.isfinite(row["ppl_inflation"])
                else row["ppl_inflation"]
            ),
            "logical_bits_per_weight": row["logical_bits_per_weight"],
            "stable_64_vs_192": row["stability"]["stable"],
        })
        for dataset, value in row["paired_vs_bf16"].items():
            paired_rows.append({
                "method": row["method"],
                "format": row["format"],
                "dataset": dataset,
                "delta_pp": value["delta_pp"],
                "ci_low": value["bootstrap_95_ci"][0],
                "ci_high": value["bootstrap_95_ci"][1],
                "conclusion": value["conclusion"],
                "mcnemar_p": value.get("mcnemar", {}).get("exact_two_sided_p", ""),
            })
    write_csv(args.output_dir / "comparison.csv", flat_rows)
    write_csv(args.output_dir / "paired_vs_bf16.csv", paired_rows)

    bf = next(row for row in rows if row["method"] == "bf16_value_fp32_compute")
    w4 = sorted([row for row in rows if row["weight_bits"] == 4], key=rank_key)
    w3 = sorted([row for row in rows if row["weight_bits"] == 3], key=rank_key)
    lines = [
        "# Qwen2.5-VL Top3 W4A16/W3A16 CPU 256样本确认报告",
        "",
        "> 协议：BF16、MBQ、OmniQuant、SqueezeLLM；decoder-only；"
        "DocVQA、MMMU、ScienceQA、TextVQA各固定256样本。RTN仅保留历史64样本参考。",
        "",
        "## 256样本主结果",
        "",
    ]
    lines.extend(table([bf]))
    lines.extend(["", "### W4A16", ""])
    lines.extend(table(w4))
    lines.extend(["", "### W3A16", ""])
    lines.extend(table(w3))
    lines.extend(["", "## 64筛选与独立192复现稳定性", ""])
    lines.extend([
        "| method | format | 64→192发生退化方向翻转的数据集 | stable |",
        "|---|---|---|---|",
    ])
    for row in [*w4, *w3]:
        flips = ", ".join(row["stability"]["sign_flips"]) or "-"
        lines.append(
            f"| {row['method']} | {row['format']} | {flips} | "
            f"{'yes' if row['stability']['stable'] else 'no'} |"
        )
    lines.extend(["", "## 方法相对BF16的配对统计", ""])
    lines.extend([
        "| method | format | dataset | Δpp | paired bootstrap 95% CI | conclusion | McNemar p |",
        "|---|---|---|---:|---|---|---:|",
    ])
    for row in [*w4, *w3]:
        for name in ACTIVE_DATASET_NAMES:
            value = row["paired_vs_bf16"][name]
            low, high = value["bootstrap_95_ci"]
            p_value = value.get("mcnemar", {}).get("exact_two_sided_p")
            p_text = "-" if p_value is None else f"{p_value:.4f}"
            lines.append(
                f"| {row['method']} | {row['format']} | {name} | "
                f"{value['delta_pp']:+.2f} | [{low:.2f}, {high:.2f}] | "
                f"{value['conclusion']} | {p_text} |"
            )
    lines.extend(["", "## RTN历史64样本参考（不参与排名）", ""])
    lines.extend([
        "| method | format | DocVQA | MMMU | ScienceQA | TextVQA | status |",
        "|---|---|---:|---:|---:|---:|---|",
    ])
    for item in historical_rtn_payload:
        score = item["scores"]
        lines.append(
            f"| rtn | {item['format']} | {score['DocVQA_VAL']:.2f} | "
            f"{score['MMMU_DEV_VAL']:.2f} | {score['ScienceQA_TEST']:.2f} | "
            f"{score['TextVQA_VAL']:.2f} | reference_baseline_64_only |"
        )
    lines.extend(["", "## 结论规则", ""])
    for key in ("w4a16", "w3a16"):
        if key not in winners:
            continue
        winner = winners[key]
        comparison = pairwise_winner_runner_up.get(key, {})
        tied = comparison.get("statistically_tied", False)
        lines.append(
            f"- {key.upper()}按固定工程排序第一名为“{winner['method']}”；"
            + (
                "与第二名总体配对差异无法明确区分。"
                if tied else "与第二名总体配对差异未被标记为统计并列。"
            )
        )
    lines.extend([
        "- high_precision/usable/degraded阈值是本项目工程标准，不是MBQ、"
        "OmniQuant或SqueezeLLM论文规定的通用可用阈值。",
        "- 对1–2个百分点的接近结果，以配对置信区间结论为准，不因均值排序"
        "宣称显著优劣。",
        "- W3/W4均为解量化后的CPU质量模拟，不代表真实打包显存或GPU加速。",
        "",
    ])
    (args.output_dir / "report_zh.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
