"""Merge quantization outcomes with modality sensitivity and produce diagnostics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .reporting import atomic_json


METHOD_NAMES = {
    "bf16_value_fp32_compute": "bf16",
    "mbq": "mbq",
    "omniquant": "omniquant",
    "rtn": "rtn",
    "squeezellm": "squeezellm",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze MBQ modality sensitivity and quantization degradation",
    )
    parser.add_argument("--top3-root", type=Path, required=True)
    parser.add_argument("--top4-root", type=Path)
    parser.add_argument("--sensitivity-jsonl", type=Path)
    parser.add_argument("--perturbations-jsonl", type=Path)
    parser.add_argument("--ablation-jsonl", type=Path)
    parser.add_argument("--residuals-jsonl", type=Path)
    parser.add_argument("--promotion-root", type=Path)
    parser.add_argument("--top2-selection", type=Path)
    parser.add_argument("--mbq-audit", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict]:
    if not path or not path.is_file():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _method_and_format(directory: str) -> tuple[str, str]:
    if directory == "bf16_value_fp32_compute":
        return "bf16", "BF16"
    method = next(
        (value for value in METHOD_NAMES if directory.startswith(value + "_")),
        directory.split("_", 1)[0],
    )
    if "_w3a16_" in directory:
        fmt = "W3A16"
    elif "_w4a16_" in directory:
        fmt = "W4A16"
    elif "_w4a8_" in directory:
        fmt = "W4A8"
    else:
        fmt = "unknown"
    return METHOD_NAMES.get(method, method), fmt


def _load_protocol_scores(root: Path, protocol: str) -> list[dict]:
    records = []
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        method, fmt = _method_and_format(run_dir.name)
        if method not in METHOD_NAMES.values():
            continue
        for dataset_dir in sorted(path for path in run_dir.iterdir() if path.is_dir()):
            samples = dataset_dir / "official_samples.jsonl"
            if not samples.is_file():
                continue
            for row in _read_jsonl(samples):
                score = row.get("official_score")
                if score is None or not math.isfinite(float(score)):
                    continue
                records.append({
                    "protocol": protocol,
                    "run_dir": run_dir.name,
                    "method": method,
                    "format": fmt,
                    "dataset": str(row.get("dataset", dataset_dir.name)),
                    "id": str(row["id"]),
                    "score": 100.0 * float(score),
                })
    return records


def build_quant_degradation(
    top3_root: Path,
    top4_root: Path | None = None,
) -> list[dict]:
    """Build long-form paired score degradation rows."""
    raw = _load_protocol_scores(top3_root, "top3_confirm_256")
    if top4_root is not None:
        raw.extend(_load_protocol_scores(top4_root, "top4_screen_64"))
    references = {
        (row["protocol"], row["dataset"], row["id"]): row["score"]
        for row in raw if row["method"] == "bf16"
    }
    result = []
    for row in raw:
        key = (row["protocol"], row["dataset"], row["id"])
        if key not in references:
            continue
        reference = references[key]
        result.append(row | {
            "bf16_score": reference,
            "score_delta_vs_bf16": row["score"] - reference,
        })
    paired = defaultdict(dict)
    for row in result:
        if row["protocol"] == "top3_confirm_256" and row["format"] == "W4A16":
            paired[(row["dataset"], row["id"])][row["method"]] = row["score"]
    mbq_squeeze = {}
    for key, values in paired.items():
        if "mbq" in values and "squeezellm" in values:
            mbq_squeeze[key] = values["mbq"] - values["squeezellm"]
    for row in result:
        row["mbq_minus_squeezellm"] = mbq_squeeze.get(
            (row["dataset"], row["id"]), math.nan,
        )
    return result


def build_nll_degradation(rows: list[dict]) -> list[dict]:
    """Pair current MBQ/SqueezeLLM target NLL with BF16 on sensitivity samples."""
    current = [
        row for row in rows
        if row.get("evaluation_split") == "sensitivity"
        and row.get("variant") in {"bf16", "mbq_current", "squeezellm_current"}
    ]
    grouped = defaultdict(dict)
    for row in current:
        grouped[(row["dataset"], row["id"])][row["variant"]] = float(
            row["target_nll"]
        )
    result = []
    for (dataset, identifier), values in sorted(grouped.items()):
        if not {"bf16", "mbq_current", "squeezellm_current"} <= set(values):
            continue
        result.append({
            "dataset": dataset,
            "id": identifier,
            "bf16_target_nll": values["bf16"],
            "mbq_target_nll": values["mbq_current"],
            "squeezellm_target_nll": values["squeezellm_current"],
            "mbq_nll_delta_vs_bf16": values["mbq_current"] - values["bf16"],
            "squeezellm_nll_delta_vs_bf16": (
                values["squeezellm_current"] - values["bf16"]
            ),
            "mbq_minus_squeezellm_nll": (
                values["mbq_current"] - values["squeezellm_current"]
            ),
        })
    return result


def enrich_score_degradation_with_nll(
    degradation: list[dict],
    nll_rows: list[dict],
) -> None:
    nll = {(row["dataset"], row["id"]): row for row in nll_rows}
    for row in degradation:
        if row["protocol"] != "top3_confirm_256":
            continue
        value = nll.get((row["dataset"], row["id"]))
        if value is None:
            continue
        row.update(value)


def _write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _finite(values) -> list[float]:
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def _analysis_ratio(row: dict) -> float:
    """Prefer the causal predictor-position ratio, retaining v1 compatibility."""
    return row.get(
        "vision_to_answer_predictor_ratio",
        row.get("vision_to_answer_ratio", math.nan),
    )


def _bootstrap_median_ci(
    values: list[float],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    values = _finite(values)
    if not values:
        return math.nan, math.nan
    rng = random.Random(seed)
    medians = [
        statistics.median(rng.choices(values, k=len(values)))
        for _ in range(samples)
    ]
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def summarize_sensitivity(
    rows: list[dict],
    *,
    bootstrap_samples: int,
    seed: int,
) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(
            row.get("sample_kind", "unknown"),
            row["dataset"],
            int(row["layer"]),
            row["kind"],
        )].append(row)
    summaries = []
    for key, values in sorted(grouped.items()):
        ratios = _finite(_analysis_ratio(row) for row in values)
        label_ratios = _finite(
            row.get("vision_to_answer_ratio") for row in values
        )
        prompt = _finite(row["vision_to_prompt_ratio"] for row in values)
        mass = _finite(row["vision_mass_share"] for row in values)
        low, high = _bootstrap_median_ci(
            ratios, samples=bootstrap_samples,
            seed=seed + key[2],
        )
        summaries.append({
            "sample_kind": key[0],
            "dataset": key[1],
            "layer": key[2],
            "kind": key[3],
            "samples": len({row["id"] for row in values}),
            "median_vision_to_answer_ratio": (
                statistics.median(ratios) if ratios else math.nan
            ),
            "median_vision_to_answer_label_ratio": (
                statistics.median(label_ratios) if label_ratios else math.nan
            ),
            "answer_ratio_basis": (
                "causal_predictor_position"
                if any("vision_to_answer_predictor_ratio" in row for row in values)
                else "upstream_label_position_legacy"
            ),
            "ratio_ci_low": low,
            "ratio_ci_high": high,
            "median_vision_to_prompt_ratio": (
                statistics.median(prompt) if prompt else math.nan
            ),
            "median_vision_mass_share": (
                statistics.median(mass) if mass else math.nan
            ),
        })
    return summaries


def gradient_aggregation_rows(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(
            row.get("sample_kind", "unknown"),
            int(row["layer"]),
            row["kind"],
        )].append(row)
    results = []
    for (sample_kind, layer, kind), values in sorted(grouped.items()):
        for mode in ("pooled_elements", "sample_mean", "source_balanced"):
            if mode == "pooled_elements":
                vision = sum(
                    row["gradients"]["vision"]["sum"] for row in values
                ) / sum(row["gradients"]["vision"]["count"] for row in values)
                language = sum(
                    row["gradients"].get(
                        "answer_predictor_language",
                        row["gradients"]["answer_language"],
                    )["sum"] for row in values
                ) / sum(
                    row["gradients"].get(
                        "answer_predictor_language",
                        row["gradients"]["answer_language"],
                    )["count"] for row in values
                )
            elif mode == "sample_mean":
                vision = statistics.fmean(
                    row["gradients"]["vision"]["mean"] for row in values
                )
                language = statistics.fmean(
                    row["gradients"].get(
                        "answer_predictor_language",
                        row["gradients"]["answer_language"],
                    )["mean"] for row in values
                )
            else:
                datasets = sorted({row["dataset"] for row in values})
                vision = statistics.fmean(
                    statistics.fmean(
                        row["gradients"]["vision"]["mean"]
                        for row in values if row["dataset"] == dataset
                    )
                    for dataset in datasets
                )
                language = statistics.fmean(
                    statistics.fmean(
                        row["gradients"].get(
                            "answer_predictor_language",
                            row["gradients"]["answer_language"],
                        )["mean"]
                        for row in values if row["dataset"] == dataset
                    )
                    for dataset in datasets
                )
            ratio = vision / language if language > 0 else math.nan
            vision_tokens = sum(
                row["token_counts"]["vision"] for row in values
            )
            language_tokens = sum(
                row["token_counts"].get(
                    "answer_predictor_language",
                    row["token_counts"]["answer_language"],
                ) for row in values
            )
            results.append({
                "sample_kind": sample_kind,
                "layer": layer,
                "kind": kind,
                "aggregation": mode,
                "samples": len({row["id"] for row in values}),
                "vision_gradient": vision,
                "answer_language_gradient": language,
                "answer_ratio_basis": (
                    "causal_predictor_position"
                    if any(
                        "answer_predictor_language" in row["gradients"]
                        for row in values
                    ) else "upstream_label_position_legacy"
                ),
                "vision_to_answer_ratio": ratio,
                "vision_tokens": vision_tokens,
                "answer_language_tokens": language_tokens,
                "vision_mass_share": (
                    ratio * vision_tokens
                    / (ratio * vision_tokens + language_tokens)
                    if ratio >= 0 and language_tokens else math.nan
                ),
            })
    return results


def sample_sensitivity_features(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        if row.get("sample_kind") == "target":
            grouped[(row["dataset"], row["id"])].append(row)
    features = []
    for (dataset, identifier), values in sorted(grouped.items()):
        ratios = _finite(_analysis_ratio(row) for row in values)
        prompt = _finite(row["vision_to_prompt_ratio"] for row in values)
        mass = _finite(row["vision_mass_share"] for row in values)
        if not ratios:
            continue
        features.append({
            "dataset": dataset,
            "id": identifier,
            "median_log10_vision_to_answer_ratio": statistics.median(
                math.log10(max(value, 1e-30)) for value in ratios
            ),
            "median_log10_vision_to_prompt_ratio": (
                statistics.median(math.log10(max(value, 1e-30)) for value in prompt)
                if prompt else math.nan
            ),
            "median_vision_mass_share": statistics.median(mass) if mass else math.nan,
        })
    return features


def dataset_sensitivity_rows(
    features: list[dict],
    *,
    bootstrap_samples: int,
    seed: int,
) -> list[dict]:
    grouped = defaultdict(list)
    for row in features:
        grouped[row["dataset"]].append(
            10 ** row["median_log10_vision_to_answer_ratio"]
        )
    results = []
    for index, (dataset, values) in enumerate(sorted(grouped.items())):
        low, high = _bootstrap_median_ci(
            values, samples=bootstrap_samples, seed=seed + index,
        )
        results.append({
            "dataset": dataset,
            "samples": len(values),
            "median_vision_to_answer_ratio": statistics.median(values),
            "ratio_ci_low": low,
            "ratio_ci_high": high,
            "motivation_supported": high < 1,
        })
    return results


def _dataset_residual(values: list[float], datasets: list[str]) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    for dataset in sorted(set(datasets)):
        indices = [index for index, item in enumerate(datasets) if item == dataset]
        result[indices] -= result[indices].mean()
    return result


def _adjusted_slope(x: list[float], y: list[float], datasets: list[str]) -> float:
    xr = _dataset_residual(x, datasets)
    yr = _dataset_residual(y, datasets)
    denominator = float(np.dot(xr, xr))
    return float(np.dot(xr, yr) / denominator) if denominator else math.nan


def correlation_rows(
    degradation: list[dict],
    sensitivity: list[dict],
    *,
    bootstrap_samples: int,
    seed: int,
) -> list[dict]:
    features = {(row["dataset"], row["id"]): row for row in sensitivity}
    grouped = defaultdict(list)
    for row in degradation:
        if row["protocol"] != "top3_confirm_256" or row["method"] == "bf16":
            continue
        feature = features.get((row["dataset"], row["id"]))
        if feature is not None:
            grouped[(row["method"], row["format"])].append((row, feature))
    results = []
    for (method, fmt), values in sorted(grouped.items()):
        x = [item[1]["median_log10_vision_to_answer_ratio"] for item in values]
        y = [item[0]["score_delta_vs_bf16"] for item in values]
        datasets = [item[0]["dataset"] for item in values]
        if len(set(x)) < 2 or len(set(y)) < 2:
            rho, p_value = math.nan, math.nan
        else:
            rho, p_value = spearmanr(x, y)
        slope = _adjusted_slope(x, y, datasets)
        rng = random.Random(f"{seed}:{method}:{fmt}")
        slopes = []
        buckets = {
            dataset: [index for index, item in enumerate(datasets) if item == dataset]
            for dataset in sorted(set(datasets))
        }
        for _ in range(bootstrap_samples):
            indices = []
            for bucket in buckets.values():
                indices.extend(rng.choices(bucket, k=len(bucket)))
            slopes.append(_adjusted_slope(
                [x[index] for index in indices],
                [y[index] for index in indices],
                [datasets[index] for index in indices],
            ))
        finite_slopes = _finite(slopes)
        results.append({
            "method": method,
            "format": fmt,
            "outcome": "score_delta_vs_bf16",
            "samples": len(values),
            "spearman_rho": float(rho),
            "spearman_p": float(p_value),
            "dataset_adjusted_slope": slope,
            "slope_ci_low": (
                float(np.quantile(finite_slopes, 0.025)) if finite_slopes else math.nan
            ),
            "slope_ci_high": (
                float(np.quantile(finite_slopes, 0.975)) if finite_slopes else math.nan
            ),
        })
    return results


def nll_correlation_rows(
    sensitivity: list[dict],
    nll_rows: list[dict],
    *,
    bootstrap_samples: int,
    seed: int,
) -> list[dict]:
    features = {(row["dataset"], row["id"]): row for row in sensitivity}
    outcomes = (
        "mbq_nll_delta_vs_bf16",
        "squeezellm_nll_delta_vs_bf16",
        "mbq_minus_squeezellm_nll",
    )
    results = []
    for outcome in outcomes:
        values = [
            (row, features[(row["dataset"], row["id"])])
            for row in nll_rows
            if (row["dataset"], row["id"]) in features
        ]
        if len(values) < 4:
            continue
        x = [item[1]["median_log10_vision_to_answer_ratio"] for item in values]
        y = [item[0][outcome] for item in values]
        datasets = [item[0]["dataset"] for item in values]
        if len(set(x)) < 2 or len(set(y)) < 2:
            rho, p_value = math.nan, math.nan
        else:
            rho, p_value = spearmanr(x, y)
        slope = _adjusted_slope(x, y, datasets)
        buckets = {
            dataset: [index for index, value in enumerate(datasets) if value == dataset]
            for dataset in sorted(set(datasets))
        }
        rng = random.Random(f"{seed}:nll:{outcome}")
        boot = []
        for _ in range(bootstrap_samples):
            indices = []
            for bucket in buckets.values():
                indices.extend(rng.choices(bucket, k=len(bucket)))
            boot.append(_adjusted_slope(
                [x[index] for index in indices],
                [y[index] for index in indices],
                [datasets[index] for index in indices],
            ))
        finite_boot = _finite(boot)
        results.append({
            "method": "paired_current",
            "format": "W4A16",
            "outcome": outcome,
            "samples": len(values),
            "spearman_rho": float(rho),
            "spearman_p": float(p_value),
            "dataset_adjusted_slope": slope,
            "slope_ci_low": float(np.quantile(finite_boot, 0.025)) if finite_boot else math.nan,
            "slope_ci_high": float(np.quantile(finite_boot, 0.975)) if finite_boot else math.nan,
        })
    return results


def _bh_fdr(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [math.nan] * count
    running = 1.0
    for rank_index in range(count - 1, -1, -1):
        original = order[rank_index]
        rank = rank_index + 1
        running = min(running, p_values[original] * count / rank)
        adjusted[original] = min(running, 1.0)
    return adjusted


def layer_correlation_rows(
    degradation: list[dict],
    sensitivity: list[dict],
    nll_rows: list[dict] | None = None,
) -> list[dict]:
    outcomes = {}
    for row in degradation:
        if (
            row["protocol"] == "top3_confirm_256"
            and row["method"] == "mbq"
            and row["format"] == "W4A16"
        ):
            outcomes[(row["dataset"], row["id"])] = {
                "score_delta_vs_bf16": row["score_delta_vs_bf16"],
                "mbq_minus_squeezellm": row["mbq_minus_squeezellm"],
            }
    for row in nll_rows or []:
        outcomes.setdefault((row["dataset"], row["id"]), {}).update({
            "mbq_nll_delta_vs_bf16": row["mbq_nll_delta_vs_bf16"],
            "mbq_minus_squeezellm_nll": row["mbq_minus_squeezellm_nll"],
        })
    grouped = defaultdict(list)
    for row in sensitivity:
        if row.get("sample_kind") != "target":
            continue
        outcome = outcomes.get((row["dataset"], row["id"]))
        ratio = _analysis_ratio(row)
        if outcome is None or ratio is None or ratio <= 0 or not math.isfinite(ratio):
            continue
        for name, value in outcome.items():
            if math.isfinite(value):
                grouped[(int(row["layer"]), row["kind"], name)].append(
                    (math.log10(ratio), value)
                )
    results = []
    for (layer, kind, outcome), values in sorted(grouped.items()):
        if len(values) < 4:
            continue
        x_values = [value[0] for value in values]
        y_values = [value[1] for value in values]
        if len(set(x_values)) < 2 or len(set(y_values)) < 2:
            rho, p_value = math.nan, math.nan
        else:
            rho, p_value = spearmanr(x_values, y_values)
        results.append({
            "method": "mbq",
            "format": "W4A16",
            "outcome": outcome,
            "layer": layer,
            "kind": kind,
            "samples": len(values),
            "spearman_rho": float(rho),
            "spearman_p": float(p_value),
            "dataset_adjusted_slope": math.nan,
            "slope_ci_low": math.nan,
            "slope_ci_high": math.nan,
        })
    finite_indices = [
        index for index, row in enumerate(results)
        if math.isfinite(row["spearman_p"])
    ]
    adjusted = _bh_fdr([results[index]["spearman_p"] for index in finite_indices])
    for index, value in zip(finite_indices, adjusted):
        results[index]["spearman_fdr"] = value
    return results


def domain_shift_rows(
    sensitivity: list[dict], *, bootstrap_samples: int, seed: int,
) -> list[dict]:
    calibration = defaultdict(lambda: defaultdict(list))
    target = defaultdict(list)
    for row in sensitivity:
        ratio = _analysis_ratio(row)
        if ratio is None or ratio <= 0 or not math.isfinite(float(ratio)):
            continue
        value = math.log10(float(ratio))
        key = (int(row["layer"]), row["kind"])
        if row.get("sample_kind") == "calibration":
            calibration[key][row["dataset"]].append(value)
        elif row.get("sample_kind") == "target":
            target[(row["dataset"], *key)].append(value)
    results = []
    for index, ((dataset, layer, kind), target_values) in enumerate(sorted(target.items())):
        source_buckets = calibration.get((layer, kind), {})
        if not source_buckets or not target_values:
            continue
        calibration_log = statistics.median(
            statistics.median(values) for values in source_buckets.values()
        )
        target_log = statistics.median(target_values)
        shift = target_log - calibration_log
        rng = random.Random(f"{seed}:domain:{index}:{dataset}:{layer}:{kind}")
        boot = []
        for _ in range(bootstrap_samples):
            cal = statistics.median(
                statistics.median(rng.choices(values, k=len(values)))
                for values in source_buckets.values()
            )
            tar = statistics.median(rng.choices(target_values, k=len(target_values)))
            boot.append(tar - cal)
        low = float(np.quantile(boot, 0.025))
        high = float(np.quantile(boot, 0.975))
        results.append({
            "dataset": dataset, "layer": layer, "kind": kind,
            "calibration_ratio": 10 ** calibration_log,
            "target_ratio": 10 ** target_log,
            "log10_shift": shift,
            "shift_ci_low": low, "shift_ci_high": high,
            "shift_ci_excludes_zero": low > 0 or high < 0,
            "exceeds_twofold": abs(shift) >= 0.3,
        })
    return results


def _svg_layer_plot(path: Path, summaries: list[dict]) -> None:
    values = [
        row for row in summaries
        if math.isfinite(row["median_vision_to_answer_ratio"])
        and row["median_vision_to_answer_ratio"] > 0
    ]
    width, height, margin = 900, 440, 55
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="55" y="28" font-size="18">Layer-wise vision/language sensitivity</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" '
        f'y2="{height-margin}" stroke="black"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" '
        f'y2="{height-margin}" stroke="black"/>',
    ]
    colors = {"calibration": "#1f77b4", "target": "#d62728"}
    for sample_kind in ("calibration", "target"):
        series = defaultdict(list)
        for row in values:
            if row["sample_kind"] == sample_kind:
                series[row["layer"]].append(
                    math.log10(row["median_vision_to_answer_ratio"])
                )
        points = []
        for layer, layer_values in sorted(series.items()):
            x = margin + layer / 35 * (width - 2 * margin)
            log_value = statistics.median(layer_values)
            y = margin + (1 - (log_value + 3) / 4) * (height - 2 * margin)
            points.append(f"{x:.1f},{y:.1f}")
        if points:
            lines.append(
                f'<polyline fill="none" stroke="{colors[sample_kind]}" '
                f'stroke-width="2" points="{" ".join(points)}"/>'
            )
    lines.extend([
        '<text x="400" y="430" font-size="13">decoder layer</text>',
        '<text x="65" y="75" font-size="12">log10 ratio; blue=calibration red=target</text>',
        "</svg>",
    ])
    path.write_text("\n".join(lines))


def _svg_scatter(
    path: Path,
    degradation: list[dict],
    features: list[dict],
) -> None:
    feature_map = {(row["dataset"], row["id"]): row for row in features}
    points = []
    for row in degradation:
        if not (
            row["protocol"] == "top3_confirm_256"
            and row["method"] == "mbq"
            and row["format"] == "W4A16"
        ):
            continue
        feature = feature_map.get((row["dataset"], row["id"]))
        outcome = row.get("mbq_minus_squeezellm")
        if feature is not None and outcome is not None and math.isfinite(float(outcome)):
            points.append((
                feature["median_log10_vision_to_answer_ratio"],
                float(outcome),
                row["dataset"],
            ))
    _write_scatter_svg(
        path,
        points,
        title="Sensitivity vs MBQ-SqueezeLLM degradation",
        x_label="median log10 vision/language sensitivity",
        y_label="MBQ - SqueezeLLM score",
        diagonal=False,
    )


def _svg_domain_shift(path: Path, shifts: list[dict]) -> None:
    points = [
        (
            math.log10(row["calibration_ratio"]),
            math.log10(row["target_ratio"]),
            row["dataset"],
        )
        for row in shifts
        if row["calibration_ratio"] > 0 and row["target_ratio"] > 0
    ]
    _write_scatter_svg(
        path,
        points,
        title="Calibration-to-target sensitivity shift",
        x_label="calibration median log10 ratio",
        y_label="target median log10 ratio",
        diagonal=True,
    )


def _write_scatter_svg(
    path: Path,
    points: list[tuple[float, float, str]],
    *,
    title: str,
    x_label: str,
    y_label: str,
    diagonal: bool,
) -> None:
    width, height, margin = 820, 500, 70
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="70" y="30" font-size="18">{title}</text>',
    ]
    if not points:
        lines.extend([
            '<text x="70" y="90" font-size="14">no matched sensitivity data</text>',
            "</svg>",
        ])
        path.write_text("\n".join(lines))
        return
    x_values = [item[0] for item in points]
    y_values = [item[1] for item in points]
    x_low, x_high = min(x_values), max(x_values)
    y_low, y_high = min(y_values), max(y_values)
    if x_low == x_high:
        x_low, x_high = x_low - 0.5, x_high + 0.5
    if y_low == y_high:
        y_low, y_high = y_low - 0.5, y_high + 0.5
    x_pad, y_pad = (x_high - x_low) * 0.08, (y_high - y_low) * 0.08
    x_low, x_high = x_low - x_pad, x_high + x_pad
    y_low, y_high = y_low - y_pad, y_high + y_pad

    def project_x(value: float) -> float:
        return margin + (value - x_low) / (x_high - x_low) * (width - 2 * margin)

    def project_y(value: float) -> float:
        return height - margin - (
            (value - y_low) / (y_high - y_low) * (height - 2 * margin)
        )

    lines.extend([
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" '
        f'y2="{height-margin}" stroke="black"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" '
        f'y2="{height-margin}" stroke="black"/>',
    ])
    if diagonal:
        common_low, common_high = max(x_low, y_low), min(x_high, y_high)
        if common_low < common_high:
            lines.append(
                f'<line x1="{project_x(common_low):.1f}" '
                f'y1="{project_y(common_low):.1f}" '
                f'x2="{project_x(common_high):.1f}" '
                f'y2="{project_y(common_high):.1f}" '
                'stroke="#777" stroke-dasharray="5,5"/>'
            )
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    datasets = sorted({item[2] for item in points})
    colors = {dataset: palette[index % len(palette)] for index, dataset in enumerate(datasets)}
    for x_value, y_value, dataset in points:
        lines.append(
            f'<circle cx="{project_x(x_value):.1f}" cy="{project_y(y_value):.1f}" '
            f'r="3" fill="{colors[dataset]}" fill-opacity="0.7"/>'
        )
    lines.extend([
        f'<text x="{width / 2 - 120:.0f}" y="{height - 18}" font-size="12">{x_label}</text>',
        f'<text x="8" y="{height / 2:.0f}" font-size="12" '
        f'transform="rotate(-90 8 {height / 2:.0f})">{y_label}</text>',
        f'<text x="{margin}" y="{height - margin + 20}" font-size="11">{x_low:.3g}</text>',
        f'<text x="{width - margin - 35}" y="{height - margin + 20}" font-size="11">{x_high:.3g}</text>',
        "</svg>",
    ])
    path.write_text("\n".join(lines))


def _file_digest(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audit_observations(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {}
    audit = json.loads(path.read_text())
    counts = audit.get("token_counts", {})
    ratio = audit.get("gradient_ratio_medians", {})
    median_ratio = statistics.median(_finite(ratio.values())) if ratio else math.nan
    return {
        "vision_tokens": counts.get("vision"),
        "language_tokens": counts.get("language"),
        "median_gradient_ratio": median_ratio,
        "vision_mass_share": (
            median_ratio * counts["vision"]
            / (median_ratio * counts["vision"] + counts["language"])
            if median_ratio > 0 and counts.get("vision") and counts.get("language")
            else math.nan
        ),
    }


def _balanced_bootstrap_mean(
    values: list[tuple[str, float]], *, samples: int, seed: str,
) -> dict:
    buckets = defaultdict(list)
    for dataset, value in values:
        if math.isfinite(value):
            buckets[dataset].append(value)
    if not buckets:
        return {"mean": math.nan, "ci_low": math.nan, "ci_high": math.nan, "samples": 0}
    mean = statistics.fmean(statistics.fmean(bucket) for bucket in buckets.values())
    rng = random.Random(seed)
    boot = [
        statistics.fmean(
            statistics.fmean(rng.choices(bucket, k=len(bucket)))
            for bucket in buckets.values()
        )
        for _ in range(samples)
    ]
    return {
        "mean": mean,
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "samples": sum(len(bucket) for bucket in buckets.values()),
        "datasets": len(buckets),
    }


def ablation_effects(rows: list[dict], *, samples: int, seed: int) -> list[dict]:
    values = defaultdict(dict)
    for row in rows:
        if row.get("evaluation_split") == "ablation":
            values[(str(row["dataset"]), str(row["id"]))][str(row["variant"])] = float(row["target_nll"])
    effects = []
    for variant in sorted({name for item in values.values() for name in item}):
        if variant in {"bf16", "mbq_current", "squeezellm_current"}:
            continue
        baseline = "squeezellm_current" if variant == "squeezellm_no_sparse" else "mbq_current"
        paired = [
            (dataset, item[baseline] - item[variant])
            for (dataset, _identifier), item in values.items()
            if baseline in item and variant in item
        ]
        effect = _balanced_bootstrap_mean(
            paired, samples=samples, seed=f"{seed}:ablation:{variant}",
        )
        effects.append({"variant": variant, "baseline": baseline, **effect})
    return effects


def perturbation_effect(perturbations: list[dict], *, samples: int, seed: int) -> dict:
    grouped = defaultdict(dict)
    for row in perturbations:
        grouped[(str(row["dataset"]), str(row["id"]), int(row["layer"]))][row["modality"]] = float(row["delta_nll"])
    paired = [
        (dataset, item["answer_language"] - item["vision"])
        for (dataset, _identifier, _layer), item in grouped.items()
        if {"answer_language", "vision"} <= set(item)
    ]
    return _balanced_bootstrap_mean(
        paired, samples=samples, seed=f"{seed}:perturbation",
    )


def w4_mbq_squeeze_effect(
    degradation: list[dict], *, samples: int, seed: int,
) -> dict:
    paired = {}
    for row in degradation:
        if (
            row["protocol"] == "top3_confirm_256"
            and row["method"] == "mbq" and row["format"] == "W4A16"
            and math.isfinite(float(row.get("mbq_minus_squeezellm", math.nan)))
        ):
            paired[(row["dataset"], row["id"])] = float(row["mbq_minus_squeezellm"])
    return _balanced_bootstrap_mean(
        [(dataset, value) for (dataset, _identifier), value in paired.items()],
        samples=samples, seed=f"{seed}:w4-mbq-squeeze",
    )


def load_promotion_scores(root: Path | None, selection_path: Path | None) -> list[dict]:
    if root is None or selection_path is None or not selection_path.is_file():
        return []
    selection = json.loads(selection_path.read_text())
    rows = []
    for variant in selection.get("selected", []):
        result_path = root / variant / "official_result.json"
        if not result_path.is_file():
            continue
        payload = json.loads(result_path.read_text())
        for dataset, score in payload.get("datasets", {}).items():
            rows.append({
                "variant": variant,
                "dataset": dataset,
                "samples": score.get("samples"),
                "score": score.get("score"),
                "status": score.get("status", payload.get("status")),
                "exploratory_promotion": next(
                    (item.get("exploratory_promotion") for item in selection.get("ranked", []) if item["variant"] == variant),
                    None,
                ),
            })
    return rows


def sensitivity_reconstruction_products(
    sensitivity: list[dict], residuals: list[dict], nll_rows: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Combine gradients and actual W4 residuals at matched linear outputs."""
    gradients = {}
    for row in sensitivity:
        if row.get("sample_kind") != "target":
            continue
        projection = "o_proj" if row["kind"] == "attention" else "down_proj"
        gradients[(row["dataset"], row["id"], int(row["layer"]), projection)] = row
    outcomes = {(row["dataset"], row["id"]): row for row in nll_rows}
    products = []
    for row in residuals:
        if (
            row.get("location") != "linear_output"
            or row.get("method") not in {"mbq_current", "squeezellm_current"}
            or row.get("modality") not in {"vision", "answer_language"}
        ):
            continue
        key = (row["dataset"], row["id"], int(row["layer"]), row["projection"])
        gradient = gradients.get(key)
        outcome = outcomes.get((row["dataset"], row["id"]))
        if gradient is None or outcome is None or not math.isfinite(float(row["rmse"])):
            continue
        modality = row["modality"]
        gradient_key = (
            "vision" if modality == "vision" else
            "answer_predictor_language"
            if "answer_predictor_language" in gradient["gradients"] else
            "answer_language"
        )
        gradient_mean = float(gradient["gradients"][gradient_key]["mean"])
        method = row["method"]
        outcome_name = (
            "mbq_nll_delta_vs_bf16" if method == "mbq_current" else
            "squeezellm_nll_delta_vs_bf16"
        )
        products.append({
            "dataset": row["dataset"], "id": row["id"],
            "layer": row["layer"], "projection": row["projection"],
            "method": method, "modality": modality,
            "gradient_mean_abs": gradient_mean,
            "reconstruction_rmse": float(row["rmse"]),
            "sensitivity_x_reconstruction_error": gradient_mean * float(row["rmse"]),
            "outcome": outcome_name,
            "nll_delta_vs_bf16": float(outcome[outcome_name]),
        })
    correlations = []
    grouped = defaultdict(list)
    for row in products:
        grouped[(row["method"], row["modality"])].append(row)
    for (method, modality), values in sorted(grouped.items()):
        x = [row["sensitivity_x_reconstruction_error"] for row in values]
        y = [row["nll_delta_vs_bf16"] for row in values]
        datasets = [row["dataset"] for row in values]
        rho, p_value = (
            spearmanr(x, y) if len(set(x)) > 1 and len(set(y)) > 1
            else (math.nan, math.nan)
        )
        correlations.append({
            "method": method, "format": "W4A16",
            "outcome": "nll_delta_vs_bf16",
            "predictor": "sensitivity_x_reconstruction_rmse",
            "modality": modality, "samples": len(values),
            "spearman_rho": float(rho), "spearman_p": float(p_value),
            "dataset_adjusted_slope": _adjusted_slope(x, y, datasets),
            "slope_ci_low": math.nan, "slope_ci_high": math.nan,
        })
    return products, correlations


def attribution_decision(
    dataset_sensitivity: list[dict],
    perturbations: list[dict],
    shifts: list[dict],
    effects: list[dict],
    w4_effect: dict,
    correlations: list[dict],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    perturb = perturbation_effect(
        perturbations, samples=bootstrap_samples, seed=seed,
    )
    motivation_gradient = sum(
        bool(row["motivation_supported"]) for row in dataset_sensitivity
    ) >= 3
    motivation_perturbation = perturb["ci_low"] > 0
    effect_map = {row["variant"]: row for row in effects}
    improved = lambda name: (
        name in effect_map and effect_map[name]["ci_low"] > 0
    )
    target_design = improved("mbq_modality_normalized") or improved("mbq_no_median_floor")
    implementation = improved("mbq_sample_average")
    sparse = effect_map.get("squeezellm_no_sparse", {})
    squeeze_capacity = bool(sparse) and sparse.get("ci_high", math.nan) < 0
    twofold = [row for row in shifts if row.get("exceeds_twofold")]
    oracle = effect_map.get("mbq_target_oracle")
    domain_shift_ci_supported = any(
        row.get("exceeds_twofold") and row.get("shift_ci_excludes_zero")
        for row in shifts
    )
    degradation_prediction_supported = any(
        row.get("outcome") == "mbq_minus_squeezellm_nll"
        and math.isfinite(float(row.get("slope_ci_low", math.nan)))
        and (
            float(row["slope_ci_low"]) > 0
            or float(row["slope_ci_high"]) < 0
        )
        for row in correlations
    )
    oracle_supported = bool(oracle) and oracle["ci_low"] > 0
    generalization = (
        bool(twofold) and domain_shift_ci_supported
        and degradation_prediction_supported and oracle_supported
    )
    w4_indistinguishable = (
        math.isfinite(w4_effect.get("ci_low", math.nan))
        and w4_effect["ci_low"] <= 0 <= w4_effect["ci_high"]
    )
    labels = [
        name for name, flag in (
            ("calibration_domain_generalization_shortfall", generalization),
            ("objective_design_mismatch", target_design),
            ("local_implementation_aggregation_issue", implementation),
            ("squeezellm_capacity_advantage", squeeze_capacity),
        ) if flag
    ]
    conclusion = (
        "没有证据证明MBQ整体泛化更差"
        if w4_indistinguishable else
        "证据不足，需结合正式生成评测与预注册消融解释"
        if not labels else "检测到可归因机制，详见matched_attributions"
    )
    return {
        "format_version": 1,
        "motivation_supported": motivation_gradient and motivation_perturbation,
        "motivation_checks": {
            "gradient_at_least_3_of_4": motivation_gradient,
            "equal_rms_language_delta_nll_greater_ci": motivation_perturbation,
            "perturbation_effect": perturb,
        },
        "calibration_generalization_shortfall": generalization,
        "calibration_generalization_checks": {
            "twofold_shift_observed": bool(twofold),
            "domain_shift_ci_excludes_zero": domain_shift_ci_supported,
            "shift_predicts_sample_degradation": degradation_prediction_supported,
            "independent_target_oracle_improves": oracle_supported,
        },
        "objective_design_mismatch": target_design,
        "local_implementation_issue": implementation,
        "squeezellm_capacity_advantage": squeeze_capacity,
        "matched_attributions": labels,
        "domain_shift_twofold_rows": len(twofold),
        "ablation_effects": effects,
        "w4_mbq_minus_squeezellm": w4_effect,
        "w4_paired_ci_contains_zero": w4_indistinguishable,
        "conclusion": conclusion,
        "caveat": (
            "域泛化判据仍要求域偏移CI不跨0以及样本级退化预测证据；"
            "缺失任一证据时不得单独归因为泛化不足。"
        ),
    }


def _write_report(
    path: Path,
    degradation: list[dict],
    summaries: list[dict],
    perturbations: list[dict],
    correlations: list[dict],
    audit: dict,
    dataset_sensitivity: list[dict],
    has_ablations: bool,
) -> None:
    w4 = [
        row for row in degradation
        if row["protocol"] == "top3_confirm_256" and row["format"] == "W4A16"
    ]
    mean_by_method = {
        method: statistics.fmean(row["score"] for row in w4 if row["method"] == method)
        for method in ("mbq", "squeezellm")
        if any(row["method"] == method for row in w4)
    }
    lines = [
        "# MBQ 模态敏感度与量化退化分析",
        "",
        "## 已有结果复核",
        "",
    ]
    if mean_by_method:
        lines.append(
            f"- W4A16 样本级平均分：MBQ {mean_by_method['mbq']:.3f}，"
            f"SqueezeLLM {mean_by_method['squeezellm']:.3f}；"
            "均值差异本身不构成泛化不足证据。"
        )
    if audit:
        lines.append(
            f"- 现有校准审计：vision/language 单 token 梯度比约 "
            f"{audit['median_gradient_ratio']:.4f}，但有效 vision 目标质量占比约 "
            f"{100 * audit['vision_mass_share']:.2f}%。"
        )
    lines.extend(["", "## 新敏感度证据", ""])
    if not summaries:
        lines.append(
            "- 尚未运行逐样本梯度采集；当前报告只完成结果表构建，"
            "不能判断校准域泛化、目标失配或实现误差。"
        )
    else:
        supported = [
            row["dataset"] for row in dataset_sensitivity
            if row["motivation_supported"]
        ]
        lines.append(
            f"- 因果对齐后的 vision/language 比值95%区间上界小于1的数据集数："
            f"{len(supported)}/{len(dataset_sensitivity)}"
            f"（{', '.join(sorted(supported)) or '-'}）。"
        )
        if any(row.get("answer_ratio_basis") == "causal_predictor_position" for row in summaries):
            lines.append(
                "- 报告同时保留官方 `labels != -100` 的 label-position 口径；"
                "相关性与扰动主分析使用 causal-LM 左移一位后的 predictor-position 口径。"
            )
    if perturbations:
        vision = _finite(
            row["delta_nll"] for row in perturbations if row["modality"] == "vision"
        )
        language = _finite(
            row["delta_nll"] for row in perturbations
            if row["modality"] == "answer_language"
        )
        lines.append(
            f"- 等幅扰动中位 ΔNLL：vision "
            f"{statistics.median(vision) if vision else math.nan:.6g}，language "
            f"{statistics.median(language) if language else math.nan:.6g}。"
        )
    lines.extend(["", "## 归因状态", ""])
    if not summaries or not perturbations or not has_ablations:
        lines.append(
            "- **待完成正式敏感度/消融实验，暂不归因。** "
            "不能将当前局部任务差异表述为"
            "“MBQ泛化能力不足”。"
        )
    elif correlations:
        lines.append(
            "- 已生成样本级相关、dataset-adjusted 系数和分层bootstrap区间；"
            "应结合消融表按预注册规则判断，而不是依据单个相关系数。"
        )
    lines.extend([
        "",
        "## 限制",
        "",
        "- 正式任务分数相关性是观察性证据；模态隔离扰动和单因素W4消融"
        "才用于因果归因。",
        "- SqueezeLLM包含非均匀码本和稀疏高精度修正，不能把方法间差异"
        "全部归因于模态敏感度。",
        "",
    ])
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    degradation = build_quant_degradation(args.top3_root, args.top4_root)
    sensitivity = _read_jsonl(args.sensitivity_jsonl)
    perturbations = _read_jsonl(args.perturbations_jsonl)
    ablations = _read_jsonl(args.ablation_jsonl)
    residuals = _read_jsonl(args.residuals_jsonl)
    nll_rows = build_nll_degradation(ablations)
    enrich_score_degradation_with_nll(degradation, nll_rows)
    summaries = summarize_sensitivity(
        sensitivity,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    aggregations = gradient_aggregation_rows(sensitivity)
    features = sample_sensitivity_features(sensitivity)
    dataset_sensitivity = dataset_sensitivity_rows(
        features,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    correlations = correlation_rows(
        degradation, features,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    correlations.extend(nll_correlation_rows(
        features, nll_rows,
        bootstrap_samples=args.bootstrap_samples, seed=args.seed,
    ))
    correlations.extend(layer_correlation_rows(
        degradation, sensitivity, nll_rows,
    ))
    shifts = domain_shift_rows(
        sensitivity, bootstrap_samples=args.bootstrap_samples, seed=args.seed,
    )
    effects = ablation_effects(
        ablations, samples=args.bootstrap_samples, seed=args.seed,
    )
    w4_effect = w4_mbq_squeeze_effect(
        degradation, samples=args.bootstrap_samples, seed=args.seed,
    )
    promotion_scores = load_promotion_scores(
        args.promotion_root, args.top2_selection,
    )
    products, product_correlations = sensitivity_reconstruction_products(
        sensitivity, residuals, nll_rows,
    )
    correlations.extend(product_correlations)
    decision = attribution_decision(
        dataset_sensitivity, perturbations, shifts, effects, w4_effect,
        correlations,
        bootstrap_samples=args.bootstrap_samples, seed=args.seed,
    )
    _write_csv(args.output_dir / "sample_quant_degradation.csv", degradation)
    _write_csv(args.output_dir / "layer_sensitivity.csv", summaries)
    _write_csv(args.output_dir / "dataset_sensitivity.csv", dataset_sensitivity)
    _write_csv(args.output_dir / "gradient_aggregations.csv", aggregations)
    _write_csv(args.output_dir / "correlations.csv", correlations)
    _write_csv(args.output_dir / "domain_shift.csv", shifts)
    _write_csv(args.output_dir / "reconstruction_residuals.csv", residuals)
    _write_csv(args.output_dir / "ablation_effects.csv", effects)
    _write_csv(args.output_dir / "promotion_scores.csv", promotion_scores)
    _write_csv(args.output_dir / "sensitivity_reconstruction_product.csv", products)
    _write_csv(
        args.output_dir / "ablation_results.csv",
        ablations,
        fields=None if ablations else [
            "variant", "dataset", "id", "target_nll", "score",
        ],
    )
    _svg_layer_plot(args.output_dir / "layer_sensitivity.svg", summaries)
    _svg_scatter(
        args.output_dir / "sensitivity_vs_degradation.svg",
        degradation,
        features,
    )
    _svg_domain_shift(
        args.output_dir / "calibration_target_shift.svg",
        shifts,
    )
    audit = _audit_observations(args.mbq_audit)
    _write_report(
        args.output_dir / "report_zh.md",
        degradation, summaries, perturbations, correlations, audit,
        dataset_sensitivity, bool(ablations),
    )
    atomic_json(args.output_dir / "attribution_decision.json", decision)
    atomic_json(args.output_dir / "summary.json", {
        "format_version": 1,
        "degradation_rows": len(degradation),
        "sensitivity_rows": len(sensitivity),
        "dataset_sensitivity_rows": dataset_sensitivity,
        "gradient_aggregation_rows": len(aggregations),
        "perturbation_rows": len(perturbations),
        "correlation_rows": correlations,
        "domain_shift_rows": shifts,
        "residual_rows": len(residuals),
        "promotion_score_rows": len(promotion_scores),
        "sensitivity_reconstruction_product_rows": len(products),
        "attribution_decision": decision,
        "mbq_audit_observations": audit,
        "attribution_status": (
            "ready_for_rule_based_review"
            if summaries and perturbations and ablations
            else "pending_required_experiments"
        ),
    })
    atomic_json(args.output_dir / "analysis_audit.json", {
        "format_version": 1,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "inputs": {
            "top3_root": str(args.top3_root.resolve()),
            "top4_root": str(args.top4_root.resolve()) if args.top4_root else None,
            "sensitivity_jsonl": str(args.sensitivity_jsonl.resolve()) if args.sensitivity_jsonl else None,
            "sensitivity_sha256": _file_digest(args.sensitivity_jsonl),
            "perturbations_sha256": _file_digest(args.perturbations_jsonl),
            "ablations_sha256": _file_digest(args.ablation_jsonl),
            "residuals_sha256": _file_digest(args.residuals_jsonl),
            "top2_selection_sha256": _file_digest(args.top2_selection),
            "mbq_audit_sha256": _file_digest(args.mbq_audit),
        },
        "row_counts": {
            "degradation": len(degradation),
            "sensitivity": len(sensitivity),
            "perturbations": len(perturbations),
            "ablations": len(ablations),
            "residuals": len(residuals),
            "promotion_scores": len(promotion_scores),
            "sensitivity_reconstruction_products": len(products),
            "correlations": len(correlations),
            "domain_shifts": len(shifts),
        },
    })


if __name__ == "__main__":
    main()
