from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


def aligned(rows_by_condition):
    maps = {}
    for condition, rows in rows_by_condition.items():
        cells = {row["id"]: row for row in rows}
        if len(cells) != len(rows) or not cells:
            raise ValueError("empty or duplicate sample IDs")
        maps[condition] = cells
    ids = sorted(next(iter(maps.values())))
    for cells in maps.values():
        if set(cells) != set(ids):
            raise ValueError("paired sample sets differ")
        for identifier in ids:
            first = next(iter(maps.values()))[identifier]
            row = cells[identifier]
            if row["dataset"] != first["dataset"] or row["target_text"]["tokens"] != first["target_text"]["tokens"]:
                raise ValueError("dataset/target-token mismatch within model")
            if not math.isfinite(row["target_text"]["total_nll"]):
                raise FloatingPointError("nonfinite loss")
    return ids, maps


def nll(rows):
    return sum(r["target_text"]["total_nll"] for r in rows) / sum(r["target_text"]["tokens"] for r in rows)


def paired_effect(baseline, quantized, candidate, *, samples=10_000, seed=42):
    ids, maps = aligned({"B": baseline, "W": quantized, "C": candidate})
    groups = defaultdict(list)
    for i, identifier in enumerate(ids):
        groups[maps["B"][identifier]["dataset"]].append(i)
    tokens = np.array([maps["B"][i]["target_text"]["tokens"] for i in ids], dtype=float)
    if (tokens <= 0).any():
        raise ValueError("target token counts must be positive")
    losses = {k: np.array([v[i]["target_text"]["total_nll"] for i in ids]) for k, v in maps.items()}

    def calculate(index):
        den = tokens[index].sum(axis=-1)
        b, w, c = (losses[k][index].sum(axis=-1) / den for k in ("B", "W", "C"))
        with np.errstate(divide="ignore", invalid="ignore"):
            return {
                "baseline_nll": b,
                "quantized_nll": w,
                "candidate_nll": c,
                "excess_loss_before": w - b,
                "excess_loss_after": c - b,
                "recovery_absolute": w - c,
                "candidate_delta_from_baseline": c - b,
                "candidate_relative_from_baseline": (c - b) / b,
                "explained_fraction": (w - c) / (w - b),
            }

    rng = np.random.default_rng(seed)
    indices = np.concatenate([rng.choice(g, (samples, len(g)), replace=True) for g in groups.values()], axis=1)
    boot = calculate(indices)
    point = calculate(np.arange(len(ids)))
    result = {
        "paired_samples": len(ids),
        "bootstrap_samples": samples,
        "seed": seed,
        "unit": "sample_id_stratified_by_dataset",
        "point": {},
        "ci95": {},
        "undefined_replicates": {},
    }
    for key in point:
        result["point"][key] = float(point[key]) if np.isfinite(point[key]) else None
        values = boot[key][np.isfinite(boot[key])]
        result["undefined_replicates"][key] = int(samples - len(values))
        result["ci95"][key] = np.quantile(values, [0.025, 0.975]).tolist() if len(values) else None
    # One-sided sample-level randomization test; target-token weights stay attached to IDs.
    differences = losses["W"] - losses["C"]
    signs = rng.choice((-1, 1), (samples, len(ids)))
    randomized = (signs * differences).sum(1)
    result["p_recovery_one_sided"] = float((1 + (randomized >= differences.sum()).sum()) / (samples + 1))
    result["p_degradation_one_sided"] = float((1 + (randomized <= differences.sum()).sum()) / (samples + 1))
    result["p_method"] = "paired_sample_sign_flip_monte_carlo"
    result["datasets"] = {
        dataset: {
            "samples": len(g),
            **{k: float(v) if np.isfinite(v) else None for k, v in calculate(np.array(g)).items()},
        }
        for dataset, g in sorted(groups.items())
    }
    qa = {}
    qa_boot = {}
    for k, cells in maps.items():
        scores = {ds: [cells[ids[i]]["score"] for i in g] for ds, g in groups.items()}
        if all(all(math.isfinite(v) for v in values) for values in scores.values()):
            qa[k] = float(100 * np.mean([np.mean(values) for values in scores.values()]))
            score_array = np.array([cells[i]["score"] for i in ids])
            offset = 0
            dataset_means = []
            for group in groups.values():
                dataset_means.append(score_array[indices[:, offset : offset + len(group)]].mean(1))
                offset += len(group)
            qa_boot[k] = 100 * np.mean(dataset_means, axis=0)
    result["qa_macro_points"] = qa or None
    result["qa_macro_ci95"] = {k: np.quantile(v, [0.025, 0.975]).tolist() for k, v in qa_boot.items()} or None
    if "C" in qa and "B" in qa:
        result["qa_delta_points"] = qa["C"] - qa["B"]
        result["qa_delta_ci95"] = np.quantile(qa_boot["C"] - qa_boot["B"], [0.025, 0.975]).tolist()
    return result


def holm(pvalues):
    if any(not 0 <= p <= 1 for p in pvalues.values()):
        raise ValueError("invalid p value")
    ordered = sorted(pvalues, key=pvalues.get)
    output, previous = {}, 0.0
    for index, name in enumerate(ordered):
        previous = max(previous, min(1.0, pvalues[name] * (len(ordered) - index)))
        output[name] = previous
    return output
