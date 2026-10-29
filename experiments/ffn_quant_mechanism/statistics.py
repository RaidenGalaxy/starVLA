from __future__ import annotations

import itertools
import math
import random
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .config import MEDIATOR_GROUPS


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    if not 0 <= probability <= 1:
        raise ValueError("probability must be in [0,1]")
    ordered = sorted(float(value) for value in values)
    if not all(math.isfinite(value) for value in ordered):
        raise FloatingPointError("non-finite bootstrap draw")
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def interval(values: Sequence[float]) -> list[float]:
    return [percentile(values, 0.025), percentile(values, 0.975)]


def _index(rows: Sequence[Mapping[str, Any]], name: str) -> dict[str, Mapping[str, Any]]:
    result = {}
    for row in rows:
        identifier = row.get("id")
        dataset = row.get("dataset")
        target = row.get("target_text")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"{name} row has no id")
        if identifier in result:
            raise ValueError(f"{name} has duplicate id {identifier}")
        if not isinstance(dataset, str) or not dataset:
            raise ValueError(f"{name}/{identifier} has no dataset")
        if not isinstance(target, Mapping):
            raise ValueError(f"{name}/{identifier} has no target metrics")
        tokens = target.get("tokens")
        total = target.get("total_nll")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
            raise ValueError(f"{name}/{identifier} has invalid token count")
        if not math.isfinite(float(total)) or float(total) < 0:
            raise ValueError(f"{name}/{identifier} has invalid total NLL")
        result[identifier] = row
    if not result:
        raise ValueError(f"{name} has no rows")
    return result


def _align(
    variants: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, dict[str, Mapping[str, Any]]], dict[str, list[str]]]:
    indexed = {name: _index(rows, name) for name, rows in variants.items()}
    if len(indexed) < 2:
        raise ValueError("paired analysis needs at least two variants")
    names = list(indexed)
    expected = set(indexed[names[0]])
    by_dataset: dict[str, list[str]] = defaultdict(list)
    for name in names[1:]:
        if set(indexed[name]) != expected:
            raise ValueError(f"variant {name} does not have the paired sample set")
    for identifier in sorted(expected):
        datasets = {str(indexed[name][identifier]["dataset"]) for name in names}
        if len(datasets) != 1:
            raise ValueError(f"dataset mismatch for {identifier}")
        by_dataset[datasets.pop()].append(identifier)
    return indexed, dict(sorted(by_dataset.items()))


def _draw(rng: random.Random, by_dataset: Mapping[str, Sequence[str]]) -> list[str]:
    return [
        rng.choice(identifiers)
        for dataset in sorted(by_dataset)
        for identifiers in (by_dataset[dataset],)
        for _ in identifiers
    ]


def _nll(ids: Sequence[str], rows: Mapping[str, Mapping[str, Any]]) -> float:
    total = sum(float(rows[item]["target_text"]["total_nll"]) for item in ids)
    tokens = sum(int(rows[item]["target_text"]["tokens"]) for item in ids)
    return total / tokens


def _dataset_relative_changes(
    indexed: Mapping[str, Mapping[str, Mapping[str, Any]]],
    by_dataset: Mapping[str, Sequence[str]],
    baseline: str,
    candidate: str,
) -> dict[str, float]:
    return {
        dataset: _nll(ids, indexed[candidate]) / _nll(ids, indexed[baseline]) - 1.0
        for dataset, ids in by_dataset.items()
    }


def _causal_point(
    ids: Sequence[str],
    indexed: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, float]:
    nll = {name: _nll(ids, rows) for name, rows in indexed.items()}

    def gap(prefix: str, condition: str) -> float:
        return nll[f"{prefix}_{condition}"] / nll[f"{prefix}_B"] - 1.0

    iw, ia, ir = (gap("infinitevl", key) for key in ("W", "A", "R"))
    qw, qa, qr = (gap("qwen", key) for key in ("W", "A", "R"))
    result = {
        "infinitevl_raw_gap": iw,
        "infinitevl_absolute_gap": ia,
        "infinitevl_relative_gap": ir,
        "qwen_raw_gap": qw,
        "qwen_absolute_gap": qa,
        "qwen_relative_gap": qr,
        "raw_did": iw - qw,
        "infinitevl_residual_transport_effect": iw - ir,
        "qwen_reverse_transport_effect": qr - qw,
        "relative_over_absolute_advantage_infinitevl": ia - ir,
        "relative_over_absolute_advantage_qwen": qr - qa,
    }
    denominator = result["raw_did"]
    result["infinitevl_did_closure"] = (
        result["infinitevl_residual_transport_effect"] / denominator if denominator else math.nan
    )
    result["qwen_did_closure"] = result["qwen_reverse_transport_effect"] / denominator if denominator else math.nan
    result["mean_bidirectional_did_closure"] = (result["infinitevl_did_closure"] + result["qwen_did_closure"]) / 2
    if all(f"infinitevl_{key}" in indexed for key in ("G", "RG")):
        ig, irg = gap("infinitevl", "G"), gap("infinitevl", "RG")
        result.update(
            {
                "infinitevl_gdn_gap": ig,
                "infinitevl_joint_gap": irg,
                "residual_explained_fraction": (iw - ir) / iw if iw else math.nan,
                "gdn_explained_fraction": (iw - ig) / iw if iw else math.nan,
                "joint_explained_fraction": (iw - irg) / iw if iw else math.nan,
                "factorial_interaction": irg - ir - ig + iw,
            }
        )
    return result


def stratified_causal_bootstrap(
    variants: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    samples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    required = {f"{model}_{condition}" for model in ("infinitevl", "qwen") for condition in ("B", "W", "A", "R")}
    if not required <= set(variants):
        raise ValueError(f"causal matrix omits variants: {sorted(required - set(variants))}")
    indexed, by_dataset = _align(variants)
    ordered = [item for dataset in sorted(by_dataset) for item in by_dataset[dataset]]
    point = _causal_point(ordered, indexed)
    draws: dict[str, list[float]] = defaultdict(list)
    rng = random.Random(seed)
    for _ in range(samples):
        values = _causal_point(_draw(rng, by_dataset), indexed)
        for key, value in values.items():
            if math.isfinite(value):
                draws[key].append(value)
    intervals = {key: interval(values) for key, values in sorted(draws.items())}
    dataset_directions = {
        "infinitevl": _dataset_relative_changes(
            indexed,
            by_dataset,
            "infinitevl_R",
            "infinitevl_W",
        ),
        "qwen": _dataset_relative_changes(
            indexed,
            by_dataset,
            "qwen_W",
            "qwen_R",
        ),
    }
    return {
        "analysis": "paired_dataset_stratified_mechanism_bootstrap",
        "paired_samples": len(ordered),
        "datasets": {key: len(value) for key, value in by_dataset.items()},
        "bootstrap_samples": samples,
        "seed": seed,
        "point": point,
        "ci95": intervals,
        "dataset_transport_directions": dataset_directions,
    }


def _factorial_point(
    ids: Sequence[str],
    indexed: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, float]:
    nll = {name: _nll(ids, rows) for name, rows in indexed.items()}
    baseline = nll["B"]
    gaps = {key: nll[key] / baseline - 1.0 for key in ("W", "R", "G", "RG")}
    raw = gaps["W"]
    return {
        "bf16_target_nll": baseline,
        "raw_gap": raw,
        "relative_transport_gap": gaps["R"],
        "gdn_carry_clamp_gap": gaps["G"],
        "joint_gap": gaps["RG"],
        "residual_explained_fraction": (raw - gaps["R"]) / raw if raw else math.nan,
        "gdn_explained_fraction": (raw - gaps["G"]) / raw if raw else math.nan,
        "joint_explained_fraction": (raw - gaps["RG"]) / raw if raw else math.nan,
        "factorial_interaction": gaps["RG"] - gaps["R"] - gaps["G"] + raw,
    }


def stratified_mediation_bootstrap(
    variants: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    samples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    required = {"B", "W", "R", "G", "RG"}
    if not required <= set(variants):
        raise ValueError(f"mediation factorial omits cells: {sorted(required - set(variants))}")
    selected = {key: variants[key] for key in sorted(required)}
    indexed, by_dataset = _align(selected)
    ordered = [item for dataset in sorted(by_dataset) for item in by_dataset[dataset]]
    point = _factorial_point(ordered, indexed)
    draws: dict[str, list[float]] = defaultdict(list)
    rng = random.Random(seed)
    for _ in range(samples):
        values = _factorial_point(_draw(rng, by_dataset), indexed)
        for key, value in values.items():
            if math.isfinite(value):
                draws[key].append(value)
    return {
        "analysis": "paired_dataset_stratified_residual_by_gdn_2x2",
        "paired_samples": len(ordered),
        "datasets": {key: len(value) for key, value in by_dataset.items()},
        "bootstrap_samples": samples,
        "seed": seed,
        "point": point,
        "ci95": {key: interval(value) for key, value in sorted(draws.items())},
    }


def stratified_gap_reduction_bootstrap(
    baseline_rows: Sequence[Mapping[str, Any]],
    raw_rows: Sequence[Mapping[str, Any]],
    repaired_rows: Sequence[Mapping[str, Any]],
    *,
    samples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    indexed, by_dataset = _align(
        {
            "B": baseline_rows,
            "W": raw_rows,
            "X": repaired_rows,
        }
    )
    ordered = [item for dataset in sorted(by_dataset) for item in by_dataset[dataset]]

    def estimate(ids: Sequence[str]) -> dict[str, float]:
        b, w, x = (_nll(ids, indexed[key]) for key in ("B", "W", "X"))
        raw = w / b - 1.0
        repaired = x / b - 1.0
        return {
            "raw_gap": raw,
            "repaired_gap": repaired,
            "absolute_gap_reduction": raw - repaired,
            "explained_fraction": (raw - repaired) / raw if raw else math.nan,
        }

    point = estimate(ordered)
    draws: dict[str, list[float]] = defaultdict(list)
    rng = random.Random(seed)
    for _ in range(samples):
        for key, value in estimate(_draw(rng, by_dataset)).items():
            if math.isfinite(value):
                draws[key].append(value)
    return {
        "analysis": "paired_dataset_stratified_gap_reduction",
        "paired_samples": len(ordered),
        "datasets": {key: len(value) for key, value in by_dataset.items()},
        "bootstrap_samples": samples,
        "seed": seed,
        "point": point,
        "ci95": {key: interval(value) for key, value in sorted(draws.items())},
    }


def length_stratified_gap_reduction_bootstrap(
    baseline_rows: Sequence[Mapping[str, Any]],
    raw_rows: Sequence[Mapping[str, Any]],
    repaired_rows: Sequence[Mapping[str, Any]],
    *,
    samples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Estimate repair benefits by preregistered length bin and a sample-level trend."""

    indexed, by_dataset = _align(
        {
            "B": baseline_rows,
            "W": raw_rows,
            "X": repaired_rows,
        }
    )
    lengths: dict[str, int] = {}
    for identifier in indexed["B"]:
        observed = {int(indexed[name][identifier].get("sequence_tokens", 0)) for name in ("B", "W", "X")}
        if len(observed) != 1 or next(iter(observed)) <= 0:
            raise ValueError(f"invalid or mismatched sequence length for {identifier}")
        lengths[identifier] = observed.pop()
    bins = {
        "le_256": [identifier for identifier, length in lengths.items() if length <= 256],
        "257_512": [identifier for identifier, length in lengths.items() if 257 <= length <= 512],
        "gt_512": [identifier for identifier, length in lengths.items() if length > 512],
    }
    bin_results = {}
    for offset, (name, identifiers) in enumerate(bins.items()):
        if not identifiers:
            bin_results[name] = {
                "status": "not_estimable",
                "paired_samples": 0,
                "reason": "no confirm192 samples in the preregistered length bin",
            }
            continue
        selected = set(identifiers)
        estimate = stratified_gap_reduction_bootstrap(
            [row for row in baseline_rows if row["id"] in selected],
            [row for row in raw_rows if row["id"] in selected],
            [row for row in repaired_rows if row["id"] in selected],
            samples=samples,
            seed=seed + offset,
        )
        bin_results[name] = {"status": "estimated", **estimate}

    def sample_benefit(identifier: str) -> float:
        def nll(name: str) -> float:
            row = indexed[name][identifier]["target_text"]
            return float(row["total_nll"]) / int(row["tokens"])

        baseline = nll("B")
        return (nll("W") - nll("X")) / baseline

    def slope(identifiers: Sequence[str]) -> float:
        x = [math.log2(lengths[identifier]) for identifier in identifiers]
        y = [sample_benefit(identifier) for identifier in identifiers]
        x_mean, y_mean = statistics.mean(x), statistics.mean(y)
        denominator = sum((value - x_mean) ** 2 for value in x)
        if denominator == 0:
            return math.nan
        return sum((left - x_mean) * (right - y_mean) for left, right in zip(x, y, strict=True)) / denominator

    ordered = [item for dataset in sorted(by_dataset) for item in by_dataset[dataset]]
    point_slope = slope(ordered)
    rng = random.Random(seed + 101)
    draws = [value for _ in range(samples) if math.isfinite(value := slope(_draw(rng, by_dataset)))]
    if not math.isfinite(point_slope) or not draws:
        raise RuntimeError("sequence-length trend is not identifiable")
    trend_ci = interval(draws)
    return {
        "analysis": "paired_confirm_length_strata_and_sample_level_trend",
        "bins": bin_results,
        "trend": {
            "definition": "sample repair benefit regressed on log2(sequence_tokens)",
            "slope_per_log2_token": point_slope,
            "ci95": trend_ci,
            "increases_with_length": trend_ci[0] > 0,
        },
        "bootstrap_samples": samples,
        "seed": seed,
    }


def screen_decisions(
    causal: Mapping[str, Any],
    *,
    gdn_gain_ratio: float | None,
    gdn_explained_fraction: float | None = None,
) -> dict[str, Any]:
    point = causal["point"]
    infinite_directions = causal["dataset_transport_directions"]["infinitevl"]
    qwen_directions = causal["dataset_transport_directions"]["qwen"]
    infinite_correct = sum(value > 0 for value in infinite_directions.values())
    qwen_correct = sum(value > 0 for value in qwen_directions.values())
    residual_pass = bool(
        point["infinitevl_residual_transport_effect"] > 0
        and point["qwen_reverse_transport_effect"] > 0
        and infinite_correct >= 3
        and qwen_correct >= 3
        and point["mean_bidirectional_did_closure"] >= 0.30
    )
    gdn_fraction = float(
        point.get("gdn_explained_fraction", math.nan) if gdn_explained_fraction is None else gdn_explained_fraction
    )
    gdn_pass = bool(
        gdn_gain_ratio is not None and gdn_gain_ratio > 1.0 and math.isfinite(gdn_fraction) and gdn_fraction >= 0.20
    )
    return {
        "H_R": {
            "pass": residual_pass,
            "required_dataset_direction_count": 3,
            "infinitevl_direction_count": infinite_correct,
            "qwen_direction_count": qwen_correct,
            "minimum_mean_did_closure": 0.30,
        },
        "H_G": {
            "pass": gdn_pass,
            "gdn_swa_gain_ratio": gdn_gain_ratio,
            "gdn_explained_fraction": gdn_fraction,
            "minimum_explained_fraction": 0.20,
        },
    }


def confirm_decisions(
    causal: Mapping[str, Any],
    *,
    gdn_gain_ratio_ci95: Sequence[float] | None,
    mediation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    point, ci = causal["point"], causal["ci95"]

    def positive(name: str) -> bool:
        values = ci.get(name)
        return bool(values and float(values[0]) > 0)

    residual = bool(
        positive("infinitevl_residual_transport_effect")
        and positive("qwen_reverse_transport_effect")
        and point["mean_bidirectional_did_closure"] >= 0.50
        and positive("relative_over_absolute_advantage_infinitevl")
        and positive("relative_over_absolute_advantage_qwen")
    )
    mediation_point = {} if mediation is None else mediation.get("point", {})
    mediation_ci = {} if mediation is None else mediation.get("ci95", {})
    gdn_fraction = float(
        mediation_point.get(
            "gdn_explained_fraction",
            point.get("gdn_explained_fraction", math.nan),
        )
    )
    gdn_fraction_ci = mediation_ci.get(
        "gdn_explained_fraction",
        ci.get("gdn_explained_fraction"),
    )
    gdn = bool(
        gdn_gain_ratio_ci95 is not None
        and float(gdn_gain_ratio_ci95[0]) > 1.0
        and gdn_fraction_ci
        and float(gdn_fraction_ci[0]) > 0
        and gdn_fraction >= 0.30
    )
    joint_fraction = float(
        mediation_point.get(
            "joint_explained_fraction",
            point.get("joint_explained_fraction", math.nan),
        )
    )
    joint_ci = mediation_ci.get(
        "joint_explained_fraction",
        ci.get("joint_explained_fraction"),
    )
    joint = bool(residual and gdn and joint_fraction >= 0.70 and joint_ci and float(joint_ci[0]) >= 0.50)
    interaction_ci = mediation_ci.get(
        "factorial_interaction",
        ci.get("factorial_interaction"),
    )
    non_additive = bool(interaction_ci and (interaction_ci[1] < 0 or interaction_ci[0] > 0))
    return {
        "H_R": residual,
        "H_G": gdn,
        "H_RG_joint": joint,
        "interaction": ("non_additive" if non_additive else "no_detected_non_additivity"),
        "rules_frozen_before_confirm": True,
    }


def gain_ratio_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str = "block_gain",
    samples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    cells: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        key = (
            str(row.get("sample_id", row["id"])),
            str(row["dataset"]),
            str(row["pair"]),
        )
        structure = str(row["structure"])
        value = float(row[metric])
        if structure not in {"gdn", "swa"} or not math.isfinite(value) or value <= 0:
            raise ValueError("invalid local gain row")
        if structure in cells[key]:
            raise ValueError(f"duplicate local gain cell {key}/{structure}")
        cells[key][structure] = value
    if not cells or any(set(value) != {"gdn", "swa"} for value in cells.values()):
        raise ValueError("local gain rows are not complete GDN/SWA pairs")
    sample_cells: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
    for key in cells:
        sample_cells[(key[0], key[1])].append(key)
    pair_sets = {sample: {key[2] for key in keys} for sample, keys in sample_cells.items()}
    expected_pairs = next(iter(pair_sets.values()))
    if any(value != expected_pairs for value in pair_sets.values()):
        raise ValueError("local gain samples do not share the same layer-pair grid")
    sample_log_ratios = {
        sample: statistics.mean(math.log(cells[key]["gdn"] / cells[key]["swa"]) for key in keys)
        for sample, keys in sample_cells.items()
    }
    by_dataset: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for sample in sample_cells:
        by_dataset[sample[1]].append(sample)

    def estimate(samples_: Sequence[tuple[str, str]]) -> float:
        return math.exp(statistics.mean(sample_log_ratios[sample] for sample in samples_))

    ordered = sorted(sample_cells)
    point = estimate(ordered)
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        chosen = [rng.choice(keys) for dataset in sorted(by_dataset) for keys in (by_dataset[dataset],) for _ in keys]
        draws.append(estimate(chosen))
    return {
        "analysis": "paired_stratified_geometric_gdn_swa_gain_ratio",
        "metric": metric,
        "paired_samples": len(sample_cells),
        "layer_pairs_per_sample": len(expected_pairs),
        "sample_pair_cells": len(cells),
        "point": point,
        "ci95": interval(draws),
        "bootstrap_samples": samples,
        "seed": seed,
    }


def powerset(groups: Sequence[str] = MEDIATOR_GROUPS) -> list[frozenset[str]]:
    return [frozenset(value) for size in range(len(groups) + 1) for value in itertools.combinations(groups, size)]


def subset_label(groups: Sequence[str]) -> str:
    selected = sorted(set(groups))
    return "none" if not selected else "+".join(selected)


def shapley_values(
    cell_metric: Mapping[frozenset[str], float],
    *,
    groups: Sequence[str] = MEDIATOR_GROUPS,
) -> dict[str, Any]:
    expected = set(powerset(groups))
    if set(cell_metric) != expected:
        raise ValueError("Shapley analysis requires all mediator subsets exactly once")
    count = len(groups)
    raw = float(cell_metric[frozenset()])

    def utility(subset: frozenset[str]) -> float:
        return raw - float(cell_metric[subset])

    values = {}
    factorial = math.factorial
    for group in groups:
        contribution = 0.0
        others = [item for item in groups if item != group]
        for subset in powerset(others):
            weight = factorial(len(subset)) * factorial(count - len(subset) - 1) / factorial(count)
            contribution += weight * (utility(subset | {group}) - utility(subset))
        values[group] = contribution
    total = utility(frozenset(groups))
    if not math.isclose(sum(values.values()), total, rel_tol=1e-9, abs_tol=1e-12):
        raise RuntimeError("Shapley contributions do not sum to the total repair effect")
    shares = {group: value / total if total else math.nan for group, value in values.items()}
    return {
        "raw_metric": raw,
        "all_repaired_metric": float(cell_metric[frozenset(groups)]),
        "total_repair_effect": total,
        "contributions": values,
        "shares": shares,
    }


def select_mediators(
    dataset_shapley: Mapping[str, Mapping[str, Any]],
    *,
    minimum_share: float = 0.15,
    maximum: int = 2,
) -> dict[str, Any]:
    aggregate = {
        group: statistics.mean(float(value["shares"][group]) for value in dataset_shapley.values())
        for group in MEDIATOR_GROUPS
    }
    aggregate_contributions = {
        group: statistics.mean(float(value["contributions"][group]) for value in dataset_shapley.values())
        for group in MEDIATOR_GROUPS
    }
    direction_counts = {
        group: sum(float(value["contributions"][group]) > 0 for value in dataset_shapley.values())
        for group in MEDIATOR_GROUPS
    }
    eligible = [group for group in MEDIATOR_GROUPS if aggregate[group] >= minimum_share and direction_counts[group] >= 3]
    selected = sorted(
        eligible,
        key=lambda group: (-abs(aggregate_contributions[group]), group),
    )[:maximum]
    return {
        "minimum_share": minimum_share,
        "minimum_same_direction_datasets": 3,
        "maximum_selected": maximum,
        "macro_shares": aggregate,
        "macro_contributions": aggregate_contributions,
        "positive_direction_datasets": direction_counts,
        "eligible": eligible,
        "selected": selected,
    }
