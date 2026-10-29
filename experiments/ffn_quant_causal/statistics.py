from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from experiments.infinitevl_quant.progress import StageProgress
from experiments.infinitevl_quant_ffn_bf16_4x64.scoring import SCORER_ID


DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 42


def _finite_number(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _validate_row(row: Mapping[str, Any], *, variant: str) -> None:
    identifier = row.get("id")
    dataset = row.get("dataset")
    if not isinstance(identifier, str) or not identifier:
        raise ValueError(f"{variant} contains a row without a non-empty id")
    if not isinstance(dataset, str) or not dataset:
        raise ValueError(f"{variant}/{identifier} has no dataset")

    target = row.get("target_text")
    if not isinstance(target, Mapping):
        raise ValueError(f"{variant}/{identifier} has no target_text metrics")
    tokens = target.get("tokens")
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
        raise ValueError(f"{variant}/{identifier} target token count must be positive")
    total_nll = _finite_number(
        target.get("total_nll"), label=f"{variant}/{identifier} total_nll",
    )
    if total_nll < 0:
        raise ValueError(f"{variant}/{identifier} total_nll must be non-negative")

    scoring = row.get("scoring")
    if scoring is not None:
        if not isinstance(scoring, Mapping) or scoring.get("scorer") != SCORER_ID:
            raise ValueError(
                f"{variant}/{identifier} uses a scorer other than {SCORER_ID}; "
                "legacy and corrected scores must never be mixed"
            )
    score = row.get("score", math.nan)
    try:
        finite_score = math.isfinite(float(score))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{variant}/{identifier} score must be numeric") from error
    if finite_score and (
        not isinstance(scoring, Mapping) or scoring.get("scorer") != SCORER_ID
    ):
        raise ValueError(
            f"{variant}/{identifier} has a QA score without the {SCORER_ID} tag"
        )


def _index_rows(rows: Sequence[Mapping[str, Any]], *, variant: str) -> dict[str, Mapping[str, Any]]:
    if not rows:
        raise ValueError(f"{variant} has no rows")
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        _validate_row(row, variant=variant)
        identifier = str(row["id"])
        if identifier in result:
            raise ValueError(f"{variant} contains duplicate sample id {identifier}")
        result[identifier] = row
    return result


def _align_variants(
    variants: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    token_match_groups: Sequence[Sequence[str]] | None = None,
) -> tuple[dict[str, dict[str, Mapping[str, Any]]], dict[str, list[str]], bool]:
    if len(variants) < 2:
        raise ValueError("paired analysis requires at least two variants")
    indexed = {
        name: _index_rows(rows, variant=name)
        for name, rows in variants.items()
    }
    names = list(indexed)
    if token_match_groups is None:
        token_match_groups = (tuple(names),)
    covered = [name for group in token_match_groups for name in group]
    if sorted(covered) != sorted(names) or len(covered) != len(set(covered)):
        raise ValueError("token match groups must partition the paired variants")
    reference_name = names[0]
    reference = indexed[reference_name]
    reference_ids = set(reference)
    for name in names[1:]:
        if set(indexed[name]) != reference_ids:
            missing = sorted(reference_ids - set(indexed[name]))
            extra = sorted(set(indexed[name]) - reference_ids)
            raise ValueError(
                f"paired variants require identical sample IDs; {name} "
                f"missing={missing[:3]} extra={extra[:3]}"
            )

    qa_availability: set[bool] = set()
    by_dataset: dict[str, list[str]] = defaultdict(list)
    for identifier in sorted(reference):
        reference_row = reference[identifier]
        dataset = str(reference_row["dataset"])
        sample_availability = []
        for name in names:
            row = indexed[name][identifier]
            if row["dataset"] != dataset:
                raise ValueError(
                    f"dataset mismatch for {identifier}: {reference_name}={dataset}, "
                    f"{name}={row['dataset']}"
                )
            sample_availability.append(math.isfinite(float(row.get("score", math.nan))))
        for group in token_match_groups:
            group_reference = group[0]
            reference_tokens = int(
                indexed[group_reference][identifier]["target_text"]["tokens"]
            )
            for name in group[1:]:
                tokens = int(indexed[name][identifier]["target_text"]["tokens"])
                if tokens != reference_tokens:
                    raise ValueError(
                        f"paired target token count mismatch for {identifier}: "
                        f"{group_reference}={reference_tokens}, {name}={tokens}"
                    )
        if len(set(sample_availability)) != 1:
            raise ValueError(f"QA score availability differs between variants for {identifier}")
        qa_availability.add(sample_availability[0])
        by_dataset[dataset].append(identifier)
    if len(qa_availability) != 1:
        raise ValueError("partial QA scoring is not valid for paired macro analysis")
    for identifiers in by_dataset.values():
        identifiers.sort()
    return indexed, dict(sorted(by_dataset.items())), qa_availability.pop()


def _macro_qa(
    selected: Sequence[str], rows: Mapping[str, Mapping[str, Any]],
) -> float:
    by_dataset: dict[str, list[float]] = defaultdict(list)
    for identifier in selected:
        row = rows[identifier]
        by_dataset[str(row["dataset"])].append(float(row["score"]))
    return 100.0 * sum(
        sum(values) / len(values) for values in by_dataset.values()
    ) / len(by_dataset)


def _token_weighted_nll(
    selected: Sequence[str], rows: Mapping[str, Mapping[str, Any]],
) -> float:
    total = sum(float(rows[item]["target_text"]["total_nll"]) for item in selected)
    tokens = sum(int(rows[item]["target_text"]["tokens"]) for item in selected)
    if tokens <= 0:
        raise ValueError("selected rows have no target tokens")
    return total / tokens


def _relative_change(candidate: float, baseline: float) -> float:
    if baseline <= 0:
        raise ValueError("relative NLL change requires positive baseline NLL")
    return candidate / baseline - 1.0


def _percentile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a percentile of an empty sequence")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("percentile probability must be in [0, 1]")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _interval(values: Sequence[float]) -> list[float]:
    if not all(math.isfinite(value) for value in values):
        raise FloatingPointError("bootstrap produced a non-finite statistic")
    return [_percentile(values, 0.025), _percentile(values, 0.975)]


def _validate_bootstrap(samples: int, seed: int) -> None:
    if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
        raise ValueError("bootstrap samples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("bootstrap seed must be an integer")


def _stratified_draw(
    rng: random.Random, by_dataset: Mapping[str, Sequence[str]],
) -> list[str]:
    selected: list[str] = []
    for dataset in sorted(by_dataset):
        identifiers = by_dataset[dataset]
        selected.extend(rng.choice(identifiers) for _ in identifiers)
    return selected


def stratified_paired_bootstrap(
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Paired, dataset-stratified QA/NLL comparison.

    The same sampled IDs are used for both variants.  QA is a macro average of
    per-dataset scores (in points); target NLL is pooled by target-token count.
    Rows without QA generation are allowed, but then QA point estimates and
    intervals are reported as ``None`` rather than inferred from NLL.
    """
    _validate_bootstrap(samples, seed)
    indexed, by_dataset, qa_available = _align_variants({
        "baseline": baseline_rows,
        "candidate": candidate_rows,
    })
    ordered = [item for dataset in sorted(by_dataset) for item in by_dataset[dataset]]

    def summarize(selected: Sequence[str]) -> dict[str, float | None]:
        baseline_nll = _token_weighted_nll(selected, indexed["baseline"])
        candidate_nll = _token_weighted_nll(selected, indexed["candidate"])
        result: dict[str, float | None] = {
            "baseline_target_nll": baseline_nll,
            "candidate_target_nll": candidate_nll,
            "target_nll_relative_change": _relative_change(candidate_nll, baseline_nll),
            "baseline_qa_macro": None,
            "candidate_qa_macro": None,
            "qa_macro_delta_points": None,
        }
        if qa_available:
            baseline_qa = _macro_qa(selected, indexed["baseline"])
            candidate_qa = _macro_qa(selected, indexed["candidate"])
            result.update({
                "baseline_qa_macro": baseline_qa,
                "candidate_qa_macro": candidate_qa,
                "qa_macro_delta_points": candidate_qa - baseline_qa,
            })
        return result

    point = summarize(ordered)
    rng = random.Random(seed)
    qa_draws: list[float] = []
    nll_draws: list[float] = []
    for _ in range(samples):
        draw = summarize(_stratified_draw(rng, by_dataset))
        nll_draws.append(float(draw["target_nll_relative_change"]))
        if qa_available:
            qa_draws.append(float(draw["qa_macro_delta_points"]))
    return {
        "analysis": "paired_dataset_stratified_bootstrap",
        "scorer": SCORER_ID,
        "qa_available": qa_available,
        "paired_samples": len(ordered),
        "datasets": {name: len(ids) for name, ids in by_dataset.items()},
        "bootstrap_samples": samples,
        "seed": seed,
        "point": point,
        "qa_macro_delta_points_ci95": _interval(qa_draws) if qa_available else None,
        "target_nll_relative_change_ci95": _interval(nll_draws),
    }


def paired_bootstrap_from_progress(
    progress: StageProgress,
    baseline_variant: str,
    candidate_variant: str,
    *,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    return stratified_paired_bootstrap(
        progress.rows(baseline_variant), progress.rows(candidate_variant),
        samples=samples, seed=seed,
    )


def stratified_difference_in_differences(
    infinitevl_baseline_rows: Sequence[Mapping[str, Any]],
    infinitevl_candidate_rows: Sequence[Mapping[str, Any]],
    qwen_baseline_rows: Sequence[Mapping[str, Any]],
    qwen_candidate_rows: Sequence[Mapping[str, Any]],
    *,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Estimate InfiniteVL-minus-Qwen degradation on the same sampled IDs."""
    _validate_bootstrap(samples, seed)
    indexed, by_dataset, qa_available = _align_variants(
        {
            "infinitevl_baseline": infinitevl_baseline_rows,
            "infinitevl_candidate": infinitevl_candidate_rows,
            "qwen_baseline": qwen_baseline_rows,
            "qwen_candidate": qwen_candidate_rows,
        },
        token_match_groups=(
            ("infinitevl_baseline", "infinitevl_candidate"),
            ("qwen_baseline", "qwen_candidate"),
        ),
    )
    ordered = [item for dataset in sorted(by_dataset) for item in by_dataset[dataset]]

    def summarize(selected: Sequence[str]) -> dict[str, float | None]:
        values = {
            name: _token_weighted_nll(selected, rows)
            for name, rows in indexed.items()
        }
        infinite_nll_change = _relative_change(
            values["infinitevl_candidate"], values["infinitevl_baseline"],
        )
        qwen_nll_change = _relative_change(
            values["qwen_candidate"], values["qwen_baseline"],
        )
        result: dict[str, float | None] = {
            "infinitevl_target_nll_relative_change": infinite_nll_change,
            "qwen_target_nll_relative_change": qwen_nll_change,
            "target_nll_relative_change_did": infinite_nll_change - qwen_nll_change,
            "target_nll_ratio_of_ratios_minus_one": (
                (values["infinitevl_candidate"] / values["infinitevl_baseline"])
                / (values["qwen_candidate"] / values["qwen_baseline"])
                - 1.0
            ),
            "infinitevl_qa_macro_delta_points": None,
            "qwen_qa_macro_delta_points": None,
            "qa_macro_did_points": None,
        }
        if qa_available:
            infinite_delta = (
                _macro_qa(selected, indexed["infinitevl_candidate"])
                - _macro_qa(selected, indexed["infinitevl_baseline"])
            )
            qwen_delta = (
                _macro_qa(selected, indexed["qwen_candidate"])
                - _macro_qa(selected, indexed["qwen_baseline"])
            )
            result.update({
                "infinitevl_qa_macro_delta_points": infinite_delta,
                "qwen_qa_macro_delta_points": qwen_delta,
                "qa_macro_did_points": infinite_delta - qwen_delta,
            })
        return result

    point = summarize(ordered)
    rng = random.Random(seed)
    qa_draws: list[float] = []
    nll_draws: list[float] = []
    ratio_draws: list[float] = []
    for _ in range(samples):
        draw = summarize(_stratified_draw(rng, by_dataset))
        nll_draws.append(float(draw["target_nll_relative_change_did"]))
        ratio_draws.append(float(draw["target_nll_ratio_of_ratios_minus_one"]))
        if qa_available:
            qa_draws.append(float(draw["qa_macro_did_points"]))
    return {
        "analysis": "infinitevl_minus_qwen_difference_in_differences",
        "scorer": SCORER_ID,
        "qa_available": qa_available,
        "paired_samples": len(ordered),
        "datasets": {name: len(ids) for name, ids in by_dataset.items()},
        "bootstrap_samples": samples,
        "seed": seed,
        "point": point,
        "qa_macro_did_points_ci95": _interval(qa_draws) if qa_available else None,
        "target_nll_relative_change_did_ci95": _interval(nll_draws),
        "target_nll_ratio_of_ratios_minus_one_ci95": _interval(ratio_draws),
    }


def difference_in_differences_from_progress(
    infinitevl_progress: StageProgress,
    qwen_progress: StageProgress,
    *,
    baseline_variant: str,
    candidate_variant: str,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    return stratified_difference_in_differences(
        infinitevl_progress.rows(baseline_variant),
        infinitevl_progress.rows(candidate_variant),
        qwen_progress.rows(baseline_variant),
        qwen_progress.rows(candidate_variant),
        samples=samples,
        seed=seed,
    )


def stratified_factorial_interaction(
    bf16_rows: Sequence[Mapping[str, Any]],
    ffn_rows: Sequence[Mapping[str, Any]],
    attn_main_rows: Sequence[Mapping[str, Any]],
    decoder_safe_rows: Sequence[Mapping[str, Any]],
    *,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Analyze the W4 FFN × attention 2×2 design.

    Cells are BF16 (neither quantized), FFN-only, attention-main-only, and
    decoder-safe (both).  The additive interaction is ``both-ffn-attn+bf16``.
    For NLL, a multiplicative interaction is also reported as
    ``both*bf16/(ffn*attn)-1``.
    """
    _validate_bootstrap(samples, seed)
    indexed, by_dataset, qa_available = _align_variants({
        "bf16": bf16_rows,
        "ffn": ffn_rows,
        "attn_main": attn_main_rows,
        "decoder_safe": decoder_safe_rows,
    })
    ordered = [item for dataset in sorted(by_dataset) for item in by_dataset[dataset]]

    def summarize(selected: Sequence[str]) -> dict[str, Any]:
        nll = {
            name: _token_weighted_nll(selected, rows)
            for name, rows in indexed.items()
        }
        if min(nll.values()) <= 0:
            raise ValueError("factorial NLL ratios require positive cell NLLs")
        result: dict[str, Any] = {
            "cell_target_nll": nll,
            "ffn_target_nll_relative_change_vs_bf16": _relative_change(
                nll["ffn"], nll["bf16"],
            ),
            "attn_main_target_nll_relative_change_vs_bf16": _relative_change(
                nll["attn_main"], nll["bf16"],
            ),
            "decoder_safe_target_nll_relative_change_vs_bf16": _relative_change(
                nll["decoder_safe"], nll["bf16"],
            ),
            "target_nll_additive_interaction": (
                nll["decoder_safe"] - nll["ffn"] - nll["attn_main"] + nll["bf16"]
            ),
            "target_nll_multiplicative_interaction": (
                nll["decoder_safe"] * nll["bf16"]
                / (nll["ffn"] * nll["attn_main"])
                - 1.0
            ),
            "cell_qa_macro": None,
            "ffn_qa_delta_points_vs_bf16": None,
            "attn_main_qa_delta_points_vs_bf16": None,
            "decoder_safe_qa_delta_points_vs_bf16": None,
            "qa_macro_additive_interaction_points": None,
        }
        if qa_available:
            qa = {name: _macro_qa(selected, rows) for name, rows in indexed.items()}
            result.update({
                "cell_qa_macro": qa,
                "ffn_qa_delta_points_vs_bf16": qa["ffn"] - qa["bf16"],
                "attn_main_qa_delta_points_vs_bf16": qa["attn_main"] - qa["bf16"],
                "decoder_safe_qa_delta_points_vs_bf16": (
                    qa["decoder_safe"] - qa["bf16"]
                ),
                "qa_macro_additive_interaction_points": (
                    qa["decoder_safe"] - qa["ffn"] - qa["attn_main"] + qa["bf16"]
                ),
            })
        return result

    point = summarize(ordered)
    rng = random.Random(seed)
    qa_draws: list[float] = []
    additive_nll_draws: list[float] = []
    multiplicative_nll_draws: list[float] = []
    for _ in range(samples):
        draw = summarize(_stratified_draw(rng, by_dataset))
        additive_nll_draws.append(float(draw["target_nll_additive_interaction"]))
        multiplicative_nll_draws.append(
            float(draw["target_nll_multiplicative_interaction"])
        )
        if qa_available:
            qa_draws.append(float(draw["qa_macro_additive_interaction_points"]))
    return {
        "analysis": "w4_ffn_by_attn_main_2x2_factorial",
        "scorer": SCORER_ID,
        "qa_available": qa_available,
        "paired_samples": len(ordered),
        "datasets": {name: len(ids) for name, ids in by_dataset.items()},
        "bootstrap_samples": samples,
        "seed": seed,
        "point": point,
        "qa_macro_additive_interaction_points_ci95": (
            _interval(qa_draws) if qa_available else None
        ),
        "target_nll_additive_interaction_ci95": _interval(additive_nll_draws),
        "target_nll_multiplicative_interaction_ci95": (
            _interval(multiplicative_nll_draws)
        ),
    }


def factorial_interaction_from_progress(
    progress: StageProgress,
    *,
    bf16_variant: str = "bf16",
    ffn_variant: str = "rtn_w4a16_ffn",
    attn_main_variant: str = "rtn_w4a16_attn_main",
    decoder_safe_variant: str = "rtn_w4a16_decoder_safe",
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    return stratified_factorial_interaction(
        progress.rows(bf16_variant),
        progress.rows(ffn_variant),
        progress.rows(attn_main_variant),
        progress.rows(decoder_safe_variant),
        samples=samples,
        seed=seed,
    )


# Compact aliases for callers that do not need to emphasize the stratification
# in every call site.  The returned payload always records the analysis design.
paired_bootstrap = stratified_paired_bootstrap
difference_in_differences = stratified_difference_in_differences
factorial_interaction = stratified_factorial_interaction
