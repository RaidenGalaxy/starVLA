from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from experiments.ffn_quant_causal.adapters import load_model_bundle
from experiments.ffn_quant_causal.budget import WallclockBudget
from experiments.ffn_quant_causal.evaluation import aggregate_rows
from experiments.infinitevl_quant.progress import atomic_json
from experiments.vlm_quant_benchmark.data import MultimodalSample

from .calibration import gamma_map
from .config import MEDIATOR_GROUPS, MechanismConfig
from .evaluation import evaluate_shadow_variant
from .gdn import aggregate_state_audits
from .protocol import audited_splits, load_audit, select_long16, stage_progress
from .stages import (
    _close,
    _scales_artifact,
    load_parent_mbq_scales,
    model_paths,
)
from .statistics import (
    confirm_decisions,
    length_stratified_gap_reduction_bootstrap,
    stratified_causal_bootstrap,
    stratified_gap_reduction_bootstrap,
    stratified_mediation_bootstrap,
)

PARENT_CONFIRM_VARIANTS = {
    "B": "{kind}__confirm192__bf16",
    "W": "{kind}__confirm192__rtn_w4a16_ffn",
}


def _rows(directory: Path) -> list[dict[str, Any]]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    rows = [json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))]
    if not rows:
        raise RuntimeError(f"no rows in {directory}")
    return rows


def parent_confirm_rows(parent: Path, kind: str, condition: str) -> list[dict[str, Any]]:
    variant = PARENT_CONFIRM_VARIANTS[condition].format(kind=kind)
    rows = _rows(parent.resolve() / "confirm" / variant / "samples")
    if len(rows) != 192 or len({row.get("id") for row in rows}) != 192:
        raise RuntimeError(f"parent confirm variant {variant} is not complete")
    return rows


def _variant(progress, name: str) -> list[dict[str, Any]]:
    rows = progress.rows(name)
    if not rows:
        raise RuntimeError(f"missing mechanism confirm rows for {name}")
    return rows


def _summary_with_argmax_change(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = aggregate_rows(rows)
    consistency = float(summary["target_argmax_consistency"])
    summary["target_argmax_change_rate"] = 1.0 - consistency if math.isfinite(consistency) else math.nan
    return summary


def _run_native_new_conditions(
    config: MechanismConfig,
    bundle,
    samples: list[MultimodalSample],
    *,
    progress,
    artifact,
    conditions: tuple[str, ...],
    wallclock_budget: WallclockBudget | None,
) -> dict[str, dict[str, Any]]:
    reference_dir = config.parent_output / "confirm/reference_artifacts" / bundle.kind
    result = {}
    for condition in conditions:
        result[condition] = evaluate_shadow_variant(
            bundle,
            samples,
            progress=progress,
            variant=f"{bundle.kind}__confirm192__rtn__{condition}",
            gammas=gamma_map(artifact, bundle.kind, condition),
            reference_dir=reference_dir,
            write_reference=False,
            gdn_reference_dir=None,
            capture_gdn=False,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            max_new_tokens=config.max_new_tokens,
            generate=condition == "R",
            bits=config.weight_bits,
            group_size=config.group_size,
            wallclock_budget=wallclock_budget,
        )
    return result


def _run_recurrent_matrix(
    config: MechanismConfig,
    bundle,
    samples: list[MultimodalSample],
    *,
    progress,
    artifact,
    method: str,
    split_label: str,
    reference_dir: Path,
    scales,
    run_residual: bool,
    selected_mediators: list[str],
    generate_core: bool,
    wallclock_budget: WallclockBudget | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    gdn_dir = progress.root / "gdn_references" / method / split_label
    evaluate_shadow_variant(
        bundle,
        samples,
        progress=progress,
        variant=f"infinitevl__{split_label}__{method}__reference_capture",
        gammas=gamma_map(artifact, "infinitevl", "B"),
        reference_dir=reference_dir,
        write_reference=not any(reference_dir.glob("*.pt")),
        gdn_reference_dir=gdn_dir,
        capture_gdn=True,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        max_new_tokens=config.max_new_tokens,
        generate=False,
        bits=config.weight_bits,
        group_size=config.group_size,
        scales=scales,
        wallclock_budget=wallclock_budget,
    )
    conditions: dict[str, tuple[str, tuple[str, ...], bool]] = {
        "B": ("B", (), False),
        "W": ("W", (), False),
        "G": ("W", (), True),
        "M": ("W", MEDIATOR_GROUPS, False),
    }
    if run_residual:
        conditions.update(
            {
                "R": ("R", (), False),
                "RG": ("R", (), True),
            }
        )
    metrics = {}
    variants = {}
    for condition, (gamma_condition, repairs, carry) in conditions.items():
        variant = f"infinitevl__{split_label}__{method}__recurrent__{condition}"
        variants[condition] = variant
        metrics[condition] = evaluate_shadow_variant(
            bundle,
            samples,
            progress=progress,
            variant=variant,
            gammas=gamma_map(artifact, "infinitevl", gamma_condition),
            reference_dir=reference_dir,
            write_reference=False,
            gdn_reference_dir=gdn_dir,
            capture_gdn=False,
            repair_groups=repairs,
            carry_clamp=carry,
            force_gdn_intervention=True,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            max_new_tokens=config.max_new_tokens,
            generate=generate_core and condition in {"G", "RG"},
            bits=config.weight_bits,
            group_size=config.group_size,
            scales=scales,
            wallclock_budget=wallclock_budget,
        )
    for mediator in selected_mediators:
        condition = f"repair_{mediator}"
        variant = f"infinitevl__{split_label}__{method}__recurrent__{condition}"
        variants[condition] = variant
        metrics[condition] = evaluate_shadow_variant(
            bundle,
            samples,
            progress=progress,
            variant=variant,
            gammas=gamma_map(artifact, "infinitevl", "W"),
            reference_dir=reference_dir,
            write_reference=False,
            gdn_reference_dir=gdn_dir,
            capture_gdn=False,
            repair_groups=(mediator,),
            force_gdn_intervention=True,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            max_new_tokens=config.max_new_tokens,
            generate=False,
            bits=config.weight_bits,
            group_size=config.group_size,
            scales=scales,
            wallclock_budget=wallclock_budget,
        )
    return metrics, variants


def _run_mbq_replication(
    config: MechanismConfig,
    paths: dict[str, Path],
    samples: list[MultimodalSample],
    *,
    progress,
    artifact,
    wallclock_budget: WallclockBudget | None,
) -> dict[str, Any]:
    native_rows: dict[str, list[dict]] = {}
    infinite_recurrent = None
    for kind in ("infinitevl", "qwen"):
        bundle = load_model_bundle(kind, paths[kind])
        try:
            scales = load_parent_mbq_scales(config.parent_output, kind)
            reference_dir = progress.root / "reference_logits/mbq/qig_acceptance32" / kind
            gdn_dir = progress.root / "gdn_references/mbq/qig_acceptance32" if kind == "infinitevl" else None
            for condition in ("B", "W", "R"):
                variant = f"{kind}__qig_acceptance32__mbq__{condition}"
                evaluate_shadow_variant(
                    bundle,
                    samples,
                    progress=progress,
                    variant=variant,
                    gammas=gamma_map(artifact, kind, condition),
                    reference_dir=reference_dir,
                    write_reference=condition == "B",
                    gdn_reference_dir=gdn_dir,
                    capture_gdn=kind == "infinitevl" and condition == "B",
                    min_pixels=config.min_pixels,
                    max_pixels=config.max_pixels,
                    max_new_tokens=config.max_new_tokens,
                    generate=False,
                    bits=config.weight_bits,
                    group_size=config.group_size,
                    scales=scales,
                    wallclock_budget=wallclock_budget,
                )
                native_rows[f"{kind}_{condition}"] = progress.rows(variant)
            if kind == "infinitevl":
                metrics, variants = _run_recurrent_matrix(
                    config,
                    bundle,
                    samples,
                    progress=progress,
                    artifact=artifact,
                    method="mbq",
                    split_label="qig_acceptance32",
                    reference_dir=reference_dir,
                    scales=scales,
                    run_residual=True,
                    selected_mediators=[],
                    generate_core=False,
                    wallclock_budget=wallclock_budget,
                )
                infinite_recurrent = {
                    "metrics": metrics,
                    "mediation": stratified_mediation_bootstrap(
                        {key: progress.rows(variants[key]) for key in ("B", "W", "R", "G", "RG")},
                        samples=config.bootstrap_samples,
                        seed=config.seed,
                    ),
                }
        finally:
            _close(bundle)
    infinite_residual = stratified_gap_reduction_bootstrap(
        native_rows["infinitevl_B"],
        native_rows["infinitevl_W"],
        native_rows["infinitevl_R"],
        samples=config.bootstrap_samples,
        seed=config.seed,
    )
    qwen_stress = stratified_gap_reduction_bootstrap(
        native_rows["qwen_B"],
        native_rows["qwen_W"],
        native_rows["qwen_R"],
        samples=config.bootstrap_samples,
        seed=config.seed,
    )
    return {
        "label": "directional_replication_only",
        "infinitevl_residual": infinite_residual,
        "qwen_reverse_stress": qwen_stress,
        "directional_checks": {
            "infinitevl_R_reduces_gap": infinite_residual["point"]["absolute_gap_reduction"] > 0,
            "qwen_R_increases_gap": qwen_stress["point"]["absolute_gap_reduction"] < 0,
            "both_directions_replicate": (
                infinite_residual["point"]["absolute_gap_reduction"] > 0
                and qwen_stress["point"]["absolute_gap_reduction"] < 0
            ),
        },
        "infinitevl_recurrent": infinite_recurrent,
    }


def run_confirm(
    config: MechanismConfig,
    *,
    infinitevl_path: Path | None,
    wallclock_budget: WallclockBudget | None,
) -> dict[str, Any]:
    load_audit(config)
    screen_path = config.output_dir / "gdn_mediation/state_and_shapley_summary.json"
    localize_path = config.output_dir / "localize/layer_pair_summary.json"
    if not screen_path.is_file() or not localize_path.is_file():
        raise FileNotFoundError("gdn_mediation and localize must complete before confirm")
    screen = json.loads(screen_path.read_text())
    localize = json.loads(localize_path.read_text())
    gates = screen["screen_decisions"]
    residual_pass = bool(gates["H_R"]["pass"])
    gdn_pass = bool(gates["H_G"]["pass"])
    progress = stage_progress(config, "confirm")
    splits, split_audit = audited_splits(config)
    samples = list(splits["confirm"])
    paths = model_paths(config, infinitevl_path)
    scales = _scales_artifact(config)
    rtn_artifact = scales["methods"]["rtn"]
    selected = list(screen["mediator_selection"]["selected"]) if gdn_pass else []

    native_new_metrics = {}
    recurrent_metrics = None
    recurrent_variants = None
    for kind in ("infinitevl", "qwen"):
        bundle = load_model_bundle(kind, paths[kind])
        try:
            if residual_pass:
                native_new_metrics[kind] = _run_native_new_conditions(
                    config,
                    bundle,
                    samples,
                    progress=progress,
                    artifact=rtn_artifact,
                    conditions=("A", "R"),
                    wallclock_budget=wallclock_budget,
                )
            if kind == "infinitevl" and gdn_pass:
                recurrent_metrics, recurrent_variants = _run_recurrent_matrix(
                    config,
                    bundle,
                    samples,
                    progress=progress,
                    artifact=rtn_artifact,
                    method="rtn",
                    split_label="confirm192",
                    reference_dir=config.parent_output / "confirm/reference_artifacts/infinitevl",
                    scales=None,
                    run_residual=residual_pass,
                    selected_mediators=selected,
                    generate_core=True,
                    wallclock_budget=wallclock_budget,
                )
        finally:
            _close(bundle)

    causal = None
    if residual_pass:
        causal = stratified_causal_bootstrap(
            {
                **{
                    f"{kind}_{condition}": parent_confirm_rows(
                        config.parent_output,
                        kind,
                        condition,
                    )
                    for kind in ("infinitevl", "qwen")
                    for condition in ("B", "W")
                },
                **{
                    f"{kind}_{condition}": _variant(
                        progress,
                        f"{kind}__confirm192__rtn__{condition}",
                    )
                    for kind in ("infinitevl", "qwen")
                    for condition in ("A", "R")
                },
            },
            samples=config.bootstrap_samples,
            seed=config.seed,
        )

    mediation = None
    gdn_only = None
    state = None
    long16 = None
    length_effects = None
    mediator_confirm_effects: dict[str, Any] = {}
    confirmed_mediators: list[str] = []
    full_mediator_repair = None
    if gdn_pass and recurrent_variants is not None:
        base_rows = {key: _variant(progress, recurrent_variants[key]) for key in ("B", "W", "G")}
        gdn_only = stratified_gap_reduction_bootstrap(
            base_rows["B"],
            base_rows["W"],
            base_rows["G"],
            samples=config.bootstrap_samples,
            seed=config.seed,
        )
        if residual_pass:
            mediation = stratified_mediation_bootstrap(
                {key: _variant(progress, recurrent_variants[key]) for key in ("B", "W", "R", "G", "RG")},
                samples=config.bootstrap_samples,
                seed=config.seed,
            )
        length_effects = {
            "carry_clamp": length_stratified_gap_reduction_bootstrap(
                base_rows["B"],
                base_rows["W"],
                base_rows["G"],
                samples=config.bootstrap_samples,
                seed=config.seed,
            ),
        }
        if residual_pass:
            length_effects["joint_R_plus_carry_clamp"] = length_stratified_gap_reduction_bootstrap(
                base_rows["B"],
                base_rows["W"],
                _variant(progress, recurrent_variants["RG"]),
                samples=config.bootstrap_samples,
                seed=config.seed,
            )
        full_mediator_repair = stratified_gap_reduction_bootstrap(
            base_rows["B"],
            base_rows["W"],
            _variant(progress, recurrent_variants["M"]),
            samples=config.bootstrap_samples,
            seed=config.seed,
        )
        for mediator in selected:
            effect = stratified_gap_reduction_bootstrap(
                base_rows["B"],
                base_rows["W"],
                _variant(progress, recurrent_variants[f"repair_{mediator}"]),
                samples=config.bootstrap_samples,
                seed=config.seed,
            )
            confirmed = effect["ci95"]["absolute_gap_reduction"][0] > 0
            mediator_confirm_effects[mediator] = {
                **effect,
                "confirmed_positive_effect": confirmed,
            }
            if confirmed:
                confirmed_mediators.append(mediator)
        token_counts = {
            row["id"]: int(row["sequence_tokens"])
            for row in parent_confirm_rows(config.parent_output, "infinitevl", "B")
        }
        selected_long = select_long16(samples, token_counts)
        long_ids = {sample.id for sample in selected_long}
        long16 = {
            "ids": sorted(long_ids),
            "per_dataset": {
                dataset: sum(sample.dataset == dataset for sample in selected_long)
                for dataset in sorted({sample.dataset for sample in selected_long})
            },
        }
        state = {
            condition: aggregate_state_audits(
                [row for row in _variant(progress, recurrent_variants[condition]) if row["id"] in long_ids]
            )
            for condition in (("W", "G", "RG", "M") if residual_pass else ("W", "G", "M"))
        }

    mbq = None
    if residual_pass and gdn_pass:
        mbq = _run_mbq_replication(
            config,
            paths,
            list(splits["qig_acceptance"]),
            progress=progress,
            artifact=scales["methods"]["mbq"],
            wallclock_budget=wallclock_budget,
        )

    decisions = {
        "H_R": False,
        "H_G": False,
        "H_RG_joint": False,
        "interaction": "not_tested",
        "rules_frozen_before_confirm": True,
    }
    if causal is not None:
        mediation_for_decision = mediation
        if mediation_for_decision is None and gdn_only is not None:
            mediation_for_decision = {
                "point": {
                    "gdn_explained_fraction": gdn_only["point"]["explained_fraction"],
                },
                "ci95": {
                    "gdn_explained_fraction": gdn_only["ci95"]["explained_fraction"],
                },
            }
        decisions = confirm_decisions(
            causal,
            gdn_gain_ratio_ci95=localize["primary_screen_exact_dose1"]["ci95"],
            mediation=mediation_for_decision,
        )
    elif gdn_only is not None:
        decisions["H_G"] = bool(
            localize["primary_screen_exact_dose1"]["ci95"][0] > 1
            and gdn_only["ci95"]["explained_fraction"][0] > 0
            and gdn_only["point"]["explained_fraction"] >= 0.30
        )

    qa_core: dict[str, dict[str, Any]] = {}
    for kind in ("infinitevl", "qwen"):
        qa_core[kind] = {
            condition: _summary_with_argmax_change(
                parent_confirm_rows(
                    config.parent_output,
                    kind,
                    condition,
                )
            )
            for condition in ("B", "W")
        }
        if residual_pass:
            qa_core[kind]["R"] = native_new_metrics[kind]["R"]
    if gdn_pass and recurrent_metrics is not None:
        qa_core["infinitevl"]["G"] = recurrent_metrics["G"]
        if residual_pass:
            qa_core["infinitevl"]["RG"] = recurrent_metrics["RG"]

    result = {
        "format_version": 1,
        "split": {
            "name": "within_protocol_holdout192",
            "samples": 192,
            "digest": split_audit["digests"]["confirm"],
            "evaluation_history": "historically_reused_evaluation",
        },
        "screen_gates": gates,
        "native_new_metrics": native_new_metrics,
        "recurrent_metrics": recurrent_metrics,
        "causal_transport": causal,
        "gdn_gap_reduction": gdn_only,
        "mediation_factorial": mediation,
        "full_mediator_repair": full_mediator_repair,
        "selected_mediator_groups": selected,
        "mediator_confirm_effects": mediator_confirm_effects,
        "confirmed_mediator_groups": confirmed_mediators,
        "length_stratified_effects": length_effects,
        "long16": long16,
        "long16_state_drift": state,
        "qa_core_conditions": qa_core,
        "mbq_directional_replication": mbq,
        "confirm_decisions": decisions,
        "labels": [
            "cpu_reference_backend",
            "fake_quantized",
            "ffn_only",
            "diagnostic_counterfactual",
            "historically_reused_evaluation",
            "no_speed_or_memory_claim",
        ],
    }
    atomic_json(progress.root / "causal_effects.json", result)
    progress.mark_complete(("causal_effects.json",))
    return result
