from __future__ import annotations

import gc
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download

from experiments.ffn_quant_causal.adapters import ModelBundle, load_model_bundle
from experiments.ffn_quant_causal.budget import WallclockBudget
from experiments.ffn_quant_causal.config import INFINITEVL_MODEL_ID, INFINITEVL_REVISION
from experiments.infinitevl_quant.progress import atomic_json, file_sha256
from experiments.vlm_quant_benchmark.data import MultimodalSample

from .calibration import derive_transport_scales, gamma_map
from .config import (
    GDN_SWA_LAYER_PAIRS,
    MEDIATOR_GROUPS,
    SCREEN_LAYER_PAIRS,
    MechanismConfig,
)
from .evaluation import (
    collect_shadow_calibration,
    evaluate_shadow_variant,
)
from .gdn import aggregate_state_audits
from .localization import run_localization_grid
from .protocol import audited_splits, load_audit, stage_progress
from .statistics import (
    gain_ratio_bootstrap,
    powerset,
    screen_decisions,
    select_mediators,
    shapley_values,
    stratified_causal_bootstrap,
    stratified_gap_reduction_bootstrap,
    stratified_mediation_bootstrap,
    subset_label,
)


def resolve_infinitevl_path(path: Path | None = None) -> Path:
    if path is not None:
        return path.resolve()
    return Path(
        snapshot_download(
            repo_id=INFINITEVL_MODEL_ID,
            revision=INFINITEVL_REVISION,
            local_files_only=True,
        )
    ).resolve()


def model_paths(
    config: MechanismConfig,
    infinitevl_path: Path | None,
) -> dict[str, Path]:
    return {
        "infinitevl": resolve_infinitevl_path(infinitevl_path),
        "qwen": config.qwen_model_path.resolve(),
    }


def _close(bundle: ModelBundle) -> None:
    bundle.close()
    gc.collect()


def _parent_scale_paths(parent: Path, model_kind: str) -> tuple[Path, Path]:
    payload = parent / "mbq_qig/scale_artifacts" / model_kind / "mbq_w4a16_ffn.pt"
    return payload, payload.with_suffix(payload.suffix + ".identity.json")


def load_parent_mbq_scales(parent: Path, model_kind: str) -> dict[str, torch.Tensor]:
    payload_path, sidecar_path = _parent_scale_paths(parent.resolve(), model_kind)
    if not payload_path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError(f"incomplete parent MBQ scale artifact for {model_kind}")
    sidecar = json.loads(sidecar_path.read_text())
    metadata = sidecar.get("payload", {})
    if metadata.get("artifact_sha256") != file_sha256(payload_path):
        raise RuntimeError(f"parent MBQ scale checksum mismatch for {model_kind}")
    value = torch.load(payload_path, map_location="cpu", weights_only=True)
    if value.get("format_version") != 2 or not isinstance(value.get("scales"), dict):
        raise RuntimeError(f"unsupported parent MBQ scale artifact for {model_kind}")
    scales = {
        str(name): tensor.detach().float().cpu()
        for name, tensor in value["scales"].items()
        if isinstance(tensor, torch.Tensor)
    }
    identity_names = set(sidecar.get("identity", {}).get("target_names", ()))
    if not scales or set(scales) != identity_names:
        raise RuntimeError(f"parent MBQ scale names mismatch for {model_kind}")
    return scales


def _scales_artifact(config: MechanismConfig) -> dict[str, Any]:
    path = config.output_dir / "calibrate/error_scales.json"
    if not path.is_file():
        raise FileNotFoundError("calibrate/error_scales.json is required")
    value = json.loads(path.read_text())
    if set(value.get("methods", {})) != {"rtn", "mbq"}:
        raise RuntimeError("error-scale artifact must contain RTN and MBQ methods")
    return value


def run_calibrate(
    config: MechanismConfig,
    *,
    infinitevl_path: Path | None,
    wallclock_budget: WallclockBudget | None,
) -> dict[str, Any]:
    load_audit(config)
    progress = stage_progress(config, "calibrate")
    splits, split_audit = audited_splits(config)
    paths = model_paths(config, infinitevl_path)
    model_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    model_audits = {}
    for kind in ("infinitevl", "qwen"):
        bundle = load_model_bundle(kind, paths[kind])
        try:
            model_audits[kind] = bundle.load_audit
            mbq_scales = load_parent_mbq_scales(config.parent_output, kind)
            model_rows[kind] = {
                "rtn": collect_shadow_calibration(
                    bundle,
                    splits["qig"],
                    progress=progress,
                    variant=f"{kind}__qig32__rtn_w4a16",
                    min_pixels=config.min_pixels,
                    max_pixels=config.max_pixels,
                    bits=config.weight_bits,
                    group_size=config.group_size,
                    wallclock_budget=wallclock_budget,
                ),
                "mbq": collect_shadow_calibration(
                    bundle,
                    splits["qig"],
                    progress=progress,
                    variant=f"{kind}__qig32__mbq_w4a16",
                    min_pixels=config.min_pixels,
                    max_pixels=config.max_pixels,
                    bits=config.weight_bits,
                    group_size=config.group_size,
                    scales=mbq_scales,
                    wallclock_budget=wallclock_budget,
                ),
            }
        finally:
            _close(bundle)
    methods = {
        method: derive_transport_scales(
            model_rows["infinitevl"][method],
            model_rows["qwen"][method],
            clip=config.transport_clip,
        )
        for method in ("rtn", "mbq")
    }
    artifact = {
        "format_version": 1,
        "calibration_split": {
            "name": "qig",
            "samples": 32,
            "digest": split_audit["digests"]["qig"],
        },
        "methods": methods,
        "confirm_data_used": False,
        "model_load_audits": model_audits,
    }
    atomic_json(progress.root / "error_scales.json", artifact)
    progress.mark_complete(("error_scales.json",))
    return artifact


def run_localize(
    config: MechanismConfig,
    *,
    infinitevl_path: Path | None,
    wallclock_budget: WallclockBudget | None,
) -> dict[str, Any]:
    load_audit(config)
    progress = stage_progress(config, "localize")
    splits, split_audit = audited_splits(config)
    bundle = load_model_bundle("infinitevl", resolve_infinitevl_path(infinitevl_path))
    try:
        diagnostic = run_localization_grid(
            bundle,
            splits["diagnostic8"],
            progress=progress,
            variant="diagnostic8_all_pairs",
            pairs=GDN_SWA_LAYER_PAIRS,
            directions=("exact", "gaussian"),
            doses=(1.0,),
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            bits=config.weight_bits,
            group_size=config.group_size,
            seed=config.seed,
            wallclock_budget=wallclock_budget,
        )
        screen = run_localization_grid(
            bundle,
            splits["screen"],
            progress=progress,
            variant="screen32_fixed_pairs",
            pairs=SCREEN_LAYER_PAIRS,
            directions=("exact",),
            doses=(0.5, 1.0, 2.0),
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            bits=config.weight_bits,
            group_size=config.group_size,
            seed=config.seed,
            wallclock_budget=wallclock_budget,
        )
    finally:
        _close(bundle)
    diagnostic_ratios = {
        direction: gain_ratio_bootstrap(
            [row for row in diagnostic if row["direction"] == direction and float(row["dose"]) == 1.0],
            samples=config.bootstrap_samples,
            seed=config.seed,
        )
        for direction in ("exact", "gaussian")
    }
    screen_ratios = {
        str(dose): gain_ratio_bootstrap(
            [row for row in screen if float(row["dose"]) == dose],
            samples=config.bootstrap_samples,
            seed=config.seed,
        )
        for dose in (0.5, 1.0, 2.0)
    }
    result = {
        "format_version": 1,
        "diagnostic_split_digest": split_audit["digests"]["diagnostic8"],
        "screen_split_digest": split_audit["digests"]["screen"],
        "diagnostic_gain_ratios": diagnostic_ratios,
        "screen_gain_ratios": screen_ratios,
        "primary_screen_exact_dose1": screen_ratios["1.0"],
    }
    atomic_json(progress.root / "layer_pair_summary.json", result)
    progress.mark_complete(("layer_pair_summary.json",))
    return result


def _evaluate_transport_model(
    config: MechanismConfig,
    bundle: ModelBundle,
    samples: Sequence[MultimodalSample],
    *,
    progress,
    scale_artifact: Mapping[str, Any],
    method: str,
    split_label: str,
    scales: Mapping[str, torch.Tensor] | None,
    capture_gdn: bool,
    generate_conditions: Sequence[str] = (),
    wallclock_budget: WallclockBudget | None,
) -> dict[str, dict[str, Any]]:
    reference_dir = progress.root / "reference_logits" / method / bundle.kind / split_label
    gdn_dir = progress.root / "gdn_references" / method / split_label if bundle.kind == "infinitevl" else None
    metrics = {}
    for condition in ("B", "W", "A", "R"):
        metrics[condition] = evaluate_shadow_variant(
            bundle,
            samples,
            progress=progress,
            variant=f"{bundle.kind}__{split_label}__{method}__{condition}",
            gammas=gamma_map(scale_artifact, bundle.kind, condition),
            reference_dir=reference_dir,
            write_reference=condition == "B",
            gdn_reference_dir=gdn_dir,
            capture_gdn=(bundle.kind == "infinitevl" and condition == "B" and capture_gdn),
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            max_new_tokens=config.max_new_tokens,
            generate=condition in set(generate_conditions),
            bits=config.weight_bits,
            group_size=config.group_size,
            scales=scales,
            wallclock_budget=wallclock_budget,
        )
    return metrics


def _transport_rows(progress, *, method: str, split_label: str) -> dict[str, list[dict]]:
    return {
        f"{kind}_{condition}": progress.rows(f"{kind}__{split_label}__{method}__{condition}")
        for kind in ("infinitevl", "qwen")
        for condition in ("B", "W", "A", "R")
    }


def run_transport(
    config: MechanismConfig,
    *,
    infinitevl_path: Path | None,
    wallclock_budget: WallclockBudget | None,
) -> dict[str, Any]:
    load_audit(config)
    progress = stage_progress(config, "transport")
    splits, split_audit = audited_splits(config)
    artifact = _scales_artifact(config)
    paths = model_paths(config, infinitevl_path)
    metrics = {}
    for kind in ("infinitevl", "qwen"):
        bundle = load_model_bundle(kind, paths[kind])
        try:
            metrics[kind] = _evaluate_transport_model(
                config,
                bundle,
                splits["screen"],
                progress=progress,
                scale_artifact=artifact["methods"]["rtn"],
                method="rtn",
                split_label="screen32",
                scales=None,
                capture_gdn=True,
                wallclock_budget=wallclock_budget,
            )
        finally:
            _close(bundle)
    causal = stratified_causal_bootstrap(
        _transport_rows(progress, method="rtn", split_label="screen32"),
        samples=config.bootstrap_samples,
        seed=config.seed,
    )
    result = {
        "format_version": 1,
        "split": "screen32",
        "split_digest": split_audit["digests"]["screen"],
        "metrics": metrics,
        "causal_transport": causal,
        "H_R_screen_without_gdn_gate": screen_decisions(
            causal,
            gdn_gain_ratio=None,
        )["H_R"],
    }
    atomic_json(progress.root / "summary.json", result)
    progress.mark_complete(("summary.json",))
    return result


def _dataset_token_nll(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["dataset"]), []).append(row)
    return {
        dataset: sum(float(row["target_text"]["total_nll"]) for row in values)
        / sum(int(row["target_text"]["tokens"]) for row in values)
        for dataset, values in grouped.items()
    }


def _mediator_shapley(progress, variants: Mapping[frozenset[str], str]) -> dict[str, Any]:
    dataset_cells: dict[str, dict[frozenset[str], float]] = {}
    overall_cells = {}
    for subset, variant in variants.items():
        rows = progress.rows(variant)
        dataset_nll = _dataset_token_nll(rows)
        for dataset, value in dataset_nll.items():
            dataset_cells.setdefault(dataset, {})[subset] = value
        all_tokens = sum(int(row["target_text"]["tokens"]) for row in rows)
        overall_cells[subset] = sum(float(row["target_text"]["total_nll"]) for row in rows) / all_tokens
    dataset_results = {dataset: shapley_values(cells) for dataset, cells in sorted(dataset_cells.items())}
    return {
        "overall": shapley_values(overall_cells),
        "datasets": dataset_results,
    }


def run_gdn_mediation(
    config: MechanismConfig,
    *,
    infinitevl_path: Path | None,
    wallclock_budget: WallclockBudget | None,
) -> dict[str, Any]:
    load_audit(config)
    transport_path = config.output_dir / "transport/summary.json"
    localize_path = config.output_dir / "localize/layer_pair_summary.json"
    if not transport_path.is_file() or not localize_path.is_file():
        raise FileNotFoundError("transport and localize must complete before GDN mediation")
    transport = json.loads(transport_path.read_text())
    localize = json.loads(localize_path.read_text())
    progress = stage_progress(config, "gdn_mediation")
    splits, split_audit = audited_splits(config)
    artifact = _scales_artifact(config)["methods"]["rtn"]
    bundle = load_model_bundle("infinitevl", resolve_infinitevl_path(infinitevl_path))
    try:
        diagnostic_reference_dir = progress.root / "reference_logits/diagnostic8"
        diagnostic_gdn_dir = progress.root / "gdn_references/diagnostic8"
        evaluate_shadow_variant(
            bundle,
            splits["diagnostic8"],
            progress=progress,
            variant="infinitevl__diagnostic8__reference_capture",
            gammas=gamma_map(artifact, "infinitevl", "B"),
            reference_dir=diagnostic_reference_dir,
            write_reference=True,
            gdn_reference_dir=diagnostic_gdn_dir,
            capture_gdn=True,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            max_new_tokens=config.max_new_tokens,
            generate=False,
            bits=config.weight_bits,
            group_size=config.group_size,
            wallclock_budget=wallclock_budget,
        )
        subset_variants = {}
        for subset in powerset():
            label = subset_label(subset)
            variant = f"infinitevl__diagnostic8__mediators__{label}"
            subset_variants[subset] = variant
            evaluate_shadow_variant(
                bundle,
                splits["diagnostic8"],
                progress=progress,
                variant=variant,
                gammas=gamma_map(artifact, "infinitevl", "W"),
                reference_dir=diagnostic_reference_dir,
                write_reference=False,
                gdn_reference_dir=diagnostic_gdn_dir,
                capture_gdn=False,
                repair_groups=tuple(subset),
                force_gdn_intervention=True,
                min_pixels=config.min_pixels,
                max_pixels=config.max_pixels,
                max_new_tokens=config.max_new_tokens,
                generate=False,
                bits=config.weight_bits,
                group_size=config.group_size,
                wallclock_budget=wallclock_budget,
            )

        screen_reference_dir = config.output_dir / "transport/reference_logits/rtn/infinitevl/screen32"
        screen_gdn_dir = config.output_dir / "transport/gdn_references/rtn/screen32"
        screen_conditions = {
            "B": ("B", (), False),
            "W": ("W", (), False),
            "R": ("R", (), False),
            "G": ("W", (), True),
            "RG": ("R", (), True),
            "M": ("W", MEDIATOR_GROUPS, False),
        }
        metrics = {}
        for condition, (gamma_condition, repairs, carry) in screen_conditions.items():
            metrics[condition] = evaluate_shadow_variant(
                bundle,
                splits["screen"],
                progress=progress,
                variant=f"infinitevl__screen32__recurrent__{condition}",
                gammas=gamma_map(artifact, "infinitevl", gamma_condition),
                reference_dir=screen_reference_dir,
                write_reference=False,
                gdn_reference_dir=screen_gdn_dir,
                capture_gdn=False,
                repair_groups=repairs,
                carry_clamp=carry,
                force_gdn_intervention=True,
                min_pixels=config.min_pixels,
                max_pixels=config.max_pixels,
                max_new_tokens=config.max_new_tokens,
                generate=False,
                bits=config.weight_bits,
                group_size=config.group_size,
                wallclock_budget=wallclock_budget,
            )
    finally:
        _close(bundle)

    mediator = _mediator_shapley(progress, subset_variants)
    selection = select_mediators(
        mediator["datasets"],
        minimum_share=config.mediator_min_share,
        maximum=config.mediator_max_confirm,
    )
    mediation_rows = {
        condition: progress.rows(f"infinitevl__screen32__recurrent__{condition}")
        for condition in ("B", "W", "R", "G", "RG")
    }
    mediation = stratified_mediation_bootstrap(
        mediation_rows,
        samples=config.bootstrap_samples,
        seed=config.seed,
    )
    full_mediator_repair = stratified_gap_reduction_bootstrap(
        mediation_rows["B"],
        mediation_rows["W"],
        progress.rows("infinitevl__screen32__recurrent__M"),
        samples=config.bootstrap_samples,
        seed=config.seed,
    )
    causal = transport["causal_transport"]
    gain = localize["primary_screen_exact_dose1"]
    decisions = screen_decisions(
        causal,
        gdn_gain_ratio=float(gain["point"]),
        gdn_explained_fraction=float(mediation["point"]["gdn_explained_fraction"]),
    )
    state = {
        condition: aggregate_state_audits(progress.rows(f"infinitevl__screen32__recurrent__{condition}"))
        for condition in ("W", "G", "RG", "M")
    }
    result = {
        "format_version": 1,
        "diagnostic_split_digest": split_audit["digests"]["diagnostic8"],
        "screen_split_digest": split_audit["digests"]["screen"],
        "screen_metrics": metrics,
        "mediation_factorial": mediation,
        "full_mediator_repair": full_mediator_repair,
        "mediator_shapley": mediator,
        "mediator_selection": selection,
        "state_drift": state,
        "local_gain": gain,
        "screen_decisions": decisions,
    }
    atomic_json(progress.root / "state_and_shapley_summary.json", result)
    progress.mark_complete(("state_and_shapley_summary.json",))
    return result
