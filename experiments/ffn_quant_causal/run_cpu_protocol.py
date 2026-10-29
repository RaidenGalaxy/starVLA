from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
from pathlib import Path

import torch
import transformers
from huggingface_hub import snapshot_download

from experiments.infinitevl_quant.progress import atomic_json, json_digest
from experiments.infinitevl_quant.config import SCHEMES

from .adapters import load_model_bundle
from .budget import (
    BUDGET_FILENAME,
    WallclockBudget,
    WallclockBudgetExhausted,
    validate_completed_stage,
)
from .audit import (
    audit_model_pair,
    build_protocol_identity,
    ensure_protocol_identity,
    implementation_audit,
    protocol_input_digests,
    require_cpu_environment,
    runtime_ffn_inventory,
    scorer_resource_audit,
    stage_progress,
)
from .config import (
    INFINITEVL_MODEL_ID,
    INFINITEVL_REVISION,
    MODEL_SPECS,
    STAGES,
    CausalConfig,
)
from .protocol import build_protocol_splits, write_split_manifests
from .rtn import run_matched_rtn
from .tracing_stage import run_model_trace
from .counterfactual_stage import (
    combine_counterfactual_summaries,
    run_model_counterfactual,
)
from .report_stage import run_report
from .dynamics_stage import (
    run_infinitevl_ffn_state_drift,
    run_relative_noise_propagation,
)
from .weight_analysis import analyze_ffn_weight_pair, summarize_rtn_weight_audits
from .mbq_qig_stage import (
    MatchedSearchSettings,
    resolve_weight_bits,
    run_matched_mbq_qig,
)
from .confirm_stage import (
    build_scale_artifact_identity,
    run_model_confirm,
    save_scale_artifact,
)
from .statistics import stratified_difference_in_differences


_FIXED_STAGE_REQUIRED = {
    "audit": (
        "environment.json", "models.json", "implementation.json",
        "scorer_resources.json", "ffn_weight_pair.json", "splits.json",
        "split_manifests.json",
    ),
    "rtn": (
        "runtime_inventory.json", "summary.json",
        "static_weight_quantization.json",
    ),
    "trace": ("summary.json",),
    "counterfactual": (
        "summary.json", "model_summaries.json", "dynamics_summary.json",
    ),
    "confirm": (
        "summary.json", "model_summaries.json",
        "infinitevl_confirm_summary.json", "qwen_confirm_summary.json",
    ),
    "report": ("analyses.json", "report_summary.json", "report_zh.md"),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Matched CPU-only FFN quantization causal protocol",
    )
    parser.add_argument("--stage", choices=(*STAGES, "all"), required=True)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/ffn_quant_causal_v2"),
    )
    parser.add_argument("--infinitevl-model-path", type=Path)
    parser.add_argument(
        "--qwen-model-path", type=Path,
        default=Path("playground/Pretrained_models/Qwen2.5-VL-3B-Instruct"),
    )
    parser.add_argument(
        "--calibration-manifest", type=Path,
        default=Path(
            "data/vlm_quant_benchmark/multimodal_calibration/"
            "multimodal_calibration_4x32.jsonl"
        ),
    )
    parser.add_argument(
        "--evaluation-manifest", type=Path,
        default=Path(
            "data/vlm_quant_benchmark/top4_protocol_v3/vlm_screen_4x64.jsonl"
        ),
    )
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--wallclock-hours", type=float, default=120.0)
    parser.add_argument("--resume", action="store_true")
    return parser


def _configure_threads(threads: int) -> None:
    if threads <= 0:
        raise ValueError("--threads must be positive")
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)


def _resolve_infinitevl(path: Path | None) -> Path:
    if path is not None:
        resolved = path.resolve()
    else:
        resolved = Path(snapshot_download(
            repo_id=INFINITEVL_MODEL_ID,
            revision=INFINITEVL_REVISION,
            local_files_only=True,
        )).resolve()
    if resolved.name != INFINITEVL_REVISION:
        # A user-supplied content-addressed copy may have another directory
        # name, so identity is ultimately enforced by the full file hashes.
        print(
            f"warning: InfiniteVL directory name is {resolved.name!r}, "
            f"expected revision {INFINITEVL_REVISION!r}; content audit is authoritative",
            flush=True,
        )
    return resolved


def _config(arguments: argparse.Namespace) -> CausalConfig:
    return CausalConfig(
        output_dir=arguments.output_dir.resolve(),
        qwen_model_path=arguments.qwen_model_path.resolve(),
        calibration_manifest=arguments.calibration_manifest.resolve(),
        evaluation_manifest=arguments.evaluation_manifest.resolve(),
        torch_threads=arguments.threads,
        wallclock_hours=arguments.wallclock_hours,
    )


def _environment() -> dict:
    return {
        **require_cpu_environment(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "labels": [
            "cpu_reference_backend",
            "fake_quantized",
            "matched_cross_model_protocol",
            "no_speed_or_memory_claim",
        ],
    }


def _require_stage_complete(
    config: CausalConfig,
    stage: str,
    input_digests: dict[str, str],
):
    progress = stage_progress(config, stage, input_digests)
    expected = _FIXED_STAGE_REQUIRED.get(stage)
    contains = (
        ("format_selection.json", "summary.json")
        if stage == "mbq_qig" else ()
    )
    if not validate_completed_stage(
        progress,
        expected_required=expected,
        required_contains=contains,
    ):
        raise FileNotFoundError(f"required stage is incomplete: {stage}")
    return progress


def _resume_stage_complete(
    config: CausalConfig,
    stage: str,
    input_digests: dict[str, str],
) -> bool:
    progress = stage_progress(config, stage, input_digests)
    expected = _FIXED_STAGE_REQUIRED.get(stage)
    contains = (
        ("format_selection.json", "summary.json")
        if stage == "mbq_qig" else ()
    )
    return validate_completed_stage(
        progress,
        expected_required=expected,
        required_contains=contains,
    )


def _execute_with_budget(stage: str, budget: WallclockBudget, function):
    """Run one stage with coarse guards around any monolithic legacy work."""

    budget.check(stage=stage, boundary="before_stage")
    try:
        result = function()
        budget.check(stage=stage, boundary="after_stage")
        return result
    except WallclockBudgetExhausted:
        # Some reused stage implementations create their marker internally.
        # A post-stage boundary that observes expiration must leave the stage
        # resumable and must never advertise it as completed within budget.
        (budget.output_dir / stage / ".complete").unlink(missing_ok=True)
        raise


def run_audit(
    config: CausalConfig,
    sources: dict[str, Path],
    *,
    wallclock_budget: WallclockBudget | None = None,
) -> tuple[dict, dict, dict]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    splits, split_audit = build_protocol_splits(
        config.calibration_manifest, config.evaluation_manifest,
    )
    pair_audit = audit_model_pair(sources, MODEL_SPECS, hash_weights=True)
    input_digests = protocol_input_digests(
        pair_audit,
        split_audit,
        calibration_manifest=config.calibration_manifest,
        evaluation_manifest=config.evaluation_manifest,
    )
    identity = build_protocol_identity(config, input_digests)
    ensure_protocol_identity(config.output_dir, identity)
    progress = stage_progress(config, "audit", input_digests)
    manifests = write_split_manifests(progress.root / "manifests", splits)
    atomic_json(progress.root / "environment.json", _environment())
    atomic_json(progress.root / "models.json", pair_audit)
    atomic_json(progress.root / "implementation.json", implementation_audit())
    atomic_json(progress.root / "scorer_resources.json", scorer_resource_audit())
    analyze_ffn_weight_pair(
        sources["infinitevl"],
        sources["qwen"],
        output_path=progress.root / "ffn_weight_pair.json",
    )
    atomic_json(progress.root / "splits.json", split_audit)
    atomic_json(progress.root / "split_manifests.json", manifests)
    atomic_json(config.output_dir / "config.json", config.to_dict() | {
        "config_digest": config.digest(),
        "protocol_identity": identity,
        "model_sources": {key: str(value) for key, value in sources.items()},
    })
    if wallclock_budget is not None:
        wallclock_budget.check(stage="audit", boundary="before_complete_marker")
    progress.mark_complete(_FIXED_STAGE_REQUIRED["audit"])
    print(f"[audit] complete: {config.output_dir}", flush=True)
    return pair_audit, split_audit, input_digests


def _load_audit_state(
    config: CausalConfig,
    sources: dict[str, Path],
) -> tuple[dict, dict, dict]:
    pair_path = config.output_dir / "audit" / "models.json"
    split_path = config.output_dir / "audit" / "splits.json"
    if not pair_path.is_file() or not split_path.is_file():
        raise FileNotFoundError("audit stage has not completed")
    stored_pair_audit = json.loads(pair_path.read_text())
    split_audit = json.loads(split_path.read_text())
    # The checkpoint directories are mutable local inputs.  Re-hash every file
    # at each invocation so a resumed stage can never silently consume weights
    # different from the audited checkpoint.
    pair_audit = audit_model_pair(sources, MODEL_SPECS, hash_weights=True)
    if pair_audit["pair_digest"] != stored_pair_audit.get("pair_digest"):
        raise RuntimeError("current model-pair content differs from the audit stage")
    for kind in ("infinitevl", "qwen"):
        current = pair_audit["models"][kind]
        stored = stored_pair_audit.get("models", {}).get(kind, {})
        for field in (
            "content_sha256", "weight_bytes", "weight_shards", "files",
            "static_ffn_inventory", "model",
        ):
            if current.get(field) != stored.get(field):
                raise RuntimeError(
                    f"current {kind} model audit field {field!r} differs "
                    "from the completed audit stage"
                )
    inputs = protocol_input_digests(
        pair_audit,
        split_audit,
        calibration_manifest=config.calibration_manifest,
        evaluation_manifest=config.evaluation_manifest,
    )
    ensure_protocol_identity(
        config.output_dir, build_protocol_identity(config, inputs),
    )
    _require_stage_complete(config, "audit", inputs)
    return pair_audit, split_audit, inputs


def run_rtn(
    config: CausalConfig,
    sources: dict[str, Path],
    input_digests: dict[str, str],
    *,
    generate: bool,
    wallclock_budget: WallclockBudget | None = None,
) -> dict:
    splits, split_audit = build_protocol_splits(
        config.calibration_manifest, config.evaluation_manifest,
    )
    if input_digests["data_splits"] != json_digest(split_audit["digests"]):
        raise RuntimeError("split digest changed after audit")
    progress = stage_progress(config, "rtn", input_digests)
    summaries: dict[str, dict] = {}
    runtime_audits = {}
    for kind in ("infinitevl", "qwen"):
        if wallclock_budget is not None:
            wallclock_budget.check(
                stage="rtn", boundary="before_model", variant=kind,
            )
        print(f"[rtn] loading {kind} from {sources[kind]}", flush=True)
        bundle = load_model_bundle(kind, sources[kind])
        try:
            runtime_targets, runtime_inventory = runtime_ffn_inventory(
                bundle.model, MODEL_SPECS[kind],
            )
            if [target.name for target in runtime_targets] != [
                target.name for target in bundle.ffn_targets
            ]:
                raise RuntimeError(f"{kind} independent FFN inventories disagree")
            runtime_audits[kind] = {
                "loader": bundle.load_audit,
                "ffn": runtime_inventory,
            }
            atomic_json(progress.root / "runtime_inventory.json", runtime_audits)
            summaries.update(run_matched_rtn(
                bundle,
                splits,
                progress=progress,
                output_dir=config.output_dir,
                min_pixels=config.min_pixels,
                max_pixels=config.max_pixels,
                max_new_tokens=config.max_new_tokens,
                generate=generate,
                wallclock_budget=wallclock_budget,
            ))
            atomic_json(progress.root / "summary.json", summaries)
        finally:
            bundle.close()
            del bundle
            gc.collect()
        if wallclock_budget is not None:
            wallclock_budget.check(
                stage="rtn", boundary="after_model", variant=kind,
            )
    atomic_json(progress.root / "runtime_inventory.json", runtime_audits)
    atomic_json(progress.root / "summary.json", summaries)
    atomic_json(
        progress.root / "static_weight_quantization.json",
        summarize_rtn_weight_audits(progress.root),
    )
    if wallclock_budget is not None:
        wallclock_budget.check(stage="rtn", boundary="before_complete_marker")
    progress.mark_complete(_FIXED_STAGE_REQUIRED["rtn"])
    print("[rtn] complete", flush=True)
    return summaries


def run_trace(
    config: CausalConfig,
    sources: dict[str, Path],
    input_digests: dict[str, str],
    *,
    wallclock_budget: WallclockBudget | None = None,
) -> dict:
    splits, _audit = build_protocol_splits(
        config.calibration_manifest, config.evaluation_manifest,
    )
    progress = stage_progress(config, "trace", input_digests)
    summaries = {}
    for kind in ("infinitevl", "qwen"):
        if wallclock_budget is not None:
            wallclock_budget.check(
                stage="trace", boundary="before_model", variant=kind,
            )
        print(f"[trace] loading {kind}", flush=True)
        bundle = load_model_bundle(kind, sources[kind])
        try:
            summaries[kind] = run_model_trace(
                bundle,
                splits["smoke4"],
                progress=progress,
                min_pixels=config.min_pixels,
                max_pixels=config.max_pixels,
            )
            atomic_json(progress.root / "summary.json", summaries)
        finally:
            bundle.close()
            del bundle
            gc.collect()
        if wallclock_budget is not None:
            wallclock_budget.check(
                stage="trace", boundary="after_model", variant=kind,
            )
    atomic_json(progress.root / "summary.json", summaries)
    if wallclock_budget is not None:
        wallclock_budget.check(stage="trace", boundary="before_complete_marker")
    progress.mark_complete(_FIXED_STAGE_REQUIRED["trace"])
    print("[trace] complete", flush=True)
    return summaries


def run_counterfactual(
    config: CausalConfig,
    sources: dict[str, Path],
    input_digests: dict[str, str],
    *,
    wallclock_budget: WallclockBudget | None = None,
) -> dict:
    _require_stage_complete(config, "trace", input_digests)
    splits, _audit = build_protocol_splits(
        config.calibration_manifest, config.evaluation_manifest,
    )
    progress = stage_progress(config, "counterfactual", input_digests)
    model_summaries = {}
    dynamics_summaries = {}
    for kind in ("infinitevl", "qwen"):
        if wallclock_budget is not None:
            wallclock_budget.check(
                stage="counterfactual", boundary="before_model", variant=kind,
            )
        print(f"[counterfactual] loading {kind} weights", flush=True)
        bundle = load_model_bundle(kind, sources[kind])
        try:
            model_summaries[kind] = run_model_counterfactual(
                bundle,
                splits["smoke4"],
                progress=progress,
                trace_root=config.output_dir / "trace",
                group_size=config.group_size,
            )
            dynamics_summaries[f"{kind}_relative_noise"] = (
                run_relative_noise_propagation(
                    bundle,
                    splits["smoke4"],
                    progress=progress,
                    trace_root=config.output_dir / "trace",
                    min_pixels=config.min_pixels,
                    max_pixels=config.max_pixels,
                    seed=config.seed,
                )
            )
            if kind == "infinitevl":
                dynamics_summaries["infinitevl_gdn_state_drift"] = (
                    run_infinitevl_ffn_state_drift(
                        bundle,
                        splits["screen"],
                        progress=progress,
                        min_pixels=config.min_pixels,
                        max_pixels=config.max_pixels,
                    )
                )
            atomic_json(progress.root / "dynamics_summary.json", dynamics_summaries)
        finally:
            bundle.close()
            del bundle
            gc.collect()
        if wallclock_budget is not None:
            wallclock_budget.check(
                stage="counterfactual", boundary="after_model", variant=kind,
            )
    combined = combine_counterfactual_summaries(progress)
    atomic_json(progress.root / "model_summaries.json", model_summaries)
    atomic_json(progress.root / "dynamics_summary.json", dynamics_summaries)
    if wallclock_budget is not None:
        wallclock_budget.check(
            stage="counterfactual", boundary="before_complete_marker",
        )
    progress.mark_complete(_FIXED_STAGE_REQUIRED["counterfactual"])
    print("[counterfactual] complete", flush=True)
    return combined


def run_mbq_qig_search(
    config: CausalConfig,
    sources: dict[str, Path],
    input_digests: dict[str, str],
    pair_audit: dict,
    split_audit: dict,
    *,
    wallclock_budget: WallclockBudget | None = None,
) -> dict:
    _require_stage_complete(config, "rtn", input_digests)
    splits, runtime_split_audit = build_protocol_splits(
        config.calibration_manifest, config.evaluation_manifest,
    )
    if runtime_split_audit["digests"] != split_audit["digests"]:
        raise RuntimeError("data split digests changed before MBQ/QIG")
    progress = stage_progress(config, "mbq_qig", input_digests)
    bits, selection = resolve_weight_bits(
        requested_bits=config.search_weight_bits,
    )
    atomic_json(progress.root / "format_selection.json", selection | {
        "scope": "decoder_ffn_gate_up_down_only",
        "activation_bits": 16,
        "reason": "preregistered matched W4 recovery test after RTN causal screen",
    })
    settings = MatchedSearchSettings(
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        seed=config.seed,
        weight_bits=bits,
        group_size=config.group_size,
        scale_grid=config.mbq_scale_grid,
        step_grid=config.qig_steps,
        coordinate_sweeps=config.qig_coordinate_sweeps,
        spearman_threshold=config.qig_spearman,
        jaccard_threshold=config.qig_jaccard,
        vvs_relative_threshold=config.vvs_threshold,
        vvs_max_layers=config.vvs_max_layers,
        vvs_bootstrap_samples=config.vvs_bootstrap,
        budget_hours=config.qig_configuration_hours,
    )
    protocol_digest = json_digest(build_protocol_identity(config, input_digests))
    summaries = {}
    scheme = settings.scheme
    artifact_required = []
    for kind in ("infinitevl", "qwen"):
        if wallclock_budget is not None:
            wallclock_budget.check(
                stage="mbq_qig", boundary="before_model", variant=kind,
            )
        print(f"[mbq_qig] loading {kind}", flush=True)
        bundle = load_model_bundle(kind, sources[kind])
        try:
            result = run_matched_mbq_qig(
                kind,
                bundle.model,
                bundle.processor,
                bundle.ffn_targets,
                splits,
                output_dir=progress.root / "search",
                protocol_identity=protocol_digest,
                settings=settings,
                cooperative_check=(
                    None
                    if wallclock_budget is None
                    else lambda boundary, model_kind=kind: wallclock_budget.check(
                        stage="mbq_qig",
                        boundary=boundary,
                        variant=model_kind,
                    )
                ),
            )
            target_names = tuple(target.name for target in bundle.ffn_targets)
            public_root = progress.root / "scale_artifacts" / kind
            paths = {
                "mbq": public_root / f"mbq_w{bits}a16_ffn.pt",
                "qig": public_root / f"qig_w{bits}a16_ffn.pt",
            }
            scales = {
                "mbq": result.mbq_scales,
                "qig": result.accepted_scales,
            }
            artifacts = {}
            search_identity = result.audit.get("identity")
            if not isinstance(search_identity, str) or not search_identity:
                raise RuntimeError(f"{kind} MBQ/QIG audit has no search identity")
            search_audit_digest = json_digest(result.audit)
            for method in ("mbq", "qig"):
                identity = build_scale_artifact_identity(
                    model_kind=kind,
                    model_digest=pair_audit["models"][kind]["content_sha256"],
                    method=method,
                    target_names=target_names,
                    scheme=scheme,
                    config_digest=config.digest(),
                    split_digests=split_audit["digests"],
                    protocol_digest=protocol_digest,
                    implementation_digest=input_digests["implementation"],
                    search_identity=search_identity,
                    search_audit_digest=search_audit_digest,
                )
                artifact = save_scale_artifact(
                    paths[method], scales[method], identity=identity,
                )
                artifacts[method] = {
                    "path": str(artifact.path),
                    "sha256": artifact.artifact_sha256,
                    "identity": identity.to_dict(),
                }
                relative = artifact.path.relative_to(progress.root)
                artifact_required.extend((
                    str(relative), str(relative) + ".identity.json",
                ))
            summaries[kind] = {
                "model_kind": kind,
                "weight_bits": bits,
                "cache_hit": result.cache_hit,
                "scale_artifacts": artifacts,
                "search_audit": result.audit,
            }
            atomic_json(progress.root / "summary.json", summaries)
        finally:
            bundle.close()
            del bundle
            gc.collect()
        if wallclock_budget is not None:
            # The legacy MBQ/VVS/QIG routines are monolithic; this is an honest
            # cooperative model boundary, not a hard interrupt inside Python.
            wallclock_budget.check(
                stage="mbq_qig", boundary="after_model", variant=kind,
            )
    atomic_json(progress.root / "summary.json", summaries)
    if wallclock_budget is not None:
        wallclock_budget.check(
            stage="mbq_qig", boundary="before_complete_marker",
        )
    progress.mark_complete(tuple([
        "format_selection.json", "summary.json", *artifact_required,
    ]))
    print("[mbq_qig] complete", flush=True)
    return summaries


def run_confirm(
    config: CausalConfig,
    sources: dict[str, Path],
    input_digests: dict[str, str],
    pair_audit: dict,
    split_audit: dict,
    *,
    wallclock_budget: WallclockBudget | None = None,
) -> dict:
    _require_stage_complete(config, "mbq_qig", input_digests)
    splits, runtime_split_audit = build_protocol_splits(
        config.calibration_manifest, config.evaluation_manifest,
    )
    if runtime_split_audit["digests"] != split_audit["digests"]:
        raise RuntimeError("data split digests changed before confirm")
    progress = stage_progress(config, "confirm", input_digests)
    bits = config.search_weight_bits
    scheme = SCHEMES[f"w{bits}a16"]
    protocol_digest = json_digest(build_protocol_identity(config, input_digests))
    search_summary_path = config.output_dir / "mbq_qig" / "summary.json"
    try:
        search_summary = json.loads(search_summary_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid MBQ/QIG summary: {search_summary_path}") from error
    model_summaries = {}
    for kind in ("infinitevl", "qwen"):
        if wallclock_budget is not None:
            wallclock_budget.check(
                stage="confirm", boundary="before_model", variant=kind,
            )
        print(f"[confirm] loading {kind}", flush=True)
        bundle = load_model_bundle(kind, sources[kind])
        try:
            search_record = search_summary.get(kind)
            if not isinstance(search_record, dict):
                raise RuntimeError(f"MBQ/QIG summary is missing model {kind}")
            search_audit = search_record.get("search_audit")
            if not isinstance(search_audit, dict):
                raise RuntimeError(f"MBQ/QIG summary has no search audit for {kind}")
            search_identity = search_audit.get("identity")
            if not isinstance(search_identity, str) or not search_identity:
                raise RuntimeError(f"MBQ/QIG search identity missing for {kind}")
            artifact_provenance = {
                "protocol_digest": protocol_digest,
                "implementation_digest": input_digests["implementation"],
                "search_identity": search_identity,
                "search_audit_digest": json_digest(search_audit),
            }
            artifact_root = config.output_dir / "mbq_qig" / "scale_artifacts" / kind
            model_summaries[kind] = run_model_confirm(
                bundle,
                splits["confirm"],
                progress=progress,
                scheme=scheme,
                mbq_artifact_path=artifact_root / f"mbq_w{bits}a16_ffn.pt",
                qig_artifact_path=artifact_root / f"qig_w{bits}a16_ffn.pt",
                model_digest=pair_audit["models"][kind]["content_sha256"],
                config_digest=config.digest(),
                split_digests=split_audit["digests"],
                artifact_provenance=artifact_provenance,
                min_pixels=config.min_pixels,
                max_pixels=config.max_pixels,
                max_new_tokens=config.max_new_tokens,
                bootstrap_samples=config.bootstrap_samples,
                seed=config.seed,
                generate=True,
                wallclock_budget=wallclock_budget,
            )
            atomic_json(progress.root / "model_summaries.json", model_summaries)
        finally:
            bundle.close()
            del bundle
            gc.collect()
        if wallclock_budget is not None:
            wallclock_budget.check(
                stage="confirm", boundary="after_model", variant=kind,
            )
    cross_model = {}
    for method in ("rtn", "mbq", "qig"):
        cross_model[method] = stratified_difference_in_differences(
            progress.rows("infinitevl__confirm192__bf16"),
            progress.rows(f"infinitevl__confirm192__{method}_w{bits}a16_ffn"),
            progress.rows("qwen__confirm192__bf16"),
            progress.rows(f"qwen__confirm192__{method}_w{bits}a16_ffn"),
            samples=config.bootstrap_samples,
            seed=config.seed,
        )
    summary = {
        "scheme": scheme.to_dict(),
        "models": model_summaries,
        "cross_model_difference_in_differences": cross_model,
        "within_protocol_holdout_samples": 192,
        "evaluation_history": "historically_reused_evaluation",
        "historically_unseen": False,
        "full256_is_descriptive_only": True,
    }
    atomic_json(progress.root / "summary.json", summary)
    if wallclock_budget is not None:
        wallclock_budget.check(stage="confirm", boundary="before_complete_marker")
    progress.mark_complete(_FIXED_STAGE_REQUIRED["confirm"])
    print("[confirm] complete", flush=True)
    return summary


def run_reporting(
    config: CausalConfig,
    input_digests: dict[str, str],
    *,
    wallclock_budget: WallclockBudget | None = None,
) -> dict:
    _require_stage_complete(config, "rtn", input_digests)
    if wallclock_budget is not None:
        wallclock_budget.check(stage="report", boundary="before_render")
    report_progress = stage_progress(config, "report", input_digests)
    rtn_progress = stage_progress(config, "rtn", input_digests)
    summary = run_report(config, report_progress, rtn_progress)
    print(f"[report] complete: {report_progress.root / 'report_zh.md'}", flush=True)
    return summary


def main() -> None:
    arguments = _parser().parse_args()
    _configure_threads(arguments.threads)
    config = _config(arguments)
    sources = {
        "infinitevl": _resolve_infinitevl(arguments.infinitevl_model_path),
        "qwen": config.qwen_model_path,
    }
    for key, source in sources.items():
        if not source.is_dir():
            raise FileNotFoundError(f"{key} model source not found: {source}")

    audit_marker = config.output_dir / "audit" / ".complete"
    if arguments.stage in {"audit", "all"}:
        if audit_marker.is_file() and not (
            config.output_dir / BUDGET_FILENAME
        ).is_file():
            raise RuntimeError(
                "completed audit predates the persistent wallclock contract; "
                "use a new --output-dir rather than inventing a later start time"
            )
        budget = WallclockBudget.open_or_create(
            config.output_dir,
            wallclock_hours=config.wallclock_hours,
            config_digest=config.digest(),
            create=True,
        )
        if arguments.resume and (config.output_dir / "audit" / ".complete").is_file():
            _pair, _splits, inputs = _load_audit_state(config, sources)
            print("[audit] verified existing complete stage", flush=True)
        else:
            _pair, _splits, inputs = _execute_with_budget(
                "audit",
                budget,
                lambda: run_audit(
                    config, sources, wallclock_budget=budget,
                ),
            )
    else:
        budget = WallclockBudget.open_or_create(
            config.output_dir,
            wallclock_hours=config.wallclock_hours,
            config_digest=config.digest(),
            create=False,
        )
        _pair, _splits, inputs = _load_audit_state(config, sources)

    if arguments.stage in {"rtn", "all"}:
        if arguments.resume and _resume_stage_complete(config, "rtn", inputs):
            print("[rtn] verified complete stage; skipping", flush=True)
        else:
            _execute_with_budget(
                "rtn",
                budget,
                lambda: run_rtn(
                    config,
                    sources,
                    inputs,
                    generate=True,
                    wallclock_budget=budget,
                ),
            )
    if arguments.stage in {"trace", "all"}:
        if arguments.resume and _resume_stage_complete(config, "trace", inputs):
            print("[trace] verified complete stage; skipping", flush=True)
        else:
            _execute_with_budget(
                "trace",
                budget,
                lambda: run_trace(
                    config, sources, inputs, wallclock_budget=budget,
                ),
            )
    if arguments.stage in {"counterfactual", "all"}:
        if arguments.resume and _resume_stage_complete(
            config, "counterfactual", inputs,
        ):
            print("[counterfactual] verified complete stage; skipping", flush=True)
        else:
            _execute_with_budget(
                "counterfactual",
                budget,
                lambda: run_counterfactual(
                    config, sources, inputs, wallclock_budget=budget,
                ),
            )
    if arguments.stage in {"mbq_qig", "all"}:
        if arguments.resume and _resume_stage_complete(config, "mbq_qig", inputs):
            print("[mbq_qig] verified complete stage; skipping", flush=True)
        else:
            _execute_with_budget(
                "mbq_qig",
                budget,
                lambda: run_mbq_qig_search(
                    config,
                    sources,
                    inputs,
                    _pair,
                    _splits,
                    wallclock_budget=budget,
                ),
            )
    if arguments.stage in {"confirm", "all"}:
        if arguments.resume and _resume_stage_complete(config, "confirm", inputs):
            print("[confirm] verified complete stage; skipping", flush=True)
        else:
            _execute_with_budget(
                "confirm",
                budget,
                lambda: run_confirm(
                    config,
                    sources,
                    inputs,
                    _pair,
                    _splits,
                    wallclock_budget=budget,
                ),
            )
    if arguments.stage in {"report", "all"}:
        if arguments.resume and _resume_stage_complete(config, "report", inputs):
            print("[report] verified complete stage; skipping", flush=True)
        else:
            _execute_with_budget(
                "report",
                budget,
                lambda: run_reporting(
                    config, inputs, wallclock_budget=budget,
                ),
            )
    if arguments.stage not in {
        "audit", "rtn", "trace", "counterfactual", "mbq_qig",
        "confirm", "report", "all",
    }:
        raise NotImplementedError(
            f"stage {arguments.stage!r} is being wired after the audited RTN foundation"
        )


if __name__ == "__main__":
    main()
