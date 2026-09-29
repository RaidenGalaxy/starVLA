from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

from experiments.infinitevl_quant.evaluation import evaluate_variant
from experiments.infinitevl_quant.inventory import build_inventory, select_scope
from experiments.infinitevl_quant.loader import ensure_snapshot, load_infinitevl_cpu
from experiments.infinitevl_quant.progress import (
    StageProgress,
    atomic_json,
    file_sha256,
    json_digest,
)
from experiments.vlm_quant_benchmark.data import sample_manifest_digest
from experiments.infinitevl_quant.data_protocol import write_split_manifests

from .attribution import MODULES, all_scope_masks, mask_key, pair_interactions, shapley_three
from .audit import audit_existing_results, audit_loaded_model, audit_quantizer
from .config import DiagnosisConfig, DiagnosticScheme, SCHEMES
from .data_protocol import build_diagnosis_splits
from .quantization import diagnostic_qdq_linears, select_targets


STAGES = ("audit", "factorial", "module", "layer", "projection")
PREREQUISITES = {
    "factorial": ("audit", "audit_summary.json"),
    "module": ("factorial", "factorial_summary.json"),
    "layer": ("module", "module_summary.json"),
    "projection": ("layer", "layer_summary.json"),
}
SCOPE_LABELS = {
    "decoder_ffn": "ffn",
    "swa_attention": "swa",
    "gdn_main": "gdn",
}
WEIGHT_ONLY_FORMATS = ("w4_asym_a16", "w3_asym_a16", "w2_asym_a16")
FACTORIAL_FORMATS = (*WEIGHT_ONLY_FORMATS, "w4_sym_channel_a8")
PROJECTIONS = {
    "decoder_ffn": ("gate_proj", "up_proj", "down_proj"),
    "swa_attention": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "gdn_main": ("q_proj", "k_proj", "v_proj", "g_proj", "o_proj"),
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument(
        "--source-output", type=Path, default=Path("outputs/infinitevl_quant_cpu_v2"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/infinitevl_quant_cpu_rootcause_v1"),
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    return parser.parse_args()


def _configure_threads(output_dir: Path, source_output: Path) -> dict:
    path = output_dir / "thread_config.json"
    if path.is_file():
        audit = json.loads(path.read_text())
    else:
        source = source_output / "thread_config.json"
        if not source.is_file():
            raise FileNotFoundError("missing validated v2 thread configuration")
        source_audit = json.loads(source.read_text())
        audit = {
            "selected": int(source_audit["selected"]),
            "interop_threads": 1,
            "source": str(source),
            "source_sha256": file_sha256(source),
        }
        atomic_json(path, audit)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    torch.set_num_threads(int(audit["selected"]))
    return audit


def _protocol(args, threads: dict) -> tuple[DiagnosisConfig, dict, dict, dict]:
    config = DiagnosisConfig(
        output_dir=args.output_dir,
        source_output=args.source_output,
        max_new_tokens=args.max_new_tokens,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    splits, split_audit = build_diagnosis_splits(
        config.calibration_manifest, config.evaluation_manifest,
    )
    package_root = Path(__file__).parent
    implementation = {
        str(path.relative_to(package_root)): file_sha256(path)
        for path in sorted(package_root.glob("*.py"))
    }
    base_files = {}
    for relative in (
        "config.py", "cpu_backend.py", "data_protocol.py", "evaluation.py",
        "inventory.py", "loader.py", "progress.py", "quantization.py",
    ):
        path = package_root.parent / "infinitevl_quant" / relative
        base_files[relative] = file_sha256(path)
    inputs = {
        "model_revision": config.model_revision,
        "calibration_manifest": file_sha256(config.calibration_manifest),
        "evaluation_manifest": file_sha256(config.evaluation_manifest),
        "source_protocol_identity": file_sha256(
            config.source_output / "protocol_identity.json"
        ),
        "splits": {
            name: sample_manifest_digest(samples) for name, samples in splits.items()
        },
        "implementation": json_digest({"diagnosis": implementation, "base": base_files}),
        "threads": json_digest(threads),
    }
    identity = {
        "format_version": 1,
        "config_digest": config.digest(),
        "input_digests": inputs,
    }
    identity_path = config.output_dir / "protocol_identity.json"
    if identity_path.is_file():
        if json.loads(identity_path.read_text()) != identity:
            raise RuntimeError("diagnosis output belongs to different code/config/inputs")
    else:
        atomic_json(identity_path, identity)
    atomic_json(
        config.output_dir / "protocol_config.json",
        config.to_dict() | {"config_digest": config.digest()},
    )
    manifests = write_split_manifests(config.output_dir / "protocol", splits)
    atomic_json(config.output_dir / "protocol/splits.json", manifests)
    atomic_json(config.output_dir / "protocol/split_audit.json", split_audit)
    atomic_json(
        config.output_dir / "protocol/implementation_sha256.json",
        {"diagnosis": implementation, "base": base_files, "combined": inputs["implementation"]},
    )
    budget_path = config.output_dir / "wallclock.json"
    if not budget_path.is_file():
        atomic_json(
            budget_path,
            {"started_unix": time.time(), "limit_hours": config.wallclock_hours},
        )
    budget = json.loads(budget_path.read_text())
    elapsed = (time.time() - float(budget["started_unix"])) / 3600.0
    if elapsed > config.wallclock_hours:
        raise TimeoutError(f"diagnosis wall-clock budget exceeded: {elapsed:.2f}h")
    return config, splits, inputs, split_audit


def _require_prerequisite(config: DiagnosisConfig, stage: str) -> None:
    requirement = PREREQUISITES.get(stage)
    if requirement is None:
        return
    upstream, result = requirement
    root = config.output_dir / upstream
    if not (root / ".complete").is_file() or not (root / result).is_file():
        raise RuntimeError(f"stage {stage} requires completed stage {upstream}")


def _model_path(args, config: DiagnosisConfig) -> Path:
    if args.model_path is not None:
        return args.model_path.resolve()
    return ensure_snapshot(
        config.model_id,
        config.model_revision,
        local_files_only=args.local_files_only,
    )


def _load_model(args, config):
    model, processor, load_audit = load_infinitevl_cpu(_model_path(args, config))
    targets, inventory = build_inventory(model)
    return model, processor, targets, inventory, load_audit


def _progress(config, stage, inputs):
    return StageProgress(
        config.output_dir, stage, config_digest=config.digest(), inputs=inputs,
    )


def _compact_quant_audit(audit: dict) -> dict:
    rows = list(audit["modules"].values())
    return {
        "scheme": audit["scheme"],
        "module_count": audit["module_count"],
        "mean_weight_nmse": (
            sum(float(row["nmse"]) for row in rows) / len(rows) if rows else 0.0
        ),
        "max_weight_nmse": max((float(row["nmse"]) for row in rows), default=0.0),
        "scope_counts": {
            scope: sum(row["scope"] == scope for row in rows) for scope in MODULES
        },
    }


def _evaluate(
    model,
    processor,
    samples,
    *,
    progress,
    variant,
    config,
    reference_dir,
    write_reference=False,
    generate=False,
    capture_hidden=False,
):
    result = evaluate_variant(
        model,
        processor,
        samples,
        progress=progress,
        variant=variant,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        max_new_tokens=config.max_new_tokens,
        generate=generate,
        reference_dir=reference_dir,
        write_reference=write_reference,
        capture_hidden=capture_hidden,
    )
    if not math.isfinite(float(result["target_nll"])):
        raise FloatingPointError(f"{variant} produced non-finite target NLL")
    return result


def stage_audit(args, config, splits, inputs, split_audit):
    progress = _progress(config, "audit", inputs)
    required = ("audit_summary.json",)
    if args.resume and progress.is_complete(required):
        return json.loads((progress.root / required[0]).read_text())
    existing = audit_existing_results(config.source_output)
    quantizer = audit_quantizer()
    model, _processor, _targets, _inventory, load_audit = _load_model(args, config)
    loaded_model = audit_loaded_model(model)
    result = {
        "passed": True,
        "existing_v2": existing,
        "quantizer": quantizer,
        "loaded_model": loaded_model,
        "split_audit": split_audit,
        "load": load_audit,
    }
    atomic_json(progress.root / required[0], result)
    progress.mark_complete(required)
    return result


def stage_factorial(args, config, splits, inputs):
    progress = _progress(config, "factorial", inputs)
    required = ("factorial_summary.json",)
    if args.resume and progress.is_complete(required):
        return json.loads((progress.root / required[0]).read_text())
    model, processor, targets, inventory, _load = _load_model(args, config)
    safe = select_scope(targets, "decoder_safe")
    metrics, audits = {}, {}
    scheme_order = (
        "a8_only", "w4_asym_a16", "w4_asym_a8",
        "w4_sym_channel_a16", "w4_sym_channel_a8",
    )
    for split_name in ("diagnosis_acceptance", "diagnosis_search"):
        samples = splits[split_name]
        short = split_name.removeprefix("diagnosis_")
        reference_dir = progress.root / f"reference_{short}"
        metrics[short] = {
            "bf16": _evaluate(
                model, processor, samples, progress=progress,
                variant=f"{short}__bf16", config=config,
                reference_dir=reference_dir, write_reference=True,
                generate=split_name == "diagnosis_search",
            )
        }
        for scheme_name in scheme_order:
            with diagnostic_qdq_linears(model, safe, SCHEMES[scheme_name]) as audit:
                metrics[short][scheme_name] = _evaluate(
                    model, processor, samples, progress=progress,
                    variant=f"{short}__{scheme_name}", config=config,
                    reference_dir=reference_dir,
                    generate=split_name == "diagnosis_search",
                )
            audits.setdefault(scheme_name, _compact_quant_audit(audit))
    search = metrics["search"]
    effects = {
        "a8_on_reference": search["a8_only"]["target_nll"] - search["bf16"]["target_nll"],
        "a8_on_asymmetric_w4": (
            search["w4_asym_a8"]["target_nll"] - search["w4_asym_a16"]["target_nll"]
        ),
        "a8_on_symmetric_w4": (
            search["w4_sym_channel_a8"]["target_nll"]
            - search["w4_sym_channel_a16"]["target_nll"]
        ),
        "symmetric_geometry_at_a16": (
            search["w4_sym_channel_a16"]["target_nll"]
            - search["w4_asym_a16"]["target_nll"]
        ),
        "symmetric_geometry_at_a8": (
            search["w4_sym_channel_a8"]["target_nll"]
            - search["w4_asym_a8"]["target_nll"]
        ),
    }
    result = {
        "metrics": metrics,
        "effects_on_search_target_nll": effects,
        "quantization": audits,
        "inventory_counts": inventory["counts"],
    }
    atomic_json(progress.root / required[0], result)
    progress.mark_complete(required)
    return result


def _mask_targets(targets, mask):
    return select_targets(targets, scopes=set(mask))


def stage_module(args, config, splits, inputs):
    progress = _progress(config, "module", inputs)
    required = ("module_summary.json",)
    if args.resume and progress.is_complete(required):
        return json.loads((progress.root / required[0]).read_text())
    model, processor, targets, inventory, _load = _load_model(args, config)
    metrics = {"acceptance": {}, "search": {}}
    audits = {}
    for split_name in ("diagnosis_acceptance", "diagnosis_search"):
        short = split_name.removeprefix("diagnosis_")
        samples = splits[split_name]
        reference_dir = progress.root / f"reference_{short}"
        metrics[short]["bf16"] = _evaluate(
            model, processor, samples, progress=progress,
            variant=f"{short}__bf16", config=config,
            reference_dir=reference_dir, write_reference=True,
            generate=short == "search",
        )
        for format_name in FACTORIAL_FORMATS:
            metrics[short][format_name] = {"none": metrics[short]["bf16"]}
            for mask in all_scope_masks()[1:]:
                key = mask_key(mask)
                members = _mask_targets(targets, mask)
                with diagnostic_qdq_linears(model, members, SCHEMES[format_name]) as audit:
                    value = _evaluate(
                        model, processor, samples, progress=progress,
                        variant=f"{short}__{format_name}__{key}", config=config,
                        reference_dir=reference_dir,
                        generate=(short == "search" and len(mask) in (1, 3)),
                    )
                metrics[short][format_name][key] = value
                audits.setdefault(
                    f"{format_name}__{key}", _compact_quant_audit(audit),
                )
    attribution = {}
    selected_modules = {"decoder_ffn"}
    for format_name in FACTORIAL_FORMATS:
        values = {
            key: float(row["target_nll"])
            for key, row in metrics["search"][format_name].items()
        }
        shapley = shapley_three(values)
        interactions = pair_interactions(values)
        single_scope_increase = {
            module: (
                values[mask_key({module})] / values["none"] - 1.0
            )
            for module in MODULES
        }
        attribution[format_name] = {
            "target_nll_shapley": shapley,
            "target_nll_pair_interactions": interactions,
            "single_scope_relative_increase": single_scope_increase,
        }
        for module in ("swa_attention", "gdn_main"):
            fraction = shapley["fraction"][module]
            if (
                (math.isfinite(fraction) and fraction >= 0.15)
                or single_scope_increase[module] >= 0.05
            ):
                selected_modules.add(module)
    result = {
        "metrics": metrics,
        "attribution": attribution,
        "selected_for_layer_scan": [
            module for module in MODULES if module in selected_modules
        ],
        "quantization": audits,
        "inventory_counts": inventory["counts"],
    }
    atomic_json(progress.root / required[0], result)
    progress.mark_complete(required)
    return result


def _mean_hidden_nmse(rows: list[dict], key: str) -> float:
    values = [float(row.get("block_nmse", {}).get(key, math.nan)) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    return sum(values) / len(values) if values else math.nan


def _variant_sensitivity(progress, variant, metrics, baseline, layer):
    rows = progress.rows(variant)
    hidden_keys = sorted(
        {int(key) for row in rows for key in row.get("block_nmse", {})},
    )
    return {
        "target_nll": metrics["target_nll"],
        "nll_relative_increase": metrics["target_nll"] / baseline["target_nll"] - 1.0,
        "mean_logits_kl": metrics["mean_logits_kl"],
        "immediate_block_nmse": _mean_hidden_nmse(rows, str(layer)),
        "final_block_nmse": (
            _mean_hidden_nmse(rows, str(hidden_keys[-1])) if hidden_keys else math.nan
        ),
    }


def _selected_module_formats(module_summary: dict) -> dict[str, tuple[str, ...]]:
    selected = module_summary["selected_for_layer_scan"]
    return {module: WEIGHT_ONLY_FORMATS for module in selected}


def stage_layer(args, config, splits, inputs):
    progress = _progress(config, "layer", inputs)
    required = ("layer_summary.json",)
    if args.resume and progress.is_complete(required):
        return json.loads((progress.root / required[0]).read_text())
    module_summary = json.loads((config.output_dir / "module/module_summary.json").read_text())
    model, processor, targets, inventory, _load = _load_model(args, config)
    scan_samples = splits["diagnosis_scan"]
    scan_reference = progress.root / "reference_scan"
    scan_baseline = _evaluate(
        model, processor, scan_samples, progress=progress,
        variant="scan__bf16", config=config, reference_dir=scan_reference,
        write_reference=True, capture_hidden=True,
    )
    scans, rankings, quant_audits = {}, {}, {}
    for module, formats in _selected_module_formats(module_summary).items():
        layers = sorted({target.layer for target in targets if target.scope == module})
        scans[module], rankings[module] = {}, {}
        for format_name in formats:
            scans[module][format_name] = {}
            for layer in layers:
                members = select_targets(targets, scopes={module}, layers={layer})
                variant = f"scan__{format_name}__{SCOPE_LABELS[module]}__layer{layer:02d}"
                with diagnostic_qdq_linears(model, members, SCHEMES[format_name]) as audit:
                    metrics = _evaluate(
                        model, processor, scan_samples, progress=progress,
                        variant=variant, config=config, reference_dir=scan_reference,
                        capture_hidden=True,
                    )
                scans[module][format_name][str(layer)] = _variant_sensitivity(
                    progress, variant, metrics, scan_baseline, layer,
                )
                quant_audits.setdefault(variant, _compact_quant_audit(audit))
            rankings[module][format_name] = sorted(
                layers,
                key=lambda layer: (
                    -scans[module][format_name][str(layer)]["nll_relative_increase"], layer
                ),
            )

    search_samples = splits["diagnosis_search"]
    search_reference = progress.root / "reference_search"
    search_baseline = _evaluate(
        model, processor, search_samples, progress=progress,
        variant="search__bf16", config=config, reference_dir=search_reference,
        write_reference=True,
    )
    interventions, random_controls = {}, {}
    safe = select_scope(targets, "decoder_safe")
    rng = random.Random(config.seed)
    for module, formats in _selected_module_formats(module_summary).items():
        interventions[module], random_controls[module] = {}, {}
        all_layers = sorted({target.layer for target in targets if target.scope == module})
        for format_name in formats:
            order = rankings[module][format_name]
            interventions[module][format_name] = {}
            random_controls[module][format_name] = {}
            for requested_k in (1, 2, 4, 8):
                k = min(requested_k, len(order))
                chosen = set(order[:k])
                quantize_top = select_targets(targets, scopes={module}, layers=chosen)
                restore_names = {
                    target.name for target in select_targets(
                        targets, scopes={module}, layers=chosen,
                    )
                }
                restore_top = [target for target in safe if target.name not in restore_names]
                values = {}
                for action, members in (
                    ("quantize_top", quantize_top), ("restore_top", restore_top),
                ):
                    variant = (
                        f"search__{format_name}__{SCOPE_LABELS[module]}__"
                        f"{action}__k{k:02d}"
                    )
                    with diagnostic_qdq_linears(model, members, SCHEMES[format_name]):
                        metrics = _evaluate(
                            model, processor, search_samples, progress=progress,
                            variant=variant, config=config, reference_dir=search_reference,
                        )
                    values[action] = metrics | {
                        "nll_relative_increase": (
                            metrics["target_nll"] / search_baseline["target_nll"] - 1.0
                        )
                    }
                interventions[module][format_name][str(k)] = {
                    "layers": order[:k], **values,
                }

                controls = []
                for control_index in range(20):
                    random_layers = set(rng.sample(all_layers, k))
                    excluded = {
                        target.name for target in select_targets(
                            targets, scopes={module}, layers=random_layers,
                        )
                    }
                    members = [target for target in safe if target.name not in excluded]
                    variant = (
                        f"random__{format_name}__{SCOPE_LABELS[module]}__"
                        f"restore__k{k:02d}__r{control_index:02d}"
                    )
                    with diagnostic_qdq_linears(model, members, SCHEMES[format_name]):
                        metrics = _evaluate(
                            model, processor, scan_samples, progress=progress,
                            variant=variant, config=config, reference_dir=scan_reference,
                            capture_hidden=True,
                        )
                    controls.append({
                        "layers": sorted(random_layers),
                        "target_nll": metrics["target_nll"],
                        "nll_relative_increase": (
                            metrics["target_nll"] / scan_baseline["target_nll"] - 1.0
                        ),
                    })
                random_controls[module][format_name][str(k)] = controls

    result = {
        "scan_baseline": scan_baseline,
        "search_baseline": search_baseline,
        "scans": scans,
        "rankings": rankings,
        "interventions": interventions,
        "random_controls": random_controls,
        "quantization": quant_audits,
        "inventory_counts": inventory["counts"],
    }
    atomic_json(progress.root / required[0], result)
    progress.mark_complete(required)
    return result


def stage_projection(args, config, splits, inputs):
    progress = _progress(config, "projection", inputs)
    required = ("projection_summary.json",)
    if args.resume and progress.is_complete(required):
        return json.loads((progress.root / required[0]).read_text())
    layer_summary = json.loads((config.output_dir / "layer/layer_summary.json").read_text())
    model, processor, targets, inventory, _load = _load_model(args, config)
    scan_samples = splits["diagnosis_scan"]
    scan_reference = progress.root / "reference_scan"
    scan_baseline = _evaluate(
        model, processor, scan_samples, progress=progress,
        variant="scan__bf16", config=config, reference_dir=scan_reference,
        write_reference=True, capture_hidden=True,
    )
    scans, rankings = {}, {}
    for module, format_rankings in layer_summary["rankings"].items():
        scans[module], rankings[module] = {}, {}
        for format_name, layer_order in format_rankings.items():
            selected_layers = list(layer_order[:6])
            if module == "decoder_ffn":
                selected_layers = list(dict.fromkeys([*selected_layers, 1, 2]))
            candidates = []
            scans[module][format_name] = {}
            for layer, projection in itertools.product(selected_layers, PROJECTIONS[module]):
                members = select_targets(
                    targets,
                    scopes={module},
                    layers={int(layer)},
                    projections={projection},
                )
                if not members:
                    continue
                key = f"layer{int(layer):02d}.{projection}"
                variant = f"scan__{format_name}__{SCOPE_LABELS[module]}__{key}"
                with diagnostic_qdq_linears(model, members, SCHEMES[format_name]):
                    metrics = _evaluate(
                        model, processor, scan_samples, progress=progress,
                        variant=variant, config=config, reference_dir=scan_reference,
                        capture_hidden=True,
                    )
                value = _variant_sensitivity(
                    progress, variant, metrics, scan_baseline, int(layer),
                )
                value.update({"layer": int(layer), "projection": projection})
                scans[module][format_name][key] = value
                candidates.append((key, value))
            rankings[module][format_name] = [
                key for key, _value in sorted(
                    candidates,
                    key=lambda item: (-item[1]["nll_relative_increase"], item[0]),
                )
            ]

    search_samples = splits["diagnosis_search"]
    search_reference = progress.root / "reference_search"
    search_baseline = _evaluate(
        model, processor, search_samples, progress=progress,
        variant="search__bf16", config=config, reference_dir=search_reference,
        write_reference=True,
    )
    safe = select_scope(targets, "decoder_safe")
    restorations = {}
    for module, format_rankings in rankings.items():
        restorations[module] = {}
        for format_name, order in format_rankings.items():
            restorations[module][format_name] = {}
            parsed = [
                (int(key.split(".", 1)[0].removeprefix("layer")), key.split(".", 1)[1])
                for key in order
            ]
            for requested_k in (1, 2, 4, 8):
                k = min(requested_k, len(parsed))
                chosen = parsed[:k]
                restore_names = set()
                for layer, projection in chosen:
                    restore_names.update(
                        target.name for target in select_targets(
                            targets,
                            scopes={module},
                            layers={layer},
                            projections={projection},
                        )
                    )
                members = [target for target in safe if target.name not in restore_names]
                variant = (
                    f"search__{format_name}__{SCOPE_LABELS[module]}__"
                    f"restore_projection__k{k:02d}"
                )
                with diagnostic_qdq_linears(model, members, SCHEMES[format_name]):
                    metrics = _evaluate(
                        model, processor, search_samples, progress=progress,
                        variant=variant, config=config, reference_dir=search_reference,
                    )
                restorations[module][format_name][str(k)] = {
                    "components": [
                        {"layer": layer, "projection": projection}
                        for layer, projection in chosen
                    ],
                    "metrics": metrics,
                    "nll_relative_increase": (
                        metrics["target_nll"] / search_baseline["target_nll"] - 1.0
                    ),
                }
    result = {
        "scan_baseline": scan_baseline,
        "search_baseline": search_baseline,
        "scans": scans,
        "rankings": rankings,
        "restorations": restorations,
        "inventory_counts": inventory["counts"],
    }
    atomic_json(progress.root / required[0], result)
    progress.mark_complete(required)
    return result


def main() -> None:
    args = _args()
    threads = _configure_threads(args.output_dir, args.source_output)
    config, splits, inputs, split_audit = _protocol(args, threads)
    _require_prerequisite(config, args.stage)
    started = time.time()
    if args.stage == "audit":
        result = stage_audit(args, config, splits, inputs, split_audit)
    elif args.stage == "factorial":
        result = stage_factorial(args, config, splits, inputs)
    elif args.stage == "module":
        result = stage_module(args, config, splits, inputs)
    elif args.stage == "layer":
        result = stage_layer(args, config, splits, inputs)
    else:
        result = stage_projection(args, config, splits, inputs)
    print(json.dumps({
        "stage": args.stage,
        "complete": True,
        "elapsed_seconds": time.time() - started,
        "result_keys": sorted(result),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

