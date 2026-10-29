from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import shutil
import time
from collections import Counter
from pathlib import Path

import torch

from experiments.infinitevl_quant.config import SCHEMES
from experiments.infinitevl_quant.data_protocol import write_split_manifests
from experiments.infinitevl_quant.evaluation import aggregate_rows, evaluate_variant, screen_gate
from experiments.infinitevl_quant.inventory import build_inventory, select_scope
from experiments.infinitevl_quant.loader import ensure_snapshot, load_infinitevl_cpu
from experiments.infinitevl_quant.progress import (
    StageProgress,
    atomic_json,
    atomic_text,
    file_sha256,
    json_digest,
    safe_id,
)
from experiments.infinitevl_quant.quantization import FakeQuantLinear, quantized_linears
from experiments.infinitevl_quant.statistics import paired_bootstrap, prediction_transitions
from experiments.vlm_quant_benchmark.data import sample_manifest_digest

from .config import EXPECTED_SCOPE_COUNTS, FORMATS, SCOPES, FullEvaluationConfig
from .data_protocol import build_full_splits
from .reporting import render_report
from .scoring import SCORER_ID, rescore_rows
from .statistics import exact_mcnemar, stratified_gate_bootstrap


STAGES = ("audit", "rescore", "smoke", "evaluate", "analyze", "report")
PREREQUISITES = {
    "rescore": ("audit", "audit_summary.json"),
    "smoke": ("rescore", "rescore_summary.json"),
    "evaluate": ("smoke", "smoke_summary.json"),
    "analyze": ("evaluate", "evaluation_summary.json"),
    "report": ("analyze", "analysis_summary.json"),
}
VARIANTS = tuple(f"{fmt}__{scope}" for fmt in FORMATS for scope in SCOPES)
BASE_IMPLEMENTATION_FILES = (
    "config.py", "cpu_backend.py", "evaluation.py", "inventory.py",
    "loader.py", "progress.py", "quantization.py", "statistics.py",
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=(*STAGES, "all"))
    parser.add_argument(
        "--source-output", type=Path,
        default=Path("outputs/infinitevl_quant_ffn_bf16_v1"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/infinitevl_quant_ffn_bf16_4x64_v2"),
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
            raise FileNotFoundError(f"missing validated thread config: {source}")
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


def _implementation_digest() -> tuple[dict, str]:
    package_root = Path(__file__).parent
    own = {
        str(path.relative_to(package_root)): file_sha256(path)
        for path in sorted(package_root.glob("*.py"))
    }
    base_root = package_root.parent / "infinitevl_quant"
    base = {name: file_sha256(base_root / name) for name in BASE_IMPLEMENTATION_FILES}
    payload = {"full_4x64": own, "base": base}
    return payload, json_digest(payload)


def _source_digest(source: Path) -> tuple[dict, str]:
    names = (
        "protocol_config.json",
        "protocol/implementation_sha256.json",
        "audit/audit_summary.json",
        "screen/.complete",
        "screen/screen_summary.json",
        "confirm/.complete",
        "confirm/confirm_summary.json",
    )
    hashes = {}
    for name in names:
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(f"missing source artifact: {path}")
        hashes[name] = file_sha256(path)
    return hashes, json_digest(hashes)


def _protocol(args: argparse.Namespace, threads: dict):
    config = FullEvaluationConfig(
        output_dir=args.output_dir,
        source_output=args.source_output,
        max_new_tokens=args.max_new_tokens,
    )
    splits, split_audit = build_full_splits(config.evaluation_manifest)
    implementation, implementation_digest = _implementation_digest()
    source_hashes, source_digest = _source_digest(config.source_output)
    inputs = {
        "model_revision": config.model_revision,
        "evaluation_manifest": file_sha256(config.evaluation_manifest),
        "splits": {name: sample_manifest_digest(rows) for name, rows in splits.items()},
        "implementation": implementation_digest,
        "source_output": source_digest,
        "threads": json_digest(threads),
        "scorer": SCORER_ID,
    }
    identity = {
        "format_version": 2,
        "config_digest": config.digest(),
        "input_digests": inputs,
    }
    identity_path = config.output_dir / "protocol_identity.json"
    if identity_path.is_file():
        if json.loads(identity_path.read_text()) != identity:
            raise RuntimeError("output belongs to different code/config/inputs; choose a new output dir")
    else:
        atomic_json(identity_path, identity)
    atomic_json(
        config.output_dir / "protocol_config.json",
        config.to_dict() | {"config_digest": config.digest()},
    )
    atomic_json(config.output_dir / "protocol/split_audit.json", split_audit)
    atomic_json(config.output_dir / "protocol/splits.json", write_split_manifests(
        config.output_dir / "protocol", splits,
    ))
    atomic_json(
        config.output_dir / "protocol/implementation_sha256.json",
        {"files": implementation, "combined": implementation_digest, "source": source_hashes},
    )
    wallclock = config.output_dir / "wallclock.json"
    if not wallclock.is_file():
        atomic_json(wallclock, {"started_unix": time.time(), "limit_hours": config.wallclock_hours})
    elapsed = (time.time() - json.loads(wallclock.read_text())["started_unix"]) / 3600.0
    if elapsed > config.wallclock_hours and args.stage not in ("analyze", "report"):
        raise TimeoutError(f"4x64 wall-clock budget exceeded: {elapsed:.2f}h")
    return config, splits, inputs, split_audit


def _progress(config: FullEvaluationConfig, stage: str, inputs: dict) -> StageProgress:
    return StageProgress(config.output_dir, stage, config_digest=config.digest(), inputs=inputs)


def _require_prerequisite(config: FullEvaluationConfig, stage: str) -> None:
    requirement = PREREQUISITES.get(stage)
    if requirement is None:
        return
    upstream, result = requirement
    root = config.output_dir / upstream
    if not (root / ".complete").is_file() or not (root / result).is_file():
        raise RuntimeError(f"stage {stage} requires completed stage {upstream}")


def _model_path(args: argparse.Namespace, config: FullEvaluationConfig) -> Path:
    if args.model_path is not None:
        return args.model_path.resolve()
    return ensure_snapshot(
        config.model_id, config.model_revision, local_files_only=args.local_files_only,
    )


def _load_model(args: argparse.Namespace, config: FullEvaluationConfig):
    model, processor, load_audit = load_infinitevl_cpu(_model_path(args, config))
    targets, inventory = build_inventory(model)
    return model, processor, targets, inventory, load_audit


def _targets_for_scope(targets, scope: str):
    members = select_scope(targets, scope)
    expected = EXPECTED_SCOPE_COUNTS[scope]
    if len(members) != expected:
        raise RuntimeError(f"scope {scope} contains {len(members)} modules, expected {expected}")
    return members


def _compact_quant_audit(audit: dict) -> dict:
    rows = list(audit["modules"].values())
    return {
        "scheme": audit["scheme"],
        "module_count": audit["module_count"],
        "scope_counts": dict(Counter(row["scope"] for row in rows)),
        "mean_weight_nmse": sum(float(row["weight_nmse"]) for row in rows) / len(rows),
        "max_weight_nmse": max(float(row["weight_nmse"]) for row in rows),
        "fake_quantized": audit["fake_quantized"],
        "deployment_kernel_claim": audit["deployment_kernel_claim"],
    }


def _ffn_identity(targets) -> tuple[tuple, str]:
    rows, digest = [], hashlib.sha256()
    for target in _targets_for_scope(targets, "ffn"):
        module, weight = target.module, target.module.weight.detach()
        rows.append((target.name, id(module), weight.data_ptr(), weight._version))
        digest.update(target.name.encode())
        digest.update(memoryview(weight.numpy()).cast("B"))
        if module.bias is not None:
            digest.update(memoryview(module.bias.detach().numpy()).cast("B"))
    return tuple(rows), digest.hexdigest()


def _assert_ffn_modules(model, signature: tuple) -> None:
    for name, module_id, pointer, version in signature:
        module = model.get_submodule(name)
        if id(module) != module_id or module.weight.data_ptr() != pointer or module.weight._version != version:
            raise RuntimeError(f"FFN reference module changed during quantization: {name}")


def _validate_source(config: FullEvaluationConfig) -> dict:
    source_config = json.loads((config.source_output / "protocol_config.json").read_text())
    expected = {
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "min_pixels": config.min_pixels,
        "max_pixels": config.max_pixels,
        "max_new_tokens": config.max_new_tokens,
    }
    mismatches = {
        key: {"source": source_config.get(key), "expected": value}
        for key, value in expected.items() if source_config.get(key) != value
    }
    if file_sha256(Path(source_config["evaluation_manifest"])) != file_sha256(config.evaluation_manifest):
        mismatches["evaluation_manifest_sha256"] = "mismatch"
    source_impl = json.loads(
        (config.source_output / "protocol/implementation_sha256.json").read_text()
    )["files"]["base"]
    current_root = Path(__file__).parent.parent / "infinitevl_quant"
    current_impl = {name: file_sha256(current_root / name) for name in BASE_IMPLEMENTATION_FILES}
    if source_impl != current_impl:
        mismatches["base_implementation"] = {
            name: {"source": source_impl.get(name), "current": current_impl.get(name)}
            for name in BASE_IMPLEMENTATION_FILES if source_impl.get(name) != current_impl.get(name)
        }
    if mismatches:
        raise RuntimeError(f"source output is not reusable: {mismatches}")
    audit = json.loads((config.source_output / "audit/audit_summary.json").read_text())
    for stage in ("screen", "confirm"):
        summary = json.loads((config.source_output / stage / f"{stage}_summary.json").read_text())
        for variant in VARIANTS:
            quant = summary["quantization"][variant]
            scope = variant.split("__", 1)[1]
            if quant["module_count"] != EXPECTED_SCOPE_COUNTS[scope]:
                raise RuntimeError(f"source {stage}/{variant} module count mismatch")
            if quant["scheme"] != SCHEMES[variant.split("__", 1)[0]].to_dict():
                raise RuntimeError(f"source {stage}/{variant} scheme mismatch")
        if summary["ffn_reference_sha256"] != audit["ffn_reference"]["sha256"]:
            raise RuntimeError(f"source {stage} FFN hash mismatch")
    return {
        "validated": True,
        "source_config_digest": source_config["config_digest"],
        "ffn_reference_sha256": audit["ffn_reference"]["sha256"],
        "base_implementation": current_impl,
    }


def _evaluate(model, processor, samples, *, progress, variant, config, reference_dir, write_reference=False):
    result = evaluate_variant(
        model, processor, samples,
        progress=progress,
        variant=variant,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        max_new_tokens=config.max_new_tokens,
        generate=True,
        reference_dir=reference_dir,
        write_reference=write_reference,
        teacher_timeout_seconds=600.0,
        generation_timeout_seconds=1800.0,
    )
    finite = (float(result["target_nll"]), float(result["target_ppl"]), float(result["mean_score"]))
    if not all(math.isfinite(value) for value in finite):
        raise FloatingPointError(f"{variant} produced non-finite metrics: {finite}")
    return result


def _run_matrix(args, config, samples, progress):
    model, processor, targets, inventory, load_audit = _load_model(args, config)
    ffn_signature, ffn_hash_before = _ffn_identity(targets)
    reference_dir = progress.root / "reference_artifacts"
    metrics = {
        "bf16": _evaluate(
            model, processor, samples, progress=progress, variant="bf16",
            config=config, reference_dir=reference_dir, write_reference=True,
        )
    }
    audits = {}
    for variant in VARIANTS:
        format_name, scope = variant.split("__", 1)
        with quantized_linears(model, _targets_for_scope(targets, scope), SCHEMES[format_name]) as audit:
            _assert_ffn_modules(model, ffn_signature)
            metrics[variant] = _evaluate(
                model, processor, samples, progress=progress, variant=variant,
                config=config, reference_dir=reference_dir,
            )
        _assert_ffn_modules(model, ffn_signature)
        audits[variant] = _compact_quant_audit(audit)
    ffn_after, ffn_hash_after = _ffn_identity(targets)
    if ffn_signature != ffn_after or ffn_hash_before != ffn_hash_after:
        raise RuntimeError("FFN reference changed during evaluation matrix")
    result = metrics, audits, inventory, load_audit, ffn_hash_before
    del model, processor
    gc.collect()
    return result


def _copy_reference(source: Path, destination: Path, sample_id: str) -> bool:
    if not source.is_file():
        return False
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if payload.get("id") != sample_id or "target_logits" not in payload:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)
    return True


def _import_rows(
    progress: StageProgress,
    samples: dict,
    source_root: Path,
    source_stage: str,
) -> dict:
    imported, missing, hashes = Counter(), Counter(), {}
    reference_source = source_root / source_stage / "reference_artifacts"
    reference_destination = progress.root / "reference_artifacts"
    for variant in ("bf16", *VARIANTS):
        directory = source_root / source_stage / variant / "samples"
        if not directory.is_dir():
            missing[variant] += 1
            continue
        for path in sorted(directory.glob("*.json")):
            row = json.loads(path.read_text())
            identifier = row.get("id")
            sample = samples.get(identifier)
            if sample is None:
                continue
            if row.get("dataset") != sample.dataset or row.get("target") != sample.target:
                raise RuntimeError(f"source row metadata mismatch: {path}")
            if progress.completed_sample(variant, identifier) is not None:
                continue
            if variant == "bf16":
                name = f"{safe_id(identifier)}.pt"
                if not _copy_reference(
                    reference_source / name, reference_destination / name, identifier,
                ):
                    missing[variant] += 1
                    continue
                hashes[f"{source_stage}/reference/{identifier}"] = file_sha256(reference_source / name)
            progress.complete_sample(variant, identifier, row)
            imported[variant] += 1
            hashes[f"{source_stage}/{variant}/{identifier}"] = file_sha256(path)
    return {
        "source": str(source_root / source_stage),
        "imported": dict(imported),
        "missing_or_invalid": dict(missing),
        "artifact_digest": json_digest(hashes),
    }


def stage_audit(args, config, splits, inputs, split_audit):
    progress = _progress(config, "audit", inputs)
    required = ("audit_summary.json",)
    if args.resume and progress.is_complete(required):
        return json.loads((progress.root / required[0]).read_text())
    source_audit = _validate_source(config)
    model, _processor, targets, inventory, load_audit = _load_model(args, config)
    counts = {scope: len(_targets_for_scope(targets, scope)) for scope in EXPECTED_SCOPE_COUNTS}
    signature, before = _ffn_identity(targets)
    target = _targets_for_scope(targets, "attn_all")[0]
    original = model.get_submodule(target.name)
    with quantized_linears(model, [target], SCHEMES["w8a16"]):
        if not isinstance(model.get_submodule(target.name), FakeQuantLinear):
            raise RuntimeError("W8 audit did not install FakeQuantLinear")
        _assert_ffn_modules(model, signature)
    if model.get_submodule(target.name) is not original:
        raise RuntimeError("W8 audit did not restore original Linear")
    after_signature, after = _ffn_identity(targets)
    if signature != after_signature or before != after:
        raise RuntimeError("FFN changed during audit")
    result = {
        "passed": True,
        "source": source_audit,
        "split_audit": split_audit,
        "scope_counts": counts,
        "inventory_counts": inventory["counts"],
        "load": load_audit,
        "ffn_reference": {"modules": len(signature), "sha256": before},
    }
    atomic_json(progress.root / required[0], result)
    progress.mark_complete(required)
    del model
    gc.collect()
    return result


def _source_rows(config: FullEvaluationConfig, variant: str) -> list[dict]:
    rows = []
    for stage in ("screen", "confirm"):
        directory = config.source_output / stage / variant / "samples"
        rows.extend(json.loads(path.read_text()) for path in sorted(directory.glob("*.json")))
    if len(rows) != 64 or len({row["id"] for row in rows}) != 64:
        raise RuntimeError(f"source {variant} expected 64 unique rows, got {len(rows)}")
    return rows


def stage_rescore(args, config, splits, inputs):
    progress = _progress(config, "rescore", inputs)
    required = ("rescore_summary.json",)
    if args.resume and progress.is_complete(required):
        return json.loads((progress.root / required[0]).read_text())
    sample_map = {sample.id: sample for sample in splits["full256"]}
    variants = {}
    for variant in ("bf16", *VARIANTS):
        raw = _source_rows(config, variant)
        scored = rescore_rows(raw, sample_map)
        legacy = aggregate_rows(raw)
        corrected = aggregate_rows(scored)
        variants[variant] = {
            "samples": len(scored),
            "changed_scores": sum(row["score"] != row["score_legacy"] for row in scored),
            "legacy": legacy,
            "v2": corrected,
            "changed_rows": [
                {
                    "id": row["id"], "dataset": row["dataset"],
                    "prediction": row["prediction"], "target": row["target"],
                    "legacy": row["score_legacy"], "v2": row["score_v2"],
                    "rule": row["scoring_v2"]["rule"],
                }
                for row in scored if row["score"] != row["score_legacy"]
            ],
        }
    result = {"scorer": SCORER_ID, "variants": variants}
    atomic_json(progress.root / required[0], result)
    progress.mark_complete(required)
    return result


def stage_smoke(args, config, splits, inputs):
    progress = _progress(config, "smoke", inputs)
    required = ("smoke_summary.json",)
    if args.resume and progress.is_complete(required):
        return json.loads((progress.root / required[0]).read_text())
    metrics, audits, inventory, load, ffn_hash = _run_matrix(
        args, config, splits["smoke4"], progress,
    )
    result = {
        "metrics": metrics,
        "quantization": audits,
        "inventory_counts": inventory["counts"],
        "load": load,
        "ffn_reference_sha256": ffn_hash,
    }
    atomic_json(progress.root / required[0], result)
    progress.mark_complete(required)
    return result


def stage_evaluate(args, config, splits, inputs):
    progress = _progress(config, "evaluate", inputs)
    required = ("evaluation_summary.json",)
    if args.resume and progress.is_complete(required):
        return json.loads((progress.root / required[0]).read_text())
    sample_map = {sample.id: sample for sample in splits["full256"]}
    imports = [
        _import_rows(progress, sample_map, config.source_output, "screen"),
        _import_rows(progress, sample_map, config.source_output, "confirm"),
        _import_rows(progress, sample_map, config.output_dir, "smoke"),
    ]
    metrics, audits, inventory, load, ffn_hash = _run_matrix(
        args, config, splits["full256"], progress,
    )
    expected_ids = set(sample_map)
    row_audit = {}
    for variant in ("bf16", *VARIANTS):
        rows = progress.rows(variant)
        identifiers = {row["id"] for row in rows}
        if len(rows) != 256 or identifiers != expected_ids:
            raise RuntimeError(
                f"{variant} rows incomplete: rows={len(rows)} missing={len(expected_ids-identifiers)} "
                f"extra={len(identifiers-expected_ids)}"
            )
        row_audit[variant] = {"samples": len(rows), "ids_digest": json_digest(sorted(identifiers))}
    result = {
        "metrics_legacy": metrics,
        "quantization": audits,
        "inventory_counts": inventory["counts"],
        "load": load,
        "ffn_reference_sha256": ffn_hash,
        "imports": imports,
        "row_audit": row_audit,
    }
    atomic_json(progress.root / required[0], result)
    progress.mark_complete(required)
    return result


def _subset(rows: list[dict], identifiers: set[str]) -> list[dict]:
    result = [row for row in rows if row["id"] in identifiers]
    if len(result) != len(identifiers):
        raise RuntimeError(f"subset expected {len(identifiers)} rows, got {len(result)}")
    return result


def _select_candidate(classification: dict[str, str]) -> dict:
    for scope, decision in (
        ("attn_all", "GO_ATTN_ALL"),
        ("attn_main", "GO_ATTN_MAIN_KEEP_GDN_DYNAMICS_REFERENCE"),
    ):
        for fmt in ("w3a16", "w4a16", "w8a16"):
            variant = f"{fmt}__{scope}"
            if classification[variant] == "ROBUST_USABLE":
                return {"candidate": variant, "decision": decision}
    return {"candidate": None, "decision": "NO_GO"}


def stage_analyze(args, config, splits, inputs):
    progress = _progress(config, "analyze", inputs)
    required = ("analysis_summary.json",)
    if args.resume and progress.is_complete(required):
        return json.loads((progress.root / required[0]).read_text())
    sample_map = {sample.id: sample for sample in splits["full256"]}
    evaluate = config.output_dir / "evaluate"
    scored = {
        variant: rescore_rows(
            [json.loads(path.read_text()) for path in sorted((evaluate / variant / "samples").glob("*.json"))],
            sample_map,
        )
        for variant in ("bf16", *VARIANTS)
    }
    views = {
        name: {sample.id for sample in splits[name]}
        for name in ("full256", "legacy128", "unseen_holdout128")
    }
    metrics, comparisons = {}, {}
    for view, identifiers in views.items():
        view_rows = {variant: _subset(rows, identifiers) for variant, rows in scored.items()}
        metrics[view] = {variant: aggregate_rows(rows) for variant, rows in view_rows.items()}
        baseline = view_rows["bf16"]
        comparisons[view] = {}
        for index, variant in enumerate(VARIANTS):
            candidate = view_rows[variant]
            bootstrap = stratified_gate_bootstrap(
                baseline, candidate,
                samples=config.bootstrap_samples,
                seed=config.seed + index,
            )
            comparisons[view][variant] = {
                "gate": screen_gate(metrics[view][variant], metrics[view]["bf16"]),
                "paired_bootstrap_by_dataset": paired_bootstrap(
                    baseline, candidate,
                    samples=config.bootstrap_samples,
                    seed=config.seed + index,
                ),
                "bootstrap": bootstrap,
                "mcnemar": exact_mcnemar(baseline, candidate),
                "prediction_transitions": prediction_transitions(baseline, candidate),
                "possible_improvement": False,
            }
    classification = {}
    for variant in VARIANTS:
        full = comparisons["full256"][variant]
        holdout = comparisons["unseen_holdout128"][variant]
        if full["gate"]["passed"] and holdout["gate"]["passed"]:
            if (
                full["bootstrap"]["gate_pass_probability"] >= config.robust_probability
                and holdout["bootstrap"]["gate_pass_probability"] >= config.robust_probability
            ):
                status = "ROBUST_USABLE"
            else:
                status = "POINT_USABLE_UNCERTAIN"
        else:
            status = "FAIL"
        classification[variant] = status
        full_point = full["bootstrap"]["point"]["macro_delta"]
        holdout_point = holdout["bootstrap"]["point"]["macro_delta"]
        possible = (
            full_point > 0
            and holdout_point > 0
            and holdout["bootstrap"]["macro_score_delta_points_ci95"][0] > 0
        )
        full["possible_improvement"] = possible
        holdout["possible_improvement"] = possible
    result = {
        "scorer": SCORER_ID,
        "variant_order": list(VARIANTS),
        "metrics": metrics,
        "comparisons": comparisons,
        "classification": classification,
        "selection": _select_candidate(classification),
    }
    atomic_json(progress.root / required[0], result)
    progress.mark_complete(required)
    return result


def stage_report(args, config, splits, inputs):
    del args, splits
    progress = _progress(config, "report", inputs)
    required = ("report_summary.json", "report_zh.md")
    if progress.is_complete(required):
        return json.loads((progress.root / required[0]).read_text())
    rescore = json.loads((config.output_dir / "rescore/rescore_summary.json").read_text())
    analysis = json.loads((config.output_dir / "analyze/analysis_summary.json").read_text())
    result = {
        "selection": analysis["selection"],
        "classification": analysis["classification"],
        "labels": config.to_dict()["labels"],
        "scorer": SCORER_ID,
    }
    atomic_json(progress.root / required[0], result)
    atomic_text(progress.root / required[1], render_report(rescore, analysis))
    progress.mark_complete(required)
    return result


STAGE_FUNCTIONS = {
    "audit": stage_audit,
    "rescore": stage_rescore,
    "smoke": stage_smoke,
    "evaluate": stage_evaluate,
    "analyze": stage_analyze,
    "report": stage_report,
}


def main() -> None:
    args = _args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    threads = _configure_threads(args.output_dir, args.source_output)
    config, splits, inputs, split_audit = _protocol(args, threads)
    stages = STAGES if args.stage == "all" else (args.stage,)
    for stage in stages:
        _require_prerequisite(config, stage)
        started = time.time()
        function = STAGE_FUNCTIONS[stage]
        result = (
            function(args, config, splits, inputs, split_audit)
            if stage == "audit"
            else function(args, config, splits, inputs)
        )
        print(json.dumps({
            "stage": stage,
            "complete": True,
            "elapsed_seconds": time.time() - started,
            "result_keys": sorted(result),
        }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
