from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import torch

from experiments.infinitevl_quant.config import MODEL_REVISION, SCHEMES
from experiments.infinitevl_quant.data_protocol import write_split_manifests
from experiments.infinitevl_quant.evaluation import evaluate_variant, screen_gate
from experiments.infinitevl_quant.inventory import build_inventory, select_scope
from experiments.infinitevl_quant.loader import ensure_snapshot, load_infinitevl_cpu
from experiments.infinitevl_quant.progress import (
    StageProgress,
    atomic_json,
    atomic_text,
    file_sha256,
    json_digest,
)
from experiments.infinitevl_quant.quantization import FakeQuantLinear, quantized_linears
from experiments.infinitevl_quant.statistics import paired_bootstrap, prediction_transitions
from experiments.infinitevl_quant_diagnosis.data_protocol import build_diagnosis_splits
from experiments.vlm_quant_benchmark.data import sample_manifest_digest

from .config import (
    CONFIRM_SCOPES,
    EXPECTED_SCOPE_COUNTS,
    FORMATS,
    SCREEN_SCOPES,
    FFNReferenceConfig,
)
from .reporting import render_report
from .statistics import stratified_paired_bootstrap


STAGES = ("audit", "smoke", "screen", "confirm", "report")
PREREQUISITES = {
    "smoke": ("audit", "audit_summary.json"),
    "screen": ("smoke", "smoke_summary.json"),
    "confirm": ("screen", "screen_summary.json"),
    "report": ("confirm", "confirm_summary.json"),
}
EXPECTED_SOURCE_STAGE_METRICS = {
    "factorial": 12,
    "module": 58,
    "layer": 1010,
    "projection": 260,
}
SOURCE_SUMMARIES = {
    "audit": "audit_summary.json",
    "factorial": "factorial_summary.json",
    "module": "module_summary.json",
    "layer": "layer_summary.json",
    "projection": "projection_summary.json",
}
SOURCE_FORMATS = {"w4a16": "w4_asym_a16", "w3a16": "w3_asym_a16"}
SOURCE_SCOPES = {
    "swa": "swa_attention",
    "gdn_main": "gdn_main",
    "attn_main": "swa_attention+gdn_main",
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument(
        "--source-rootcause-output", type=Path,
        default=Path("outputs/infinitevl_quant_cpu_rootcause_v1"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/infinitevl_quant_ffn_bf16_v1"),
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
            raise FileNotFoundError("missing validated root-cause thread configuration")
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
    base = {
        name: file_sha256(base_root / name)
        for name in (
            "config.py", "cpu_backend.py", "evaluation.py", "inventory.py",
            "loader.py", "progress.py", "quantization.py", "statistics.py",
        )
    }
    diagnosis = package_root.parent / "infinitevl_quant_diagnosis" / "data_protocol.py"
    payload = {"ffn_reference": own, "base": base, "diagnosis_data_protocol.py": file_sha256(diagnosis)}
    return payload, json_digest(payload)


def _protocol(args: argparse.Namespace, threads: dict) -> tuple[FFNReferenceConfig, dict, dict, dict]:
    config = FFNReferenceConfig(
        output_dir=args.output_dir,
        source_rootcause_output=args.source_rootcause_output,
        max_new_tokens=args.max_new_tokens,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    splits, split_audit = build_diagnosis_splits(
        config.calibration_manifest, config.evaluation_manifest,
    )
    smoke = []
    seen = set()
    for sample in splits["diagnosis_search"]:
        if sample.dataset not in seen:
            smoke.append(sample)
            seen.add(sample.dataset)
    if len(smoke) != 4:
        raise RuntimeError("smoke split must contain one sample per evaluation dataset")
    splits["ffn_reference_smoke"] = smoke

    implementation, implementation_digest = _implementation_digest()
    source_hashes = {}
    for stage, summary in SOURCE_SUMMARIES.items():
        root = config.source_rootcause_output / stage
        for name in (".complete", summary):
            path = root / name
            if not path.is_file():
                raise FileNotFoundError(f"missing source root-cause artifact: {path}")
            source_hashes[f"{stage}/{name}"] = file_sha256(path)
    inputs = {
        "model_revision": config.model_revision,
        "calibration_manifest": file_sha256(config.calibration_manifest),
        "evaluation_manifest": file_sha256(config.evaluation_manifest),
        "source_rootcause": json_digest(source_hashes),
        "splits": {name: sample_manifest_digest(rows) for name, rows in splits.items()},
        "implementation": implementation_digest,
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
            raise RuntimeError("output belongs to different code/config/inputs; choose a new output dir")
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
        {"files": implementation, "combined": implementation_digest, "source": source_hashes},
    )
    wallclock_path = config.output_dir / "wallclock.json"
    if not wallclock_path.is_file():
        atomic_json(
            wallclock_path,
            {"started_unix": time.time(), "limit_hours": config.wallclock_hours},
        )
    elapsed = (
        time.time() - float(json.loads(wallclock_path.read_text())["started_unix"])
    ) / 3600.0
    if elapsed > config.wallclock_hours and args.stage != "report":
        raise TimeoutError(f"FFN-reference wall-clock budget exceeded: {elapsed:.2f}h")
    return config, splits, inputs, split_audit


def _require_prerequisite(config: FFNReferenceConfig, stage: str) -> None:
    requirement = PREREQUISITES.get(stage)
    if requirement is None:
        return
    upstream, result = requirement
    root = config.output_dir / upstream
    if not (root / ".complete").is_file() or not (root / result).is_file():
        raise RuntimeError(f"stage {stage} requires completed stage {upstream}")


def _progress(config: FFNReferenceConfig, stage: str, inputs: dict) -> StageProgress:
    return StageProgress(
        config.output_dir, stage, config_digest=config.digest(), inputs=inputs,
    )


def _model_path(args: argparse.Namespace, config: FFNReferenceConfig) -> Path:
    if args.model_path is not None:
        return args.model_path.resolve()
    return ensure_snapshot(
        config.model_id, config.model_revision, local_files_only=args.local_files_only,
    )


def _load_model(args: argparse.Namespace, config: FFNReferenceConfig):
    model, processor, load_audit = load_infinitevl_cpu(_model_path(args, config))
    targets, inventory = build_inventory(model)
    return model, processor, targets, inventory, load_audit


def _targets_for_scope(targets, scope: str):
    members = select_scope(targets, scope)
    expected = EXPECTED_SCOPE_COUNTS[scope]
    if len(members) != expected:
        raise RuntimeError(f"scope {scope} contains {len(members)} Linear modules, expected {expected}")
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
    rows = []
    digest = hashlib.sha256()
    for target in _targets_for_scope(targets, "ffn"):
        module = target.module
        current = module.weight.detach()
        rows.append((target.name, id(module), current.data_ptr(), current._version))
        digest.update(target.name.encode())
        digest.update(memoryview(current.numpy()).cast("B"))
        if module.bias is not None:
            digest.update(memoryview(module.bias.detach().numpy()).cast("B"))
    return tuple(rows), digest.hexdigest()


def _assert_ffn_modules(model, signature: tuple) -> None:
    for name, module_id, pointer, version in signature:
        module = model.get_submodule(name)
        if id(module) != module_id or module.weight.data_ptr() != pointer or module.weight._version != version:
            raise RuntimeError(f"FFN reference module changed during quantization: {name}")


def _evaluate(
    model, processor, samples, *, progress, variant, config, reference_dir,
    write_reference=False,
):
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
    finite = (
        float(result["target_nll"]), float(result["target_ppl"]),
        float(result["mean_score"]),
    )
    if not all(math.isfinite(value) for value in finite):
        raise FloatingPointError(f"{variant} produced non-finite metrics: {finite}")
    return result


def _source_metrics_audit(source: Path) -> dict:
    identities = {}
    metrics_checked = aggregate_errors = nonfinite = sample_rows = 0
    for stage in SOURCE_SUMMARIES:
        identity = json.loads((source / stage / "identity.json").read_text())
        identities[stage] = identity["config_digest"]
    if len(set(identities.values())) != 1:
        raise RuntimeError(f"source stage config digests disagree: {identities}")
    stage_metrics = {}
    for stage, expected in EXPECTED_SOURCE_STAGE_METRICS.items():
        paths = list((source / stage).glob("*/metrics.json"))
        stage_metrics[stage] = len(paths)
        if len(paths) != expected:
            raise RuntimeError(f"source {stage} has {len(paths)} metrics, expected {expected}")
        for metrics_path in paths:
            rows = [json.loads(path.read_text()) for path in (metrics_path.parent / "samples").glob("*.json")]
            metrics = json.loads(metrics_path.read_text())
            tokens = sum(int(row["target_text"]["tokens"]) for row in rows)
            total = sum(float(row["target_text"]["total_nll"]) for row in rows)
            observed = total / tokens
            value = float(metrics["target_nll"])
            metrics_checked += 1
            sample_rows += len(rows)
            aggregate_errors += int(
                len(rows) != int(metrics["samples"]) or abs(observed - value) > 1e-10
            )
            nonfinite += int(not math.isfinite(value))
    if aggregate_errors or nonfinite:
        raise RuntimeError(
            f"source metric audit failed: aggregate_errors={aggregate_errors}, nonfinite={nonfinite}"
        )
    return {
        "config_digest": next(iter(identities.values())),
        "stage_metrics": stage_metrics,
        "metrics_checked": metrics_checked,
        "sample_rows": sample_rows,
        "aggregate_errors": aggregate_errors,
        "nonfinite_metrics": nonfinite,
    }


def stage_audit(args, config, splits, inputs, split_audit):
    progress = _progress(config, "audit", inputs)
    required = ("audit_summary.json",)
    if args.resume and progress.is_complete(required):
        return json.loads((progress.root / required[0]).read_text())
    source_config = json.loads(
        (config.source_rootcause_output / "protocol_config.json").read_text()
    )
    if source_config["model_revision"] != MODEL_REVISION:
        raise RuntimeError("source root-cause model revision mismatch")
    source_audit = _source_metrics_audit(config.source_rootcause_output)
    model, _processor, targets, inventory, load_audit = _load_model(args, config)
    scope_counts = {
        scope: len(_targets_for_scope(targets, scope)) for scope in EXPECTED_SCOPE_COUNTS
    }
    ffn_before, ffn_hash_before = _ffn_identity(targets)
    target = _targets_for_scope(targets, "attn_all")[0]
    original = model.get_submodule(target.name)
    with quantized_linears(model, [target], SCHEMES["w8a16"]) as quant_audit:
        if not isinstance(model.get_submodule(target.name), FakeQuantLinear):
            raise RuntimeError("W8 context did not install FakeQuantLinear")
        _assert_ffn_modules(model, ffn_before)
    if model.get_submodule(target.name) is not original:
        raise RuntimeError("W8 context did not restore the original Linear")
    ffn_after, ffn_hash_after = _ffn_identity(targets)
    if ffn_before != ffn_after or ffn_hash_before != ffn_hash_after:
        raise RuntimeError("FFN reference weights changed during W8 audit")
    result = {
        "passed": True,
        "source_audit": source_audit,
        "split_audit": split_audit,
        "scope_counts": scope_counts,
        "inventory_counts": inventory["counts"],
        "load": load_audit,
        "ffn_reference": {
            "modules": len(ffn_before), "sha256": ffn_hash_before,
            "definition": "official BF16 checkpoint values widened to FP32 storage and compute",
        },
        "w8_context": _compact_quant_audit(quant_audit),
    }
    atomic_json(progress.root / required[0], result)
    progress.mark_complete(required)
    return result


def _run_matrix(args, config, samples, progress, variants):
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
    for format_name, scope in variants:
        variant = f"{format_name}__{scope}"
        members = _targets_for_scope(targets, scope)
        with quantized_linears(model, members, SCHEMES[format_name]) as audit:
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
    return metrics, audits, inventory, load_audit, ffn_hash_before


def stage_smoke(args, config, splits, inputs):
    progress = _progress(config, "smoke", inputs)
    required = ("smoke_summary.json",)
    if args.resume and progress.is_complete(required):
        return json.loads((progress.root / required[0]).read_text())
    variants = [(fmt, scope) for fmt in FORMATS for scope in CONFIRM_SCOPES]
    metrics, audits, inventory, load, ffn_hash = _run_matrix(
        args, config, splits["ffn_reference_smoke"], progress, variants,
    )
    result = {
        "metrics": metrics, "quantization": audits,
        "inventory_counts": inventory["counts"], "load": load,
        "ffn_reference_sha256": ffn_hash,
    }
    atomic_json(progress.root / required[0], result)
    progress.mark_complete(required)
    return result


def _source_reproduction(config, metrics: dict) -> dict:
    source = json.loads(
        (config.source_rootcause_output / "module/module_summary.json").read_text()
    )["metrics"]["search"]
    checks = {}
    for format_name, source_format in SOURCE_FORMATS.items():
        for scope, source_scope in SOURCE_SCOPES.items():
            variant = f"{format_name}__{scope}"
            observed = float(metrics[variant]["target_nll"])
            expected = float(source[source_format][source_scope]["target_nll"])
            error = abs(observed - expected)
            checks[variant] = {
                "observed": observed, "expected": expected,
                "absolute_error": error, "passed": error <= config.reproduction_atol,
            }
    failures = {key: value for key, value in checks.items() if not value["passed"]}
    if failures:
        raise RuntimeError(f"A-D reproduction mismatch: {failures}")
    return checks


def stage_screen(args, config, splits, inputs):
    progress = _progress(config, "screen", inputs)
    required = ("screen_summary.json",)
    if args.resume and progress.is_complete(required):
        return json.loads((progress.root / required[0]).read_text())
    variants = [(fmt, scope) for fmt in FORMATS for scope in SCREEN_SCOPES]
    metrics, audits, inventory, load, ffn_hash = _run_matrix(
        args, config, splits["diagnosis_search"], progress, variants,
    )
    gates = {
        variant: screen_gate(value, metrics["bf16"])
        for variant, value in metrics.items() if variant != "bf16"
    }
    result = {
        "metrics": metrics, "gates": gates,
        "reproduction": _source_reproduction(config, metrics),
        "quantization": audits, "inventory_counts": inventory["counts"],
        "load": load, "ffn_reference_sha256": ffn_hash,
    }
    atomic_json(progress.root / required[0], result)
    progress.mark_complete(required)
    return result


def select_candidate(comparisons: dict) -> dict:
    for scope, decision in (
        ("attn_all", "GO_ATTN_ALL"),
        ("attn_main", "GO_ATTN_MAIN_KEEP_GDN_DYNAMICS_REFERENCE"),
    ):
        for format_name in ("w3a16", "w4a16", "w8a16"):
            variant = f"{format_name}__{scope}"
            if comparisons[variant]["gate"]["passed"]:
                return {"candidate": variant, "decision": decision}
    return {"candidate": None, "decision": "NO_GO"}


def stage_confirm(args, config, splits, inputs):
    progress = _progress(config, "confirm", inputs)
    required = ("confirm_summary.json",)
    if args.resume and progress.is_complete(required):
        return json.loads((progress.root / required[0]).read_text())
    variants = [(fmt, scope) for fmt in FORMATS for scope in CONFIRM_SCOPES]
    metrics, audits, inventory, load, ffn_hash = _run_matrix(
        args, config, splits["diagnosis_confirm"], progress, variants,
    )
    baseline_rows = progress.rows("bf16")
    comparisons = {}
    for variant in metrics:
        if variant == "bf16":
            continue
        rows = progress.rows(variant)
        comparisons[variant] = {
            "gate": screen_gate(metrics[variant], metrics["bf16"]),
            "paired_bootstrap_by_dataset": paired_bootstrap(
                baseline_rows, rows, samples=config.bootstrap_samples, seed=config.seed,
            ),
            "stratified_paired_bootstrap": stratified_paired_bootstrap(
                baseline_rows, rows, samples=config.bootstrap_samples, seed=config.seed,
            ),
            "prediction_transitions": prediction_transitions(baseline_rows, rows),
        }
    result = {
        "metrics": metrics, "comparisons": comparisons,
        "selection": select_candidate(comparisons),
        "quantization": audits, "inventory_counts": inventory["counts"],
        "load": load, "ffn_reference_sha256": ffn_hash,
    }
    atomic_json(progress.root / required[0], result)
    progress.mark_complete(required)
    return result


def stage_report(config, inputs):
    progress = _progress(config, "report", inputs)
    required = ("report_summary.json", "report_zh.md")
    if progress.is_complete(required):
        return json.loads((progress.root / required[0]).read_text())
    audit = json.loads((config.output_dir / "audit/audit_summary.json").read_text())
    screen = json.loads((config.output_dir / "screen/screen_summary.json").read_text())
    confirm = json.loads((config.output_dir / "confirm/confirm_summary.json").read_text())
    result = {
        "selection": confirm["selection"],
        "confirm_gates": {
            key: value["gate"] for key, value in confirm["comparisons"].items()
        },
        "labels": config.to_dict()["labels"],
    }
    atomic_json(progress.root / required[0], result)
    atomic_text(progress.root / required[1], render_report(audit, screen, confirm))
    progress.mark_complete(required)
    return result


def main() -> None:
    args = _args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    threads = _configure_threads(args.output_dir, args.source_rootcause_output)
    config, splits, inputs, split_audit = _protocol(args, threads)
    _require_prerequisite(config, args.stage)
    started = time.time()
    if args.stage == "audit":
        result = stage_audit(args, config, splits, inputs, split_audit)
    elif args.stage == "smoke":
        result = stage_smoke(args, config, splits, inputs)
    elif args.stage == "screen":
        result = stage_screen(args, config, splits, inputs)
    elif args.stage == "confirm":
        result = stage_confirm(args, config, splits, inputs)
    else:
        result = stage_report(config, inputs)
    print(json.dumps({
        "stage": args.stage,
        "complete": True,
        "elapsed_seconds": time.time() - started,
        "result_keys": sorted(result),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

