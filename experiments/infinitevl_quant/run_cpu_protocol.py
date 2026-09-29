from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

from experiments.vlm_quant_benchmark.data import sample_manifest_digest

from .config import MODEL_REVISION, ProtocolConfig, SCHEMES
from .cpu_backend import (
    CPUFusedRMSNormGated,
    CPUShortConvolution,
    FLA_REFERENCE_COMMIT,
    FLA_REFERENCE_SHA256,
    FLA_REFERENCE_URL,
    choose_torch_threads,
    chunk_gated_delta_rule,
    recurrent_gated_delta_rule,
)
from .data_protocol import build_splits, write_split_manifests
from .evaluation import evaluate_variant, screen_gate
from .inventory import build_inventory, select_scope
from .loader import (
    ensure_snapshot,
    environment_record,
    load_infinitevl_cpu,
    snapshot_provenance,
)
from .mbq_qig import (
    accept_qig_layers,
    run_mbq,
    run_qig_layers,
    run_vvs,
)
from .progress import StageProgress, atomic_json, file_sha256, json_digest
from .quantization import fp16_rounded_parameters, quantized_linears
from .reporting import build_report
from .state_probe import collect_states, compare_states, state_probe_token_count
from .statistics import paired_bootstrap, prediction_transitions


STAGES = ("preflight", "smoke", "scope", "screen", "qig", "confirm", "report")
STAGE_PREREQUISITES = {
    "smoke": ("preflight", ("preflight.json",)),
    "scope": ("smoke", ("smoke_summary.json",)),
    "screen": ("scope", ("scope_sensitivity.json",)),
    "qig": ("screen", ("screen_summary.json",)),
    "confirm": ("qig", ("qig_summary.json", "scales.pt")),
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/infinitevl_quant_cpu_v2"))
    parser.add_argument("--resume", action="store_true", help="reuse matching per-sample artifacts")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-weight-hash", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    return parser.parse_args()


def _protocol(args, threads: dict) -> tuple[ProtocolConfig, dict, dict]:
    config = ProtocolConfig(
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    splits = build_splits(config.calibration_manifest, config.evaluation_manifest)
    implementation_files = {
        str(path.relative_to(Path(__file__).parent)): file_sha256(path)
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }
    benchmark_root = (Path(__file__).parents[1] / "vlm_quant_benchmark").resolve()
    imported_benchmark_files = {}
    for module in tuple(sys.modules.values()):
        source = getattr(module, "__file__", None)
        if not source:
            continue
        path = Path(source)
        if path.suffix == ".pyc" and getattr(module, "__spec__", None) is not None:
            origin = getattr(module.__spec__, "origin", None)
            path = Path(origin) if origin else path
        try:
            resolved = path.resolve()
            relative = resolved.relative_to(benchmark_root)
        except (OSError, ValueError):
            continue
        if resolved.is_file() and resolved.suffix == ".py":
            imported_benchmark_files[str(relative)] = file_sha256(resolved)
    implementation_record = {
        "infinitevl_quant": implementation_files,
        "imported_vlm_quant_benchmark": dict(sorted(imported_benchmark_files.items())),
    }
    inputs = {
        "model_revision": MODEL_REVISION,
        "calibration_manifest": file_sha256(config.calibration_manifest),
        "evaluation_manifest": file_sha256(config.evaluation_manifest),
        "textvqa_reference": file_sha256(
            Path("data/vlm_quant_benchmark/LMUData/TextVQA_VAL_local.tsv")
        ),
        "splits": {
            name: sample_manifest_digest(samples) for name, samples in splits.items()
        },
        "implementation": json_digest(implementation_record),
        "thread_config": json_digest({
            "selected": threads["selected"],
            "interop_threads": threads["interop_threads"],
        }),
    }
    protocol_identity = {
        "format_version": 1, "config_digest": config.digest(),
        "input_digests": inputs,
    }
    identity_path = config.output_dir / "protocol_identity.json"
    if identity_path.is_file():
        existing = json.loads(identity_path.read_text())
        if existing != protocol_identity:
            raise RuntimeError(
                "output directory belongs to a different protocol/config; "
                "choose a new --output-dir"
            )
    else:
        atomic_json(identity_path, protocol_identity)
    atomic_json(
        config.output_dir / "protocol_config.json",
        config.to_dict() | {"config_digest": config.digest()},
    )
    manifests = write_split_manifests(config.output_dir / "protocol", splits)
    if {name: value["digest"] for name, value in manifests.items()} != inputs["splits"]:
        raise RuntimeError("written split manifest digest does not match protocol identity")
    atomic_json(config.output_dir / "protocol" / "splits.json", manifests)
    atomic_json(
        config.output_dir / "protocol" / "implementation_sha256.json",
        implementation_record | {"combined_sha256": inputs["implementation"]},
    )
    budget_path = config.output_dir / "wallclock.json"
    if not budget_path.exists():
        atomic_json(budget_path, {"started_unix": time.time(), "limit_hours": config.wallclock_hours})
    budget = json.loads(budget_path.read_text())
    elapsed_hours = (time.time() - budget["started_unix"]) / 3600.0
    if elapsed_hours > config.wallclock_hours and args.stage not in {"report"}:
        raise TimeoutError(f"protocol wall-clock budget exceeded: {elapsed_hours:.2f}h")
    return config, splits, inputs


def _require_prerequisite(config: ProtocolConfig, stage: str) -> None:
    requirement = STAGE_PREREQUISITES.get(stage)
    if requirement is None:
        return
    upstream, required_files = requirement
    root = config.output_dir / upstream
    marker = root / ".complete"
    if not marker.is_file() or any(not (root / name).is_file() for name in required_files):
        raise RuntimeError(
            f"stage {stage} requires completed stage {upstream}: {required_files}"
        )
    try:
        payload = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid completion marker for stage {upstream}") from error
    if payload.get("config_digest") != config.digest():
        raise RuntimeError(f"stage {upstream} belongs to a different protocol config")
    if tuple(payload.get("required", ())) != required_files:
        raise RuntimeError(f"stage {upstream} completion marker is incomplete")


def _configure_threads(output_dir: Path) -> dict:
    path = output_dir / "thread_config.json"
    if path.is_file():
        audit = json.loads(path.read_text())
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            if torch.get_num_interop_threads() != 1:
                raise
        torch.set_num_threads(int(audit["selected"]))
        return audit | {"reused": True}
    audit = choose_torch_threads()
    atomic_json(path, audit)
    return audit | {"reused": False}


def _model_path(args, config) -> Path:
    if args.model_path is not None:
        return args.model_path.resolve()
    return ensure_snapshot(
        config.model_id, config.model_revision, local_files_only=args.local_files_only,
    )


def _backend_parity() -> dict:
    generator = torch.Generator().manual_seed(123)
    q = torch.randn(2, 19, 3, 5, generator=generator)
    k = torch.randn(2, 19, 3, 5, generator=generator)
    v = torch.randn(2, 19, 3, 7, generator=generator)
    g = -torch.rand(2, 19, 3, generator=generator)
    beta = torch.rand(2, 19, 3, generator=generator)
    recurrent, recurrent_state = recurrent_gated_delta_rule(
        q, k, v, g, beta, output_final_state=True,
    )
    chunk, chunk_state = chunk_gated_delta_rule(
        q, k, v, g, beta, output_final_state=True, chunk_size=4,
    )
    delta_error = max(
        float((recurrent - chunk).abs().max()),
        float((recurrent_state - chunk_state).abs().max()),
    )
    conv = CPUShortConvolution(11, 4, activation="silu")
    x = torch.randn(2, 17, 11, generator=generator)
    one_shot, _ = conv(x)
    left, state = conv(x[:, :7], output_final_state=True)
    right, _ = conv(x[:, 7:], cache=state, output_final_state=True)
    conv_error = float((one_shot - torch.cat((left, right), dim=1)).abs().max())
    norm = CPUFusedRMSNormGated(13, eps=1e-6)
    value = torch.randn(2, 7, 13, generator=generator)
    gate = torch.randn(2, 7, 13, generator=generator)
    actual = norm(value, gate)
    expected = (
        value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + 1e-6)
        * norm.weight.float() * torch.nn.functional.silu(gate.float())
    )
    norm_error = float((actual - expected).abs().max())
    result = {
        "gated_delta_recurrent_chunk_max_error": delta_error,
        "short_convolution_one_shot_cached_max_error": conv_error,
        "rmsnorm_gate_max_error": norm_error,
        "passed": delta_error <= 1e-4 and conv_error <= 1e-5 and norm_error <= 1e-6,
    }
    if not result["passed"]:
        raise RuntimeError(f"CPU backend parity failed: {result}")
    return result


def _model_cache_parity(model, processor) -> dict:
    seed = processor.tokenizer(
        "CPU cache parity check for the pinned chunk reference.",
        add_special_tokens=False,
    )["input_ids"]
    if not seed:
        raise RuntimeError("tokenizer returned no IDs for cache parity seed")
    input_ids = torch.tensor((seed * (65 // len(seed) + 1))[:65]).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        full = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
        cached = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True)
        next_token = cached.logits[:, -1].argmax(dim=-1, keepdim=True)
        extended_ids = torch.cat((input_ids, next_token), dim=-1)
        extended_mask = torch.ones_like(extended_ids)
        recomputed = model(
            input_ids=extended_ids, attention_mask=extended_mask,
            use_cache=False, return_dict=True,
        )
        incremental = model(
            input_ids=next_token,
            attention_mask=extended_mask,
            past_key_values=cached.past_key_values,
            cache_position=torch.tensor([input_ids.shape[-1]], dtype=torch.long),
            use_cache=True,
            return_dict=True,
        )
    enabled_error = float((full.logits - cached.logits).abs().max())
    incremental_error = float(
        (recomputed.logits[:, -1] - incremental.logits[:, -1]).abs().max()
    )
    maximum = max(enabled_error, incremental_error)
    if maximum > 1e-4:
        raise RuntimeError(f"one-shot/cached-chunk logits parity exceeded 1e-4: {maximum}")
    return {
        "sequence_tokens": int(input_ids.shape[-1]),
        "prefill_backend": "pinned_fla_naive_chunk_reference",
        "incremental_backend": "differentiable_recurrent_reference",
        "one_shot_cache_enabled_logits_max_error": enabled_error,
        "one_shot_cached_chunk_logits_max_error": incremental_error,
        "incremental_argmax_equal": bool(
            torch.equal(
                recomputed.logits[:, -1].argmax(-1),
                incremental.logits[:, -1].argmax(-1),
            )
        ),
        "threshold": 1e-4,
    }


def stage_preflight(args, config, splits, inputs, threads):
    progress = StageProgress(config.output_dir, "preflight", config_digest=config.digest(), inputs=inputs)
    if args.resume and progress.is_complete(("preflight.json",)):
        return json.loads((progress.root / "preflight.json").read_text())
    diagnostics = {
        "status": "running", "last_completed_step": "initialized",
        "model_revision": MODEL_REVISION, "threads": threads,
        "environment": environment_record(),
        "fla_reference": {
            "commit": FLA_REFERENCE_COMMIT,
            "sha256": FLA_REFERENCE_SHA256,
            "url": FLA_REFERENCE_URL,
            "implementation": "ported_naive_chunk_plus_qk_l2norm_fp32_state",
        },
        "labels": ["cpu_reference_backend", "fake_quantized", "no_speed_or_memory_claim"],
    }
    diagnostics_path = progress.root / "diagnostics.json"
    atomic_json(diagnostics_path, diagnostics)

    def checkpoint(step: str, **values) -> None:
        diagnostics.update(values)
        diagnostics["last_completed_step"] = step
        atomic_json(diagnostics_path, diagnostics)

    try:
        model_path = _model_path(args, config)
        provenance = snapshot_provenance(
            model_path, hash_weights=not args.skip_weight_hash,
        )
        checkpoint("snapshot_verified", snapshot=provenance)
        parity = _backend_parity()
        checkpoint("backend_parity", backend_parity=parity)
        model, processor, load_audit = load_infinitevl_cpu(model_path)
        checkpoint("model_loaded", load=load_audit)
        _targets, inventory = build_inventory(model)
        checkpoint("inventory_verified", inventory=inventory)
        cache_parity = _model_cache_parity(model, processor)
        checkpoint("model_cache_parity", model_cache_parity=cache_parity)
        package_root = Path(__file__).parent
        patch_hashes = {
            path.name: file_sha256(path) for path in sorted(package_root.glob("*.py"))
        }
        result = {
            "model_revision": MODEL_REVISION, "snapshot": provenance,
            "cpu_patch_sha256": patch_hashes, "threads": threads,
            "environment": diagnostics["environment"], "backend_parity": parity,
            "model_cache_parity": cache_parity, "load": load_audit,
            "inventory": inventory, "fla_reference": diagnostics["fla_reference"],
            "labels": diagnostics["labels"],
        }
        atomic_json(progress.root / "preflight.json", result)
        diagnostics.update({"status": "complete", "last_completed_step": "complete"})
        atomic_json(diagnostics_path, diagnostics)
        progress.mark_complete(("preflight.json",))
        return result
    except Exception as error:
        diagnostics.update({
            "status": "failed", "failure_type": type(error).__name__,
            "failure_message": str(error),
        })
        atomic_json(diagnostics_path, diagnostics)
        raise


def _load_model(args, config):
    model_path = _model_path(args, config)
    model, processor, audit = load_infinitevl_cpu(model_path)
    targets, inventory = build_inventory(model)
    return model, processor, targets, inventory, audit


def stage_smoke(args, config, splits, inputs):
    progress = StageProgress(config.output_dir, "smoke", config_digest=config.digest(), inputs=inputs)
    if args.resume and progress.is_complete(("smoke_summary.json",)):
        return json.loads((progress.root / "smoke_summary.json").read_text())
    model, processor, _targets, _inventory, load_audit = _load_model(args, config)
    reference_dir = progress.root / "reference_artifacts"
    metrics = {}
    metrics["bf16"] = evaluate_variant(
        model, processor, splits["smoke"], progress=progress, variant="bf16",
        min_pixels=config.min_pixels, max_pixels=config.max_pixels,
        max_new_tokens=config.max_new_tokens, generate=True,
        reference_dir=reference_dir, write_reference=True,
        teacher_timeout_seconds=600, generation_timeout_seconds=1800,
    )
    with fp16_rounded_parameters(model) as rounding:
        metrics["fp16"] = evaluate_variant(
            model, processor, splits["smoke"], progress=progress, variant="fp16_rounded",
            min_pixels=config.min_pixels, max_pixels=config.max_pixels,
            max_new_tokens=config.max_new_tokens, generate=True,
            reference_dir=reference_dir,
            teacher_timeout_seconds=600, generation_timeout_seconds=1800,
        )
    if not all(math.isfinite(value["target_nll"]) for value in metrics.values()):
        raise FloatingPointError("smoke produced non-finite target NLL")
    result = {"metrics": metrics, "fp16_definition": rounding, "load": load_audit}
    atomic_json(progress.root / "smoke_summary.json", result)
    progress.mark_complete(("smoke_summary.json",))
    return result


def _mean_block_nmse(rows):
    values = [value for row in rows for value in row.get("block_nmse", {}).values()]
    return sum(values) / len(values) if values else math.nan


def stage_scope(args, config, splits, inputs):
    progress = StageProgress(config.output_dir, "scope", config_digest=config.digest(), inputs=inputs)
    if args.resume and progress.is_complete(("scope_sensitivity.json",)):
        return json.loads((progress.root / "scope_sensitivity.json").read_text())
    model, processor, targets, inventory, _audit = _load_model(args, config)
    reference_dir = progress.root / "reference_artifacts"
    baseline = evaluate_variant(
        model, processor, splits["acceptance"], progress=progress, variant="bf16",
        min_pixels=config.min_pixels, max_pixels=config.max_pixels,
        max_new_tokens=0, generate=False, reference_dir=reference_dir,
        write_reference=True, capture_hidden=True,
    )
    variants = {}
    for scope in ("ffn", "swa", "gdn_main", "gdn_dynamics", "decoder_all"):
        members = select_scope(targets, scope)
        with quantized_linears(model, members, SCHEMES["w4a16"]) as quant_audit:
            metrics = evaluate_variant(
                model, processor, splits["acceptance"], progress=progress,
                variant=f"{scope}_w4a16", min_pixels=config.min_pixels,
                max_pixels=config.max_pixels, max_new_tokens=0, generate=False,
                reference_dir=reference_dir, capture_hidden=True,
            )
        rows = progress.rows(f"{scope}_w4a16")
        parameters = sum(member.parameter_count for member in members)
        variants[scope] = metrics | {
            "nll_relative_increase": metrics["target_nll"] / baseline["target_nll"] - 1.0,
            "mean_block_nmse": _mean_block_nmse(rows),
            "parameters": parameters,
            "parameter_fraction": parameters / inventory["total_model_parameters"],
            "quantization": quant_audit,
        }
    result = {"baseline": baseline, "variants": variants, "inventory": inventory}
    atomic_json(progress.root / "scope_sensitivity.json", result)
    progress.mark_complete(("scope_sensitivity.json",))
    return result


def stage_screen(args, config, splits, inputs):
    progress = StageProgress(config.output_dir, "screen", config_digest=config.digest(), inputs=inputs)
    if args.resume and progress.is_complete(("screen_summary.json",)):
        return json.loads((progress.root / "screen_summary.json").read_text())
    model, processor, targets, _inventory, _audit = _load_model(args, config)
    safe = select_scope(targets, "decoder_safe")
    reference_dir = progress.root / "reference_artifacts"
    metrics = {}
    metrics["bf16"] = evaluate_variant(
        model, processor, splits["screen"], progress=progress, variant="bf16",
        min_pixels=config.min_pixels, max_pixels=config.max_pixels,
        max_new_tokens=config.max_new_tokens, generate=True,
        reference_dir=reference_dir, write_reference=True,
    )
    with fp16_rounded_parameters(model):
        metrics["fp16"] = evaluate_variant(
            model, processor, splits["screen"], progress=progress, variant="fp16_rounded",
            min_pixels=config.min_pixels, max_pixels=config.max_pixels,
            max_new_tokens=config.max_new_tokens, generate=True,
            reference_dir=reference_dir,
        )
    for key in ("w4a16", "w4a8", "w3a16", "w2a16"):
        with quantized_linears(model, safe, SCHEMES[key]):
            metrics[key] = evaluate_variant(
                model, processor, splits["screen"], progress=progress, variant=key,
                min_pixels=config.min_pixels, max_pixels=config.max_pixels,
                max_new_tokens=config.max_new_tokens, generate=True,
                reference_dir=reference_dir,
            )
    gates = {key: screen_gate(metrics[key], metrics["bf16"]) for key in (
        "w2a16", "w3a16", "w4a8", "w4a16",
    )}
    candidate = next((key for key in ("w2a16", "w3a16", "w4a8", "w4a16") if gates[key]["passed"]), "w4a16")
    fallback = not any(gate["passed"] for gate in gates.values())
    result = {
        "metrics": metrics, "gates": gates, "candidate": candidate,
        "candidate_scheme": SCHEMES[candidate].to_dict(), "fallback_to_w4": fallback,
    }
    atomic_json(progress.root / "screen_summary.json", result)
    progress.mark_complete(("screen_summary.json",))
    return result


def _load_screen_candidate(config) -> str:
    path = config.output_dir / "screen" / "screen_summary.json"
    if not path.is_file():
        raise FileNotFoundError("run --stage screen before --stage qig")
    return json.loads(path.read_text())["candidate"]


def _save_scale_bundle(path, candidate, mbq, qig, audit):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({
        "format_version": 1, "candidate": candidate,
        "mbq_scales": mbq, "qig_scales": qig, "audit": audit,
    }, temporary)
    temporary.replace(path)


def stage_qig(args, config, splits, inputs):
    progress = StageProgress(config.output_dir, "qig", config_digest=config.digest(), inputs=inputs)
    if args.resume and progress.is_complete(("qig_summary.json", "scales.pt")):
        return json.loads((progress.root / "qig_summary.json").read_text())
    candidate = _load_screen_candidate(config)
    scheme = SCHEMES[candidate]
    model, processor, targets, _inventory, _audit = _load_model(args, config)
    safe = select_scope(targets, "decoder_safe")
    mbq, cache_hit = run_mbq(
        model, processor, safe, splits["mbq"], scheme,
        artifact=progress.root / "mbq_search.pt", min_pixels=config.min_pixels,
        max_pixels=config.max_pixels, seed=config.seed,
    )
    vvs_path = progress.root / "vvs.json"
    if vvs_path.is_file():
        gate = json.loads(vvs_path.read_text())
        layers = [int(value) for value in gate["selected"]]
    else:
        layers, gate = run_vvs(
            model, processor, safe, splits["vvs"], scheme, mbq,
            min_pixels=config.min_pixels, max_pixels=config.max_pixels,
            relative_threshold=config.vvs_threshold, max_layers=config.vvs_max_layers,
            bootstrap_samples=config.vvs_bootstrap, seed=config.seed,
        )
        atomic_json(vvs_path, gate)
    raw_path = progress.root / "qig_raw.pt"
    if raw_path.is_file():
        payload = torch.load(raw_path, map_location="cpu", weights_only=True)
        qig_raw, qig_audit = payload["scales"], payload["audit"]
    else:
        qig_raw, qig_audit = run_qig_layers(
            model, processor, safe, splits["qig"], scheme, mbq, layers,
            min_pixels=config.min_pixels, max_pixels=config.max_pixels,
            step_grid=config.qig_steps, alpha_grid=config.qig_alphas,
            spearman_threshold=config.qig_spearman,
            jaccard_threshold=config.qig_jaccard,
            budget_hours=config.qig_budget_hours, seed=config.seed,
        )
        temporary = raw_path.with_suffix(".tmp")
        torch.save({"scales": qig_raw, "audit": qig_audit}, temporary)
        temporary.replace(raw_path)
    ordered = sorted(
        [int(value) for value in qig_audit.get("effective_layers", [])],
        key=lambda layer: -float(gate.get("scores", {}).get(str(layer), gate.get("scores", {}).get(layer, 0))),
    )
    final_scales, acceptance = accept_qig_layers(
        model, processor, safe, splits["acceptance"], scheme,
        mbq.scales, qig_raw, ordered, min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
    )
    audit = {
        "mbq": mbq.audit, "mbq_cache_hit": cache_hit,
        "vulnerability_gate": gate, "qig": qig_audit,
        "acceptance": acceptance,
    }
    _save_scale_bundle(
        progress.root / "scales.pt", candidate, mbq.scales, final_scales, audit,
    )
    result = {
        "candidate": candidate, "selected_layers": layers,
        "accepted_layers": [int(key) for key, value in acceptance["layers"].items() if value["accepted"]],
        "mbq_cache_hit": cache_hit, "qig_skipped": qig_audit.get("skipped", False),
        "audit": audit,
    }
    atomic_json(progress.root / "qig_summary.json", result)
    progress.mark_complete(("qig_summary.json", "scales.pt"))
    return result


def _load_scales(config):
    path = config.output_dir / "qig" / "scales.pt"
    if not path.is_file():
        raise FileNotFoundError("run --stage qig before --stage confirm")
    return torch.load(path, map_location="cpu", weights_only=True)


def _aggregate_state(records):
    if not records:
        raise ValueError("cannot aggregate an empty state-probe record set")
    by_prefix = defaultdict(list)
    aucs = []
    for record in records:
        aucs.append(record["state_drift_auc"])
        for prefix, metrics in record["by_prefix"].items():
            by_prefix[prefix].append(metrics)
    return {
        "state_drift_auc": sum(aucs) / len(aucs),
        "by_prefix": {
            prefix: {key: sum(value[key] for value in values) / len(values) for key in values[0]}
            for prefix, values in sorted(by_prefix.items(), key=lambda item: int(item[0]))
        },
    }


def _run_state_probe(model, processor, samples, safe, scheme, bundle, config, progress):
    output_path = progress.root / "state_probe.json"
    if output_path.is_file():
        return json.loads(output_path.read_text())
    records = defaultdict(list)
    selected_samples = []
    for sample, expected_tokens in samples:
        cached = progress.completed_sample("state_probe", sample.id)
        if cached is not None:
            if (
                cached.get("dataset") != sample.dataset
                or int(cached.get("sequence_tokens", -1)) != expected_tokens
                or set(cached.get("comparisons", {})) != {"rtn", "mbq", "vg_qig"}
            ):
                raise RuntimeError(f"invalid resumed state-probe artifact for {sample.id}")
            for name, comparison in cached["comparisons"].items():
                records[name].append(comparison)
            selected_samples.append({
                "id": sample.id, "dataset": sample.dataset,
                "sequence_tokens": expected_tokens, "resumed": True,
            })
            continue
        reference, tokens = collect_states(
            model, processor, sample, min_pixels=config.min_pixels, max_pixels=config.max_pixels,
        )
        if tokens != expected_tokens:
            raise RuntimeError(
                f"processor token count changed for {sample.id}: {expected_tokens} -> {tokens}"
            )
        contexts = {
            "rtn": lambda: quantized_linears(model, safe, scheme),
            "mbq": lambda: quantized_linears(model, safe, scheme, scales=bundle["mbq_scales"]),
            "vg_qig": lambda: quantized_linears(model, safe, scheme, scales=bundle["qig_scales"]),
        }
        sample_comparisons = {}
        for name, factory in contexts.items():
            with factory():
                candidate, _ = collect_states(
                    model, processor, sample, min_pixels=config.min_pixels,
                    max_pixels=config.max_pixels,
                )
            comparison = compare_states(reference, candidate)
            comparison.update({"id": sample.id, "dataset": sample.dataset, "sequence_tokens": tokens})
            records[name].append(comparison)
            sample_comparisons[name] = comparison
        progress.complete_sample("state_probe", sample.id, {
            "id": sample.id, "dataset": sample.dataset, "sequence_tokens": tokens,
            "comparisons": sample_comparisons,
        })
        selected_samples.append({
            "id": sample.id, "dataset": sample.dataset,
            "sequence_tokens": tokens, "resumed": False,
        })
    if len(selected_samples) != 4 or len({row["dataset"] for row in selected_samples}) != 4:
        raise RuntimeError("state probe requires exactly one sample from each of four datasets")
    result = {
        "prefixes": [64, 128, 256, 512],
        "selected_samples": selected_samples,
        "samples": {name: values for name, values in records.items()},
        "aggregate": {name: _aggregate_state(values) for name, values in records.items()},
    }
    atomic_json(output_path, result)
    return result


def stage_confirm(args, config, splits, inputs):
    progress = StageProgress(config.output_dir, "confirm", config_digest=config.digest(), inputs=inputs)
    if args.resume and progress.is_complete(("confirm_summary.json", "state_probe.json")):
        return json.loads((progress.root / "confirm_summary.json").read_text())
    bundle = _load_scales(config)
    candidate, scheme = bundle["candidate"], SCHEMES[bundle["candidate"]]
    model, processor, targets, _inventory, _audit = _load_model(args, config)
    safe = select_scope(targets, "decoder_safe")
    reference_dir = progress.root / "reference_artifacts"
    metrics = {}
    metrics["bf16"] = evaluate_variant(
        model, processor, splits["confirm"], progress=progress, variant="bf16",
        min_pixels=config.min_pixels, max_pixels=config.max_pixels,
        max_new_tokens=config.max_new_tokens, generate=True,
        reference_dir=reference_dir, write_reference=True,
    )
    contexts = {
        "rtn": lambda: quantized_linears(model, safe, scheme),
        "mbq": lambda: quantized_linears(model, safe, scheme, scales=bundle["mbq_scales"]),
        "vg_qig": lambda: quantized_linears(model, safe, scheme, scales=bundle["qig_scales"]),
    }
    for name, factory in contexts.items():
        with factory():
            metrics[name] = evaluate_variant(
                model, processor, splits["confirm"], progress=progress, variant=name,
                min_pixels=config.min_pixels, max_pixels=config.max_pixels,
                max_new_tokens=config.max_new_tokens, generate=True,
                reference_dir=reference_dir,
            )
    bf16_rows = progress.rows("bf16")
    comparisons = {}
    for name in contexts:
        rows = progress.rows(name)
        comparisons[name] = {
            "paired_bootstrap": paired_bootstrap(
                bf16_rows, rows, samples=config.bootstrap_samples, seed=config.seed,
            ),
            "prediction_transitions": prediction_transitions(bf16_rows, rows),
        }
    probe_samples = []
    seen = set()
    for sample in splits["confirm"]:
        if sample.dataset in seen:
            continue
        tokens = state_probe_token_count(
            processor, sample, min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
        )
        if tokens >= 512:
            probe_samples.append((sample, tokens))
            seen.add(sample.dataset)
    expected_datasets = {sample.dataset for sample in splits["confirm"]}
    if len(expected_datasets) != 4 or seen != expected_datasets:
        missing = sorted(expected_datasets - seen)
        raise RuntimeError(
            "confirm split does not contain a >=512-token state-probe sample "
            f"for each dataset; missing={missing}"
        )
    state = _run_state_probe(
        model, processor, probe_samples, safe, scheme, bundle, config, progress,
    )
    result = {
        "candidate": candidate, "metrics": metrics,
        "comparisons_to_bf16": comparisons,
        "state_probe_summary": state["aggregate"],
    }
    atomic_json(progress.root / "confirm_summary.json", result)
    progress.mark_complete(("confirm_summary.json", "state_probe.json"))
    return result


def stage_report(args, config, inputs):
    progress = StageProgress(
        config.output_dir, "report", config_digest=config.digest(), inputs=inputs,
    )
    required = ("report_summary.json",)
    upstream = {}
    for relative in (
        "preflight/preflight.json", "preflight/diagnostics.json",
        "smoke/smoke_summary.json",
        "scope/scope_sensitivity.json",
        "screen/screen_summary.json", "qig/qig_summary.json",
        "confirm/confirm_summary.json", "confirm/state_probe.json",
    ):
        path = config.output_dir / relative
        upstream[relative] = file_sha256(path) if path.is_file() else "missing"
    if args.resume and progress.is_complete(required):
        cached = json.loads((progress.root / required[0]).read_text())
        if cached.get("report_input_sha256") == upstream:
            return cached
    result = build_report(config.output_dir)
    result["report_input_sha256"] = upstream
    atomic_json(progress.root / required[0], result)
    progress.mark_complete(required)
    return result


def main() -> None:
    args = _args()
    threads = _configure_threads(args.output_dir)
    config, splits, inputs = _protocol(args, threads)
    _require_prerequisite(config, args.stage)
    if args.stage == "preflight":
        result = stage_preflight(args, config, splits, inputs, threads)
    elif args.stage == "smoke":
        result = stage_smoke(args, config, splits, inputs)
    elif args.stage == "scope":
        result = stage_scope(args, config, splits, inputs)
    elif args.stage == "screen":
        result = stage_screen(args, config, splits, inputs)
    elif args.stage == "qig":
        result = stage_qig(args, config, splits, inputs)
    elif args.stage == "confirm":
        result = stage_confirm(args, config, splits, inputs)
    else:
        result = stage_report(args, config, inputs)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
