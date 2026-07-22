"""Run CPU Qwen2.5-VL full-model Linear fake-quantization ablations."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig

from experiments.ffn_quant.data import generate_tiny_dataset, load_manifest, select_samples
from experiments.ffn_quant.modeling import build_teacher_forced_inputs, load_qwen_cpu

from .awq import AWQResult, ActivationCache, apply_awq_, search_awq
from .evaluation import (
    create_reference_cache,
    evaluate_variant,
    reference_logits_metrics,
    reference_metrics,
)
from .modeling import (
    CONFIGS,
    FFNTarget,
    GROUPS,
    LinearTarget,
    build_inventory,
    config_assignment,
    discover_full_model,
    make_fp16_weight_reference_,
)
from .outputs import atomic_json, atomic_jsonl, write_report, write_summary
from .quantization import ParameterSnapshot, WeightStats, fingerprint, quantize_linear_

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preset", choices=("smoke", "cpu"), default="smoke")
    parser.add_argument("--reference", choices=("fp16_rounded_fp32_compute",), default="fp16_rounded_fp32_compute")
    parser.add_argument("--configs", help="comma-separated fixed configuration names")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.group_size <= 0 or args.max_new_tokens <= 0:
        parser.error("group size and max new tokens must be positive")
    if args.configs:
        args.configs = [value.strip() for value in args.configs.split(",") if value.strip()]
        invalid = sorted(set(args.configs) - set(CONFIGS))
        if invalid:
            parser.error(f"unsupported configs: {invalid}")
    return args


def _variant_configs(args: argparse.Namespace) -> list[str]:
    if args.configs:
        return [name for name in args.configs if name != "fp16_weight_reference"]
    if args.preset == "smoke":
        return ["rtn_w4_all_linear", "hybrid_awq_w4_all_linear"]
    return [name for name in CONFIGS if name != "fp16_weight_reference"]


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _environment() -> dict[str, Any]:
    packages = {}
    for name in ("torch", "torchvision", "transformers", "qwen-vl-utils", "Pillow", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cpu_count": os.cpu_count(),
        "cuda_available": torch.cuda.is_available(),
    }


def _atomic_torch(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


@torch.inference_mode()
def _collect_calibration(model: torch.nn.Module, processor: Any, samples: list[Any], ffns: list[FFNTarget]) -> ActivationCache:
    cache = ActivationCache(ffns)
    with cache:
        for sample in samples:
            inputs, _mask = build_teacher_forced_inputs(processor, sample)
            model(**inputs, use_cache=False, output_hidden_states=False, return_dict=True)
    return cache


def _serialise_awq(result: AWQResult) -> dict[str, Any]:
    return {
        "metrics": result.metrics_dict(),
        "input_scale": result.input_scale,
        "internal_scale": result.internal_scale,
    }


def _deserialise_awq(value: dict[str, Any]) -> AWQResult:
    metrics = value["metrics"]
    return AWQResult(
        metrics["domain"], metrics["layer"], value["input_scale"], value["internal_scale"],
        metrics["rtn_normalized_mse"], metrics["awq_normalized_mse"], metrics["equivalence_max_error"],
        metrics["selected_input"], metrics["selected_internal"], metrics["fallback_reason"],
        metrics["input_search"], metrics["internal_search"], metrics["elapsed_seconds"],
    )


def _aggregate_weight(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def aggregate(selected: list[dict[str, Any]]) -> dict[str, Any]:
        count = sum(row["parameters"] for row in selected)
        error = sum(row["mse"] * row["parameters"] for row in selected)
        signal = sum((row["mse"] * row["parameters"] / row["normalized_mse"]) if row["normalized_mse"] else 0.0 for row in selected)
        nmse = error / max(signal, 1e-30)
        return {
            "modules": len(selected),
            "parameters": count,
            "mse": error / max(count, 1),
            "normalized_mse": nmse,
            "sqnr_db": float("inf") if error == 0 else 10.0 * __import__("math").log10(max(signal, 1e-30) / error),
            "max_abs_error": max((row["max_abs_error"] for row in selected), default=0.0),
        }
    return {
        "overall": aggregate(rows),
        "by_group": {group: aggregate([row for row in rows if row["group"] == group]) for group in GROUPS},
        "per_module": rows,
    }


def _write_all(
    output: Path,
    results: list[dict[str, Any]],
    generation_rows: list[dict[str, Any]],
    inventory: dict[str, Any],
    run_config: dict[str, Any],
    conversion: dict[str, Any],
    awq_metrics: list[dict[str, Any]],
) -> None:
    atomic_json(output / "metrics.json", {result["name"]: result for result in results})
    atomic_jsonl(output / "generations.jsonl", generation_rows)
    write_summary(output / "summary.csv", results)
    write_report(output / "report.md", results, inventory, run_config, conversion, awq_metrics)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.manifest or generate_tiny_dataset(args.output_dir / "tiny_tabletop", seed=args.seed)
    samples = load_manifest(manifest)
    calibration_count, evaluation_count = (2, 2) if args.preset == "smoke" else (8, 16)
    calibration = select_samples(samples, split="calibration", limit=calibration_count)
    evaluation = select_samples(samples, split="evaluation", limit=evaluation_count)
    variants = _variant_configs(args)
    identity = {
        "model_path": args.model_path,
        "reference": args.reference,
        "preset": args.preset,
        "manifest_sha256": _file_hash(Path(manifest)),
        "calibration_samples": [sample.id for sample in calibration],
        "evaluation_samples": [sample.id for sample in evaluation],
        "configs": variants,
        "group_size": args.group_size,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
    }
    run_hash = _hash(identity)
    config_path = args.output_dir / "config.json"
    if config_path.is_file():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous.get("run_hash") != run_hash:
            raise RuntimeError("output directory belongs to a different run configuration")
    run_config = {**identity, "run_hash": run_hash, "environment": _environment()}
    atomic_json(config_path, run_config)

    checkpoint_config = AutoConfig.from_pretrained(args.model_path)
    checkpoint_dtype = getattr(checkpoint_config, "dtype", None)
    if checkpoint_dtype is None and hasattr(checkpoint_config, "text_config"):
        checkpoint_dtype = getattr(checkpoint_config.text_config, "dtype", None)
    print(f"Loading {args.model_path} on CPU in FP32...", flush=True)
    model, processor = load_qwen_cpu(args.model_path)
    conversion = make_fp16_weight_reference_(model)
    conversion["checkpoint_declared_dtype"] = str(checkpoint_dtype or "bfloat16")
    atomic_json(args.output_dir / "reference_conversion.json", conversion)
    targets, ffns = discover_full_model(model, strict_qwen3b=True)
    inventory = build_inventory(model, targets, ffns)
    atomic_json(args.output_dir / "module_inventory.json", inventory)
    print(f"Selected {len(targets)} Linear modules with {inventory['target_parameters']:,} weights.", flush=True)

    mutable: list[tuple[str, torch.nn.Parameter]] = [(f"{target.name}.weight", target.module.weight) for target in targets]
    for ffn in ffns:
        mutable.extend(ffn.named_scale_parameters())
    snapshot = ParameterSnapshot(mutable)
    mutable_ids = {id(parameter) for _name, parameter in snapshot.parameters}
    non_target = fingerprint(model.named_parameters(), mutable_ids)

    reference_dir = args.output_dir / "reference_cache"
    reference_generations = create_reference_cache(
        model, processor, evaluation, ffns, reference_dir,
        max_new_tokens=args.max_new_tokens, reuse=args.resume,
    )
    logits = reference_logits_metrics(reference_dir, evaluation)
    reference_generation_rows = [{**row, "config": "fp16_weight_reference", "candidate": row["reference"]} for row in reference_generations]
    results = [{
        "name": "fp16_weight_reference",
        "bits": "fp16_weight_values",
        "config_hash": run_hash,
        "quantized_linears": 0,
        "awq_linears": 0,
        "weight_metrics": _aggregate_weight([]),
        "metrics": reference_metrics(reference_generations, logits),
    }]
    all_generation_rows = reference_generation_rows[:]

    needs_awq = any("awq" in name for name in variants)
    awq_results: dict[tuple[str, int], AWQResult] = {}
    awq_path = args.output_dir / "awq_scales.pt"
    if needs_awq and args.resume and awq_path.is_file():
        raw = torch.load(awq_path, map_location="cpu", weights_only=False)
        awq_results = {tuple(key.split(":")): value for key, value in raw.items()}  # type: ignore[assignment]
        awq_results = {(domain, int(index)): _deserialise_awq(value) for (domain, index), value in awq_results.items()}
        print(f"Loaded {len(awq_results)} cached AWQ scales.", flush=True)
    elif needs_awq:
        search_ffns = ffns if args.preset == "cpu" else [next(ffn for ffn in reversed(ffns) if ffn.domain == domain) for domain in ("vision", "decoder")]
        print(f"Collecting calibration activations for {len(search_ffns)} FFNs...", flush=True)
        cache = _collect_calibration(model, processor, calibration, search_ffns)
        for position, ffn in enumerate(search_ffns, 1):
            result = search_awq(
                ffn,
                cache.reconstruction_inputs(ffn),
                cache.mean_abs(ffn, "input"),
                cache.mean_abs(ffn, "down"),
                group_size=args.group_size,
            )
            awq_results[(ffn.domain, ffn.index)] = result
            outcome = result.fallback_reason or f"nmse={result.awq_nmse:.6g}"
            print(f"AWQ {position}/{len(search_ffns)} {ffn.domain}.{ffn.index}: {outcome}", flush=True)
        _atomic_torch({f"{domain}:{index}": _serialise_awq(value) for (domain, index), value in awq_results.items()}, awq_path)
    awq_metrics = [value.metrics_dict() for value in awq_results.values()]
    atomic_json(args.output_dir / "awq_search.json", awq_metrics)
    _write_all(args.output_dir, results, all_generation_rows, inventory, run_config, conversion, awq_metrics)

    target_by_name = {target.name: target for target in targets}
    for name in variants:
        result_path = args.output_dir / name / "metrics.json"
        config_hash = _hash({**identity, "name": name})
        if args.resume and result_path.is_file():
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            if existing.get("config_hash") == config_hash:
                print(f"Skipping completed {name}", flush=True)
                results.append(existing)
                generation_path = args.output_dir / name / "generations.jsonl"
                if generation_path.is_file():
                    all_generation_rows.extend(json.loads(line) for line in generation_path.read_text().splitlines() if line)
                _write_all(args.output_dir, results, all_generation_rows, inventory, run_config, conversion, awq_metrics)
                continue
        snapshot.restore()
        snapshot.assert_restored()
        assignment = config_assignment(name, targets)
        started = time.monotonic()
        weight_rows: list[dict[str, Any]] = []
        applied_awq: set[str] = set()
        try:
            for ffn in ffns:
                names = [next(target.name for target in targets if target.module is linear) for linear in ffn.linears]
                if not all(assignment[module_name] == "awq_w4" for module_name in names):
                    continue
                awq = awq_results.get((ffn.domain, ffn.index))
                if awq is None:
                    # Smoke validates only the last FFN per domain; all other FFNs safely use RTN W4.
                    for module_name in names:
                        assignment[module_name] = "rtn_w4"
                    continue
                awq_weight_stats = apply_awq_(ffn, awq, group_size=args.group_size)
                applied_awq.update(names)
                for module_name, stats in zip(names, awq_weight_stats):
                    target = target_by_name[module_name]
                    weight_rows.append({
                        "name": module_name, "group": target.group, "mode": "awq_w4", "parameters": target.module.weight.numel(),
                        **stats.to_dict(),
                    })
            for target in targets:
                mode = assignment[target.name]
                if mode == "reference" or target.name in applied_awq:
                    continue
                bits = 8 if mode == "rtn_w8" else 4
                stats = quantize_linear_(target.module, bits=bits, group_size=args.group_size)
                weight_rows.append({"name": target.name, "group": target.group, "mode": mode, "parameters": target.module.weight.numel(), **stats.to_dict()})

            metrics, generated = evaluate_variant(
                model, processor, evaluation, ffns, reference_dir, reference_generations,
                max_new_tokens=args.max_new_tokens,
            )
            result = {
                "name": name,
                "bits": 8 if name == "rtn_w8_all_linear" else 4,
                "config_hash": config_hash,
                "quantized_linears": len(weight_rows),
                "awq_linears": len(applied_awq),
                "elapsed_seconds": time.monotonic() - started,
                "weight_metrics": _aggregate_weight(weight_rows),
                "metrics": metrics,
            }
            atomic_json(result_path, result)
            config_generations = [{**row, "config": name} for row in generated]
            atomic_jsonl(args.output_dir / name / "generations.jsonl", config_generations)
            all_generation_rows.extend(config_generations)
            results.append(result)
            _write_all(args.output_dir, results, all_generation_rows, inventory, run_config, conversion, awq_metrics)
            print(f"Completed {name} in {result['elapsed_seconds']:.1f}s", flush=True)
        except Exception as exc:
            atomic_json(args.output_dir / name / "error.json", {"name": name, "type": type(exc).__name__, "message": str(exc), "config_hash": config_hash})
            if isinstance(exc, (MemoryError, torch.OutOfMemoryError)):
                print("OOM: use --preset smoke or select fewer --configs; parameters were not changed automatically.", flush=True)
            raise
        finally:
            snapshot.restore()
            snapshot.assert_restored()
            if fingerprint(model.named_parameters(), mutable_ids) != non_target:
                raise RuntimeError(f"non-target parameter changed during {name}")

    print(f"Results written to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
