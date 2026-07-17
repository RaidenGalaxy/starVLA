"""CLI for CPU Qwen2.5-VL decoder-FFN fake-quantization ablations."""

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

from .awq import search_awq_layer
from .data import generate_tiny_dataset, load_manifest, select_samples
from .evaluation import baseline_metrics, collect_calibration_activations, create_reference_cache, evaluate_variant
from .modeling import (
    PROJECTION_NAMES,
    build_module_inventory,
    discover_decoder_ffns,
    load_qwen_cpu,
    resolve_scope,
)
from .outputs import atomic_json, write_report, write_summary_csv
from .quantization import ParameterSnapshot, parameter_fingerprint, quantize_linear_

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
METHODS = ("rtn_w8", "rtn_w4", "awq_w4")
SCOPES = ("last1", "last4", "all")


def _parse_csv(value: str | None, allowed: tuple[str, ...]) -> list[str] | None:
    if value is None:
        return None
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(parsed) - set(allowed))
    if invalid:
        raise argparse.ArgumentTypeError(f"unsupported values {invalid}; choose from {allowed}")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preset", choices=("smoke", "cpu"), default="smoke")
    parser.add_argument("--methods", help="comma-separated: rtn_w8,rtn_w4,awq_w4")
    parser.add_argument("--scopes", help="comma-separated: last1,last4,all")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    args.methods = _parse_csv(args.methods, METHODS)
    args.scopes = _parse_csv(args.scopes, SCOPES)
    if args.group_size <= 0:
        parser.error("--group-size must be positive")
    return args


def _default_experiments(preset: str) -> list[tuple[str, str]]:
    if preset == "smoke":
        return [("rtn_w4", "last1"), ("awq_w4", "last1")]
    return [
        ("rtn_w8", "all"),
        ("rtn_w4", "last1"),
        ("rtn_w4", "last4"),
        ("rtn_w4", "all"),
        ("awq_w4", "last1"),
        ("awq_w4", "last4"),
        ("awq_w4", "all"),
    ]


def _experiments(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.methods is None and args.scopes is None:
        return _default_experiments(args.preset)
    methods = args.methods or list(METHODS)
    scopes = args.scopes or list(SCOPES)
    return [(method, scope) for method in methods for scope in scopes]


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configuration_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def _environment() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ("torch", "torchvision", "transformers", "qwen-vl-utils", "Pillow", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": os.cpu_count(),
    }


def _atomic_torch_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _write_outputs(output_dir: Path, results: list[dict[str, Any]], inventory: dict[str, Any]) -> None:
    atomic_json(output_dir / "metrics.json", {result["name"]: result for result in results})
    write_summary_csv(output_dir / "summary.csv", results)
    write_report(output_dir / "report.md", results, inventory)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = args.manifest or generate_tiny_dataset(args.output_dir / "tiny_tabletop", seed=args.seed)
    samples = load_manifest(manifest)
    calibration_limit, evaluation_limit = (2, 2) if args.preset == "smoke" else (8, 16)
    calibration_samples = select_samples(samples, split="calibration", limit=calibration_limit)
    evaluation_samples = select_samples(samples, split="evaluation", limit=evaluation_limit)
    experiment_pairs = _experiments(args)

    run_identity = {
        "model_path": args.model_path,
        "manifest_sha256": _hash_file(Path(manifest)),
        "preset": args.preset,
        "group_size": args.group_size,
        "seed": args.seed,
        "calibration_samples": [sample.id for sample in calibration_samples],
        "evaluation_samples": [sample.id for sample in evaluation_samples],
        "experiments": experiment_pairs,
    }
    run_hash = _configuration_hash(run_identity)
    existing_config = args.output_dir / "config.json"
    if existing_config.is_file():
        previous = json.loads(existing_config.read_text(encoding="utf-8"))
        if previous.get("run_hash") != run_hash:
            raise RuntimeError("output directory contains a different run; choose a new --output-dir")
    atomic_json(existing_config, {**run_identity, "run_hash": run_hash, "environment": _environment()})

    print(f"Loading {args.model_path} on CPU in FP32...", flush=True)
    model, processor = load_qwen_cpu(args.model_path)
    entries = discover_decoder_ffns(model, strict_qwen3b=True)
    inventory = build_module_inventory(model, entries)
    if inventory["linear_modules"] != 108 or inventory["target_parameters"] != 2_434_793_472:
        raise RuntimeError(f"unexpected Qwen2.5-VL-3B inventory: {inventory['linear_modules']} modules, {inventory['target_parameters']} weights")
    atomic_json(args.output_dir / "module_inventory.json", inventory)
    print(
        f"Selected {inventory['linear_modules']} FFN Linear modules, "
        f"{inventory['target_parameters']:,} weights ({inventory['target_fraction']:.2%} of model).",
        flush=True,
    )

    all_mutable = [parameter for entry in entries for parameter in entry.named_mutable_parameters()]
    snapshot = ParameterSnapshot(all_mutable)
    mutable_ids = {id(parameter) for _name, parameter in all_mutable}
    non_target_before = parameter_fingerprint(model.named_parameters(), excluded_ids=mutable_ids)

    reference_dir = args.output_dir / "reference_cache"
    reference_logits = create_reference_cache(
        model, processor, evaluation_samples, entries, reference_dir, reuse=args.resume
    )
    results: list[dict[str, Any]] = [
        {
            "name": "original_reference",
            "method": "original",
            "scope": "none",
            "bits": "original",
            "config_hash": run_hash,
            "metrics": baseline_metrics(reference_logits, entries),
        }
    ]
    _write_outputs(args.output_dir, results, inventory)

    needs_awq = any(method == "awq_w4" for method, _scope in experiment_pairs)
    activation_cache = None
    if needs_awq:
        print("Collecting bounded calibration activations...", flush=True)
        activation_cache = collect_calibration_activations(model, processor, calibration_samples, entries)

    awq_scales: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    for method, scope in experiment_pairs:
        name = f"{method}_{scope}"
        result_path = args.output_dir / name / "metrics.json"
        config_hash = _configuration_hash({**run_identity, "method": method, "scope": scope})
        if args.resume and result_path.is_file():
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            if existing.get("config_hash") == config_hash:
                print(f"Skipping completed {name}", flush=True)
                results.append(existing)
                _write_outputs(args.output_dir, results, inventory)
                continue

        snapshot.restore()
        snapshot.assert_restored()
        selected = resolve_scope(entries, scope)
        started = time.monotonic()
        weight_stats: list[dict[str, Any]] = []
        awq_metrics: list[dict[str, Any]] = []
        try:
            if method.startswith("rtn_"):
                bits = 8 if method == "rtn_w8" else 4
                for entry in selected:
                    for projection_name, linear in zip(PROJECTION_NAMES, entry.linears):
                        stats = quantize_linear_(linear, bits=bits, group_size=args.group_size)
                        weight_stats.append(
                            {"layer": entry.index, "projection": projection_name, **stats.to_dict()}
                        )
            else:
                bits = 4
                assert activation_cache is not None
                awq_scales[name] = {}
                for entry in selected:
                    layer_result = search_awq_layer(
                        entry,
                        activation_cache.reconstruction_inputs(entry.index),
                        activation_cache.mean_abs(entry.index),
                        activation_cache.mean_abs(entry.index, down=True),
                        group_size=args.group_size,
                    )
                    awq_metrics.append(layer_result.metrics_dict())
                    awq_scales[name][entry.index] = {
                        "input_scale": layer_result.input_scale,
                        "internal_scale": layer_result.internal_scale,
                    }
                _atomic_torch_save(awq_scales, args.output_dir / "awq_scales.pt")

            metrics = evaluate_variant(model, processor, evaluation_samples, entries, reference_dir)
            result = {
                "name": name,
                "method": method,
                "scope": scope,
                "bits": bits,
                "group_size": args.group_size,
                "config_hash": config_hash,
                "elapsed_seconds": time.monotonic() - started,
                "weight_quantization": weight_stats,
                "awq_search": awq_metrics,
                "metrics": metrics,
            }
            atomic_json(result_path, result)
            results.append(result)
            _write_outputs(args.output_dir, results, inventory)
            print(f"Completed {name} in {result['elapsed_seconds']:.1f}s", flush=True)
        except RuntimeError as exc:
            guidance = None
            if "out of memory" in str(exc).lower():
                guidance = "Use --preset smoke or restrict --scopes; parameters are never changed automatically."
            atomic_json(
                args.output_dir / name / "error.json",
                {"name": name, "error": str(exc), "oom_guidance": guidance},
            )
            raise
        finally:
            snapshot.restore()
            snapshot.assert_restored()
            if parameter_fingerprint(model.named_parameters(), excluded_ids=mutable_ids) != non_target_before:
                raise RuntimeError(f"a non-target parameter changed while running {name}")

    print(f"Results written to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
