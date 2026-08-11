"""Evaluate streaming-dequant or native AVX-512 W4A8 Qwen2.5-VL checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch

from experiments.ffn_quant.data import load_manifest, select_samples
from experiments.full_model_quant.evaluation import evaluate_variant
from experiments.full_model_quant.outputs import atomic_json, atomic_jsonl
from experiments.packed_model_quant.checkpoint import load_packed_checkpoint, sha256_file, stable_hash
from experiments.packed_model_quant.run_ablation import _ffns, _reference_generations, _rss, _safe, _storage

from . import BACKENDS, KERNEL_FORMAT_VERSION
from .extension import build_identity, load_native_extension, require_avx512_vnni
from .linear import NativePackedLinear, make_linear_factory

BASE_NAMES = {
    "rtn_w8_all_linear": "rtn_w8",
    "rtn_w4_all_linear": "rtn_w4",
    "hybrid_awq_w4_all_linear": "hybrid_awq_w4",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--output-tile-size", type=int, default=256)
    parser.add_argument("--preset", choices=("smoke", "cpu"), default="smoke")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.output_tile_size <= 0 or args.max_new_tokens <= 0 or args.threads <= 0:
        parser.error("tile size, max new tokens, and threads must be positive")
    return args


def _cpu_flags() -> list[str]:
    path = Path("/proc/cpuinfo")
    if not path.is_file():
        return []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("flags"):
            return sorted(line.split(":", 1)[1].split())
    return []


def _runtime(model) -> dict[str, Any]:
    modules = [(name, module) for name, module in model.named_modules() if isinstance(module, NativePackedLinear)]
    per_module = {name: module.runtime.to_dict() for name, module in modules}
    summed = {}
    for field in (
        "calls", "unpack_seconds", "dequant_seconds", "unpack_dequant_seconds", "linear_seconds",
        "activation_quant_seconds", "native_gemm_seconds", "activation_metrics_seconds", "tiles",
        "integer_macs", "fallback_calls", "activation_error_power", "activation_signal_power",
        "activation_values",
    ):
        summed[field] = sum(getattr(module.runtime, field) for _name, module in modules)
    error_power = float(summed["activation_error_power"])
    signal_power = float(summed["activation_signal_power"])
    values = int(summed["activation_values"])
    activation_metrics = {
        "mse": error_power / max(values, 1),
        "normalized_mse": error_power / max(signal_power, 1e-30),
        "sqnr_db": math.inf if error_power == 0 else 10 * math.log10(max(signal_power, 1e-30) / error_power),
        "max_abs_error": max((module.runtime.activation_max_abs_error for _name, module in modules), default=0.0),
        "values": values,
    }
    max_float = max((module.runtime.max_float_weight_tile_bytes for _name, module in modules), default=0)
    max_full = max((module.runtime.full_weight_bytes for _name, module in modules), default=0)
    return {
        **summed,
        "modules": len(modules),
        "max_temporary_bytes": max((module.runtime.max_temporary_bytes for _name, module in modules), default=0),
        "max_float_weight_tile_bytes": max_float,
        "max_full_weight_bytes": max_full,
        "max_float_tile_fraction": max_float / max(max_full, 1),
        "full_float_weight_materializations": sum(
            module.runtime.max_float_weight_tile_bytes >= module.runtime.full_weight_bytes
            for _name, module in modules
            if module.runtime.calls
        ),
        "activation_quantization": activation_metrics,
        "per_module": per_module,
    }


def _name(config: str, backend: str) -> str:
    return f"native_{BASE_NAMES[config]}_{backend}"


def _write_csv(path: Path, result: dict[str, Any]) -> None:
    metrics = result["metrics"]
    runtime = result["runtime"]
    hidden, logits, generation = metrics["hidden_states"]["final"], metrics["logits"], metrics["generation"]
    row = {
        "name": result["name"],
        "backend": result["backend"],
        "final_hidden_nmse": hidden["normalized_mse"],
        "final_hidden_cosine": hidden["cosine_similarity"],
        "logits_kl": logits["kl_divergence"],
        "top1_agreement": logits["top1_agreement"],
        "target_nll": logits["target_nll"],
        "generation_reference_agreement": generation["reference_exact_agreement"],
        "generation_token_agreement": generation["token_agreement"],
        "activation_nmse": runtime["activation_quantization"]["normalized_mse"],
        "fallback_calls": runtime["fallback_calls"],
        "elapsed_seconds": result["elapsed_seconds"],
    }
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    temporary.replace(path)


def _write_report(path: Path, result: dict[str, Any], storage: dict[str, Any]) -> None:
    metrics, runtime = result["metrics"], result["runtime"]
    hidden, logits, generation = metrics["hidden_states"]["final"], metrics["logits"], metrics["generation"]
    activation = runtime["activation_quantization"]
    if result["backend"] == "avx512_w4a8":
        method = (
            "激活按token/group-128动态量化为INT8；packed INT4权重直接由AVX-512 VNNI执行"
            "无符号权重×有符号激活点积，经-8·sum(qx)修正后以INT32累加。"
        )
    else:
        method = "权重只按输出通道tile解包和反量化；每个tile使用后立即释放，未生成完整浮点权重矩阵。"
    lines = [
        "# Qwen2.5-VL Native Packed CPU实验报告", "",
        f"配置：`{result['name']}`；backend：`{result['backend']}`；样本数：{generation['samples']}。", "",
        method, "",
        "## 执行完整性", "",
        f"- Packed Linear：{runtime['modules']} / 414",
        f"- 浮点整权重物化次数：{runtime['full_float_weight_materializations']}",
        f"- 浮点fallback调用：{runtime['fallback_calls']}",
        f"- 最大浮点权重tile：{runtime['max_float_weight_tile_bytes'] / 2**20:.3f} MiB",
        f"- 最大完整FP32权重理论值：{runtime['max_full_weight_bytes'] / 2**20:.3f} MiB", "",
        "## 端到端结果", "",
        "| Final Hidden NMSE↓ | Cosine↑ | Logits KL↓ | Top-1 Agreement↑ | Target NLL↓ | Reference Exact↑ | Token Agreement↑ |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {hidden['normalized_mse']:.6g} | {hidden['cosine_similarity']:.6f} | {logits['kl_divergence']:.6g} | "
        f"{logits['top1_agreement']:.2%} | {logits['target_nll']:.6g} | "
        f"{generation['reference_exact_agreement']:.2%} | {generation['token_agreement']:.2%} |", "",
        "## 激活量化与运行时", "",
        f"- 激活NMSE：{activation['normalized_mse']:.6g}",
        f"- 激活SQNR：{activation['sqnr_db']:.4f} dB",
        f"- 激活最大绝对误差：{activation['max_abs_error']:.6g}",
        f"- activation quant时间：{runtime['activation_quant_seconds']:.3f}s",
        f"- native GEMM时间：{runtime['native_gemm_seconds']:.3f}s",
        f"- unpack+dequant时间：{runtime['unpack_dequant_seconds']:.3f}s",
        f"- 完整评估时间：{result['elapsed_seconds']:.3f}s", "",
        "## 结论边界", "",
        "该结果验证CPU packed权重执行和AVX-512 W4A8整数核心点积，不是GPU W4A16 kernel、显存或吞吐结果。",
        f"Checkpoint磁盘大小为{storage['checkpoint_disk_bytes'] / 2**30:.3f} GiB；逐子系统结果见`metrics.json`。", "",
    ]
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)


def main(argv=None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    manifest = args.manifest or args.source_run / "tiny_tabletop" / "manifest.jsonl"
    identity = {
        "kernel_format": KERNEL_FORMAT_VERSION,
        "checkpoint_hash": sha256_file(args.checkpoint / "quantization_config.json"),
        "source_run_hash": json.loads((args.source_run / "config.json").read_text())["run_hash"],
        "manifest_hash": sha256_file(manifest),
        "backend": args.backend,
        "output_tile_size": args.output_tile_size,
        "threads": args.threads,
        "preset": args.preset,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "cpu_flags": _cpu_flags(),
        "kernel_build": build_identity(),
        "activation_quantization": "symmetric_per_token_group128_int8_-127_127",
    }
    run_hash = stable_hash(identity)
    result_path = args.output_dir / "metrics.json"
    if args.resume and result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("run_hash") == run_hash:
            print(f"Skipping completed native run at {args.output_dir}")
            return 0
    atomic_json(args.output_dir / "config.json", {**identity, "run_hash": run_hash})
    try:
        if args.backend == "avx512_w4a8":
            require_avx512_vnni(load_native_extension())
        load_started = time.monotonic()
        model, processor, quant, inventory = load_packed_checkpoint(
            args.checkpoint,
            linear_factory=make_linear_factory(args.backend, args.output_tile_size),
        )
        load_seconds = time.monotonic() - load_started
        modules = [(name, module) for name, module in model.named_modules() if isinstance(module, NativePackedLinear)]
        if len(modules) != 414 or quant["quantized_parameters"] != 3_441_197_056:
            raise RuntimeError("native loader did not preserve the expected 414-Linear inventory")
        if any(hasattr(module, "weight") for _name, module in modules):
            raise RuntimeError("a native packed target exposes a floating weight")
        atomic_json(args.output_dir / "packed_inventory.json", inventory)
        samples = load_manifest(manifest)
        evaluation = select_samples(samples, split="evaluation", limit=2 if args.preset == "smoke" else 16)
        references = _reference_generations(args.source_run, evaluation)
        started = time.monotonic()
        metrics, generations = evaluate_variant(
            model, processor, evaluation, _ffns(model), args.source_run / "reference_cache", references,
            max_new_tokens=args.max_new_tokens,
        )
        elapsed = time.monotonic() - started
        runtime = _runtime(model)
        if runtime["modules"] != 414 or runtime["fallback_calls"] != 0:
            raise RuntimeError("native execution did not cover all packed modules without fallback")
        if runtime["full_float_weight_materializations"] != 0:
            raise RuntimeError("a complete floating target weight was materialized")
        name = _name(quant["config"], args.backend)
        result = {
            "name": name,
            "run_hash": run_hash,
            "backend": args.backend,
            "load_seconds": load_seconds,
            "elapsed_seconds": elapsed,
            "metrics": metrics,
            "runtime": runtime,
            "memory": {"after_evaluation": _rss()},
        }
        storage = _storage(args.checkpoint, quant)
        atomic_json(result_path, _safe(result))
        atomic_json(args.output_dir / "runtime.json", _safe({**runtime, "memory": result["memory"]}))
        atomic_json(args.output_dir / "storage_metrics.json", _safe(storage))
        atomic_jsonl(args.output_dir / "generations.jsonl", [{**row, "config": name} for row in generations])
        _write_csv(args.output_dir / "summary.csv", result)
        _write_report(args.output_dir / "report.md", result, storage)
        print(f"Completed {name} in {elapsed:.1f}s")
        return 0
    except Exception as exc:
        atomic_json(
            args.output_dir / "error.json",
            {"type": type(exc).__name__, "message": str(exc), "run_hash": run_hash, "backend": args.backend},
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
