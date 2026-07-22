"""Evaluate an independently loadable packed Qwen2.5-VL checkpoint on CPU."""
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
import torch.nn as nn

from experiments.ffn_quant.data import load_manifest, select_samples
from experiments.full_model_quant.evaluation import evaluate_variant
from experiments.full_model_quant.modeling import FFNTarget
from experiments.full_model_quant.outputs import atomic_json, atomic_jsonl
from .checkpoint import load_packed_checkpoint, sha256_file, stable_hash
from .linear import COMPUTE_MODES, PackedWeightOnlyLinear

BASE_NAMES = {
    "rtn_w8_all_linear": "packed_rtn_w8",
    "rtn_w4_all_linear": "packed_rtn_w4",
    "hybrid_awq_w4_all_linear": "packed_hybrid_awq_w4",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preset", choices=("smoke", "cpu"), default="smoke")
    parser.add_argument("--compute-mode", choices=COMPUTE_MODES, default="dequant_fp32")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.max_new_tokens <= 0:
        parser.error("max new tokens must be positive")
    return args


def _rss() -> dict[str, int]:
    values = {"rss_bytes": 0, "peak_rss_bytes": 0}
    path = Path("/proc/self/status")
    if not path.is_file():
        return values
    mapping = {"VmRSS:": "rss_bytes", "VmHWM:": "peak_rss_bytes"}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts and parts[0] in mapping:
            values[mapping[parts[0]]] = int(parts[1]) * 1024
    return values


def _ffns(model: nn.Module) -> list[FFNTarget]:
    names = {id(module): name for name, module in model.named_modules()}
    result: list[FFNTarget] = []
    for index, layer in enumerate(model.model.language_model.layers):
        result.append(FFNTarget("decoder", index, names[id(layer.mlp)], names[id(layer.post_attention_layernorm)], layer.mlp, layer.post_attention_layernorm))
    for index, block in enumerate(model.model.visual.blocks):
        result.append(FFNTarget("vision", index, names[id(block.mlp)], names[id(block.norm2)], block.mlp, block.norm2))
    if len(result) != 68:
        raise RuntimeError(f"expected 68 FFN blocks, found {len(result)}")
    return result


def _reference_generations(source_run: Path, samples: list[Any]) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        path = source_run / "reference_cache" / f"{sample.id}.generation.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        answer = value.get("reference", value.get("answer", value.get("generation")))
        if answer is None:
            raise RuntimeError(f"reference generation has no answer field: {path}")
        rows.append({"id": sample.id, "target": sample.target, "reference": answer})
    return rows


def _runtime(model: nn.Module) -> dict[str, Any]:
    modules = [(name, module) for name, module in model.named_modules() if isinstance(module, PackedWeightOnlyLinear)]
    return {
        "calls": sum(module.runtime.calls for _name, module in modules),
        "unpack_dequant_seconds": sum(module.runtime.unpack_dequant_seconds for _name, module in modules),
        "linear_seconds": sum(module.runtime.linear_seconds for _name, module in modules),
        "max_module_temporary_bytes": max((module.runtime.max_temporary_bytes for _name, module in modules), default=0),
        "per_module": {name: {
            "calls": module.runtime.calls,
            "unpack_dequant_seconds": module.runtime.unpack_dequant_seconds,
            "linear_seconds": module.runtime.linear_seconds,
            "max_temporary_bytes": module.runtime.max_temporary_bytes,
        } for name, module in modules},
    }


def _storage(checkpoint: Path, quant: dict[str, Any]) -> dict[str, Any]:
    actual = sum(path.stat().st_size for path in checkpoint.rglob("*") if path.is_file())
    target_packed = quant["qweight_bytes"] + quant["scale_bytes"]
    total_parameters = 3_754_622_976
    return {
        "checkpoint_disk_bytes": actual,
        "tensor_bytes": quant["tensor_bytes"],
        "qweight_bytes": quant["qweight_bytes"],
        "scale_bytes": quant["scale_bytes"],
        "non_quantized_bytes": quant["non_quantized_bytes"],
        "target_effective_bits_per_weight": target_packed * 8 / quant["quantized_parameters"],
        "target_compression_vs_fp16": quant["quantized_parameters"] * 2 / target_packed,
        "whole_tensor_compression_vs_fp16": total_parameters * 2 / quant["tensor_bytes"],
        "whole_tensor_compression_vs_fp32_fake": total_parameters * 4 / quant["tensor_bytes"],
    }


def _safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe(item) for item in value]
    return value


def _write_csv(path: Path, result: dict[str, Any]) -> None:
    metrics = result["metrics"]
    row = {
        "name": result["name"], "compute_mode": result["compute_mode"],
        "final_hidden_nmse": metrics["hidden_states"]["final"]["normalized_mse"],
        "final_hidden_cosine": metrics["hidden_states"]["final"]["cosine_similarity"],
        "logits_kl": metrics["logits"]["kl_divergence"],
        "top1_agreement": metrics["logits"]["top1_agreement"],
        "target_nll": metrics["logits"]["target_nll"],
        "target_perplexity": metrics["logits"]["target_perplexity"],
        "generation_reference_agreement": metrics["generation"]["reference_exact_agreement"],
        "generation_target_accuracy": metrics["generation"]["target_exact_accuracy"],
        "generation_token_agreement": metrics["generation"]["token_agreement"],
        "generation_edit_similarity": metrics["generation"]["normalized_edit_similarity"],
        "elapsed_seconds": result["elapsed_seconds"],
    }
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    temporary.replace(path)


def _write_report(path: Path, result: dict[str, Any], storage: dict[str, Any]) -> None:
    metrics = result["metrics"]
    hidden = metrics["hidden_states"]["final"]
    logits = metrics["logits"]
    generation = metrics["generation"]
    lines = [
        "# Packed Qwen2.5-VL 端到端CPU量化报告", "",
        f"配置：`{result['name']}`；计算模式：`{result['compute_mode']}`；evaluation样本：{generation['samples']}。", "",
        "414个目标Linear以真实INT4/INT8整数码和group scale保存。CPU前向按模块解包并执行FP32矩阵乘法；A16模式仅模拟Linear边界FP16舍入，不是GPU融合kernel性能测试。", "",
        "## 存储", "",
        f"- Checkpoint磁盘大小：{storage['checkpoint_disk_bytes'] / 2**30:.3f} GiB",
        f"- Packed整数码：{storage['qweight_bytes'] / 2**30:.3f} GiB",
        f"- Group scale：{storage['scale_bytes'] / 2**20:.3f} MiB",
        f"- 目标权重有效位宽：{storage['target_effective_bits_per_weight']:.4f} bit/weight",
        f"- 目标权重相对FP16压缩比：{storage['target_compression_vs_fp16']:.3f}×", "",
        "## 系统级端到端指标", "",
        "| Final Hidden NMSE↓ | Final Hidden Cosine↑ | SQNR dB↑ | Logits KL↓ | Top-1 Agreement↑ | Target NLL↓ | Target PPL↓ |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {hidden['normalized_mse']:.6g} | {hidden['cosine_similarity']:.6f} | {hidden['sqnr_db']:.4f} | {logits['kl_divergence']:.6g} | {logits['top1_agreement']:.2%} | {logits['target_nll']:.6g} | {logits['target_perplexity']:.6g} |", "",
        "| Reference Exact↑ | Target Exact↑ | Token Agreement↑ | Edit Similarity↑ |",
        "|---:|---:|---:|---:|",
        f"| {generation['reference_exact_agreement']:.2%} | {generation['target_exact_accuracy']:.2%} | {generation['token_agreement']:.2%} | {generation['normalized_edit_similarity']:.2%} |", "",
        "逐Vision Block、Merger、Decoder Attention、FFN和hidden-state结果保存在`metrics.json`。CPU解包耗时不代表真实INT4推理速度。", "",
    ]
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    manifest = args.manifest or args.source_run / "tiny_tabletop" / "manifest.jsonl"
    identity = {
        "checkpoint_hash": sha256_file(args.checkpoint / "quantization_config.json"),
        "source_run_hash": json.loads((args.source_run / "config.json").read_text())["run_hash"],
        "manifest_hash": sha256_file(manifest), "preset": args.preset,
        "compute_mode": args.compute_mode, "max_new_tokens": args.max_new_tokens, "seed": args.seed,
    }
    run_hash = stable_hash(identity)
    result_path = args.output_dir / "metrics.json"
    if args.resume and result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("run_hash") == run_hash:
            print(f"Skipping completed packed run at {args.output_dir}")
            return 0
    atomic_json(args.output_dir / "config.json", {**identity, "run_hash": run_hash})
    try:
        load_started = time.monotonic()
        model, processor, quant, inventory = load_packed_checkpoint(args.checkpoint, compute_mode=args.compute_mode)
        load_seconds = time.monotonic() - load_started
        memory_after_load = _rss()
        atomic_json(args.output_dir / "packed_inventory.json", inventory)
        samples = load_manifest(manifest)
        evaluation = select_samples(samples, split="evaluation", limit=2 if args.preset == "smoke" else 16)
        references = _reference_generations(args.source_run, evaluation)
        started = time.monotonic()
        metrics, generations = evaluate_variant(model, processor, evaluation, _ffns(model), args.source_run / "reference_cache", references, max_new_tokens=args.max_new_tokens)
        elapsed = time.monotonic() - started
        runtime = _runtime(model)
        memory_after_eval = _rss()
        suffix = "fp32" if args.compute_mode == "dequant_fp32" else "a16"
        name = f"{BASE_NAMES[quant['config']]}_{suffix}"
        result = {
            "name": name, "run_hash": run_hash, "compute_mode": args.compute_mode,
            "load_seconds": load_seconds, "elapsed_seconds": elapsed, "metrics": metrics,
            "runtime": runtime, "memory": {"after_load": memory_after_load, "after_evaluation": memory_after_eval},
        }
        storage = _storage(args.checkpoint, quant)
        atomic_json(result_path, _safe(result))
        atomic_json(args.output_dir / "storage_metrics.json", _safe({**storage, **result["memory"], "runtime": runtime}))
        atomic_jsonl(args.output_dir / "generations.jsonl", [{**row, "config": name} for row in generations])
        _write_csv(args.output_dir / "summary.csv", result)
        _write_report(args.output_dir / "report.md", result, storage)
        print(f"Completed {name} in {elapsed:.1f}s")
        return 0
    except Exception as exc:
        atomic_json(args.output_dir / "error.json", {"type": type(exc).__name__, "message": str(exc), "run_hash": run_hash})
        if isinstance(exc, (MemoryError, torch.OutOfMemoryError)):
            print("OOM: use --preset smoke; experiment parameters were not changed automatically.")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
