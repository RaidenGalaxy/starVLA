"""Run the gated native CPU smoke suite and the two full W4A8 configurations."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from experiments.full_model_quant.outputs import atomic_json

from .detailed_report import write_detailed_report

SMOKE_MATRIX = {
    "streaming_rtn_w8": ("rtn_w8", "streaming_dequant_fp32"),
    "streaming_rtn_w4": ("rtn_w4", "streaming_dequant_fp32"),
    "streaming_hybrid_awq_w4": ("hybrid_awq_w4", "streaming_dequant_fp32"),
    "native_rtn_w4a8": ("rtn_w4", "avx512_w4a8"),
    "native_hybrid_awq_w4a8": ("hybrid_awq_w4", "avx512_w4a8"),
}
FULL_MATRIX = {
    "native_rtn_w4a8": ("rtn_w4", "avx512_w4a8"),
    "native_hybrid_awq_w4a8": ("hybrid_awq_w4", "avx512_w4a8"),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("outputs/packed_model_quant/checkpoints"))
    parser.add_argument(
        "--w8-checkpoint",
        type=Path,
        default=Path("outputs/packed_model_quant/matrix/checkpoints/rtn_w8"),
    )
    parser.add_argument(
        "--packed-results-root",
        type=Path,
        default=Path("outputs/packed_model_quant/matrix/results"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/native_model_quant/matrix"))
    parser.add_argument("--phase", choices=("smoke", "cpu", "all"), default="all")
    parser.add_argument("--output-tile-size", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _checkpoint(args, name):
    if name == "rtn_w8":
        return args.w8_checkpoint
    return args.checkpoint_root / name


def _run_one(args, name, checkpoint_name, backend, preset):
    output = args.output_dir / preset / name
    command = [
        sys.executable,
        "-m",
        "experiments.native_model_quant.run_ablation",
        "--checkpoint",
        str(_checkpoint(args, checkpoint_name)),
        "--source-run",
        str(args.source_run),
        "--output-dir",
        str(output),
        "--backend",
        backend,
        "--preset",
        preset,
        "--output-tile-size",
        str(args.output_tile_size),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--threads",
        str(args.threads),
        "--seed",
        str(args.seed),
    ]
    if args.resume:
        command.append("--resume")
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    return json.loads((output / "metrics.json").read_text(encoding="utf-8"))


def _gate(results):
    failures = []
    for name, result in results.items():
        runtime = result["runtime"]
        if runtime["modules"] != 414:
            failures.append(f"{name}: modules={runtime['modules']}")
        if runtime["fallback_calls"] != 0:
            failures.append(f"{name}: fallback_calls={runtime['fallback_calls']}")
        if runtime["full_float_weight_materializations"] != 0:
            failures.append(f"{name}: full floating weight materialized")
        if result["metrics"]["generation"]["samples"] != 2:
            failures.append(f"{name}: smoke did not evaluate two samples")
    return failures


def _row(phase, name, result):
    metrics, runtime = result["metrics"], result["runtime"]
    hidden, logits, generation = metrics["hidden_states"]["final"], metrics["logits"], metrics["generation"]
    return {
        "phase": phase,
        "name": name,
        "backend": result["backend"],
        "samples": generation["samples"],
        "final_hidden_nmse": hidden["normalized_mse"],
        "final_hidden_cosine": hidden["cosine_similarity"],
        "logits_kl": logits["kl_divergence"],
        "top1_agreement": logits["top1_agreement"],
        "activation_nmse": runtime["activation_quantization"]["normalized_mse"],
        "fallback_calls": runtime["fallback_calls"],
        "full_weight_materializations": runtime["full_float_weight_materializations"],
        "elapsed_seconds": result["elapsed_seconds"],
    }


def _aggregate(output, rows, gate_failures, packed_results_root):
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "summary.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output / "summary.csv")
    atomic_json(output / "metrics.json", {row["phase"] + ":" + row["name"]: row for row in rows})
    lines = [
        "# Native Packed Qwen2.5-VL CPU矩阵", "",
        f"Smoke门控：{'通过' if not gate_failures else '失败'}。", "",
        "| 阶段 | 配置 | Backend | 样本 | Final Hidden NMSE↓ | Cosine↑ | Logits KL↓ | Top-1↑ | Activation NMSE↓ |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['phase']} | {row['name']} | {row['backend']} | {row['samples']} | "
            f"{row['final_hidden_nmse']:.6g} | {row['final_hidden_cosine']:.6f} | "
            f"{row['logits_kl']:.6g} | {row['top1_agreement']:.2%} | {row['activation_nmse']:.6g} |"
        )
    if gate_failures:
        lines.extend(["", "## 门控失败", ""] + [f"- {item}" for item in gate_failures])
    lines.extend([
        "", "AVX-512配置使用INT8激活、packed INT4权重和INT32核心点积；结果不代表GPU W4A16性能。", "",
    ])
    report = output / "report.md.tmp"
    report.write_text("\n".join(lines), encoding="utf-8")
    report.replace(output / "report.md")
    complete = all(
        (output / "cpu" / name / "metrics.json").is_file()
        for name in FULL_MATRIX
    )
    if complete:
        write_detailed_report(output, packed_results_root)


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, smoke_results = [], {}
    if args.phase in ("smoke", "all"):
        for name, (checkpoint, backend) in SMOKE_MATRIX.items():
            result = _run_one(args, name, checkpoint, backend, "smoke")
            smoke_results[name] = result
            rows.append(_row("smoke", name, result))
            _aggregate(args.output_dir, rows, _gate(smoke_results), args.packed_results_root)
    failures = _gate(smoke_results) if smoke_results else []
    if failures:
        _aggregate(args.output_dir, rows, failures, args.packed_results_root)
        raise RuntimeError("smoke gate failed; full CPU matrix was not started: " + "; ".join(failures))
    if args.phase in ("cpu", "all"):
        if args.phase == "cpu":
            for name in SMOKE_MATRIX:
                path = args.output_dir / "smoke" / name / "metrics.json"
                if not path.is_file():
                    raise RuntimeError("--phase cpu requires completed smoke gate results")
                smoke_results[name] = json.loads(path.read_text(encoding="utf-8"))
                rows.append(_row("smoke", name, smoke_results[name]))
            failures = _gate(smoke_results)
            if failures:
                raise RuntimeError("saved smoke gate failed: " + "; ".join(failures))
        for name, (checkpoint, backend) in FULL_MATRIX.items():
            result = _run_one(args, name, checkpoint, backend, "cpu")
            rows.append(_row("cpu", name, result))
            _aggregate(args.output_dir, rows, [], args.packed_results_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
