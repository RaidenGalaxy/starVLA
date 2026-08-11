"""Export and evaluate the fixed six-configuration packed quantization matrix."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from .detailed_report import write_detailed_report

MATRIX = {
    "packed_rtn_w8_fp32": ("rtn_w8_all_linear", "dequant_fp32"),
    "packed_rtn_w8_a16": ("rtn_w8_all_linear", "a16_round_fp32_accum"),
    "packed_rtn_w4_fp32": ("rtn_w4_all_linear", "dequant_fp32"),
    "packed_rtn_w4_a16": ("rtn_w4_all_linear", "a16_round_fp32_accum"),
    "packed_hybrid_awq_w4_fp32": ("hybrid_awq_w4_all_linear", "dequant_fp32"),
    "packed_hybrid_awq_w4_a16": ("hybrid_awq_w4_all_linear", "a16_round_fp32_accum"),
}
CHECKPOINT_NAMES = {
    "rtn_w8_all_linear": "rtn_w8",
    "rtn_w4_all_linear": "rtn_w4",
    "hybrid_awq_w4_all_linear": "hybrid_awq_w4",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preset", choices=("smoke", "cpu"), default="smoke")
    parser.add_argument("--configs", help="comma-separated matrix configuration names")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--scale-dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    args.configs = list(MATRIX) if not args.configs else [item.strip() for item in args.configs.split(",") if item.strip()]
    invalid = sorted(set(args.configs) - set(MATRIX))
    if invalid:
        parser.error(f"unsupported configs: {invalid}")
    return args


def _run(command):
    print("Running:", " ".join(map(str, command)), flush=True)
    subprocess.run(command, check=True)


def _row(name, result, source_run, comparable_fake):
    metrics = result["metrics"]
    hidden, logits, generation = metrics["hidden_states"]["final"], metrics["logits"], metrics["generation"]
    source_name = MATRIX[name][0]
    fake = json.loads((source_run / source_name / "metrics.json").read_text())["metrics"]
    fake_hidden, fake_logits = fake["hidden_states"]["final"], fake["logits"]
    delta_nmse = hidden["normalized_mse"] - fake_hidden["normalized_mse"] if comparable_fake else None
    delta_kl = logits["kl_divergence"] - fake_logits["kl_divergence"] if comparable_fake else None
    return {
        "name": name,
        "final_hidden_nmse": hidden["normalized_mse"],
        "final_hidden_cosine": hidden["cosine_similarity"],
        "logits_kl": logits["kl_divergence"],
        "top1_agreement": logits["top1_agreement"],
        "target_nll": logits["target_nll"],
        "target_perplexity": logits["target_perplexity"],
        "generation_reference_agreement": generation["reference_exact_agreement"],
        "generation_token_agreement": generation["token_agreement"],
        "generation_edit_similarity": generation["normalized_edit_similarity"],
        "delta_nmse_vs_fake": delta_nmse,
        "delta_kl_vs_fake": delta_kl,
        "elapsed_seconds": result["elapsed_seconds"],
    }


def _delta(value):
    return "n/a" if value is None else f"{value:.6g}"


def _aggregate(output, rows):
    temporary = output / "summary.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output / "summary.csv")
    if set(MATRIX).issubset({row["name"] for row in rows}):
        write_detailed_report(output, rows)
    else:
        lines = [
            "# Packed Qwen2.5-VL 矩阵运行进度", "",
            f"已完成 {len(rows)}/{len(MATRIX)} 个配置。六组齐全后将自动生成详细方法与结果报告。", "",
            "| 配置 | Final Hidden NMSE↓ | Cosine↑ | Logits KL↓ | Top-1↑ |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(f"| {row['name']} | {row['final_hidden_nmse']:.6g} | {row['final_hidden_cosine']:.6f} | {row['logits_kl']:.6g} | {row['top1_agreement']:.2%} |")
        report = output / "report.md.tmp"
        report.write_text("\n".join(lines) + "\n", encoding="utf-8")
        report.replace(output / "report.md")
    (output / "metrics.json.tmp").write_text(json.dumps({row["name"]: row for row in rows}, indent=2) + "\n")
    (output / "metrics.json.tmp").replace(output / "metrics.json")


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = args.output_dir / "checkpoints"
    results = args.output_dir / "results"
    checkpoints.mkdir(exist_ok=True)
    results.mkdir(exist_ok=True)
    needed_exports = dict.fromkeys(MATRIX[name][0] for name in args.configs)
    for export_config in needed_exports:
        checkpoint = checkpoints / CHECKPOINT_NAMES[export_config]
        command = [sys.executable, "-m", "experiments.packed_model_quant.export_checkpoint", "--model-path", args.model_path, "--source-run", str(args.source_run), "--config", export_config, "--output-dir", str(checkpoint), "--group-size", str(args.group_size), "--scale-dtype", args.scale_dtype]
        if args.resume:
            command.append("--resume")
        _run(command)
    rows = []
    for name in args.configs:
        export_config, compute_mode = MATRIX[name]
        command = [sys.executable, "-m", "experiments.packed_model_quant.run_ablation", "--checkpoint", str(checkpoints / CHECKPOINT_NAMES[export_config]), "--source-run", str(args.source_run), "--output-dir", str(results / name), "--preset", args.preset, "--compute-mode", compute_mode, "--max-new-tokens", str(args.max_new_tokens), "--seed", str(args.seed)]
        if args.resume:
            command.append("--resume")
        _run(command)
        rows.append(_row(name, json.loads((results / name / "metrics.json").read_text()), args.source_run, args.preset == "cpu"))
        _aggregate(args.output_dir, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
