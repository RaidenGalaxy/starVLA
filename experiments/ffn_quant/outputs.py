"""Atomic machine-readable outputs and a compact Markdown report."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_safe(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def summary_row(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result["metrics"]
    return {
        "name": result["name"],
        "method": result["method"],
        "scope": result["scope"],
        "bits": result.get("bits", "original"),
        "ffn_nmse": metrics["ffn"]["overall"]["normalized_mse"],
        "ffn_cosine": metrics["ffn"]["overall"]["cosine_similarity"],
        "final_hidden_nmse": metrics["hidden_states"]["final"]["normalized_mse"],
        "final_hidden_cosine": metrics["hidden_states"]["final"]["cosine_similarity"],
        "logits_kl": metrics["logits"].get("kl_divergence", 0.0),
        "top1_agreement": metrics["logits"].get("top1_agreement", 1.0),
        "target_nll": metrics["logits"]["target_nll"],
        "target_perplexity": metrics["logits"]["target_perplexity"],
    }


def write_summary_csv(path: Path, results: list[dict[str, Any]]) -> None:
    rows = [summary_row(result) for result in results]
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_report(path: Path, results: list[dict[str, Any]], inventory: dict[str, Any]) -> None:
    rows = [summary_row(result) for result in results]
    lines = [
        "# Qwen2.5-VL Decoder FFN Fake-Quantization Report",
        "",
        "> The reference uses original checkpoint values with FP32 CPU compute. This does not measure GPU FP16/W4A16 kernels, memory, or latency.",
        "",
        "## Quantization scope",
        "",
        f"- Decoder layers: {inventory['decoder_layers']}",
        f"- Target Linear modules: {inventory['linear_modules']}",
        f"- Target parameters: {inventory['target_parameters']:,} ({inventory['target_fraction']:.2%} of model)",
        "",
        "## Results",
        "",
        "| Method | Scope | Bits | FFN NMSE | Final hidden cosine | Logits KL | Top-1 agreement | Target NLL |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['scope']} | {row['bits']} | {row['ffn_nmse']:.6g} | "
            f"{row['final_hidden_cosine']:.6g} | {row['logits_kl']:.6g} | "
            f"{row['top1_agreement']:.4f} | {row['target_nll']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Layer sensitivity",
            "",
            "Per-layer metrics are stored in each configuration's `metrics.json`.",
            "AWQ search curves and selected scales are stored in `awq_scales.pt`.",
            "",
        ]
    )
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)
