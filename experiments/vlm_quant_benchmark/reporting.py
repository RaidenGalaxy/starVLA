from __future__ import annotations

import csv
import json
from pathlib import Path


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_report(path: Path, config: dict, module_inventory: dict, metrics: dict, audit: dict) -> None:
    diagnostic = metrics.get("target_text_diagnostic", {})
    precision = (
        "reference precision"
        if config["weight_bits"] is None
        else f"W{config['weight_bits']}A{config.get('activation_bits') or 16} CPU simulation"
    )
    content = f"""# Qwen2.5-VL quantization report

## Experiment definition

- Method: `{config['method']}`
- Scheme: `{config.get('scheme_id', 'unspecified')}`
- Implementation tier: `{config.get('implementation_tier', 'unspecified')}`
- Scope: `{config['scope']}`
- Calibration protocol: `{config.get('calibration_protocol', 'unspecified')}`
- Model revision: `{config.get('model_revision', 'unspecified')}`
- Processor revision: `{config.get('processor_revision', 'unspecified')}`
- Pixel bounds: {config.get('min_pixels', 'n/a')} to {config.get('max_pixels', 'n/a')}
- Weight/activation: {precision}
- Group size: {config['group_size']}
- Reference: bf16_value_fp32_compute
- Selected Linear modules: {module_inventory['linear_count']}
- Selected weights: {module_inventory['weight_count']:,}

This is an algorithmic CPU quality experiment, not an integer-kernel memory or
latency benchmark. Logical BPW may be reported, but FP32 materialized storage is
never presented as deployment compression. Target-text PPL is diagnostic only
and is not WikiText/C4/PTB corpus perplexity.

## Task-level diagnostics

- Samples: {metrics.get('samples', 'n/a')}
- Exact accuracy: {metrics.get('exact_accuracy', float('nan')):.6f}
- ANLS: {metrics.get('anls', float('nan')):.6f}
- Normalized edit similarity: {metrics.get('normalized_edit_similarity', float('nan')):.6f}
- Target-text tokens: {diagnostic.get('tokens', 0)}
- Target-text NLL (diagnostic only): {diagnostic.get('nll', float('nan')):.6f}
- Target-text PPL (diagnostic only): {diagnostic.get('ppl', float('nan')):.6f}

Target-text PPL excludes the prompt, chat terminator and template newlines. It
is never used for method ranking. Official MMMU, DocVQA, MMBench and MathVista
scores must be produced by the pinned VLMEvalKit scorer. Local exact/ANLS fields
are diagnostics and are not substitutes for official benchmark metrics.

## Quantization audit

- Replaced modules: {audit.get('module_count', 0)}
- Quantization time: {audit.get('elapsed_seconds', 0.0):.2f} seconds

Machine-readable per-module search curves and weight errors are stored in
`quantization_audit.json`; per-sample outputs are stored in `generations.jsonl`.
"""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)
