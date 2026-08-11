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
    conditional = metrics.get("conditional_answer", {})
    content = f"""# Qwen2.5-VL quantization report

## Experiment definition

- Method: `{config['method']}`
- Scope: `{config['scope']}`
- Weight/activation: W{config['weight_bits']}A{config.get('activation_bits') or 32} CPU simulation
- Group size: {config['group_size']}
- Reference: BF16 checkpoint values widened to FP32 for CPU execution
- Selected Linear modules: {module_inventory['linear_count']}
- Selected weights: {module_inventory['weight_count']:,}

This is an algorithmic CPU experiment, not a GPU integer-kernel benchmark.
Conditional-answer perplexity is computed only on gold answer tokens and is not
WikiText/C4/PTB corpus perplexity.

## Task-level diagnostics

- Samples: {metrics.get('samples', 'n/a')}
- Exact accuracy: {metrics.get('exact_accuracy', float('nan')):.6f}
- ANLS: {metrics.get('anls', float('nan')):.6f}
- Normalized edit similarity: {metrics.get('normalized_edit_similarity', float('nan')):.6f}
- Conditional answer NLL: {conditional.get('nll', float('nan')):.6f}
- Conditional answer PPL: {conditional.get('ppl', float('nan')):.6f}

Official MMMU, DocVQA, MMBench and MathVista scores must be produced by the
pinned VLMEvalKit scorer. The local exact/ANLS fields are diagnostics and are
not substitutes for official benchmark metrics.

## Quantization audit

- Replaced modules: {audit.get('module_count', 0)}
- Quantization time: {audit.get('elapsed_seconds', 0.0):.2f} seconds

Machine-readable per-module search curves and weight errors are stored in
`quantization_audit.json`; per-sample outputs are stored in `generations.jsonl`.
"""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)
