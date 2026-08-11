from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import torch

from .calibration import collect_calibration
from .config import ExperimentConfig, MethodName, QuantScope
from .corpus_evaluation import evaluate_token_sequences_ppl
from .methods import apply_method
from .reporting import atomic_json
from .runner import CALIBRATION_METHODS, environment_record, load_reference
from .selection import inventory, select_linears


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WikiText-2/PTB/C4 corpus perplexity")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--corpora-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", choices=[item.value for item in MethodName], required=True)
    parser.add_argument("--scope", choices=[item.value for item in QuantScope], default="decoder_canonical")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--allow-nonstandard-model", action="store_true")
    return parser.parse_args()


def load_sequences(path: Path) -> list[torch.Tensor]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, list) or not all(isinstance(item, torch.Tensor) for item in value):
        raise ValueError(f"invalid prepared corpus: {path}")
    return value


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    method, scope = MethodName(args.method), QuantScope(args.scope)
    activation = 8 if method in {MethodName.SMOOTHQUANT, MethodName.LLM_INT8, MethodName.TENDER} else None
    config = ExperimentConfig(
        method, scope=scope, weight_bits=8 if activation else 4,
        activation_bits=activation, zero_point=activation is None,
        group_size=args.group_size,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "config.json", config.to_dict() | {
        "config_hash": config.digest, "environment": environment_record()})
    try:
        model, _processor, conversion = load_reference(args.model_path)
        atomic_json(args.output_dir / "reference_conversion.json", conversion)
        selected = select_linears(model, scope, strict_qwen_3b=not args.allow_nonstandard_model)
        atomic_json(args.output_dir / "module_inventory.json", inventory(selected))
        calibration = None
        if method in CALIBRATION_METHODS:
            sequences = load_sequences(args.corpora_dir / "c4_calibration_128x2048.pt")
            batches = ({"input_ids": sequence.unsqueeze(0)} for sequence in sequences)
            calibration = collect_calibration(model, selected, batches, reservoir_rows=32)
        applied = apply_method(model, selected, config, calibration)
        try:
            results = {}
            files = {
                "wikitext2_test": "wikitext2_test.pt",
                "ptb_test": "ptb_test.pt",
                "c4_validation": "c4_validation_256x2048.pt",
            }
            for name, filename in files.items():
                results[name] = evaluate_token_sequences_ppl(
                    model, load_sequences(args.corpora_dir / filename), sequence_length=2048)
            atomic_json(args.output_dir / "corpus_metrics.json", results)
            atomic_json(args.output_dir / "quantization_audit.json", applied.audit)
        finally:
            applied.restore()
    except Exception as error:
        atomic_json(args.output_dir / "error.json", {
            "type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc(), "config_hash": config.digest})
        raise


if __name__ == "__main__":
    main()
