from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import torch

from .config import ExperimentConfig, MethodName, QuantScope
from .data import assert_disjoint, fixed_subset, load_manifest
from .reporting import atomic_json, write_summary
from .runner import load_reference, run_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper-aligned Qwen2.5-VL quantization benchmark")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--calibration-manifest", type=Path)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preset", choices=("smoke", "screen", "full"), default="smoke")
    parser.add_argument("--methods", default="bf16_reference,rtn,awq,gptq,smoothquant,llm_int8,tender")
    parser.add_argument("--scopes", default="decoder_canonical,full_vlm_extended")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-nonstandard-model", action="store_true")
    return parser.parse_args()


def method_config(method: MethodName, scope: QuantScope, args: argparse.Namespace) -> ExperimentConfig:
    common = dict(scope=scope, group_size=args.group_size, seed=args.seed,
                  max_new_tokens=args.max_new_tokens)
    if method in (MethodName.SMOOTHQUANT, MethodName.LLM_INT8, MethodName.TENDER):
        return ExperimentConfig(method, weight_bits=8, activation_bits=8, zero_point=False, **common)
    return ExperimentConfig(method, weight_bits=4, activation_bits=None, zero_point=True, **common)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    methods = [MethodName(value.strip()) for value in args.methods.split(",") if value.strip()]
    scopes = [QuantScope(value.strip()) for value in args.scopes.split(",") if value.strip()]
    calibration = load_manifest(args.calibration_manifest) if args.calibration_manifest else []
    evaluation = load_manifest(args.evaluation_manifest)
    if calibration:
        assert_disjoint(calibration, evaluation)
    sample_limit = {"smoke": 2, "screen": 256, "full": len(evaluation)}[args.preset]
    evaluation = fixed_subset(evaluation, sample_limit, args.seed)

    model, processor, conversion = load_reference(args.model_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "reference_conversion.json", conversion)

    rows = []
    for method in methods:
        method_scopes = scopes[:1] if method == MethodName.BF16_REFERENCE else scopes
        for scope in method_scopes:
            config = method_config(method, scope, args)
            name = f"{method.value}__{scope.value}"
            destination = args.output_dir / name
            try:
                metrics = run_config(
                    model, processor, config, calibration, evaluation, destination,
                    strict_qwen_3b=not args.allow_nonstandard_model,
                    resume=args.resume,
                )
                conditional = metrics.get("conditional_answer", {})
                rows.append({
                    "config": name,
                    "method": method.value,
                    "scope": scope.value,
                    "samples": metrics.get("samples"),
                    "exact_accuracy": metrics.get("exact_accuracy"),
                    "anls": metrics.get("anls"),
                    "conditional_answer_nll": conditional.get("nll"),
                    "conditional_answer_ppl": conditional.get("ppl"),
                    "status": "complete",
                })
            except Exception as error:
                atomic_json(destination / "error.json", {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                    "config_hash": config.digest,
                })
                rows.append({"config": name, "method": method.value, "scope": scope.value,
                             "status": "failed", "error": str(error)})
            write_summary(args.output_dir / "summary.csv", rows)


if __name__ == "__main__":
    main()
