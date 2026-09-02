from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import torch

from .all_methods import apply_method
from .calibration import (
    calibration_cache_key,
    collect_calibration_cached,
)
from .config import ExperimentConfig, ImplementationTier, MethodName, QuantScope
from .corpus_evaluation import evaluate_token_sequences_ppl, paired_window_delta
from .data import load_manifest, sample_manifest_digest
from .mbq import (
    collect_mbq_search_cached,
    mbq_cache_key,
    mbq_calibration_batches,
    mbq_statistics_cache_key,
)
from .provenance import completed_run, mark_complete, sha256_json, source_digest
from .reporting import atomic_json
from .runner import CALIBRATION_METHODS, environment_record, load_reference
from .selection import inventory, select_linears


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WikiText-2/PTB/C4 corpus perplexity")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--corpora-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", choices=sorted({item.value for item in MethodName}), required=True)
    parser.add_argument("--scope", choices=[item.value for item in QuantScope], default="decoder_canonical")
    parser.add_argument("--preset", choices=("smoke", "screen", "full"), default="screen")
    parser.add_argument("--weight-bits", type=int, choices=(3, 4, 8))
    parser.add_argument("--activation-bits", type=int, choices=(8,))
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--scheme-id")
    parser.add_argument("--calibration-manifest", type=Path)
    parser.add_argument(
        "--weight-granularity", choices=("groupwise", "per_output_channel"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--reference-metrics", type=Path)
    parser.add_argument("--calibration-cache-dir", type=Path)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-nonstandard-model", action="store_true")
    return parser.parse_args()


def load_sequences(path: Path) -> list[torch.Tensor]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, list) or not all(isinstance(item, torch.Tensor) for item in value):
        raise ValueError(f"invalid prepared corpus: {path}")
    return value


def load_partial_results(
    path: Path,
    *,
    config_digest: str,
    input_digests: dict[str, str],
) -> dict:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text())
    if payload.get("config_digest") != config_digest:
        raise ValueError("corpus partial config digest mismatch")
    if payload.get("input_digests") != input_digests:
        raise ValueError("corpus partial input digest mismatch")
    results = payload.get("results")
    if not isinstance(results, dict):
        raise ValueError("corpus partial results must be a mapping")
    return results


def save_partial_results(
    path: Path,
    results: dict,
    *,
    config_digest: str,
    input_digests: dict[str, str],
) -> None:
    atomic_json(path, {
        "format_version": 1,
        "config_digest": config_digest,
        "input_digests": input_digests,
        "results": results,
    })


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    method, scope = MethodName(args.method), QuantScope(args.scope)
    if scope != QuantScope.DECODER_CANONICAL:
        raise ValueError("corpus PPL is decoder_canonical only; text inputs do not execute vision modules")
    activation = args.activation_bits
    if activation is None and method in {MethodName.SMOOTHQUANT, MethodName.LLM_INT8, MethodName.VECTORWISE_W8, MethodName.TENDER}:
        activation = 8
    weight_bits = args.weight_bits
    if method == MethodName.BF16_VALUE_FP32_COMPUTE:
        weight_bits, activation = None, None
    elif weight_bits is None:
        weight_bits = 8 if activation == 8 else 4
    weight_granularity = args.weight_granularity or (
        "per_output_channel"
        if method == MethodName.MBQ and activation == 8
        else "groupwise"
    )
    tier = (
        ImplementationTier.REFERENCE
        if method == MethodName.BF16_VALUE_FP32_COMPUTE
        else ImplementationTier.NAIVE_BASELINE
        if method in {MethodName.RTN, MethodName.VECTORWISE_W8}
        else ImplementationTier.CPU_RESEARCH
        if method in {MethodName.OMNIQUANT, MethodName.SPQR, MethodName.SQUEEZELLM}
        else ImplementationTier.LOCAL_VALIDATED
    )
    config = ExperimentConfig(
        method,
        scope=scope,
        weight_bits=weight_bits,
        activation_bits=activation,
        zero_point=weight_bits is not None and activation is None,
        weight_granularity=weight_granularity,
        group_size=args.group_size,
        scheme_id=args.scheme_id or "",
        implementation_tier=tier,
        calibration_protocol=(
            "none" if method == MethodName.BF16_VALUE_FP32_COMPUTE
            else "multimodal_train_4x32" if method == MethodName.MBQ
            else "c4_128x2048"
        ),
        calibration_samples=2 if args.preset == "smoke" else 128,
        awq_alpha_grid=(0.0, 0.5, 1.0) if args.preset == "smoke" else ExperimentConfig.awq_alpha_grid,
        awq_clip_grid=(1.0, 0.9) if args.preset == "smoke" else ExperimentConfig.awq_clip_grid,
        omniquant_steps=2 if args.preset == "smoke" else 20,
        squeezellm_kmeans_steps=1 if args.preset == "smoke" else 5,
    )
    if method == MethodName.MBQ and args.calibration_manifest is None:
        raise ValueError("MBQ corpus evaluation requires --calibration-manifest")
    model_digest = source_digest(args.model_path)
    corpus_files = {
        "wikitext2_test": args.corpora_dir / "wikitext2_test.pt",
        "ptb_test": args.corpora_dir / "ptb_test.pt",
        "c4_validation": args.corpora_dir / "c4_validation_256x2048.pt",
    }
    input_digests = {
        "model": model_digest,
        "calibration": source_digest(
            args.calibration_manifest
            if method == MethodName.MBQ
            else args.corpora_dir / "c4_calibration_128x2048.pt"
        ),
        "preset": sha256_json(args.preset),
        "max_windows": sha256_json({"smoke": 2, "screen": 16, "full": None}[args.preset]),
        **{name: source_digest(path) for name, path in corpus_files.items()},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and completed_run(
        args.output_dir, config_digest=config.digest, input_digests=input_digests,
        required_artifacts=("corpus_metrics.json",),
    ):
        return
    partial_path = args.output_dir / "corpus_metrics.partial.json"
    for stale in (".complete", "corpus_metrics.json", "error.json"):
        (args.output_dir / stale).unlink(missing_ok=True)
    if not args.resume:
        partial_path.unlink(missing_ok=True)
    atomic_json(args.output_dir / "config.json", config.to_dict() | {
        "config_hash": config.digest,
        "input_digests": input_digests,
        "environment": environment_record(),
    })
    try:
        model, processor, conversion = load_reference(args.model_path)
        atomic_json(args.output_dir / "reference_conversion.json", conversion)
        selected = select_linears(model, scope, strict_qwen_3b=not args.allow_nonstandard_model)
        atomic_json(args.output_dir / "module_inventory.json", inventory(selected))
        calibration, cache_hit = None, False
        if method == MethodName.MBQ:
            cache_dir = (
                args.calibration_cache_dir or args.output_dir.parent / "_calibration_cache"
            )
            samples = load_manifest(args.calibration_manifest)
            samples = samples[:config.calibration_samples]
            if len(samples) != config.calibration_samples:
                raise ValueError("MBQ calibration manifest does not contain enough samples")
            key = mbq_cache_key(
                model_revision=model_digest,
                calibration_digest=sample_manifest_digest(samples),
                selected_names=[item.name for item in selected],
                config=config,
                reservoir_rows=32,
                attention_implementation="sdpa",
            )
            calibration, cache_hit = collect_mbq_search_cached(
                model, selected,
                mbq_calibration_batches(
                    processor, samples,
                    min_pixels=config.min_pixels, max_pixels=config.max_pixels,
                ),
                config,
                cache_dir=cache_dir,
                key=key,
                raw_key=mbq_statistics_cache_key(
                    model_revision=model_digest,
                    calibration_digest=sample_manifest_digest(samples),
                    selected_names=[item.name for item in selected],
                    reservoir_rows=32,
                    seed=config.seed,
                    min_pixels=config.min_pixels,
                    max_pixels=config.max_pixels,
                    attention_implementation="sdpa",
                ),
                reservoir_rows=32,
                attention_implementation="sdpa",
            )
        elif method in CALIBRATION_METHODS:
            sequences = load_sequences(args.corpora_dir / "c4_calibration_128x2048.pt")
            if args.preset == "smoke":
                sequences = sequences[:2]
            calibration_subset_digest = sha256_json({
                "source": input_digests["calibration"],
                "documents": len(sequences),
                "token_lengths": [int(sequence.numel()) for sequence in sequences],
            })
            key = calibration_cache_key(
                model_revision=model_digest,
                calibration_digest=calibration_subset_digest,
                selected_names=[item.name for item in selected],
                min_pixels=config.min_pixels,
                max_pixels=config.max_pixels,
                reservoir_rows=32,
                seed=config.seed,
            )
            calibration, cache_hit = collect_calibration_cached(
                model,
                selected,
                ({"input_ids": sequence.unsqueeze(0)} for sequence in sequences),
                cache_dir=args.calibration_cache_dir or args.output_dir.parent / "_calibration_cache",
                key=key,
                reservoir_rows=32,
                seed=config.seed,
            )
        applied = apply_method(model, selected, config, calibration)
        try:
            results = (
                load_partial_results(
                    partial_path,
                    config_digest=config.digest,
                    input_digests=input_digests,
                )
                if args.resume else {}
            )
            unexpected = set(results) - set(corpus_files)
            if unexpected:
                raise ValueError(f"corpus partial contains unexpected datasets: {sorted(unexpected)}")
            max_windows = {"smoke": 2, "screen": 16, "full": None}[args.preset]
            for name, path in corpus_files.items():
                if name in results:
                    continue
                results[name] = evaluate_token_sequences_ppl(
                    model,
                    load_sequences(path),
                    sequence_length=2048,
                    batch_size=args.batch_size,
                    max_windows=max_windows,
                )
                save_partial_results(
                    partial_path, results,
                    config_digest=config.digest,
                    input_digests=input_digests,
                )
            if args.reference_metrics is not None:
                reference = json.loads(args.reference_metrics.read_text())
                for name in results:
                    results[name]["paired_vs_bf16"] = paired_window_delta(
                        results[name], reference[name], seed=config.seed,
                    )
            atomic_json(args.output_dir / "corpus_metrics.json", results)
            applied.audit["calibration_cache_hit"] = cache_hit
            atomic_json(args.output_dir / "quantization_audit.json", applied.audit)
            mark_complete(
                args.output_dir,
                config_digest=config.digest,
                input_digests=input_digests,
                required_artifacts=("corpus_metrics.json",),
            )
            partial_path.unlink(missing_ok=True)
        finally:
            applied.restore()
    except Exception as error:
        atomic_json(args.output_dir / "error.json", {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "config_hash": config.digest,
            "input_digests": input_digests,
        })
        raise


if __name__ == "__main__":
    main()
