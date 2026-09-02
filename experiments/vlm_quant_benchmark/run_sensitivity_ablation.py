"""Resumable W4 teacher-forced ablations for MBQ sensitivity attribution."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import time
from pathlib import Path

import torch

from .all_methods import apply_method
from .calibration import calibration_cache_key, collect_calibration_cached
from .config import ExperimentConfig, ImplementationTier, MethodName, QuantScope
from .data import load_manifest, sample_manifest_digest
from .mbq import (
    collect_mbq_search_cached,
    mbq_cache_key,
    mbq_calibration_batches,
    mbq_statistics_cache_key,
)
from .provenance import source_digest
from .reporting import atomic_json
from .runner import calibration_batches, environment_record, load_reference
from .selection import select_linears
from .sensitivity import evaluate_sample_target_nll, stable_hash_subset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run controlled W4 MBQ/SqueezeLLM sensitivity ablations",
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-cache-dir", type=Path)
    parser.add_argument("--calibration-samples", type=int, default=128)
    parser.add_argument("--adaptation-per-dataset", type=int, default=32)
    parser.add_argument("--nll-per-dataset", type=int, default=32)
    parser.add_argument("--promotion-per-dataset", type=int, default=64)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--variants",
        help="Optional comma-separated variant names; default executes all.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--oracle-promotion-nll", action="store_true",
        help="Also evaluate target-oracle NLL on the disjoint promotion set.",
    )
    parser.add_argument("--allow-nonstandard-model", action="store_true")
    return parser.parse_args()


def ablation_configs(
    *,
    calibration_samples: int,
    seed: int,
) -> list[tuple[str, ExperimentConfig | None, str]]:
    common = {
        "scope": QuantScope.DECODER_CANONICAL,
        "weight_bits": 4,
        "activation_bits": None,
        "group_size": 128,
        "zero_point": True,
        "weight_granularity": "groupwise",
        "calibration_samples": calibration_samples,
        "calibration_protocol": "multimodal_train_4x32",
        "seed": seed,
        "implementation_tier": ImplementationTier.LOCAL_VALIDATED,
    }
    base = ExperimentConfig(
        MethodName.MBQ, scheme_id="mbq_current_w4a16", **common,
    )
    variants = [
        ("bf16", None, "reference"),
        ("mbq_current", base, "calibration"),
        (
            "mbq_no_reweight",
            dataclasses.replace(
                base, mbq_reweight=False, scheme_id="mbq_no_reweight_w4a16",
            ),
            "calibration",
        ),
        (
            "mbq_sample_average",
            dataclasses.replace(
                base,
                mbq_gradient_aggregation="sample_mean",
                scheme_id="mbq_sample_average_w4a16",
            ),
            "calibration",
        ),
        (
            "mbq_modality_normalized",
            dataclasses.replace(
                base,
                mbq_objective_normalization="modality_mean",
                scheme_id="mbq_modality_normalized_w4a16",
            ),
            "calibration",
        ),
        (
            "mbq_no_median_floor",
            dataclasses.replace(
                base,
                mbq_median_floor=False,
                scheme_id="mbq_no_median_floor_w4a16",
            ),
            "calibration",
        ),
        (
            "mbq_block_reconstruction",
            dataclasses.replace(
                base,
                mbq_reconstruction_level="block_output",
                scheme_id="mbq_block_reconstruction_w4a16",
            ),
            "unsupported_backend",
        ),
        (
            "squeezellm_current",
            ExperimentConfig(
                MethodName.SQUEEZELLM,
                scheme_id="squeezellm_current_w4a16",
                implementation_tier=ImplementationTier.CPU_RESEARCH,
                **{key: value for key, value in common.items()
                   if key != "implementation_tier"},
            ),
            "calibration",
        ),
        (
            "squeezellm_no_sparse",
            ExperimentConfig(
                MethodName.SQUEEZELLM,
                scheme_id="squeezellm_no_sparse_w4a16",
                squeezellm_sparse_fraction=0.0,
                implementation_tier=ImplementationTier.CPU_RESEARCH,
                **{key: value for key, value in common.items()
                   if key != "implementation_tier"},
            ),
            "calibration",
        ),
        (
            "mbq_target_oracle",
            dataclasses.replace(
                base,
                calibration_samples=4 * 32,
                calibration_protocol="target_domain_diagnostic_oracle",
                scheme_id="mbq_target_oracle_w4a16",
            ),
            "target_adaptation",
        ),
    ]
    return variants


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _sample_cache_path(
    result_dir: Path, variant: str, config_digest: str, sample_id: str,
) -> Path:
    digest = hashlib.sha256(sample_id.encode()).hexdigest()
    return result_dir / variant / config_digest / f"{digest}.json"


def _evaluate_samples_cached(
    model,
    processor,
    samples,
    *,
    variant: str,
    config_digest: str,
    result_dir: Path,
    analysis_ids: set[str],
    promotion_ids: set[str],
    min_pixels: int,
    max_pixels: int,
    resume: bool,
) -> list[dict]:
    """Evaluate and atomically commit one NLL sample at a time."""
    rows = []
    for sample in samples:
        path = _sample_cache_path(result_dir, variant, config_digest, sample.id)
        if resume and path.is_file():
            payload = json.loads(path.read_text())
            if payload.get("status") != "complete":
                raise RuntimeError(f"incomplete cached sample: {path}")
            rows.extend(payload["rows"])
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            batch = mbq_calibration_batches(
                processor, [sample], min_pixels=min_pixels, max_pixels=max_pixels,
            )
            sample_rows = evaluate_sample_target_nll(
                model, batch, variant=variant,
            )
            sample_rows = [
                row | {
                    "evaluation_split": (
                        "sensitivity" if row["id"] in analysis_ids else
                        "diagnostic_promotion_oracle"
                        if row["id"] in promotion_ids else "ablation"
                    ),
                }
                for row in sample_rows
            ]
            atomic_json(path, {
                "status": "complete",
                "variant": variant,
                "config_digest": config_digest,
                "id": sample.id,
                "rows": sample_rows,
            })
        except Exception as error:
            atomic_json(path.with_suffix(".error.json"), {
                "status": "failed",
                "variant": variant,
                "config_digest": config_digest,
                "id": sample.id,
                "error_type": type(error).__name__,
                "error": str(error),
            })
            raise
        rows.extend(sample_rows)
    return rows


def main() -> None:
    args = parse_args()
    if min(
        args.calibration_samples,
        args.adaptation_per_dataset,
        args.nll_per_dataset,
        args.promotion_per_dataset,
    ) <= 0:
        raise ValueError("all ablation sample counts must be positive")
    calibration_all = load_manifest(args.calibration_manifest)
    if len(calibration_all) < args.calibration_samples:
        raise ValueError("calibration manifest does not contain enough samples")
    calibration = calibration_all[:args.calibration_samples]
    evaluation = load_manifest(args.evaluation_manifest)
    adaptation = stable_hash_subset(
        evaluation, args.adaptation_per_dataset,
        salt=f"sensitivity:{args.seed}",
    )
    adaptation_ids = {sample.id for sample in adaptation}
    nll_samples = stable_hash_subset(
        evaluation, args.nll_per_dataset,
        salt=f"ablation-nll:{args.seed}",
        exclude_ids=adaptation_ids,
    )
    used_ids = adaptation_ids | {sample.id for sample in nll_samples}
    promotion = stable_hash_subset(
        evaluation, args.promotion_per_dataset,
        salt=f"ablation-promotion:{args.seed}",
        exclude_ids=used_ids,
    )
    configs = ablation_configs(
        calibration_samples=args.calibration_samples,
        seed=args.seed,
    )
    plan_configs = list(configs)
    if args.variants:
        requested_variants = {
            value.strip() for value in args.variants.split(",") if value.strip()
        }
        known_variants = {name for name, _config, _source in configs}
        missing = requested_variants - known_variants
        if missing:
            raise ValueError(f"unknown ablation variants: {sorted(missing)}")
        configs = [item for item in configs if item[0] in requested_variants]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "ablation_plan.json", {
        "format_version": 1,
        "environment": environment_record(),
        "selection_is_outcome_independent": True,
        "adaptation_ids": [sample.id for sample in adaptation],
        "nll_ids": [sample.id for sample in nll_samples],
        "promotion_ids": [sample.id for sample in promotion],
        "sets_are_disjoint": not (
            adaptation_ids & {sample.id for sample in nll_samples}
            or used_ids & {sample.id for sample in promotion}
        ),
        "variants": [
            {
                "variant": name,
                "calibration_source": source,
                "executable": source != "unsupported_backend",
                "config": None if config is None else config.to_dict(),
                "config_hash": None if config is None else config.digest,
            }
            for name, config, source in plan_configs
        ],
        "promotion_rule": (
            "Select the two non-reference variants with the largest paired "
            "target-NLL improvement over mbq_current; generation is a separate "
            "explicit stage on promotion_ids."
        ),
    })
    if args.dry_run:
        return

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    model_digest = source_digest(args.model_path)
    model, processor, _conversion = load_reference(args.model_path)
    selected = select_linears(
        model, QuantScope.DECODER_CANONICAL,
        strict_qwen_3b=not args.allow_nonstandard_model,
    )
    cache_dir = args.calibration_cache_dir or args.output_dir / "_calibration_cache"
    result_dir = args.output_dir / "_variant_cache"
    result_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        for variant, config, source in configs:
            summary_path = result_dir / f"{variant}.json"
            if source == "unsupported_backend":
                atomic_json(summary_path, {
                    "variant": variant,
                    "status": "not_executed",
                    "reason": (
                        "The local CPU port does not yet provide an exact "
                        "attention/MLP block-output reconstruction backend."
                    ),
                    "rows": [],
                })
                continue
            evaluation_samples = (
                [*adaptation, *nll_samples]
                if variant in {"bf16", "mbq_current", "squeezellm_current"}
                else nll_samples
            )
            if variant == "mbq_target_oracle" and args.oracle_promotion_nll:
                evaluation_samples = [*nll_samples, *promotion]
            analysis_ids = {sample.id for sample in adaptation}
            promotion_ids = {sample.id for sample in promotion}
            if config is None:
                rows = _evaluate_samples_cached(
                    model, processor, evaluation_samples,
                    variant=variant,
                    config_digest="bf16_reference",
                    result_dir=result_dir,
                    analysis_ids=analysis_ids,
                    promotion_ids=promotion_ids,
                    min_pixels=256 * 28 * 28,
                    max_pixels=1280 * 28 * 28,
                    resume=args.resume,
                )
                atomic_json(summary_path, {
                    "variant": variant, "status": "complete", "rows": len(rows),
                    "config_digest": "bf16_reference",
                })
                continue
            calibration_samples = adaptation if source == "target_adaptation" else calibration
            if source == "target_adaptation":
                config = dataclasses.replace(
                    config, calibration_samples=len(calibration_samples),
                )
            calibration_digest = sample_manifest_digest(calibration_samples)
            if config.method == MethodName.MBQ:
                key = mbq_cache_key(
                    model_revision=model_digest,
                    calibration_digest=calibration_digest,
                    selected_names=[item.name for item in selected],
                    config=config,
                    reservoir_rows=32,
                    attention_implementation="sdpa",
                )
                stats, _cache_hit = collect_mbq_search_cached(
                    model, selected,
                    mbq_calibration_batches(
                        processor, calibration_samples,
                        min_pixels=config.min_pixels,
                        max_pixels=config.max_pixels,
                    ),
                    config,
                    cache_dir=cache_dir,
                    key=key,
                    raw_key=mbq_statistics_cache_key(
                        model_revision=model_digest,
                        calibration_digest=calibration_digest,
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
            else:
                key = calibration_cache_key(
                    model_revision=model_digest,
                    calibration_digest=calibration_digest,
                    selected_names=[item.name for item in selected],
                    min_pixels=config.min_pixels,
                    max_pixels=config.max_pixels,
                    reservoir_rows=32,
                    seed=config.seed,
                )
                stats, _cache_hit = collect_calibration_cached(
                    model, selected,
                    calibration_batches(
                        processor, calibration_samples,
                        min_pixels=config.min_pixels,
                        max_pixels=config.max_pixels,
                    ),
                    cache_dir=cache_dir,
                    key=key,
                    reservoir_rows=32,
                    seed=config.seed,
                )
            applied = apply_method(model, selected, config, stats)
            try:
                rows = _evaluate_samples_cached(
                    model, processor, evaluation_samples,
                    variant=variant,
                    config_digest=config.digest,
                    result_dir=result_dir,
                    analysis_ids=analysis_ids,
                    promotion_ids=promotion_ids,
                    min_pixels=config.min_pixels,
                    max_pixels=config.max_pixels,
                    resume=args.resume,
                )
                atomic_json(summary_path, {
                    "variant": variant,
                    "status": "complete",
                    "config": config.to_dict(),
                    "config_digest": config.digest,
                    "rows": len(rows),
                })
            finally:
                applied.restore()
    finally:
        del model, processor
    rows = []
    statuses = []
    for path in sorted(result_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        statuses.append({
            "variant": payload["variant"],
            "status": payload["status"],
            "reason": payload.get("reason", ""),
        })
        config_digest = payload.get("config_digest")
        if payload["status"] == "complete" and config_digest:
            for sample_path in sorted(
                (result_dir / payload["variant"] / config_digest).glob("*.json")
            ):
                if sample_path.name.endswith(".error.json"):
                    continue
                rows.extend(json.loads(sample_path.read_text()).get("rows", []))
    _atomic_jsonl(args.output_dir / "ablation_results.jsonl", rows)
    atomic_json(args.output_dir / "run_audit.json", {
        "format_version": 1,
        "statuses": statuses,
        "rows": len(rows),
        "elapsed_seconds": time.perf_counter() - started,
    })


if __name__ == "__main__":
    main()
