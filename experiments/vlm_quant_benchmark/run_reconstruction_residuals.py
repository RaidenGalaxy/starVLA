"""Collect formal BF16/MBQ/SqueezeLLM block and projection residuals."""
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
from .config import MethodName, QuantScope
from .data import load_manifest, sample_manifest_digest
from .mbq import (
    collect_mbq_search_cached,
    mbq_cache_key,
    mbq_calibration_batches,
    mbq_statistics_cache_key,
)
from .provenance import source_digest
from .reporting import atomic_json
from .residuals import (
    RESIDUAL_FORMAT_VERSION,
    aggregate_mbq_block_rms,
    capture_reconstruction_outputs,
    reconstruction_rows,
)
from .run_sensitivity_ablation import ablation_configs
from .runner import calibration_batches, environment_record, load_reference
from .selection import select_linears
from .sensitivity import sensitivity_anchors, stable_hash_subset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect W4 reconstruction residuals")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-cache-dir", type=Path)
    parser.add_argument("--calibration-samples", type=int, default=128)
    parser.add_argument("--target-per-dataset", type=int, default=32)
    parser.add_argument("--samples-per-dataset", type=int, default=8)
    parser.add_argument("--layers", default="0,6,12,18,24,30,34")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--fast-smoke", action="store_true",
        help="Preflight only: use a two-point MBQ grid and one k-means step.",
    )
    parser.add_argument("--allow-nonstandard-model", action="store_true")
    return parser.parse_args()


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_sample_rows(cache_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(cache_dir.glob("*.json")):
        if path.name.endswith(".error.json"):
            continue
        payload = json.loads(path.read_text())
        if payload.get("status") != "complete":
            raise RuntimeError(f"incomplete residual cache: {path}")
        rows.extend(payload["rows"])
    return rows


def main() -> None:
    args = parse_args()
    layers = tuple(int(value) for value in args.layers.split(",") if value.strip())
    calibration = load_manifest(args.calibration_manifest)[:args.calibration_samples]
    if len(calibration) != args.calibration_samples:
        raise ValueError("calibration manifest does not contain enough samples")
    evaluation = load_manifest(args.evaluation_manifest)
    target = stable_hash_subset(
        evaluation, args.target_per_dataset, salt=f"sensitivity:{args.seed}",
    )
    samples = stable_hash_subset(
        target, args.samples_per_dataset, salt=f"perturbation:{args.seed}",
    )
    variants = {name: config for name, config, _source in ablation_configs(
        calibration_samples=args.calibration_samples, seed=args.seed,
    )}
    mbq_config = variants["mbq_current"]
    squeeze_config = variants["squeezellm_current"]
    assert mbq_config is not None and squeeze_config is not None
    if args.fast_smoke:
        mbq_config = dataclasses.replace(
            mbq_config, mbq_scale_grid=(0.0, 0.5),
            scheme_id="mbq_preflight_fast_w4a16",
        )
        squeeze_config = dataclasses.replace(
            squeeze_config, squeezellm_kmeans_steps=1,
            scheme_id="squeezellm_preflight_fast_w4a16",
        )
    cache_identity = hashlib.sha256(
        f"{RESIDUAL_FORMAT_VERSION}:{mbq_config.digest}:{squeeze_config.digest}:"
        f"{','.join(map(str, layers))}".encode()
    ).hexdigest()[:20]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "residual_plan.json", {
        "format_version": RESIDUAL_FORMAT_VERSION,
        "environment": environment_record(),
        "models": ["bf16", "mbq_current", "squeezellm_current"],
        "layers": list(layers),
        "sample_ids": [sample.id for sample in samples],
        "sample_count": len(samples),
        "mbq_config": mbq_config.to_dict(),
        "squeezellm_config": squeeze_config.to_dict(),
        "cache_identity": cache_identity,
        "preflight_fast_search": args.fast_smoke,
    })
    if args.dry_run:
        return

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    model_digest = source_digest(args.model_path)
    bf16_model, processor, _conversion = load_reference(args.model_path)
    mbq_model, mbq_processor, _ = load_reference(args.model_path)
    squeeze_model, squeeze_processor, _ = load_reference(args.model_path)
    del mbq_processor, squeeze_processor
    strict = not args.allow_nonstandard_model
    bf16_selected = select_linears(bf16_model, QuantScope.DECODER_CANONICAL, strict_qwen_3b=strict)
    mbq_selected = select_linears(mbq_model, QuantScope.DECODER_CANONICAL, strict_qwen_3b=strict)
    squeeze_selected = select_linears(squeeze_model, QuantScope.DECODER_CANONICAL, strict_qwen_3b=strict)
    anchor_names = [
        item.name for item in sensitivity_anchors(bf16_selected)
        if any(f".layers.{layer}." in item.name for layer in layers)
    ]
    if len(anchor_names) != len(layers) * 2:
        raise ValueError(f"expected {len(layers) * 2} anchors, found {len(anchor_names)}")
    cache_dir = args.calibration_cache_dir or args.output_dir / "_calibration_cache"
    calibration_digest = sample_manifest_digest(calibration)
    selected_names = [item.name for item in mbq_selected]
    mbq_key = mbq_cache_key(
        model_revision=model_digest,
        calibration_digest=calibration_digest,
        selected_names=selected_names,
        config=mbq_config,
        reservoir_rows=32,
        attention_implementation="sdpa",
    )
    mbq_stats, _ = collect_mbq_search_cached(
        mbq_model, mbq_selected,
        mbq_calibration_batches(
            processor, calibration,
            min_pixels=mbq_config.min_pixels, max_pixels=mbq_config.max_pixels,
        ),
        mbq_config,
        cache_dir=cache_dir,
        key=mbq_key,
        raw_key=mbq_statistics_cache_key(
            model_revision=model_digest,
            calibration_digest=calibration_digest,
            selected_names=selected_names,
            reservoir_rows=32,
            seed=mbq_config.seed,
            min_pixels=mbq_config.min_pixels,
            max_pixels=mbq_config.max_pixels,
            attention_implementation="sdpa",
            modality_mode=mbq_config.mbq_modality_mode,
            algorithm_version=mbq_config.mbq_algorithm_version,
        ),
        reservoir_rows=32,
        attention_implementation="sdpa",
    )
    squeeze_names = [item.name for item in squeeze_selected]
    squeeze_key = calibration_cache_key(
        model_revision=model_digest,
        calibration_digest=calibration_digest,
        selected_names=squeeze_names,
        min_pixels=squeeze_config.min_pixels,
        max_pixels=squeeze_config.max_pixels,
        reservoir_rows=32,
        seed=squeeze_config.seed,
    )
    squeeze_stats, _ = collect_calibration_cached(
        squeeze_model, squeeze_selected,
        calibration_batches(
            processor, calibration,
            min_pixels=squeeze_config.min_pixels,
            max_pixels=squeeze_config.max_pixels,
        ),
        cache_dir=cache_dir,
        key=squeeze_key,
        reservoir_rows=32,
        seed=squeeze_config.seed,
    )
    mbq_applied = apply_method(mbq_model, mbq_selected, mbq_config, mbq_stats)
    squeeze_applied = apply_method(
        squeeze_model, squeeze_selected, squeeze_config, squeeze_stats,
    )
    sample_cache = args.output_dir / "_sample_cache" / cache_identity
    sample_cache.mkdir(parents=True, exist_ok=True)
    try:
        for sample in samples:
            digest = hashlib.sha256(sample.id.encode()).hexdigest()
            path = sample_cache / f"{digest}.json"
            if args.resume and path.is_file():
                continue
            try:
                batch = next(mbq_calibration_batches(
                    processor, [sample],
                    min_pixels=mbq_config.min_pixels,
                    max_pixels=mbq_config.max_pixels,
                ))
                reference = capture_reconstruction_outputs(
                    bf16_model, batch, layers_to_test=layers, anchor_names=anchor_names,
                )
                mbq = capture_reconstruction_outputs(
                    mbq_model, batch, layers_to_test=layers, anchor_names=anchor_names,
                )
                squeeze = capture_reconstruction_outputs(
                    squeeze_model, batch, layers_to_test=layers, anchor_names=anchor_names,
                )
                rows = []
                for method, candidate in (
                    ("bf16", reference), ("mbq_current", mbq),
                    ("squeezellm_current", squeeze),
                ):
                    rows.extend(reconstruction_rows(
                        batch, reference, candidate, method=method,
                    ))
                atomic_json(path, {
                    "status": "complete", "id": sample.id, "rows": rows,
                })
            except Exception as error:
                atomic_json(path.with_suffix(".error.json"), {
                    "status": "failed", "id": sample.id,
                    "error_type": type(error).__name__, "error": str(error),
                })
                raise
    finally:
        mbq_applied.restore()
        squeeze_applied.restore()
        del bf16_model, mbq_model, squeeze_model, processor
    rows = _load_sample_rows(sample_cache)
    layer_rms = aggregate_mbq_block_rms(rows)
    _atomic_jsonl(args.output_dir / "reconstruction_residuals.jsonl", rows)
    atomic_json(args.output_dir / "mbq_layer_residual_rms.json", {
        "format_version": RESIDUAL_FORMAT_VERSION,
        "location": "decoder_block_output",
        "method": "mbq_current",
        "aggregation": f"pooled_elements_across_{len(samples)}_samples",
        "layer_rms": {str(key): value for key, value in layer_rms.items()},
    })
    atomic_json(args.output_dir / "run_audit.json", {
        "format_version": RESIDUAL_FORMAT_VERSION,
        "status": "complete",
        "sample_count": len(samples),
        "rows": len(rows),
        "layer_rms": {str(key): value for key, value in layer_rms.items()},
        "elapsed_seconds": time.perf_counter() - started,
    })


if __name__ == "__main__":
    main()
