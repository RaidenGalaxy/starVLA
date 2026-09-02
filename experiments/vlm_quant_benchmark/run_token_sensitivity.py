"""Resumable CPU entrypoint for sample-level VLM modality sensitivity."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from .data import load_manifest, sample_manifest_digest
from .mbq import mbq_calibration_batches
from .provenance import source_digest
from .reporting import atomic_json
from .runner import environment_record, load_reference
from .selection import inventory, select_linears
from .sensitivity import (
    SENSITIVITY_FORMAT_VERSION,
    collect_modality_perturbations,
    collect_sample_sensitivity,
    stable_hash_subset,
)
from .config import QuantScope


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Qwen2.5-VL vision/language token sensitivity",
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-samples", type=int, default=128)
    parser.add_argument("--target-per-dataset", type=int, default=32)
    parser.add_argument(
        "--datasets",
        help="Optional comma-separated evaluation datasets; default uses all.",
    )
    parser.add_argument("--perturbation-per-dataset", type=int, default=8)
    parser.add_argument("--perturbation-layers", default="0,6,12,18,24,30,34")
    parser.add_argument(
        "--perturbation-rms", type=float, default=0.0,
        help="Hidden-state noise RMS; zero disables causal perturbations.",
    )
    parser.add_argument(
        "--perturbation-rms-json", type=Path,
        help=(
            "JSON mapping decoder layer to the measured MBQ block-output RMS; "
            "takes precedence over --perturbation-rms."
        ),
    )
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-nonstandard-model", action="store_true")
    return parser.parse_args()


def _sample_key(kind: str, identifier: str) -> str:
    return hashlib.sha256(f"{kind}:{identifier}".encode()).hexdigest()


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_cache_rows(cache_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(cache_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        rows.extend(payload["rows"])
    return rows


def main() -> None:
    args = parse_args()
    if args.calibration_samples <= 0 or args.target_per_dataset <= 0:
        raise ValueError("sample counts must be positive")
    if args.perturbation_per_dataset < 0 or args.perturbation_rms < 0:
        raise ValueError("perturbation settings must be non-negative")
    layers = tuple(
        int(value) for value in args.perturbation_layers.split(",") if value.strip()
    )
    noise_rms: dict[int, float] | float = args.perturbation_rms
    if args.perturbation_rms_json:
        raw_rms = json.loads(args.perturbation_rms_json.read_text())
        raw_rms = raw_rms.get("layer_rms", raw_rms)
        noise_rms = {int(key): float(value) for key, value in raw_rms.items()}
        missing_layers = set(layers) - set(noise_rms)
        if missing_layers:
            raise ValueError(f"RMS JSON is missing layers: {sorted(missing_layers)}")
        if any(noise_rms[layer] <= 0 for layer in layers):
            raise ValueError("all measured perturbation RMS values must be positive")
    rms_payload = (
        {str(key): noise_rms[key] for key in sorted(noise_rms)}
        if isinstance(noise_rms, dict) else noise_rms
    )
    rms_digest = hashlib.sha256(
        json.dumps(rms_payload, sort_keys=True).encode()
    ).hexdigest()[:16]
    calibration_all = load_manifest(args.calibration_manifest)
    if len(calibration_all) < args.calibration_samples:
        raise ValueError("calibration manifest does not contain enough samples")
    calibration = calibration_all[:args.calibration_samples]
    evaluation_all = load_manifest(args.evaluation_manifest)
    if args.datasets:
        requested = {value.strip() for value in args.datasets.split(",") if value.strip()}
        available = {sample.dataset for sample in evaluation_all}
        missing = requested - available
        if missing:
            raise ValueError(f"evaluation datasets not found: {sorted(missing)}")
        evaluation_all = [
            sample for sample in evaluation_all if sample.dataset in requested
        ]
    target = stable_hash_subset(
        evaluation_all, args.target_per_dataset,
        salt=f"sensitivity:{args.seed}",
    )
    perturbation = (
        stable_hash_subset(
            target, args.perturbation_per_dataset,
            salt=f"perturbation:{args.seed}",
        )
        if args.perturbation_per_dataset else []
    )
    selected_rows = [
        {
            "kind": kind,
            "id": sample.id,
            "dataset": sample.dataset,
            "official_index": sample.official_index,
        }
        for kind, samples in (("calibration", calibration), ("target", target))
        for sample in samples
    ]
    input_digests = {
        "calibration": sample_manifest_digest(calibration),
        "target": sample_manifest_digest(target),
        "model": (
            str(Path(args.model_path).resolve())
            if args.dry_run else source_digest(args.model_path)
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "analysis_plan.json", {
        "format_version": 1,
        "environment": environment_record(),
        "input_digests": input_digests,
        "selection_is_outcome_independent": True,
        "calibration_samples": len(calibration),
        "target_samples": len(target),
        "perturbation_samples": len(perturbation),
        "perturbation_layers": list(layers),
        "perturbation_rms": rms_payload,
        "perturbation_location": "decoder_block_output",
        "selected": selected_rows,
    })
    if args.dry_run:
        return

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    model, processor, conversion = load_reference(args.model_path)
    selected = select_linears(
        model, QuantScope.DECODER_CANONICAL,
        strict_qwen_3b=not args.allow_nonstandard_model,
    )
    atomic_json(args.output_dir / "module_inventory.json", inventory(selected))
    gradient_cache = (
        args.output_dir / "_sample_cache"
        / f"gradients_v{SENSITIVITY_FORMAT_VERSION}"
    )
    perturbation_cache = (
        args.output_dir / "_sample_cache"
        / f"perturbations_v{SENSITIVITY_FORMAT_VERSION}_{rms_digest}"
    )
    gradient_cache.mkdir(parents=True, exist_ok=True)
    perturbation_cache.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        for kind, samples in (("calibration", calibration), ("target", target)):
            for sample in samples:
                path = gradient_cache / f"{_sample_key(kind, sample.id)}.json"
                if args.resume and path.is_file():
                    continue
                batch = next(mbq_calibration_batches(
                    processor, [sample],
                    min_pixels=args.min_pixels,
                    max_pixels=args.max_pixels,
                ))
                rows = collect_sample_sensitivity(model, selected, [batch])
                atomic_json(path, {
                    "kind": kind,
                    "id": sample.id,
                    "rows": [row | {"sample_kind": kind} for row in rows],
                })
        if isinstance(noise_rms, dict) or noise_rms > 0:
            for sample in perturbation:
                path = perturbation_cache / f"{_sample_key('target', sample.id)}.json"
                if args.resume and path.is_file():
                    continue
                batch = next(mbq_calibration_batches(
                    processor, [sample],
                    min_pixels=args.min_pixels,
                    max_pixels=args.max_pixels,
                ))
                rows = collect_modality_perturbations(
                    model, batch,
                    layers_to_test=layers,
                    noise_rms=noise_rms,
                    seed=args.seed,
                )
                atomic_json(path, {"id": sample.id, "rows": rows})
    finally:
        del model, processor

    gradient_rows = _load_cache_rows(gradient_cache)
    perturbation_rows = _load_cache_rows(perturbation_cache)
    _atomic_jsonl(args.output_dir / "sample_sensitivity.jsonl", gradient_rows)
    _atomic_jsonl(args.output_dir / "perturbations.jsonl", perturbation_rows)
    atomic_json(args.output_dir / "run_audit.json", {
        "format_version": 1,
        "status": "complete",
        "input_digests": input_digests,
        "reference_conversion": conversion,
        "gradient_rows": len(gradient_rows),
        "perturbation_rows": len(perturbation_rows),
        "elapsed_seconds": time.perf_counter() - started,
    })


if __name__ == "__main__":
    main()
