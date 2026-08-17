from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import torch

from .config import ExperimentConfig
from .data import assert_disjoint, fixed_subset, load_manifest, sample_manifest_digest
from .provenance import completed_run, run_input_digests, sha256_json
from .reporting import atomic_json
from .runner import load_reference, run_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one isolated Qwen2.5-VL quantization config")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-cache-dir", type=Path, required=True)
    parser.add_argument("--input-digests-json", type=Path)
    parser.add_argument("--preset", choices=("smoke", "screen", "full"), default="smoke")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-nonstandard-model", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    config = ExperimentConfig.from_dict(json.loads(args.config_json.read_text()))
    calibration = load_manifest(args.calibration_manifest) if args.calibration_manifest else []
    evaluation = load_manifest(args.evaluation_manifest)
    if calibration:
        assert_disjoint(calibration, evaluation)
    limit = {"smoke": 2, "screen": 256, "full": len(evaluation)}[args.preset]
    evaluation = fixed_subset(evaluation, limit, config.seed)
    digests = (
        json.loads(args.input_digests_json.read_text())
        if args.input_digests_json is not None
        else run_input_digests(
            model_source=args.model_path,
            calibration_manifest=args.calibration_manifest,
            evaluation_manifest=args.evaluation_manifest,
            processor_revision=config.processor_revision,
        )
    )
    digests = digests | {
        "preset": sha256_json(args.preset),
        "evaluation_subset": sample_manifest_digest(evaluation),
        "calibration_subset": (
            sample_manifest_digest(calibration[:config.calibration_samples])
            if calibration else ""
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and completed_run(
        args.output_dir,
        config_digest=config.digest,
        input_digests=digests,
    ):
        return
    try:
        model, processor, conversion = load_reference(args.model_path)
        atomic_json(args.output_dir / "reference_conversion.json", conversion)
        run_config(
            model,
            processor,
            config,
            calibration,
            evaluation,
            args.output_dir,
            strict_qwen_3b=not args.allow_nonstandard_model,
            resume=args.resume,
            input_digests=digests,
            calibration_cache_dir=args.calibration_cache_dir,
        )
    except Exception as error:
        atomic_json(args.output_dir / "error.json", {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "config_hash": config.digest,
            "input_digests": digests,
        })
        raise


if __name__ == "__main__":
    main()
