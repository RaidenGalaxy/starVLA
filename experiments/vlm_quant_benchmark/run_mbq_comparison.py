"""Opt-in matched MBQ comparison matrix; planning is the default behavior."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .config import ExperimentConfig, ImplementationTier, MethodName, QuantScope
from .data import assert_disjoint, load_manifest, manifest_digest
from .provenance import run_input_digests
from .reporting import atomic_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Matched MBQ W4A16/W4A8 comparison")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-cache-dir", type=Path)
    parser.add_argument("--preset", choices=("smoke", "screen", "full"), default="smoke")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--execute", action="store_true",
        help="execute serial child runs; without this flag only write the plan",
    )
    parser.add_argument("--allow-nonstandard-model", action="store_true")
    return parser.parse_args()


def comparison_configs(args: argparse.Namespace) -> list[ExperimentConfig]:
    calibration_samples = 2 if args.preset == "smoke" else 128
    common = {
        "scope": QuantScope.DECODER_CANONICAL,
        "weight_bits": 4,
        "calibration_samples": calibration_samples,
        "calibration_protocol": "multimodal_train_4x32",
        "calibration_manifest_sha256": manifest_digest(args.calibration_manifest),
        "evaluation_manifest_sha256": manifest_digest(args.evaluation_manifest),
        "seed": args.seed,
    }
    return [
        ExperimentConfig(
            MethodName.RTN, activation_bits=None, zero_point=True,
            group_size=128, weight_granularity="groupwise",
            scheme_id="naive_w4a16_g128_asym",
            implementation_tier=ImplementationTier.NAIVE_BASELINE, **common,
        ),
        ExperimentConfig(
            MethodName.MBQ, activation_bits=None, zero_point=True,
            group_size=128, weight_granularity="groupwise",
            scheme_id="mbq_w4a16_g128_asym", **common,
        ),
        ExperimentConfig(
            MethodName.RTN, activation_bits=8, zero_point=False,
            group_size=128, weight_granularity="per_output_channel",
            scheme_id="naive_w4a8_perchannel_sym",
            implementation_tier=ImplementationTier.NAIVE_BASELINE, **common,
        ),
        ExperimentConfig(
            MethodName.SMOOTHQUANT, activation_bits=8, zero_point=False,
            group_size=128, weight_granularity="per_output_channel",
            scheme_id="smoothquant_w4a8_perchannel_sym", **common,
        ),
        ExperimentConfig(
            MethodName.MBQ, activation_bits=8, zero_point=False,
            group_size=128, weight_granularity="per_output_channel",
            scheme_id="mbq_w4a8_perchannel_sym", **common,
        ),
    ]


def main() -> None:
    args = parse_args()
    calibration = load_manifest(args.calibration_manifest)
    evaluation = load_manifest(args.evaluation_manifest)
    assert_disjoint(calibration, evaluation)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_dir = args.output_dir / "_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.calibration_cache_dir or args.output_dir / "_calibration_cache"
    digests = run_input_digests(
        model_source=args.model_path,
        calibration_manifest=args.calibration_manifest,
        evaluation_manifest=args.evaluation_manifest,
        processor_revision="transformers-4.57.0-use_fast",
    )
    input_path = args.output_dir / "_input_digests.json"
    atomic_json(input_path, digests)
    rows = []
    for config in comparison_configs(args):
        run_id = f"{config.method.value}__{config.scheme_id}__{config.digest[:12]}"
        config_path = config_dir / f"{config.digest}.json"
        config.dump(config_path)
        command = [
            sys.executable,
            "-m", "experiments.vlm_quant_benchmark.run_single",
            "--model-path", args.model_path,
            "--config-json", str(config_path),
            "--calibration-manifest", str(args.calibration_manifest),
            "--evaluation-manifest", str(args.evaluation_manifest),
            "--output-dir", str(args.output_dir / run_id),
            "--calibration-cache-dir", str(cache_dir),
            "--input-digests-json", str(input_path),
            "--preset", args.preset,
            "--threads", str(args.threads),
        ]
        if args.resume:
            command.append("--resume")
        if args.allow_nonstandard_model:
            command.append("--allow-nonstandard-model")
        row = {
            "run_id": run_id,
            "method": config.method.value,
            "scheme_id": config.scheme_id,
            "format": f"W{config.weight_bits}A{config.activation_bits or 16}",
            "command": command,
            "status": "planned",
        }
        if args.execute:
            completed = subprocess.run(command, check=False)
            row["status"] = "complete" if completed.returncode == 0 else "failed"
            row["returncode"] = completed.returncode
        rows.append(row)
        atomic_json(args.output_dir / "comparison_plan.json", {
            "execute": args.execute,
            "default_scheduler_modified": False,
            "input_digests": digests,
            "runs": rows,
        })


if __name__ == "__main__":
    main()
