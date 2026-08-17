from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .config import ExperimentConfig, ImplementationTier, MethodName, QuantScope
from .data import assert_disjoint, load_manifest, manifest_digest
from .provenance import run_input_digests
from .reporting import atomic_json, write_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isolated Qwen2.5-VL quantization matrix")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", default="")
    parser.add_argument("--processor-revision", default="transformers-4.57.0-use_fast")
    parser.add_argument("--calibration-manifest", type=Path)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-cache-dir", type=Path)
    parser.add_argument("--preset", choices=("smoke", "screen", "full"), default="smoke")
    parser.add_argument(
        "--methods",
        default=(
            "bf16_value_fp32_compute,rtn,awq,gptq,smoothquant,llm_int8,"
            "tender,vectorwise_w8,omniquant,spqr,squeezellm"
        ),
    )
    parser.add_argument("--scopes", default="decoder_canonical,full_vlm_extended")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-nonstandard-model", action="store_true")
    return parser.parse_args()


def method_configs(
    method: MethodName,
    scope: QuantScope,
    args: argparse.Namespace,
    *,
    calibration_sha256: str,
    evaluation_sha256: str,
    model_revision: str,
) -> list[ExperimentConfig]:
    tier = (
        ImplementationTier.CPU_RESEARCH
        if method in {MethodName.OMNIQUANT, MethodName.SPQR, MethodName.SQUEEZELLM}
        else ImplementationTier.LOCAL_VALIDATED
    )
    common = {
        "scope": scope,
        "group_size": args.group_size,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "calibration_protocol": "multimodal_train_4x32",
        "calibration_manifest_sha256": calibration_sha256,
        "evaluation_manifest_sha256": evaluation_sha256,
        "model_revision": model_revision,
        "processor_revision": args.processor_revision,
        "implementation_tier": tier,
    }
    if args.preset == "smoke":
        common.update({
            "calibration_samples": 2,
            "awq_alpha_grid": (0.0, 0.5, 1.0),
            "awq_clip_grid": (1.0, 0.9),
            "omniquant_steps": 2,
            "squeezellm_kmeans_steps": 1,
        })
    if method == MethodName.BF16_VALUE_FP32_COMPUTE:
        return [ExperimentConfig(
            method,
            weight_bits=None,
            activation_bits=None,
            zero_point=False,
            scheme_id="bf16_value_fp32_compute",
            implementation_tier=ImplementationTier.REFERENCE,
            calibration_protocol="none",
            **{key: value for key, value in common.items()
               if key not in {"implementation_tier", "calibration_protocol"}},
        )]
    if method == MethodName.RTN:
        baseline = common | {"implementation_tier": ImplementationTier.NAIVE_BASELINE}
        return [
            ExperimentConfig(
                method, weight_bits=4, activation_bits=None, zero_point=True,
                scheme_id="naive_w4a16_g128_asym", **baseline,
            ),
            ExperimentConfig(
                method, weight_bits=8, activation_bits=8, zero_point=False,
                scheme_id="naive_w8a8_g128_sym", **baseline,
            ),
            ExperimentConfig(
                method, weight_bits=4, activation_bits=4, zero_point=False,
                scheme_id="naive_w4a4_g128_sym", **baseline,
            ),
        ]
    if method == MethodName.TENDER:
        return [
            ExperimentConfig(
                method, weight_bits=8, activation_bits=8, zero_point=False,
                scheme_id="tender_w8a8_g128_sym", **common,
            ),
            ExperimentConfig(
                method, weight_bits=4, activation_bits=4, zero_point=False,
                scheme_id="tender_w4a4_g128_sym", **common,
            ),
        ]
    if method == MethodName.SMOOTHQUANT:
        return [ExperimentConfig(
            method, weight_bits=8, activation_bits=8, zero_point=False,
            scheme_id="smoothquant_w8a8_g128_sym", **common,
        )]
    if method == MethodName.LLM_INT8:
        return [ExperimentConfig(
            method, weight_bits=8, activation_bits=8, zero_point=False,
            scheme_id="llm_int8_vectorwise_mixed_w8", **common,
        )]
    if method == MethodName.VECTORWISE_W8:
        return [ExperimentConfig(
            method, weight_bits=8, activation_bits=8, zero_point=False,
            scheme_id="vectorwise_w8_baseline", **(
                common | {"implementation_tier": ImplementationTier.NAIVE_BASELINE}
            ),
        )]
    scheme = {
        MethodName.AWQ: "local_awq_w4a16_g128_asym",
        MethodName.GPTQ: "blockwise_gptq_w4a16_g128_asym",
        MethodName.OMNIQUANT: "local_omniquant_w4a16_g128_asym",
        MethodName.SPQR: "local_spqr_w4a16_g128_asym",
        MethodName.SQUEEZELLM: "local_squeezellm_w4a16_g128_asym",
    }[method]
    return [ExperimentConfig(
        method, weight_bits=4, activation_bits=None, zero_point=True,
        scheme_id=scheme, **common,
    )]


def main() -> None:
    args = parse_args()
    methods = [MethodName(value.strip()) for value in args.methods.split(",") if value.strip()]
    scopes = [QuantScope(value.strip()) for value in args.scopes.split(",") if value.strip()]
    calibration = load_manifest(args.calibration_manifest) if args.calibration_manifest else []
    evaluation = load_manifest(args.evaluation_manifest)
    if calibration:
        assert_disjoint(calibration, evaluation)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_dir = args.output_dir / "_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.calibration_cache_dir or args.output_dir / "_calibration_cache"
    input_digests = run_input_digests(
        model_source=args.model_path,
        calibration_manifest=args.calibration_manifest,
        evaluation_manifest=args.evaluation_manifest,
        processor_revision=args.processor_revision,
    )
    input_path = args.output_dir / "_input_digests.json"
    atomic_json(input_path, input_digests)
    atomic_json(args.output_dir / "legacy_results_notice.json", {
        "legacy_glob": "outputs/vlm_quant_benchmark/smoke_*",
        "status": "legacy_metric_schema_do_not_rank",
        "reason": "old conditional-answer PPL included chat-template suffix tokens",
    })

    calibration_sha = (
        manifest_digest(args.calibration_manifest) if args.calibration_manifest else ""
    )
    evaluation_sha = manifest_digest(args.evaluation_manifest)
    model_revision = args.model_revision or input_digests["model"]
    rows = []
    for method in methods:
        for scope in scopes:
            for config in method_configs(
                method,
                scope,
                args,
                calibration_sha256=calibration_sha,
                evaluation_sha256=evaluation_sha,
                model_revision=model_revision,
            ):
                run_id = (
                    f"{method.value}__{config.scheme_id}__{scope.value}__"
                    f"{config.digest[:12]}"
                )
                destination = args.output_dir / run_id
                config_path = config_dir / f"{config.digest}.json"
                config.dump(config_path)
                command = [
                    sys.executable,
                    "-m", "experiments.vlm_quant_benchmark.run_single",
                    "--model-path", args.model_path,
                    "--config-json", str(config_path),
                    "--evaluation-manifest", str(args.evaluation_manifest),
                    "--output-dir", str(destination),
                    "--calibration-cache-dir", str(cache_dir),
                    "--input-digests-json", str(input_path),
                    "--preset", args.preset,
                    "--threads", str(args.threads),
                ]
                if args.calibration_manifest:
                    command.extend(["--calibration-manifest", str(args.calibration_manifest)])
                if args.resume:
                    command.append("--resume")
                if args.allow_nonstandard_model:
                    command.append("--allow-nonstandard-model")
                completed = subprocess.run(command, check=False)
                if completed.returncode == 0:
                    metrics = json.loads((destination / "metrics.json").read_text())
                    diagnostic = metrics.get("target_text_diagnostic", {})
                    rows.append({
                        "run_id": run_id,
                        "method": method.value,
                        "scheme_id": config.scheme_id,
                        "scope": scope.value,
                        "implementation_tier": str(config.implementation_tier),
                        "samples": metrics.get("samples"),
                        "exact_accuracy": metrics.get("exact_accuracy"),
                        "anls": metrics.get("anls"),
                        "target_text_tokens_diagnostic_only": diagnostic.get("tokens"),
                        "status": "complete",
                    })
                else:
                    error_path = destination / "error.json"
                    error = json.loads(error_path.read_text()) if error_path.is_file() else {}
                    rows.append({
                        "run_id": run_id,
                        "method": method.value,
                        "scheme_id": config.scheme_id,
                        "scope": scope.value,
                        "implementation_tier": str(config.implementation_tier),
                        "status": "failed",
                        "error": error.get("message", f"child exit {completed.returncode}"),
                    })
                write_summary(args.output_dir / "summary.csv", rows)


if __name__ == "__main__":
    main()
