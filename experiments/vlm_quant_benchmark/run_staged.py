from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .reporting import atomic_json


LOCAL_METHODS = (
    "rtn,awq,gptq,smoothquant,llm_int8,vectorwise_w8,"
    "tender,omniquant,spqr,squeezellm"
)
SCHEMES = (
    ("rtn", "naive_w4a16_g128_asym", 4, None),
    ("awq", "local_awq_w4a16_g128_asym", 4, None),
    ("gptq", "blockwise_gptq_w4a16_g128_asym", 4, None),
    ("omniquant", "local_omniquant_w4a16_g128_asym", 4, None),
    ("spqr", "local_spqr_w4a16_g128_asym", 4, None),
    ("squeezellm", "local_squeezellm_w4a16_g128_asym", 4, None),
    ("rtn", "naive_w8a8_g128_sym", 8, 8),
    ("smoothquant", "smoothquant_w8a8_g128_sym", 8, 8),
    ("tender", "tender_w8a8_g128_sym", 8, 8),
    ("vectorwise_w8", "vectorwise_w8_baseline", 8, 8),
    ("llm_int8", "llm_int8_vectorwise_mixed_w8", 8, 8),
)


@dataclass
class StageTask:
    task_id: str
    stage: str
    command: list[str]
    estimate_seconds: float | None = None
    status: str = "pending"
    elapsed_seconds: float | None = None
    returncode: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serial staged Qwen2.5-VL benchmark scheduler")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--official-awq-path", required=True)
    parser.add_argument("--vlmevalkit-root", type=Path, required=True)
    parser.add_argument("--lmu-data-root", type=Path)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--screen-manifest", type=Path, required=True)
    parser.add_argument("--corpora-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("correctness", "timing", "corpus_screen", "vlm_screen", "full", "all"),
        default="all",
    )
    parser.add_argument("--best-local-w4", type=Path)
    parser.add_argument("--budget-hours", type=float, default=72.0)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    return parser.parse_args()


def module(name: str, *arguments: str) -> list[str]:
    return [sys.executable, "-m", f"experiments.vlm_quant_benchmark.{name}", *arguments]


def common(args) -> list[str]:
    values = [
        "--base-model-path", args.model_path,
        "--vlmevalkit-root", str(args.vlmevalkit_root),
        "--lmu-data-root", str(args.lmu_data_root),
        "--calibration-manifest", str(args.calibration_manifest),
        "--screen-manifest", str(args.screen_manifest),
        "--threads", str(args.threads),
    ]
    if args.resume:
        values.append("--resume")
    return values


def official_command(args, *, output: Path, preset: str, datasets: str,
                     method: str = "bf16_value_fp32_compute",
                     scope: str = "decoder_canonical",
                     scheme_id: str | None = None,
                     weight_bits: int | None = None,
                     activation_bits: int | None = None,
                     official_awq: bool = False) -> list[str]:
    command = module(
        "run_vlmeval_official",
        *common(args),
        "--method", method,
        "--scope", scope,
        "--preset", preset,
        "--datasets", datasets,
        "--output-dir", str(output),
    )
    if scheme_id is not None:
        command.extend(["--scheme-id", scheme_id])
    if weight_bits is not None:
        command.extend(["--weight-bits", str(weight_bits)])
    if activation_bits is not None:
        command.extend(["--activation-bits", str(activation_bits)])
    if official_awq:
        command.extend(["--official-awq-path", args.official_awq_path])
    return command


def matrix_command(args, *, output: Path, preset: str, methods: str) -> list[str]:
    command = module(
        "run_matrix",
        "--model-path", args.model_path,
        "--calibration-manifest", str(args.calibration_manifest),
        "--evaluation-manifest", str(args.screen_manifest),
        "--output-dir", str(output),
        "--preset", preset,
        "--methods", methods,
        "--scopes", "decoder_canonical,full_vlm_extended",
        "--threads", str(args.threads),
    )
    if args.resume:
        command.append("--resume")
    return command


def corpus_command(args, *, method: str, output: Path, preset: str,
                   scheme_id: str | None = None,
                   weight_bits: int | None = None,
                   activation_bits: int | None = None,
                   reference: Path | None = None) -> list[str]:
    command = module(
        "run_corpus_ppl",
        "--model-path", args.model_path,
        "--corpora-dir", str(args.corpora_dir),
        "--output-dir", str(output),
        "--method", method,
        "--scope", "decoder_canonical",
        "--preset", preset,
        "--threads", str(args.threads),
    )
    if scheme_id is not None:
        command.extend(["--scheme-id", scheme_id])
    if weight_bits is not None:
        command.extend(["--weight-bits", str(weight_bits)])
    if activation_bits is not None:
        command.extend(["--activation-bits", str(activation_bits)])
    if reference is not None:
        command.extend(["--reference-metrics", str(reference)])
    if args.resume:
        command.append("--resume")
    return command


def build_tasks(args) -> list[StageTask]:
    output = args.output_dir
    tasks = [
        StageTask(
            "unit_tests", "correctness",
            [sys.executable, "-m", "pytest", "-q", "experiments/vlm_quant_benchmark/tests"],
        ),
        StageTask(
            "official_awq_checkpoint_gate", "correctness",
            module(
                "validate_official_awq_checkpoint",
                "--checkpoint", args.official_awq_path,
                "--output", str(output / "correctness" / "official_awq_checkpoint.json"),
            ),
        ),
        StageTask(
            "local_two_sample_smoke", "correctness",
            matrix_command(
                args, output=output / "correctness" / "local_smoke",
                preset="smoke", methods=f"bf16_value_fp32_compute,{LOCAL_METHODS}",
            ),
        ),
        StageTask(
            "official_awq_four_dataset_smoke", "correctness",
            official_command(
                args, output=output / "correctness" / "official_awq_smoke",
                preset="smoke", datasets=(
                    "DocVQA_VAL,MathVista_MINI,MMMU_DEV_VAL,MMBench_DEV_EN"
                ),
                official_awq=True,
            ),
        ),
        StageTask(
            "bf16_vlm_4x32_timing", "timing",
            official_command(
                args, output=output / "timing" / "bf16_vlm_4x32",
                preset="timing", datasets=",".join((
                    "DocVQA_VAL", "MathVista_MINI", "MMMU_DEV_VAL", "MMBench_DEV_EN",
                )),
            ),
        ),
        StageTask(
            "bf16_corpus_16_window_timing", "timing",
            corpus_command(
                args, method="bf16_value_fp32_compute",
                output=output / "timing" / "bf16_corpus_16", preset="screen",
            ),
        ),
    ]
    corpus_root = output / "corpus_screen"
    reference_metrics = corpus_root / "bf16" / "corpus_metrics.json"
    tasks.append(StageTask(
        "corpus_screen_bf16", "corpus_screen",
        corpus_command(
            args, method="bf16_value_fp32_compute",
            output=corpus_root / "bf16", preset="screen",
        ),
    ))
    for method, scheme, weight_bits, activation_bits in SCHEMES:
        tasks.append(StageTask(
            f"corpus_screen_{scheme}", "corpus_screen",
            corpus_command(
                args, method=method,
                scheme_id=scheme,
                weight_bits=weight_bits,
                activation_bits=activation_bits,
                output=corpus_root / scheme, preset="screen",
                reference=reference_metrics,
            ),
        ))
    tasks.extend([
        StageTask(
            "vlm_screen_bf16", "vlm_screen",
            official_command(
                args, output=output / "vlm_screen" / "bf16",
                preset="screen", datasets=",".join((
                    "DocVQA_VAL", "MathVista_MINI", "MMMU_DEV_VAL", "MMBench_DEV_EN",
                )),
            ),
        ),
        StageTask(
            "vlm_screen_official_awq", "vlm_screen",
            official_command(
                args, output=output / "vlm_screen" / "official_awq",
                preset="screen", datasets=",".join((
                    "DocVQA_VAL", "MathVista_MINI", "MMMU_DEV_VAL", "MMBench_DEV_EN",
                )), official_awq=True,
            ),
        ),
    ])
    for method, scheme, weight_bits, activation_bits in SCHEMES:
        for scope in ("decoder_canonical", "full_vlm_extended"):
            tasks.append(StageTask(
                f"vlm_screen_{scheme}_{scope}", "vlm_screen",
                official_command(
                    args,
                    output=output / "vlm_screen" / f"{scheme}__{scope}",
                    preset="screen",
                    datasets=",".join((
                        "DocVQA_VAL", "MathVista_MINI", "MMMU_DEV_VAL", "MMBench_DEV_EN",
                    )),
                    method=method,
                    scope=scope,
                    scheme_id=scheme,
                    weight_bits=weight_bits,
                    activation_bits=activation_bits,
                ),
            ))

    if args.best_local_w4 is not None:
        selected = json.loads(args.best_local_w4.read_text())
        method = str(selected["method"])
        scope = str(selected.get("scope", "decoder_canonical"))
        scheme = str(selected["scheme_id"])
        weight_bits = int(selected.get("weight_bits", 4))
        activation_bits = selected.get("activation_bits")
        if activation_bits is not None:
            activation_bits = int(activation_bits)
        tasks.extend([
            StageTask(
                "full_bf16", "full",
                official_command(
                    args, output=output / "full" / "bf16",
                    preset="full", datasets=",".join((
                        "DocVQA_VAL", "MathVista_MINI", "MMMU_DEV_VAL", "MMBench_DEV_EN",
                    )),
                ),
            ),
            StageTask(
                "full_official_awq", "full",
                official_command(
                    args, output=output / "full" / "official_awq",
                    preset="full", datasets=",".join((
                        "DocVQA_VAL", "MathVista_MINI", "MMMU_DEV_VAL", "MMBench_DEV_EN",
                    )), official_awq=True,
                ),
            ),
            StageTask(
                "full_best_local_w4", "full",
                official_command(
                    args, output=output / "full" / "best_local_w4",
                    preset="full", datasets="MathVista_MINI,MMMU_DEV_VAL",
                    method=method, scope=scope, scheme_id=scheme,
                    weight_bits=weight_bits, activation_bits=activation_bits,
                ),
            ),
            StageTask(
                "corpus_full_bf16", "full",
                corpus_command(
                    args, method="bf16_value_fp32_compute",
                    output=output / "corpus_full" / "bf16", preset="full",
                ),
            ),
            StageTask(
                "corpus_full_rtn_w4", "full",
                corpus_command(
                    args, method="rtn",
                    output=output / "corpus_full" / "rtn_w4", preset="full",
                    reference=output / "corpus_full" / "bf16" / "corpus_metrics.json",
                ),
            ),
            StageTask(
                "corpus_full_official_awq", "full",
                module(
                    "run_official_awq_corpus",
                    "--base-model-path", args.model_path,
                    "--awq-model-path", args.official_awq_path,
                    "--corpora-dir", str(args.corpora_dir),
                    "--output-dir", str(output / "corpus_full" / "official_awq"),
                    "--preset", "full",
                    "--threads", str(args.threads),
                    "--reference-metrics",
                    str(output / "corpus_full" / "bf16" / "corpus_metrics.json"),
                    *(("--resume",) if args.resume else ()),
                ),
            ),
            StageTask(
                "corpus_full_best_local_w4", "full",
                corpus_command(
                    args, method=method, scheme_id=scheme,
                    weight_bits=weight_bits, activation_bits=activation_bits,
                    output=output / "corpus_full" / "best_local_w4", preset="full",
                    reference=output / "corpus_full" / "bf16" / "corpus_metrics.json",
                ),
            ),
        ])
    return tasks


def selected_tasks(tasks: list[StageTask], stage: str) -> list[StageTask]:
    order = ("correctness", "timing", "corpus_screen", "vlm_screen", "full")
    if stage == "all":
        return tasks
    return [task for task in tasks if task.stage == stage]


def estimate_from_timing(output_dir: Path) -> dict[str, float]:
    estimates = {}
    corpus = output_dir / "timing" / "bf16_corpus_16" / "corpus_metrics.json"
    if corpus.is_file():
        metrics = json.loads(corpus.read_text())
        estimates["corpus_full"] = sum(
            value.get("elapsed_seconds", 0.0) for value in metrics.values()
        ) * 16
    vlm = output_dir / "timing" / "bf16_vlm_4x32" / "official_result.json"
    if vlm.is_file():
        estimates["vlm_full"] = sum(
            value.get("elapsed_seconds", 0.0)
            for value in json.loads(vlm.read_text()).get("datasets", {}).values()
        ) * 4
    return estimates


def main() -> None:
    args = parse_args()
    if args.lmu_data_root is None:
        args.lmu_data_root = args.screen_manifest.parent.parent / "LMUData"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = selected_tasks(build_tasks(args), args.stage)
    plan_path = args.output_dir / "stage_plan.json"
    budget_seconds = args.budget_hours * 3600 * 0.9
    started_all = time.monotonic()
    timing_estimates = estimate_from_timing(args.output_dir)

    if args.stage in {"full", "all"} and args.best_local_w4 is None:
        atomic_json(args.output_dir / "full_status.json", {
            "status": "awaiting_selection",
            "required": "--best-local-w4 JSON containing method and scope",
            "matrix_is_not_reduced": True,
        })
        if args.stage == "full":
            return

    for task in tasks:
        if task.stage == "full" and args.best_local_w4 is None:
            task.status = "awaiting_selection"
            continue
        if task.stage == "full":
            task.estimate_seconds = (
                timing_estimates.get("corpus_full")
                if task.task_id.startswith("corpus_")
                else timing_estimates.get("vlm_full")
            )
            remaining = budget_seconds - (time.monotonic() - started_all)
            if task.estimate_seconds and task.estimate_seconds * 1.1 > remaining:
                task.status = "deferred_budget_no_matrix_reduction"
                continue
        atomic_json(plan_path, {
            "format_version": 1,
            "threads": args.threads,
            "budget_hours": args.budget_hours,
            "recovery_reserve_fraction": 0.1,
            "serial_execution": True,
            "tasks": [asdict(item) for item in tasks],
        })
        if args.dry_run:
            task.status = "planned"
            continue
        task.status = "running"
        atomic_json(plan_path, {
            "format_version": 1,
            "threads": args.threads,
            "budget_hours": args.budget_hours,
            "recovery_reserve_fraction": 0.1,
            "serial_execution": True,
            "tasks": [asdict(item) for item in tasks],
        })
        started = time.monotonic()
        environment = os.environ.copy()
        pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = os.pathsep.join(
            value for value in (str(args.vlmevalkit_root), pythonpath) if value
        )
        completed = subprocess.run(task.command, check=False, env=environment)
        task.elapsed_seconds = time.monotonic() - started
        task.returncode = completed.returncode
        task.status = "complete" if completed.returncode == 0 else "failed"
        if completed.returncode != 0 and not args.continue_on_failure:
            atomic_json(plan_path, {
                "format_version": 1,
                "threads": args.threads,
                "budget_hours": args.budget_hours,
                "recovery_reserve_fraction": 0.1,
                "serial_execution": True,
                "tasks": [asdict(item) for item in tasks],
            })
            raise SystemExit(completed.returncode)
    atomic_json(plan_path, {
        "format_version": 1,
        "threads": args.threads,
        "budget_hours": args.budget_hours,
        "recovery_reserve_fraction": 0.1,
        "serial_execution": True,
        "tasks": [asdict(item) for item in tasks],
    })


if __name__ == "__main__":
    main()
