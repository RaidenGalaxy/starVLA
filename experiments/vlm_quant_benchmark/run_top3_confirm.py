"""Resumable CPU scheduler for the Top3 4x256 confirmation protocol."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .dataset_protocol import ACTIVE_DATASET_NAMES
from .reporting import atomic_json


PROTOCOL_VERSION = 4
SAMPLES_PER_DATASET = 256
ACTIVE_METHODS = ("mbq", "omniquant", "squeezellm")


@dataclass
class Task:
    task_id: str
    stage: str
    command: list[str]
    status: str = "pending"
    returncode: int | None = None
    elapsed_seconds: float | None = None
    log_path: str | None = None


def confirm_configs() -> list[dict]:
    rows = [{
        "method": "bf16_value_fp32_compute",
        "scheme": "bf16_value_fp32_compute",
        "bits": None,
    }]
    rows.extend({
        "method": method,
        "scheme": f"{method}_w4a16_g128_asym",
        "bits": 4,
    } for method in ACTIVE_METHODS)
    rows.extend({
        "method": method,
        "scheme": f"{method}_w3a16_g128_asym",
        "bits": 3,
    } for method in ACTIVE_METHODS)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Top3 W4/W3 CPU 4x256 confirmation")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vlmevalkit-root", type=Path, required=True)
    parser.add_argument("--lmu-data-root", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--confirm-manifest", type=Path, required=True)
    parser.add_argument("--screen64-root", type=Path, required=True)
    parser.add_argument("--bf16-full-donor", type=Path)
    parser.add_argument("--calibration-cache-dir", type=Path)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--legacy-corpus-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("correctness", "smoke", "confirm", "report", "all"),
        default="all",
    )
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--wait-for-pid", type=int,
        help="Wait for an existing scheduler PID before starting this pipeline.",
    )
    return parser.parse_args()


def module(name: str, *args: str) -> list[str]:
    return [sys.executable, "-m", f"experiments.vlm_quant_benchmark.{name}", *args]


def vlm_command(
    args: argparse.Namespace,
    config: dict,
    output: Path,
    *,
    preset: str,
) -> list[str]:
    command = module(
        "run_vlmeval_official",
        "--base-model-path", args.model_path,
        "--vlmevalkit-root", str(args.vlmevalkit_root),
        "--lmu-data-root", str(args.lmu_data_root),
        "--calibration-manifest", str(args.calibration_manifest),
        "--screen-manifest", str(args.confirm_manifest),
        "--datasets", ",".join(ACTIVE_DATASET_NAMES),
        "--method", config["method"],
        "--scope", "decoder_canonical",
        "--scheme-id", config["scheme"],
        "--weight-granularity", "groupwise",
        "--preset", preset,
        "--threads", str(args.threads),
        "--output-dir", str(output),
    )
    if args.calibration_cache_dir is not None:
        command.extend(["--calibration-cache-dir", str(args.calibration_cache_dir)])
    if config["bits"] is not None:
        command.extend(["--weight-bits", str(config["bits"])])
    if preset == "confirm":
        command.extend([
            "--expected-samples-per-dataset", str(SAMPLES_PER_DATASET),
            "--reuse-predictions-from", str(args.screen64_root / config["scheme"]),
        ])
        if config["bits"] is None and args.bf16_full_donor is not None:
            command.extend([
                "--reuse-predictions-from", str(args.bf16_full_donor),
            ])
    if args.resume:
        command.append("--resume")
    return command


def report_command(args: argparse.Namespace) -> list[str]:
    command = module(
        "top3_confirm_report",
        "--confirm-root", str(args.output_dir / "confirm"),
        "--screen64-root", str(args.screen64_root),
        "--corpus-root", str(args.corpus_root),
        "--output-dir", str(args.output_dir / "summary"),
        "--samples-per-dataset", str(SAMPLES_PER_DATASET),
    )
    if args.legacy_corpus_root is not None:
        command.extend(["--legacy-corpus-root", str(args.legacy_corpus_root)])
    return command


def tasks_for_stage(args: argparse.Namespace, stage: str) -> list[Task]:
    configs = confirm_configs()
    if stage == "correctness":
        return [Task(
            "unit_tests", stage,
            [sys.executable, "-m", "pytest", "-q", "experiments/vlm_quant_benchmark/tests"],
        )]
    if stage == "smoke":
        return [
            Task(
                f"smoke_{config['scheme']}", stage,
                vlm_command(
                    args, config, args.output_dir / "smoke" / config["scheme"],
                    preset="smoke",
                ),
            )
            for config in configs
        ]
    if stage == "confirm":
        return [
            Task(
                f"confirm_{config['scheme']}", stage,
                vlm_command(
                    args, config, args.output_dir / "confirm" / config["scheme"],
                    preset="confirm",
                ),
            )
            for config in configs
        ]
    if stage == "report":
        return [Task("top3_confirm_report", stage, report_command(args))]
    raise ValueError(stage)


def write_plan(path: Path, tasks: list[Task]) -> None:
    atomic_json(path, {
        "protocol_version": PROTOCOL_VERSION,
        "samples_per_dataset": SAMPLES_PER_DATASET,
        "active_methods": list(ACTIVE_METHODS),
        "serial_execution": True,
        "tasks": [asdict(task) for task in tasks],
    })


def run_stage(args: argparse.Namespace, stage: str, *, continue_on_failure: bool) -> bool:
    tasks = tasks_for_stage(args, stage)
    plan_path = args.output_dir / f"stage_plan_{stage}.json"
    write_plan(plan_path, tasks)
    if args.dry_run:
        for task in tasks:
            task.status = "planned"
        write_plan(plan_path, tasks)
        return True
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (
            str(args.vlmevalkit_root), environment.get("PYTHONPATH"),
        ) if value
    )
    log_root = args.output_dir / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    succeeded = True
    for task in tasks:
        log_path = log_root / f"{task.task_id}.log"
        task.log_path = str(log_path)
        task.status = "running"
        write_plan(plan_path, tasks)
        started = time.monotonic()
        with log_path.open("a") as handle:
            completed = subprocess.run(
                task.command,
                check=False,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        task.elapsed_seconds = time.monotonic() - started
        task.returncode = completed.returncode
        task.status = "complete" if completed.returncode == 0 else "failed"
        succeeded = succeeded and completed.returncode == 0
        write_plan(plan_path, tasks)
        if completed.returncode and not continue_on_failure:
            return False
    return succeeded


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.wait_for_pid is not None:
        while Path(f"/proc/{args.wait_for_pid}").exists():
            time.sleep(30)
    stages = (
        ("correctness", False),
        ("smoke", False),
        ("confirm", True),
        ("report", False),
    ) if args.stage == "all" else ((args.stage, args.stage == "confirm"),)
    for stage, continue_on_failure in stages:
        if not run_stage(
            args, stage, continue_on_failure=continue_on_failure,
        ):
            raise SystemExit(f"{stage} stage failed; inspect {args.output_dir / 'logs'}")


if __name__ == "__main__":
    main()
