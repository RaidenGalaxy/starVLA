"""Resumable scheduler for the active BF16/Top4 W4A16/W3A16 protocol."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .dataset_protocol import ACTIVE_DATASET_NAMES, PROTOCOL_VERSION
from .provenance import sha256_json, source_digest
from .reporting import atomic_json


TOP4 = (
    ("mbq", "mbq_w4a16_g128_asym", "mbq_w3a16_g128_asym", "mbq_w4a16_g128_asym"),
    ("rtn", "rtn_w4a16_g128_asym", "rtn_w3a16_g128_asym", "naive_w4a16_g128_asym"),
    ("omniquant", "omniquant_w4a16_g128_asym", "omniquant_w3a16_g128_asym", "local_omniquant_w4a16_g128_asym"),
    ("squeezellm", "squeezellm_w4a16_g128_asym", "squeezellm_w3a16_g128_asym", "local_squeezellm_w4a16_g128_asym"),
)


@dataclass
class Task:
    task_id: str
    stage: str
    command: list[str]
    status: str = "pending"
    returncode: int | None = None
    elapsed_seconds: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Top4 W4A16/W3A16 VLM protocol")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vlmevalkit-root", type=Path, required=True)
    parser.add_argument("--lmu-data-root", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--screen-manifest", type=Path, required=True)
    parser.add_argument("--corpora-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--legacy-vlm-root", type=Path)
    parser.add_argument("--legacy-corpus-root", type=Path)
    parser.add_argument(
        "--stage", choices=("correctness", "smoke", "corpus", "screen", "report", "full", "all"),
        default="all",
    )
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    return parser.parse_args()


def module(name: str, *args: str) -> list[str]:
    return [sys.executable, "-m", f"experiments.vlm_quant_benchmark.{name}", *args]


def quant_configs() -> list[dict]:
    rows = [{
        "method": "bf16_value_fp32_compute", "scheme": "bf16_value_fp32_compute",
        "bits": None, "legacy_dir": "bf16",
    }]
    for method, w4, w3, legacy_w4 in TOP4:
        rows.append({"method": method, "scheme": w4, "bits": 4, "legacy_dir": f"{legacy_w4}__decoder_canonical"})
        rows.append({"method": method, "scheme": w3, "bits": 3, "legacy_dir": None})
    return rows


def vlm_command(args, config: dict, output: Path, preset: str, *, reuse: bool = False) -> list[str]:
    command = module(
        "run_vlmeval_official",
        "--base-model-path", args.model_path,
        "--vlmevalkit-root", str(args.vlmevalkit_root),
        "--lmu-data-root", str(args.lmu_data_root),
        "--calibration-manifest", str(args.calibration_manifest),
        "--screen-manifest", str(args.screen_manifest),
        "--datasets", ",".join(ACTIVE_DATASET_NAMES),
        "--method", config["method"],
        "--scope", "decoder_canonical",
        "--scheme-id", config["scheme"],
        "--weight-granularity", "groupwise",
        "--preset", preset,
        "--threads", str(args.threads),
        "--output-dir", str(output),
    )
    if config["bits"] is not None:
        command.extend(["--weight-bits", str(config["bits"])])
    if reuse and args.legacy_vlm_root is not None and config["legacy_dir"] is not None:
        command.extend([
            "--reuse-predictions-from", str(args.legacy_vlm_root / config["legacy_dir"]),
        ])
    if args.resume:
        command.append("--resume")
    return command


def corpus_command(args, config: dict, output: Path, reference: Path | None = None) -> list[str]:
    command = module(
        "run_corpus_ppl",
        "--model-path", args.model_path,
        "--corpora-dir", str(args.corpora_dir),
        "--output-dir", str(output),
        "--method", config["method"],
        "--scope", "decoder_canonical",
        "--scheme-id", config["scheme"],
        "--preset", "screen",
        "--threads", str(args.threads),
    )
    if config["bits"] is not None:
        command.extend(["--weight-bits", str(config["bits"])])
    if config["method"] == "mbq":
        command.extend(["--calibration-manifest", str(args.calibration_manifest)])
    if reference is not None:
        command.extend(["--reference-metrics", str(reference)])
    if args.resume:
        command.append("--resume")
    return command


def reusable_bf16_reference(args) -> Path | None:
    if args.legacy_corpus_root is None:
        return None
    directory = args.legacy_corpus_root / "bf16"
    config_path = directory / "config.json"
    metrics_path = directory / "corpus_metrics.json"
    complete_path = directory / ".complete"
    if not (config_path.is_file() and metrics_path.is_file() and complete_path.is_file()):
        return None
    config = json.loads(config_path.read_text())
    corpus_files = {
        "wikitext2_test": args.corpora_dir / "wikitext2_test.pt",
        "ptb_test": args.corpora_dir / "ptb_test.pt",
        "c4_validation": args.corpora_dir / "c4_validation_256x2048.pt",
    }
    expected = {
        "model": source_digest(args.model_path),
        "calibration": source_digest(args.corpora_dir / "c4_calibration_128x2048.pt"),
        "preset": sha256_json("screen"),
        "max_windows": sha256_json(16),
        **{name: source_digest(path) for name, path in corpus_files.items()},
    }
    if (
        config.get("method") != "bf16_value_fp32_compute"
        or config.get("scope") != "decoder_canonical"
        or config.get("input_digests") != expected
    ):
        return None
    return metrics_path


def base_tasks(args) -> list[Task]:
    configs = quant_configs()
    tasks = [Task(
        "unit_tests", "correctness",
        [sys.executable, "-m", "pytest", "-q", "experiments/vlm_quant_benchmark/tests"],
    )]
    for config in configs:
        tasks.append(Task(
            f'smoke_{config["scheme"]}', "smoke",
            vlm_command(args, config, args.output_dir / "smoke" / config["scheme"], "smoke"),
        ))
    bf16 = configs[0]
    reference = reusable_bf16_reference(args)
    if reference is None:
        reference = args.output_dir / "corpus_screen" / "bf16" / "corpus_metrics.json"
        tasks.append(Task(
            "corpus_bf16", "corpus",
            corpus_command(args, bf16, args.output_dir / "corpus_screen" / "bf16"),
        ))
    for config in configs:
        if config["bits"] == 3:
            tasks.append(Task(
                f'corpus_{config["scheme"]}', "corpus",
                corpus_command(
                    args, config, args.output_dir / "corpus_screen" / config["scheme"], reference,
                ),
            ))
    for config in configs:
        tasks.append(Task(
            f'screen_{config["scheme"]}', "screen",
            vlm_command(
                args, config, args.output_dir / "vlm_screen" / config["scheme"],
                "screen", reuse=True,
            ),
        ))
    report = module(
        "top4_report",
        "--screen-root", str(args.output_dir / "vlm_screen"),
        "--corpus-root", str(args.output_dir / "corpus_screen"),
        "--output-dir", str(args.output_dir / "summary"),
    )
    if args.legacy_corpus_root is not None:
        report.extend(["--legacy-corpus-root", str(args.legacy_corpus_root)])
    if (args.output_dir / "full").is_dir():
        report.extend(["--full-root", str(args.output_dir / "full")])
    tasks.append(Task("top4_report", "report", report))
    return tasks


def full_tasks(args) -> list[Task]:
    selection_path = args.output_dir / "summary" / "selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError(f"screen selection is not ready: {selection_path}")
    winners = json.loads(selection_path.read_text())["winners"]
    configs = [{
        "method": "bf16_value_fp32_compute", "scheme": "bf16_value_fp32_compute",
        "bits": None, "legacy_dir": None,
    }]
    for key in ("w4a16", "w3a16"):
        row = winners[key]
        configs.append({
            "method": row["method"], "scheme": row["scheme_id"],
            "bits": row["weight_bits"], "legacy_dir": None,
        })
    tasks = [Task(
        f'full_{config["scheme"]}', "full",
        vlm_command(args, config, args.output_dir / "full" / config["scheme"], "full"),
    ) for config in configs]
    report = module(
        "top4_report",
        "--screen-root", str(args.output_dir / "vlm_screen"),
        "--full-root", str(args.output_dir / "full"),
        "--corpus-root", str(args.output_dir / "corpus_screen"),
        "--output-dir", str(args.output_dir / "summary"),
    )
    if args.legacy_corpus_root is not None:
        report.extend(["--legacy-corpus-root", str(args.legacy_corpus_root)])
    tasks.append(Task("final_report", "full", report))
    return tasks


def run_tasks(args, tasks: list[Task], plan_path: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(args.vlmevalkit_root), environment.get("PYTHONPATH")) if value
    )
    for task in tasks:
        atomic_json(plan_path, {
            "protocol_version": PROTOCOL_VERSION,
            "serial_execution": True,
            "tasks": [asdict(item) for item in tasks],
        })
        if args.dry_run:
            task.status = "planned"
            continue
        task.status = "running"
        atomic_json(plan_path, {
            "protocol_version": PROTOCOL_VERSION,
            "serial_execution": True,
            "tasks": [asdict(item) for item in tasks],
        })
        started = time.monotonic()
        completed = subprocess.run(task.command, check=False, env=environment)
        task.elapsed_seconds = time.monotonic() - started
        task.returncode = completed.returncode
        task.status = "complete" if completed.returncode == 0 else "failed"
        atomic_json(plan_path, {
            "protocol_version": PROTOCOL_VERSION,
            "serial_execution": True,
            "tasks": [asdict(item) for item in tasks],
        })
        if completed.returncode and not args.continue_on_failure:
            raise SystemExit(completed.returncode)
    atomic_json(plan_path, {
        "protocol_version": PROTOCOL_VERSION,
        "serial_execution": True,
        "tasks": [asdict(item) for item in tasks],
    })


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_dir / (
        "stage_plan.json" if args.stage == "all" else f"stage_plan_{args.stage}.json"
    )
    if args.stage == "full":
        tasks = full_tasks(args)
        run_tasks(args, tasks, plan_path)
        return
    tasks = base_tasks(args)
    if args.stage != "all":
        tasks = [task for task in tasks if task.stage == args.stage]
    run_tasks(args, tasks, plan_path)
    if args.stage == "all" and not args.dry_run:
        run_tasks(args, full_tasks(args), args.output_dir / "full_stage_plan.json")


if __name__ == "__main__":
    main()
