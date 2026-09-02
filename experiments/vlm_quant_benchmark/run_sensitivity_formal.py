"""Serial, locked, resumable scheduler for the formal MBQ sensitivity study."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import psutil

from .data import file_digest
from .reporting import atomic_json


STAGES = (
    "preflight", "gradients", "w4_nll", "residuals", "perturbations",
    "analysis", "promotion", "final",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run formal MBQ sensitivity analysis")
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/vlm_quant_benchmark/sensitivity_analysis_v1/formal_128"),
    )
    parser.add_argument(
        "--model-path", default="playground/Pretrained_models/Qwen2.5-VL-3B-Instruct",
    )
    parser.add_argument(
        "--calibration-manifest", type=Path,
        default=Path("data/vlm_quant_benchmark/multimodal_calibration/multimodal_calibration_4x32.jsonl"),
    )
    parser.add_argument(
        "--evaluation-manifest", type=Path,
        default=Path("data/vlm_quant_benchmark/top3_protocol_v4/vlm_confirm_4x256.jsonl"),
    )
    parser.add_argument(
        "--top3-root", type=Path,
        default=Path("outputs/vlm_quant_benchmark/top3_protocol_v4/confirm"),
    )
    parser.add_argument(
        "--top4-root", type=Path,
        default=Path("outputs/vlm_quant_benchmark/top4_protocol_v3/vlm_screen"),
    )
    parser.add_argument("--vlmevalkit-root", type=Path, default=Path("third_party/VLMEvalKit"))
    parser.add_argument("--lmu-data-root", type=Path, default=Path("data/vlm_quant_benchmark/LMUData"))
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-real-smoke", action="store_true")
    parser.add_argument("--allow-nonstandard-model", action="store_true")
    return parser.parse_args()


@contextmanager
def pid_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as error:
        try:
            pid = int(path.read_text().strip())
            alive = psutil.pid_exists(pid)
        except Exception:
            pid, alive = -1, False
        if alive:
            raise RuntimeError(f"formal sensitivity run is already active: PID {pid}") from error
        raise RuntimeError(f"stale PID lock must be inspected before retry: {path}") from error
    os.write(descriptor, f"{os.getpid()}\n".encode())
    os.close(descriptor)
    try:
        yield
    finally:
        if path.is_file() and path.read_text().strip() == str(os.getpid()):
            path.unlink()


def _line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open() as handle:
        return sum(bool(line.strip()) for line in handle)


def _active_quant_processes() -> list[dict]:
    needles = (
        "run_sensitivity_ablation", "run_token_sensitivity",
        "run_reconstruction_residuals", "run_vlmeval_official",
        "run_top3_confirm", "run_top4_protocol",
    )
    found = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        if process.pid == os.getpid():
            continue
        command = " ".join(process.info.get("cmdline") or [])
        if any(needle in command for needle in needles):
            found.append({"pid": process.pid, "command": command})
    return found


def _run(
    command: list[str], log_path: Path, *, environment: dict[str, str] | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as handle:
        handle.write(f"\n[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {' '.join(command)}\n")
        handle.flush()
        result = subprocess.run(
            command, stdout=handle, stderr=subprocess.STDOUT, env=environment,
        )
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)


def _common(args: argparse.Namespace) -> list[str]:
    values = [
        "--model-path", str(args.model_path),
        "--calibration-manifest", str(args.calibration_manifest),
        "--evaluation-manifest", str(args.evaluation_manifest),
        "--threads", str(args.threads), "--seed", str(args.seed),
    ]
    if args.resume:
        values.append("--resume")
    if args.allow_nonstandard_model:
        values.append("--allow-nonstandard-model")
    return values


def _analysis_command(args: argparse.Namespace) -> list[str]:
    root = args.output_dir
    return [
        sys.executable, "-m", "experiments.vlm_quant_benchmark.analyze_sensitivity_correlation",
        "--top3-root", str(args.top3_root), "--top4-root", str(args.top4_root),
        "--sensitivity-jsonl", str(root / "collection/sample_sensitivity.jsonl"),
        "--perturbations-jsonl", str(root / "collection/perturbations.jsonl"),
        "--ablation-jsonl", str(root / "w4_nll/ablation_results.jsonl"),
        "--residuals-jsonl", str(root / "residuals/reconstruction_residuals.jsonl"),
        "--promotion-root", str(root / "promotion/runs"),
        "--top2-selection", str(root / "promotion/top2_selection.json"),
        "--output-dir", str(root / "analysis"),
        "--bootstrap-samples", "2000", "--seed", str(args.seed),
    ]


def execute_stage(args: argparse.Namespace, stage: str) -> None:
    root, py = args.output_dir, sys.executable
    logs = root / "logs"
    cache = root / "_calibration_cache"
    common = _common(args)
    if stage == "preflight":
        available_memory = psutil.virtual_memory().available
        disk_free = shutil.disk_usage(root.parent if root.parent.exists() else Path(".")).free
        active = _active_quant_processes()
        if available_memory < 80 * 1024**3:
            raise RuntimeError(f"preflight requires 80 GiB available memory; found {available_memory / 1024**3:.1f}")
        if disk_free < 100 * 1024**3:
            raise RuntimeError(f"preflight requires 100 GiB free disk; found {disk_free / 1024**3:.1f}")
        if active:
            raise RuntimeError(f"other quantization processes are active: {active}")
        for path in (
            Path(args.model_path), args.calibration_manifest, args.evaluation_manifest,
            args.top3_root, args.top4_root, args.vlmevalkit_root,
        ):
            if not path.exists():
                raise FileNotFoundError(path)
        test_environment = dict(os.environ)
        existing_pythonpath = test_environment.get("PYTHONPATH", "")
        test_environment["PYTHONPATH"] = os.pathsep.join(
            value for value in (str(args.vlmevalkit_root.resolve()), existing_pythonpath)
            if value
        )
        _run(
            [
                py, "-m", "pytest", "-q",
                "experiments/vlm_quant_benchmark/tests",
            ],
            logs / "preflight_tests.log",
            environment=test_environment,
        )
        _run([
            py, "-m", "experiments.vlm_quant_benchmark.run_token_sensitivity",
            *common, "--output-dir", str(root / "_preflight/gradient_dry"),
            "--dry-run",
        ], logs / "preflight_gradient_dry.log")
        _run([
            py, "-m", "experiments.vlm_quant_benchmark.run_sensitivity_ablation",
            *common, "--output-dir", str(root / "_preflight/ablation_dry"),
            "--dry-run",
        ], logs / "preflight_ablation_dry.log")
        if not args.skip_real_smoke:
            smoke_common = [
                "--model-path", str(args.model_path),
                "--calibration-manifest", str(args.calibration_manifest),
                "--evaluation-manifest", str(args.evaluation_manifest),
                "--threads", str(args.threads), "--seed", str(args.seed),
            ]
            if args.allow_nonstandard_model:
                smoke_common.append("--allow-nonstandard-model")
            _run([
                py, "-m", "experiments.vlm_quant_benchmark.run_reconstruction_residuals",
                *smoke_common,
                "--calibration-samples", "2", "--target-per-dataset", "1",
                "--samples-per-dataset", "1", "--layers", "0",
                "--fast-smoke",
                "--output-dir", str(root / "_preflight/three_model_residual_smoke"),
                "--calibration-cache-dir", str(root / "_preflight/smoke_cache"),
            ], logs / "preflight_three_model_smoke.log")
        atomic_json(root / "preflight_audit.json", {
            "status": "complete", "available_memory_bytes": available_memory,
            "disk_free_bytes": disk_free, "active_quant_processes": active,
            "threads": args.threads, "seed": args.seed,
            "model_path": str(Path(args.model_path).resolve()),
            "calibration_manifest_sha256": file_digest(args.calibration_manifest),
            "evaluation_manifest_sha256": file_digest(args.evaluation_manifest),
            "real_three_model_smoke": not args.skip_real_smoke,
        })
    elif stage == "gradients":
        _run([
            py, "-m", "experiments.vlm_quant_benchmark.run_token_sensitivity",
            *common, "--output-dir", str(root / "collection"),
            "--perturbation-rms", "0", "--resume",
        ], logs / "gradients.log")
        if _line_count(root / "collection/sample_sensitivity.jsonl") != 18432:
            raise RuntimeError("formal gradient row count must equal 18,432")
    elif stage == "w4_nll":
        _run([
            py, "-m", "experiments.vlm_quant_benchmark.run_sensitivity_ablation",
            *common, "--output-dir", str(root / "w4_nll"),
            "--calibration-cache-dir", str(cache), "--resume",
        ], logs / "w4_nll.log")
        if _line_count(root / "w4_nll/ablation_results.jsonl") != 1536:
            raise RuntimeError("base formal W4 NLL row count must equal 1,536")
    elif stage == "residuals":
        _run([
            py, "-m", "experiments.vlm_quant_benchmark.run_reconstruction_residuals",
            *common, "--output-dir", str(root / "residuals"),
            "--calibration-cache-dir", str(cache), "--resume",
        ], logs / "residuals.log")
        if _line_count(root / "residuals/reconstruction_residuals.jsonl") != 8064:
            raise RuntimeError("formal residual row count must equal 8,064")
    elif stage == "perturbations":
        rms = root / "residuals/mbq_layer_residual_rms.json"
        if not rms.is_file():
            raise FileNotFoundError(rms)
        _run([
            py, "-m", "experiments.vlm_quant_benchmark.run_token_sensitivity",
            *common, "--output-dir", str(root / "collection"),
            "--perturbation-rms-json", str(rms), "--resume",
        ], logs / "perturbations.log")
        if _line_count(root / "collection/perturbations.jsonl") != 672:
            raise RuntimeError("formal perturbation row count must equal 672")
    elif stage == "analysis":
        _run(_analysis_command(args), logs / "analysis_mid.log")
    elif stage == "promotion":
        promotion = root / "promotion"
        _run([
            py, "-m", "experiments.vlm_quant_benchmark.select_sensitivity_promotion",
            "--ablation-jsonl", str(root / "w4_nll/ablation_results.jsonl"),
            "--ablation-plan", str(root / "w4_nll/ablation_plan.json"),
            "--evaluation-manifest", str(args.evaluation_manifest),
            "--output-dir", str(promotion), "--seed", str(args.seed),
        ], logs / "promotion_select.log")
        selection = json.loads((promotion / "top2_selection.json").read_text())
        for variant in selection["selected"]:
            _run([
                py, "-m", "experiments.vlm_quant_benchmark.run_vlmeval_official",
                "--vlmevalkit-root", str(args.vlmevalkit_root),
                "--lmu-data-root", str(args.lmu_data_root),
                "--base-model-path", str(args.model_path),
                "--calibration-manifest", str(args.calibration_manifest),
                "--calibration-cache-dir", str(cache),
                "--screen-manifest", str(promotion / "promotion_manifest.jsonl"),
                "--expected-samples-per-dataset", "64", "--preset", "screen",
                "--experiment-config-json", str(promotion / f"{variant}_config.json"),
                "--threads", str(args.threads), "--seed", str(args.seed),
                "--output-dir", str(promotion / "runs" / variant), "--resume",
                *( ["--allow-nonstandard-model"] if args.allow_nonstandard_model else [] ),
            ], logs / f"promotion_{variant}.log")
        _run([
            py, "-m", "experiments.vlm_quant_benchmark.run_sensitivity_ablation",
            *common, "--output-dir", str(root / "w4_nll"),
            "--calibration-cache-dir", str(cache), "--variants", "mbq_target_oracle",
            "--oracle-promotion-nll", "--resume",
        ], logs / "promotion_oracle_nll.log")
    elif stage == "final":
        _run(_analysis_command(args), logs / "analysis_final.log")
        required = {
            "sample_sensitivity": 18432,
            "perturbations": 672,
            "residuals": 8064,
        }
        observed = {
            "sample_sensitivity": _line_count(root / "collection/sample_sensitivity.jsonl"),
            "perturbations": _line_count(root / "collection/perturbations.jsonl"),
            "residuals": _line_count(root / "residuals/reconstruction_residuals.jsonl"),
            "ablation_nll": _line_count(root / "w4_nll/ablation_results.jsonl"),
        }
        if any(observed[key] != value for key, value in required.items()):
            raise RuntimeError(f"formal row-count validation failed: {observed}")
        selection = json.loads((root / "promotion/top2_selection.json").read_text())
        promotion_rows = {}
        for variant in selection["selected"]:
            result = root / "promotion/runs" / variant / "official_result.json"
            if not result.is_file() or json.loads(result.read_text()).get("status") != "complete":
                raise RuntimeError(f"promotion result is incomplete: {variant}")
            promotion_rows[variant] = sum(
                int(item["samples"]) for item in json.loads(result.read_text())["datasets"].values()
            )
            if promotion_rows[variant] != 256:
                raise RuntimeError(f"promotion {variant} must contain 256 samples")
        atomic_json(root / "formal_run_audit.json", {
            "status": "complete", "required_rows": required,
            "observed_rows": observed, "promotion_rows": promotion_rows,
            "selected": selection["selected"],
            "input_digests": {
                "calibration_manifest": file_digest(args.calibration_manifest),
                "evaluation_manifest": file_digest(args.evaluation_manifest),
            },
            "attribution_decision": str((root / "analysis/attribution_decision.json").resolve()),
        })
    else:  # pragma: no cover
        raise ValueError(stage)


def main() -> None:
    args = parse_args()
    if args.threads != 16:
        raise ValueError("the preregistered formal protocol requires exactly 16 threads")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "stage_state.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {
        "format_version": 1,
        "stages": {stage: {"status": "pending"} for stage in STAGES},
    }
    selected_stages = STAGES if args.stage == "all" else (args.stage,)
    with pid_lock(args.output_dir / "formal_run.pid"):
        for stage in selected_stages:
            if args.resume and state["stages"].get(stage, {}).get("status") == "complete":
                continue
            started = time.time()
            state["stages"][stage] = {"status": "running", "started_unix": started}
            atomic_json(state_path, state)
            try:
                execute_stage(args, stage)
            except Exception as error:
                state["stages"][stage] = {
                    "status": "failed", "started_unix": started,
                    "finished_unix": time.time(), "error_type": type(error).__name__,
                    "error": str(error),
                }
                atomic_json(state_path, state)
                raise
            state["stages"][stage] = {
                "status": "complete", "started_unix": started,
                "finished_unix": time.time(), "elapsed_seconds": time.time() - started,
            }
            atomic_json(state_path, state)


if __name__ == "__main__":
    main()
