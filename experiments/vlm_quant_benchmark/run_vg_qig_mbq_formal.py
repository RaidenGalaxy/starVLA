"""Resumable CPU experiment driver for Vulnerability-Gated QIG-MBQ."""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import torch

from .config import ExperimentConfig, ImplementationTier, MethodName
from .data import assert_disjoint, load_manifest
from .reporting import atomic_json


DEFAULT_MODEL = Path("playground/Pretrained_models/Qwen2.5-VL-3B-Instruct")
DEFAULT_CALIBRATION = Path(
    "data/vlm_quant_benchmark/multimodal_calibration/multimodal_calibration_4x32.jsonl"
)
DEFAULT_EVALUATION = Path("data/vlm_quant_benchmark/top4_protocol_v3/vlm_screen_4x64.jsonl")
DEFAULT_OUTPUT = Path("outputs/vlm_quant_benchmark/vg_qig_mbq_v1")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--calibration-manifest", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--evaluation-manifest", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--mode", choices=("preflight", "smoke", "full", "w4a8", "w2a16"), default="smoke",
    )
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--resume", action="store_true", default=True)
    return parser.parse_args()


def config_matrix(mode: str) -> list[ExperimentConfig]:
    smoke = mode == "smoke"
    common = {
        "calibration": "multimodal_4source",
        "calibration_samples": 1 if smoke else 128,
        "max_new_tokens": 8 if smoke else 128,
        "mbq_gradient_aggregation": "source_balanced",
        "mbq_objective_normalization": "modality_mean",
        "mbq_scale_grid": (0.0, 0.5) if smoke else tuple(i / 20 for i in range(20)),
    }
    vg = {
        "implementation_tier": ImplementationTier.CPU_RESEARCH,
        "vg_vvs_samples": 1 if smoke else 16,
        "vg_qig_samples": 1 if smoke else 32,
        "vg_vvs_max_layers": 2 if smoke else 4,
        "vg_gate_bootstrap_samples": 20 if smoke else 2000,
        "vg_ig_step_grid": (2,) if smoke else (8, 16, 32),
        "vg_coordinate_sweeps": 1 if smoke else 2,
        "vg_candidate_layers": (0, 1) if smoke else (),
    }
    if smoke:
        return [ExperimentConfig(MethodName.VG_QIG_MBQ, weight_bits=3, **common, **vg)]
    if mode == "w4a8":
        w4a8 = {
            "weight_bits": 4,
            "activation_bits": 8,
            "weight_granularity": "per_output_channel",
            "zero_point": False,
        }
        return [
            ExperimentConfig(MethodName.VG_QIG_MBQ, **w4a8, **common, **vg),
            ExperimentConfig(MethodName.MBQ, **w4a8, **common),
        ]
    if mode == "w2a16":
        return [
            ExperimentConfig(MethodName.VG_QIG_MBQ, weight_bits=2, **common, **vg),
            ExperimentConfig(MethodName.MBQ, weight_bits=2, **common),
        ]
    return [
        ExperimentConfig(MethodName.VG_QIG_MBQ, weight_bits=3, **common, **vg),
        ExperimentConfig(MethodName.VG_UNIFORM_MBQ, weight_bits=3, **common, **vg),
        ExperimentConfig(MethodName.MBQ, weight_bits=3, **common),
        ExperimentConfig(MethodName.VG_QIG_MBQ, weight_bits=4, **common, **vg),
        ExperimentConfig(MethodName.MBQ, weight_bits=4, **common),
    ]


def label(config: ExperimentConfig) -> str:
    activations = config.activation_bits or 16
    return f"{config.method.value}__w{config.weight_bits}a{activations}"


def preflight(args) -> dict:
    if torch.cuda.is_available():
        raise RuntimeError("this protocol is CPU-only but PyTorch reports CUDA available")
    if not (args.model_path / "config.json").is_file():
        raise FileNotFoundError(args.model_path / "config.json")
    calibration = load_manifest(args.calibration_manifest)
    evaluation = load_manifest(args.evaluation_manifest)
    assert_disjoint(calibration, evaluation)
    sources = {sample.dataset for sample in calibration}
    if len(sources) != 4:
        raise ValueError(f"formal VG calibration requires four sources, got {sorted(sources)}")
    return {
        "cpu_only": True, "torch_cuda_available": False,
        "torch_threads": args.threads, "model_path": str(args.model_path.resolve()),
        "calibration_samples": len(calibration), "calibration_sources": sorted(sources),
        "evaluation_samples": len(evaluation),
        "implementation_tier": "cpu_research_approximation",
        "official_reproduction_claim": False, "deployment_claim": False,
    }


def _rows(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {row["id"]: row for row in map(json.loads, path.read_text().splitlines())}


def _paired_ci(base: dict[str, dict], candidate: dict[str, dict], seed: int = 42) -> dict:
    ids = sorted(set(base) & set(candidate))
    differences = [
        float(candidate[item]["exact_match"]) - float(base[item]["exact_match"])
        for item in ids
    ]
    if not differences:
        return {"samples": 0, "mean": None, "ci95": [None, None]}
    rng = random.Random(seed)
    estimates = []
    for _ in range(10000):
        estimates.append(sum(rng.choice(differences) for _ in differences) / len(differences))
    estimates.sort()
    return {"samples": len(ids), "mean": sum(differences) / len(differences),
            "ci95": [estimates[249], estimates[9749]]}


def summarize(root: Path) -> dict:
    rows = {
        path.parent.name: _rows(path)
        for path in root.glob("runs/*/generations.jsonl")
    }
    output = {"paired": {}, "replacement_gate": None}
    for run_key, summary_key in (
        ("w2a16", "w2"), ("w3a16", "w3"),
        ("w4a16", "w4"), ("w4a8", "w4a8"),
    ):
        base = rows.get(f"mbq__{run_key}", {})
        qig = rows.get(f"vg_qig_mbq__{run_key}", {})
        if base and qig:
            output["paired"][summary_key] = _paired_ci(base, qig)
    if "w3" in output["paired"] and "w4" in output["paired"]:
        w3, w4 = output["paired"]["w3"], output["paired"]["w4"]
        output["replacement_gate"] = {
            "w3_ci_lower_gt_zero": w3["ci95"][0] > 0,
            "w4_noninferiority_ge_minus_0_5pt": w4["mean"] >= -0.005,
            "passed": w3["ci95"][0] > 0 and w4["mean"] >= -0.005,
            "dataset_guardrail_requires_manual_review": True,
        }
    return output


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    record = preflight(args)
    atomic_json(args.output_root / "preflight.json", record)
    if args.mode == "preflight":
        return
    configs = config_matrix(args.mode)
    config_root = args.output_root / "configs"
    cache_root = args.output_root / "cache"
    run_root = args.output_root / ("smoke" if args.mode == "smoke" else "runs")
    state = {"mode": args.mode, "started_at": time.time(), "runs": []}
    atomic_json(args.output_root / f"{args.mode}_state.json", state)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    for config in configs:
        config_path = config_root / f"{label(config)}.json"
        config.dump(config_path)
        output = run_root / label(config)
        command = [
            sys.executable, "-m", "experiments.vlm_quant_benchmark.run_single",
            "--model-path", str(args.model_path), "--config-json", str(config_path),
            "--calibration-manifest", str(args.calibration_manifest),
            "--evaluation-manifest", str(args.evaluation_manifest),
            "--output-dir", str(output), "--calibration-cache-dir", str(cache_root),
            "--threads", str(args.threads), "--preset", "smoke" if args.mode == "smoke" else "full",
            "--resume",
        ]
        started = time.time()
        completed = subprocess.run(command, cwd=Path.cwd(), env=environment, check=False)
        row = {"label": label(config), "returncode": completed.returncode,
               "elapsed_seconds": time.time() - started, "output": str(output)}
        state["runs"].append(row)
        atomic_json(args.output_root / f"{args.mode}_state.json", state)
        if completed.returncode:
            raise SystemExit(completed.returncode)
    state["completed_at"] = time.time()
    state["summary"] = summarize(args.output_root)
    atomic_json(args.output_root / f"{args.mode}_state.json", state)
    atomic_json(args.output_root / "comparison.json", state["summary"])


if __name__ == "__main__":
    main()
