from __future__ import annotations

import argparse
from pathlib import Path

import torch

from experiments.ffn_quant_causal.budget import (
    WallclockBudget,
    validate_completed_stage,
)

from .config import STAGES, MechanismConfig
from .confirm import run_confirm
from .protocol import create_audit, stage_progress
from .reporting import run_report
from .stages import (
    run_calibrate,
    run_gdn_mediation,
    run_localize,
    run_transport,
)

REQUIRED = {
    "audit": ("audit.json", "manifests"),
    "calibrate": ("error_scales.json",),
    "localize": ("layer_pair_summary.json",),
    "transport": ("summary.json",),
    "gdn_mediation": ("state_and_shapley_summary.json",),
    "confirm": ("causal_effects.json",),
    "report": ("analyses.json", "report_summary.json", "report_zh.md"),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="InfiniteVL/Qwen FFN quantization strong-causal mechanism protocol",
    )
    parser.add_argument("--stage", required=True, choices=(*STAGES, "all"))
    parser.add_argument(
        "--parent-output",
        type=Path,
        default=Path("outputs/ffn_quant_causal_v3"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ffn_quant_mechanism_v1"),
    )
    parser.add_argument("--infinitevl-model-path", type=Path)
    parser.add_argument(
        "--qwen-model-path",
        type=Path,
        default=Path("playground/Pretrained_models/Qwen2.5-VL-3B-Instruct"),
    )
    parser.add_argument(
        "--calibration-manifest",
        type=Path,
        default=Path("data/vlm_quant_benchmark/multimodal_calibration/multimodal_calibration_4x32.jsonl"),
    )
    parser.add_argument(
        "--evaluation-manifest",
        type=Path,
        default=Path("data/vlm_quant_benchmark/top4_protocol_v3/vlm_screen_4x64.jsonl"),
    )
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--wallclock-hours", type=float, default=120.0)
    parser.add_argument("--resume", action="store_true")
    return parser


def _config(args: argparse.Namespace) -> MechanismConfig:
    return MechanismConfig(
        output_dir=args.output_dir.resolve(),
        parent_output=args.parent_output.resolve(),
        qwen_model_path=args.qwen_model_path.resolve(),
        calibration_manifest=args.calibration_manifest.resolve(),
        evaluation_manifest=args.evaluation_manifest.resolve(),
        torch_threads=args.threads,
        wallclock_hours=args.wallclock_hours,
    )


def _configure_threads(config: MechanismConfig) -> None:
    torch.set_num_threads(config.torch_threads)
    try:
        torch.set_num_interop_threads(config.interop_threads)
    except RuntimeError:
        if torch.get_num_interop_threads() != config.interop_threads:
            raise


def _complete(config: MechanismConfig, stage: str) -> bool:
    try:
        progress = stage_progress(config, stage)
    except FileNotFoundError:
        return False
    return validate_completed_stage(
        progress,
        expected_required=REQUIRED[stage],
    )


def _run_stage(
    stage: str,
    config: MechanismConfig,
    *,
    infinitevl_path: Path | None,
    budget: WallclockBudget,
) -> None:
    budget.check(stage=stage, boundary="before_stage")
    if stage == "calibrate":
        run_calibrate(
            config,
            infinitevl_path=infinitevl_path,
            wallclock_budget=budget,
        )
    elif stage == "localize":
        run_localize(
            config,
            infinitevl_path=infinitevl_path,
            wallclock_budget=budget,
        )
    elif stage == "transport":
        run_transport(
            config,
            infinitevl_path=infinitevl_path,
            wallclock_budget=budget,
        )
    elif stage == "gdn_mediation":
        run_gdn_mediation(
            config,
            infinitevl_path=infinitevl_path,
            wallclock_budget=budget,
        )
    elif stage == "confirm":
        run_confirm(
            config,
            infinitevl_path=infinitevl_path,
            wallclock_budget=budget,
        )
    elif stage == "report":
        run_report(config)
    else:
        raise ValueError(f"unknown executable stage {stage}")
    budget.check(stage=stage, boundary="after_stage")


def main() -> None:
    args = _parser().parse_args()
    config = _config(args)
    _configure_threads(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage in {"audit", "all"}:
        budget = WallclockBudget.open_or_create(
            config.output_dir,
            wallclock_hours=config.wallclock_hours,
            config_digest=config.digest(),
            create=True,
        )
        if args.resume and _complete(config, "audit"):
            print("[audit] verified complete stage; skipping", flush=True)
        else:
            budget.check(stage="audit", boundary="before_stage")
            create_audit(config)
            progress = stage_progress(config, "audit")
            progress.mark_complete(REQUIRED["audit"])
            budget.check(stage="audit", boundary="after_stage")
            print("[audit] complete", flush=True)
    else:
        budget = WallclockBudget.open_or_create(
            config.output_dir,
            wallclock_hours=config.wallclock_hours,
            config_digest=config.digest(),
            create=False,
        )
    for stage in STAGES[1:]:
        if args.stage not in {stage, "all"}:
            continue
        if args.resume and _complete(config, stage):
            print(f"[{stage}] verified complete stage; skipping", flush=True)
            continue
        _run_stage(
            stage,
            config,
            infinitevl_path=args.infinitevl_model_path,
            budget=budget,
        )
        print(f"[{stage}] complete", flush=True)


if __name__ == "__main__":
    main()
