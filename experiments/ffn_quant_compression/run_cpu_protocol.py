from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import traceback
from pathlib import Path

import torch

from experiments.ffn_quant_causal.budget import WallclockBudget, WallclockBudgetExhausted
from experiments.ffn_quant_rootcause.stages import load_summary
from experiments.infinitevl_quant.progress import atomic_json, file_sha256

from .analysis import analyze
from .config import MODELS, STAGES, Config
from .evidence import Progress, prepare, write_status
from .reporting import report
from .sweep import audit_sentinels, complete_rows, sweep
from .verification import verify


def parser():
    p = argparse.ArgumentParser(description="Fixed W4 FFN compression/quality curves, one persistent CPU budget")
    p.add_argument("--stage", required=True, choices=(*STAGES, "all"))
    for name in (
        "output-dir",
        "rootcause-output",
        "parent-output",
        "mechanism-output",
        "infinitevl-model-path",
        "qwen-model-path",
        "calibration-manifest",
        "evaluation-manifest",
    ):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--wallclock-hours", type=float, default=120)
    p.add_argument("--reserve-hours", type=float, default=5)
    p.add_argument("--bootstrap-samples", type=int, default=10_000)
    p.add_argument("--resume", action="store_true")
    return p


def progress_for(config, stage, payload, design):
    upstream = {}
    dependencies = {"audit": (), "sweep": ("audit",), "analyze": ("audit", "sweep")}
    for dependency in dependencies[stage]:
        load_summary(config, dependency)
        upstream[dependency] = file_sha256(config.output_dir / dependency / "summary.json")
    return Progress(config, stage, payload, design, upstream)


def mandatory_complete(progress, design, splits):
    for kind in MODELS:
        for split, key, order in (("confirm192", "confirm", "qa_order"), ("screen32", "screen", "main_screen_order")):
            if any(
                complete_rows(progress, f"{kind}__{split}__{label}", splits[key], required=False) is None
                for label in design["models"][kind][order]
            ):
                return False
    return True


def snapshot(config, payload, splits, design, progress, omissions=None):
    summary, _ = analyze(config, payload, splits, design, progress, partial=True, omissions=omissions)
    return report(config, summary)


def main():
    args = parser().parse_args()
    values = {k: v for k, v in vars(args).items() if k not in ("stage", "resume") and v is not None}
    config = Config(**values)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config.threads)
    torch.set_num_interop_threads(1)
    with (config.output_dir / ".writer.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another compression experiment writer is running") from error
        current = "audit"
        payload = splits = design = sweep_progress = None
        try:
            if args.stage == "report":
                # Read only verified completed analysis, or generate an explicitly partial snapshot.
                if (config.output_dir / "analyze/.complete").exists():
                    summary = load_summary(config, "analyze")
                    print(json.dumps(report(config, summary), ensure_ascii=False), flush=True)
                    return
            budget = WallclockBudget.open_or_create(
                config.output_dir / "budget",
                wallclock_hours=config.wallclock_hours,
                config_digest=config.digest(),
                create=args.stage in ("all", "audit"),
            )
            budget.check(stage="audit", boundary="before_identity_audit")
            write_status(config, "running", "audit", action="verify_models_rankings_data_runtime_and_code")
            payload, splits, design = prepare(config)
            if args.stage == "report":
                sweep_progress = progress_for(config, "sweep", payload, design)
                print(
                    json.dumps(snapshot(config, payload, splits, design, sweep_progress), ensure_ascii=False), flush=True
                )
                return
            requested = STAGES[:-1] if args.stage == "all" else (args.stage,)
            for current in requested:
                progress = progress_for(config, current, payload, design)
                if args.resume and progress.verified_complete():
                    print(f"[{current}] identity and checksums verified; skipping", flush=True)
                    continue
                if (progress.root / ".complete").exists():
                    raise RuntimeError("completed stage exists; use --resume or a new output directory")
                budget.check(stage=current, boundary="before_stage")
                write_status(
                    config,
                    "running",
                    current,
                    deadline_utc=datetime.datetime.fromtimestamp(
                        budget.deadline_unix, datetime.timezone.utc
                    ).isoformat(),
                )
                print(f"[{current}] starting remaining_hours={budget.remaining_seconds() / 3600:.2f}", flush=True)
                if current == "audit":
                    report(config)
                    summary = audit_sentinels(config, payload, splits, design, progress, budget)
                elif current == "sweep":
                    sweep_progress = progress
                    audit_summary = load_summary(config, "audit")
                    try:
                        summary = sweep(
                            config,
                            payload,
                            splits,
                            design,
                            audit_summary,
                            progress,
                            budget,
                            on_checkpoint=lambda omissions, target_progress=progress: snapshot(
                                config, payload, splits, design, target_progress, omissions
                            ),
                        )
                    except WallclockBudgetExhausted:
                        if budget.remaining_seconds() <= 0 or not mandatory_complete(progress, design, splits):
                            raise
                        summary = {
                            "mandatory_conditions_complete": True,
                            "budgeted_omissions": {"random_nll": {"reason": "analysis_reserve_reached"}},
                            "history_reuse_is_explicit": True,
                        }
                else:
                    sweep_progress = progress_for(config, "sweep", payload, design)
                    sweep_summary = load_summary(config, "sweep")
                    summary, raw = analyze(
                        config, payload, splits, design, sweep_progress, omissions=sweep_summary["budgeted_omissions"]
                    )
                    budget.check(stage=current, boundary="before_independent_verification")
                    summary["verification"] = verify(summary, raw, splits, design, config.bootstrap_samples)
                    atomic_json(config.output_dir / "verification.json", summary["verification"])
                budget.check(stage=current, boundary="before_complete_marker")
                progress.finish(summary)
                print(f"[{current}] complete", flush=True)
                if current == "sweep":
                    snapshot(config, payload, splits, design, progress, summary["budgeted_omissions"])
            write_status(config, "complete" if args.stage == "all" else "stage_complete", current)
            print(json.dumps(report(config), ensure_ascii=False), flush=True)
        except BaseException as error:
            state = (
                "budget_exhausted"
                if isinstance(error, WallclockBudgetExhausted)
                else ("interrupted" if isinstance(error, KeyboardInterrupt) else "failed")
            )
            write_status(config, state, current, error_type=type(error).__name__, error=str(error))
            try:
                if sweep_progress is not None:
                    snapshot(config, payload, splits, design, sweep_progress)
                else:
                    report(config)
            except Exception:
                traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
