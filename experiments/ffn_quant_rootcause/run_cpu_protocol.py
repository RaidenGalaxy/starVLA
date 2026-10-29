from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import traceback
from pathlib import Path

import torch

from experiments.ffn_quant_causal.budget import WallclockBudget
from experiments.infinitevl_quant.progress import atomic_json, file_sha256

from . import stages
from .config import STAGES, Config
from .evidence import EvidenceProgress, audit
from .reporting import report

DEPENDENCIES = {
    "audit": (),
    "ratio": ("audit",),
    "local": ("ratio",),
    "dose": ("local", "ratio"),
    "propagation": ("dose", "local"),
    "confirm": ("ratio", "local", "dose", "propagation"),
}


def parser():
    p = argparse.ArgumentParser(
        description="CPU-only, two-round FFN root-cause validation; immutable historical evidence"
    )
    p.add_argument("--stage", required=True, choices=(*STAGES, "all"))
    for name in (
        "output-dir",
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
    p.add_argument("--bootstrap-samples", type=int, default=10_000)
    p.add_argument(
        "--resume", action="store_true", help="Validate identity and checksums before skipping completed stages"
    )
    return p


def make_progress(config, stage, audit_digest):
    upstream = {}
    for dependency in DEPENDENCIES[stage]:
        stages.load_summary(config, dependency)
        upstream[dependency] = file_sha256(config.output_dir / dependency / "summary.json")
    return EvidenceProgress(
        config.output_dir, stage, config_digest=config.digest(), inputs={"audit": audit_digest, "upstream": upstream}
    )


def status(config, state, stage, **extra):
    atomic_json(
        config.output_dir / "status.json",
        {
            "status": state,
            "stage": stage,
            "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            **extra,
        },
    )


def entry_budget(config, requested_stage, round1):
    if requested_stage in ("propagation", "confirm") or (
        requested_stage == "all" and (config.output_dir / "dose/.complete").is_file()
    ):
        stages.load_summary(config, "dose")
        return WallclockBudget.open_or_create(
            config.output_dir / "budgets/round2",
            wallclock_hours=config.wallclock_hours,
            config_digest=config.digest(),
            create=True,
        )
    return round1


def main():
    args = parser().parse_args()
    options = {k: v for k, v in vars(args).items() if k not in ("stage", "resume") and v is not None}
    config = Config(**options)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config.threads)
    torch.set_num_interop_threads(1)
    # Prevent concurrent writers, including independent CLI invocations in the same output.
    with (config.output_dir / ".writer.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another root-cause writer is running in this output directory") from error
        if args.stage == "report":
            print(json.dumps(report(config), ensure_ascii=False), flush=True)
            return
        current = "audit"
        try:
            round1 = WallclockBudget.open_or_create(
                config.output_dir / "budgets/round1",
                wallclock_hours=config.wallclock_hours,
                config_digest=config.digest(),
                create=args.stage in ("audit", "all"),
            )
            verification_budget = entry_budget(config, args.stage, round1)
            verification_budget.check(stage="audit", boundary="before_identity_audit")
            status(config, "running", "audit", action="verify_models_code_data_and_history")
            payload, splits, audit_digest = audit(config)
            requested = STAGES[:-1] if args.stage == "all" else (args.stage,)
            for current in requested:
                progress = make_progress(config, current, audit_digest)
                if args.resume and progress.verified_complete():
                    print(f"[{current}] identity and checksums verified; skipping", flush=True)
                    continue
                if (progress.root / ".complete").exists():
                    raise RuntimeError("completed output exists; use --resume or a new output directory")
                round_number = 1 if current in ("audit", "ratio", "local", "dose") else 2
                budget = (
                    round1
                    if round_number == 1
                    else WallclockBudget.open_or_create(
                        config.output_dir / "budgets/round2",
                        wallclock_hours=config.wallclock_hours,
                        config_digest=config.digest(),
                        create=True,
                    )
                )
                budget.check(stage=current, boundary="before_stage")
                status(
                    config,
                    "running",
                    current,
                    round=round_number,
                    deadline_utc=datetime.datetime.fromtimestamp(
                        budget.deadline_unix, datetime.timezone.utc
                    ).isoformat(),
                )
                report(config)
                print(
                    f"[{current}] starting round={round_number} remaining_hours={budget.remaining_seconds() / 3600:.2f}",
                    flush=True,
                )
                if current == "audit":
                    summary = stages.scalar_audit(config, payload, progress, budget)
                    summary["audit_digest"] = audit_digest
                else:
                    summary = getattr(stages, current)(config, payload, splits, progress, budget)
                budget.check(stage=current, boundary="before_complete_marker")
                progress.finish(summary)
                report(config)
                print(f"[{current}] complete", flush=True)
            status(config, "complete" if args.stage == "all" else "stage_complete", current)
            print(json.dumps(report(config), ensure_ascii=False), flush=True)
        except BaseException as error:
            status(
                config,
                "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                current,
                error_type=type(error).__name__,
                error=str(error),
            )
            try:
                report(config)
            except Exception:
                traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
