from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import os
import tarfile
import traceback
from pathlib import Path

import torch

from experiments.ffn_quant_causal.budget import WallclockBudgetExhausted
from experiments.infinitevl_quant.progress import atomic_json, file_sha256

from .analysis import analyze
from .config import MODELS, STAGES, Config
from .design import build_design
from .prepare import prepare
from .protocol import audit_live, bridge, execute_matrix, optional_old_sweep, qa, rank
from .reporting import report
from .runtime import Budget, audit_identity, frozen_json, load_splits, progress, rows, verify_frozen_sources


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


def finalize_stage(evidence, design, splits, split, keys):
    counts = {}
    for kind in MODELS:
        labels = set(label for key in keys for label in design["models"][kind][key])
        for label in labels:
            rows(evidence, kind, split, label, splits[split])
        counts[kind] = {"conditions": len(labels), "samples_per_condition": len(splits[split])}
    evidence.finish({"complete": True, "counts": counts})


def archive(config):
    root = config.output_dir
    target = root / "report/sample_records.tar.gz"
    files = sorted(
        p
        for stage in ("audit", "rank", "bridge", "confirm", "sweep", "qa", "old_supplement")
        for p in (root / stage).glob("*/samples/*.json")
    )
    with tarfile.open(target, "w:gz") as tar:
        for path in files:
            tar.add(path, arcname=str(path.relative_to(root)))
        for path in sorted((root / "prepare").glob("*.jsonl")):
            tar.add(path, arcname=str(path.relative_to(root)))
    expected = {str(p.relative_to(root)): file_sha256(p) for p in files}
    import hashlib

    with tarfile.open(target, "r:gz") as tar:
        for name, digest in expected.items():
            if hashlib.sha256(tar.extractfile(name).read()).hexdigest() != digest:
                raise RuntimeError("exported sample archive differs from source rows")
    atomic_json(
        root / "report/sample_archive.json",
        {"path": str(target), "sha256": file_sha256(target), "sample_records": len(files), "source_hashes": expected},
    )


def main():
    parser = argparse.ArgumentParser(
        description="Three-model W4 FFN layerwise generalization, 240h persistent CPU budget"
    )
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=Config().output_dir)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--total-hours", type=float, default=240)
    parser.add_argument("--round-hours", type=float, default=120)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = Config(
        output_dir=args.output_dir.resolve(),
        threads=args.threads,
        total_hours=args.total_hours,
        round_hours=args.round_hours,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(16)
    torch.set_num_interop_threads(1)
    budget = Budget(config)
    audit = splits = design = None
    current = "prepare"
    with (config.output_dir / ".writer.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another generalization experiment writer is active") from error
        try:
            if args.stage == "report":
                path = config.output_dir / "analyze/summary.json"
                if path.exists():
                    saved = json.loads(path.read_text())
                    design = json.loads((config.output_dir / "audit/locked_design.json").read_text())
                    audit = json.loads((config.output_dir / "audit/audit.json").read_text())
                    if not progress(config, "analyze", audit, design).verified_complete():
                        raise RuntimeError("analysis summary lacks a verified completion manifest")
                    print(json.dumps(report(config, saved, design), ensure_ascii=False), flush=True)
                    return
            budget.check(stage=current, boundary="before_preparation")
            status(
                config,
                "running",
                current,
                pid=os.getpid(),
                deadline_utc=datetime.datetime.fromtimestamp(budget.deadline_unix, datetime.timezone.utc).isoformat(),
            )
            paths = prepare(config)
            splits = load_splits(config)
            audit = audit_identity(config, paths, splits)
            current = "audit"
            status(config, "running", current, action="native_RTN_scalar_and_history_checks")
            audit_summary = audit_live(config, audit, splits, budget)
            if args.stage == "audit":
                status(config, "stage_complete", current)
                return
            current = "rank"
            status(config, "running", current, action="28_layer_recovery_on_old_diagnostic8")
            orders = rank(config, audit, splits, budget)
            design = frozen_json(config.output_dir / "audit/locked_design.json", build_design(orders))
            if args.stage == "rank":
                status(config, "stage_complete", current)
                return
            if args.stage in ("all", "confirm", "sweep"):
                current = "bridge"
                status(config, "running", current)
                bridge(config, audit, splits, design, budget, audit_summary)
            if args.stage in ("all", "confirm"):
                current = "confirm"
                status(config, "running", current)
                evidence = execute_matrix(config, audit, splits, design, budget, "confirm", "confirm128", ("confirm",))
                finalize_stage(evidence, design, splits, "confirm128", ("confirm",))
                if args.stage == "confirm":
                    status(config, "stage_complete", current)
                    return
            if args.stage in ("all", "sweep"):
                current = "sweep"
                status(config, "running", current, action="random_controls_before_dense_scan")
                execute_matrix(config, audit, splits, design, budget, "sweep", "screen32", ("random",))
                evidence = execute_matrix(config, audit, splits, design, budget, "sweep", "screen32", ("main",))
                finalize_stage(evidence, design, splits, "screen32", ("main", "random"))
                if args.stage == "sweep":
                    status(config, "stage_complete", current)
                    return
            if args.stage == "all":
                current = "qa"
                status(config, "running", current)
                try:
                    qa(config, audit, splits, design, budget)
                except WallclockBudgetExhausted:
                    atomic_json(
                        config.output_dir / "qa_omission.json",
                        {"reason": "analysis_reserve_reached", "partial_rows_retained": True},
                    )
                if budget.remaining_seconds() > 0:
                    current = "old_supplement"
                    status(config, "running", current)
                    try:
                        optional_old_sweep(config, audit, splits, design, budget, audit_summary)
                    except WallclockBudgetExhausted:
                        atomic_json(
                            config.output_dir / "optional_omissions.json",
                            {"old_qwen7b_dense": {"reason": "analysis_reserve_reached", "partial_rows_retained": True}},
                        )
            current = "analyze"
            budget.inference = False
            status(config, "running", current)
            budget.check(stage=current, boundary="before_independent_analysis")
            verify_frozen_sources(config, audit)
            summary = analyze(config, audit, splits, design)
            atomic_json(config.output_dir / "verification.json", summary["verification"])
            analysis_progress = progress(config, "analyze", audit, design)
            analysis_progress.finish(summary)
            current = "report"
            status(config, "running", current)
            report(config, summary, design)
            archive(config)
            atomic_json(
                config.output_dir / "report/final_identity.json",
                {
                    "analysis_sha256": file_sha256(config.output_dir / "analyze/summary.json"),
                    "verification_sha256": file_sha256(config.output_dir / "verification.json"),
                    "archive_sha256": file_sha256(config.output_dir / "report/sample_records.tar.gz"),
                },
            )
            status(
                config,
                "complete",
                current,
                core_complete=True,
                qa_complete=(config.output_dir / "qa/.complete").exists(),
                elapsed_hours=(__import__("time").time() - budget.actual.started_unix) / 3600,
            )
        except BaseException as error:
            interrupted = isinstance(error, (KeyboardInterrupt, SystemExit))
            exhausted = isinstance(error, WallclockBudgetExhausted)
            status(
                config,
                "budget_exhausted" if exhausted else "interrupted" if interrupted else "failed",
                current,
                error_type=type(error).__name__,
                error=str(error),
            )
            if audit and splits and design:
                try:
                    budget.inference = False
                    summary = analyze(config, audit, splits, design, partial=True)
                    report(config, summary, design)
                    atomic_json(config.output_dir / "verification.json", summary["verification"])
                except Exception:
                    traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
