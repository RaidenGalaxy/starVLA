from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from experiments.ffn_quant_causal.budget import (
    BUDGET_FILENAME,
    EXHAUSTED_FILENAME,
    WallclockBudget,
    WallclockBudgetExhausted,
    validate_completed_stage,
)
from experiments.ffn_quant_causal.evaluation import validate_variant_rows
from experiments.ffn_quant_causal.run_cpu_protocol import _execute_with_budget
from experiments.infinitevl_quant.progress import StageProgress, atomic_json


def test_persistent_budget_reopen_keeps_original_deadline(tmp_path) -> None:
    first = WallclockBudget.open_or_create(
        tmp_path,
        wallclock_hours=120.0,
        config_digest="config",
        create=True,
        now=100.0,
    )
    reopened = WallclockBudget.open_or_create(
        tmp_path,
        wallclock_hours=120.0,
        config_digest="config",
        create=False,
        now=10_000.0,
    )
    assert first.started_unix == reopened.started_unix == 100.0
    assert first.deadline_unix == reopened.deadline_unix == 100.0 + 120.0 * 3600.0
    record = json.loads((tmp_path / BUDGET_FILENAME).read_text())
    assert record["enforcement"] == "cooperative_boundaries_only"
    assert record["hard_interrupts"] is False


def test_budget_cannot_be_created_by_non_audit_or_reset_by_config(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="only the first audit"):
        WallclockBudget.open_or_create(
            tmp_path,
            wallclock_hours=120.0,
            config_digest="config",
            create=False,
        )
    WallclockBudget.open_or_create(
        tmp_path,
        wallclock_hours=120.0,
        config_digest="config",
        create=True,
        now=0.0,
    )
    with pytest.raises(RuntimeError, match="another config"):
        WallclockBudget.open_or_create(
            tmp_path,
            wallclock_hours=120.0,
            config_digest="changed",
            create=True,
        )
    with pytest.raises(RuntimeError, match="duration differs"):
        WallclockBudget.open_or_create(
            tmp_path,
            wallclock_hours=121.0,
            config_digest="config",
            create=True,
        )


def test_exhaustion_is_recorded_as_cooperative_boundary(tmp_path) -> None:
    budget = WallclockBudget.open_or_create(
        tmp_path,
        wallclock_hours=1.0,
        config_digest="config",
        create=True,
        now=10.0,
    )
    budget.check(stage="rtn", boundary="before_sample", now=3609.0)
    with pytest.raises(WallclockBudgetExhausted, match="after_sample_committed"):
        budget.check(
            stage="rtn",
            boundary="after_sample_committed",
            variant="w4",
            sample_id="sample-1",
            now=3610.0,
        )
    event = json.loads((tmp_path / EXHAUSTED_FILENAME).read_text())
    assert event["sample_id"] == "sample-1"
    assert event["in_flight_python_calls_are_not_hard_interrupted"] is True
    assert event["complete_marker_must_not_be_written"] is True


def test_resume_requires_exact_identity_and_declared_artifacts(tmp_path) -> None:
    progress = StageProgress(
        tmp_path, "rtn", config_digest="config", inputs={"model": "digest"},
    )
    atomic_json(progress.root / "summary.json", {"ok": True})
    progress.mark_complete(("summary.json",))
    assert validate_completed_stage(
        progress, expected_required=("summary.json",),
    )
    (progress.root / "summary.json").unlink()
    with pytest.raises(RuntimeError, match="identity or required artifacts"):
        validate_completed_stage(
            progress, expected_required=("summary.json",),
        )


def test_post_stage_expiration_removes_new_complete_marker(tmp_path) -> None:
    class BoundaryBudget:
        output_dir = tmp_path

        def __init__(self) -> None:
            self.calls = 0

        def check(self, **_kwargs) -> None:
            self.calls += 1
            if self.calls == 2:
                raise WallclockBudgetExhausted("expired after monolithic stage")

    budget = BoundaryBudget()

    def action() -> None:
        marker = tmp_path / "trace" / ".complete"
        marker.parent.mkdir(parents=True)
        marker.write_text("{}")

    with pytest.raises(WallclockBudgetExhausted):
        _execute_with_budget("trace", budget, action)
    assert not (tmp_path / "trace" / ".complete").exists()


def test_variant_rows_reject_foreign_duplicate_and_dataset_mismatch() -> None:
    samples = [
        SimpleNamespace(id="a", dataset="left"),
        SimpleNamespace(id="b", dataset="right"),
    ]
    valid = [{"id": "a", "dataset": "left"}, {"id": "b", "dataset": "right"}]
    assert validate_variant_rows(valid, samples, variant="toy") == valid
    with pytest.raises(RuntimeError, match="sample set mismatch"):
        validate_variant_rows(
            [{"id": "a", "dataset": "left"}, {"id": "foreign", "dataset": "right"}],
            samples,
            variant="toy",
        )
    with pytest.raises(RuntimeError, match="duplicate"):
        validate_variant_rows(
            [{"id": "a", "dataset": "left"}, {"id": "a", "dataset": "left"}],
            samples,
            variant="toy",
        )
    with pytest.raises(RuntimeError, match="mismatched datasets"):
        validate_variant_rows(
            [{"id": "a", "dataset": "wrong"}, {"id": "b", "dataset": "right"}],
            samples,
            variant="toy",
        )
