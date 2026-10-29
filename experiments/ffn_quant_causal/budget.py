from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from experiments.infinitevl_quant.progress import StageProgress, atomic_json


BUDGET_FORMAT_VERSION = 1
BUDGET_FILENAME = "wallclock_budget.json"
EXHAUSTED_FILENAME = "wallclock_budget_exhausted.json"


class WallclockBudgetExhausted(RuntimeError):
    """Raised only at a cooperative protocol boundary.

    A model forward/generation that is already executing is never claimed to
    be preempted.  The exception is raised before starting the next bounded
    unit of work, or immediately after the current unit has been committed.
    """


def _utc_timestamp(value: float) -> str:
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


@dataclass(frozen=True)
class WallclockBudget:
    output_dir: Path
    config_digest: str
    started_unix: float
    deadline_unix: float
    budget_seconds: float

    @property
    def path(self) -> Path:
        return self.output_dir / BUDGET_FILENAME

    @property
    def exhausted_path(self) -> Path:
        return self.output_dir / EXHAUSTED_FILENAME

    @classmethod
    def open_or_create(
        cls,
        output_dir: Path,
        *,
        wallclock_hours: float,
        config_digest: str,
        create: bool,
        now: float | None = None,
    ) -> "WallclockBudget":
        """Open one persistent deadline, creating it only for the audit entry.

        The deadline is an absolute Unix wall-clock timestamp so a new Python
        process cannot reset the 120-hour allowance.  ``create=False`` is used
        by every non-audit entry point and rejects legacy/incomplete output
        directories that have no trustworthy original start time.
        """

        output_dir = Path(output_dir).resolve()
        if (
            isinstance(wallclock_hours, bool)
            or not isinstance(wallclock_hours, (int, float))
            or not math.isfinite(float(wallclock_hours))
            or wallclock_hours <= 0
        ):
            raise ValueError("wallclock_hours must be a finite positive number")
        if not isinstance(config_digest, str) or not config_digest:
            raise ValueError("config_digest must be non-empty")
        path = output_dir / BUDGET_FILENAME
        requested_seconds = float(wallclock_hours) * 3600.0
        if path.is_file():
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                raise RuntimeError(f"invalid persistent wallclock budget: {path}") from error
            required = {
                "format_version", "config_digest", "started_unix",
                "started_utc", "deadline_unix", "deadline_utc",
                "budget_seconds", "wallclock_hours", "clock",
                "enforcement", "hard_interrupts",
            }
            if set(payload) != required:
                raise RuntimeError(f"unexpected persistent wallclock budget schema: {path}")
            if payload["format_version"] != BUDGET_FORMAT_VERSION:
                raise RuntimeError(f"unsupported persistent wallclock budget version: {path}")
            if payload["config_digest"] != config_digest:
                raise RuntimeError("persistent wallclock budget belongs to another config")
            if not math.isclose(
                float(payload["budget_seconds"]), requested_seconds,
                rel_tol=0.0, abs_tol=1e-6,
            ):
                raise RuntimeError("persistent wallclock duration differs from this config")
            if payload["clock"] != "unix_wall_clock":
                raise RuntimeError("persistent wallclock budget uses an unknown clock")
            if payload["enforcement"] != "cooperative_boundaries_only":
                raise RuntimeError("persistent wallclock budget has an unknown enforcement mode")
            if payload["hard_interrupts"] is not False:
                raise RuntimeError("persistent budget must not claim hard interruption")
            started = float(payload["started_unix"])
            deadline = float(payload["deadline_unix"])
            if not all(math.isfinite(value) for value in (started, deadline)):
                raise RuntimeError("persistent wallclock timestamps must be finite")
            if not math.isclose(
                deadline - started, requested_seconds,
                rel_tol=0.0, abs_tol=1e-6,
            ):
                raise RuntimeError("persistent wallclock deadline is internally inconsistent")
            if payload["started_utc"] != _utc_timestamp(started) or (
                payload["deadline_utc"] != _utc_timestamp(deadline)
            ):
                raise RuntimeError("persistent wallclock ISO timestamps are inconsistent")
            return cls(output_dir, config_digest, started, deadline, requested_seconds)

        if not create:
            raise FileNotFoundError(
                f"missing {BUDGET_FILENAME}; only the first audit may create the deadline"
            )
        observed = time.time() if now is None else float(now)
        if not math.isfinite(observed):
            raise ValueError("wallclock start must be finite")
        deadline = observed + requested_seconds
        payload = {
            "format_version": BUDGET_FORMAT_VERSION,
            "config_digest": config_digest,
            "started_unix": observed,
            "started_utc": _utc_timestamp(observed),
            "deadline_unix": deadline,
            "deadline_utc": _utc_timestamp(deadline),
            "budget_seconds": requested_seconds,
            "wallclock_hours": float(wallclock_hours),
            "clock": "unix_wall_clock",
            "enforcement": "cooperative_boundaries_only",
            "hard_interrupts": False,
        }
        atomic_json(path, payload)
        return cls(output_dir, config_digest, observed, deadline, requested_seconds)

    def remaining_seconds(self, *, now: float | None = None) -> float:
        observed = time.time() if now is None else float(now)
        if not math.isfinite(observed):
            raise ValueError("wallclock observation must be finite")
        return self.deadline_unix - observed

    def check(
        self,
        *,
        stage: str,
        boundary: str,
        variant: str | None = None,
        sample_id: str | None = None,
        now: float | None = None,
    ) -> None:
        """Raise at a cooperative boundary once the absolute deadline passes."""

        observed = time.time() if now is None else float(now)
        if not math.isfinite(observed):
            raise ValueError("wallclock observation must be finite")
        if observed < self.deadline_unix:
            return
        event = {
            "format_version": BUDGET_FORMAT_VERSION,
            "status": "exhausted",
            "stage": str(stage),
            "boundary": str(boundary),
            "variant": None if variant is None else str(variant),
            "sample_id": None if sample_id is None else str(sample_id),
            "observed_unix": observed,
            "observed_utc": _utc_timestamp(observed),
            "deadline_unix": self.deadline_unix,
            "deadline_utc": _utc_timestamp(self.deadline_unix),
            "seconds_over_budget": observed - self.deadline_unix,
            "enforcement": "cooperative_boundaries_only",
            "in_flight_python_calls_are_not_hard_interrupted": True,
            "complete_marker_must_not_be_written": True,
        }
        atomic_json(self.exhausted_path, event)
        location = f"{stage}/{boundary}"
        if variant is not None:
            location += f"/{variant}"
        if sample_id is not None:
            location += f"/{sample_id}"
        raise WallclockBudgetExhausted(
            f"persistent wallclock budget exhausted at cooperative boundary {location}"
        )


def validate_completed_stage(
    progress: StageProgress,
    *,
    expected_required: tuple[str, ...] | None = None,
    required_contains: Iterable[str] = (),
) -> bool:
    """Strictly validate a present completion marker before resume skips work."""

    marker = progress.root / ".complete"
    if not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid completion marker: {marker}") from error
    required = payload.get("required")
    if (
        not isinstance(required, list)
        or not required
        or any(not isinstance(item, str) or not item for item in required)
        or len(required) != len(set(required))
    ):
        raise RuntimeError(f"invalid required-artifact list in completion marker: {marker}")
    required_tuple = tuple(required)
    if expected_required is not None and required_tuple != expected_required:
        raise RuntimeError(
            f"completion marker required artifacts differ for {progress.root.name}: "
            f"{required_tuple} != {expected_required}"
        )
    missing_declared = set(required_contains) - set(required_tuple)
    if missing_declared:
        raise RuntimeError(
            f"completion marker for {progress.root.name} omits required declarations: "
            f"{sorted(missing_declared)}"
        )
    if not progress.is_complete(required_tuple):
        raise RuntimeError(
            f"completion marker identity or required artifacts are invalid: {marker}"
        )
    return True
