from __future__ import annotations

import json
import sys

from experiments.ffn_quant_causal.budget import WallclockBudgetExhausted
from experiments.ffn_quant_generalization import run_cpu_protocol as driver
from experiments.ffn_quant_generalization.config import MODELS


def test_driver_keeps_analysis_after_optional_budget_stop(tmp_path, monkeypatch):
    output = tmp_path / "run"
    events = []
    monkeypatch.setattr(sys, "argv", ["run_cpu_protocol", "--output-dir", str(output), "--stage", "all", "--resume"])
    monkeypatch.setattr(driver.torch, "set_num_threads", lambda value: None)
    monkeypatch.setattr(driver.torch, "set_num_interop_threads", lambda value: None)
    monkeypatch.setattr(driver, "prepare", lambda config: {})
    monkeypatch.setattr(driver, "load_splits", lambda config: {})
    monkeypatch.setattr(driver, "audit_identity", lambda *args: {"test": True})
    monkeypatch.setattr(driver, "audit_live", lambda *args: events.append("audit") or {})
    monkeypatch.setattr(
        driver, "rank", lambda *args: events.append("rank") or {k: list(range(v[0])) for k, v in MODELS.items()}
    )
    monkeypatch.setattr(driver, "bridge", lambda *args: events.append("bridge"))
    monkeypatch.setattr(driver, "execute_matrix", lambda *args: events.append(args[-1][0]))
    monkeypatch.setattr(driver, "finalize_stage", lambda *args: None)
    monkeypatch.setattr(driver, "qa", lambda *args: events.append("qa"))

    def optional(*args):
        events.append("optional")
        raise WallclockBudgetExhausted("analysis reserve")

    monkeypatch.setattr(driver, "optional_old_sweep", optional)
    monkeypatch.setattr(driver, "verify_frozen_sources", lambda *args: None)
    monkeypatch.setattr(
        driver,
        "analyze",
        lambda *args: events.append("analyze") or {"core_complete": True, "verification": {"passed": True}},
    )

    def report(config, *args):
        events.append("report")
        (config.output_dir / "report").mkdir()

    def archive(config):
        events.append("archive")
        (config.output_dir / "report/sample_records.tar.gz").write_bytes(b"test")

    monkeypatch.setattr(driver, "report", report)
    monkeypatch.setattr(driver, "archive", archive)
    driver.main()
    assert events == [
        "audit",
        "rank",
        "bridge",
        "confirm",
        "random",
        "main",
        "qa",
        "optional",
        "analyze",
        "report",
        "archive",
    ]
    assert json.loads((output / "status.json").read_text())["status"] == "complete"
    assert (
        json.loads((output / "optional_omissions.json").read_text())["old_qwen7b_dense"]["reason"]
        == "analysis_reserve_reached"
    )
