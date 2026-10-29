import json
from types import SimpleNamespace

import pytest
import torch

from experiments.ffn_quant_causal.budget import WallclockBudget, WallclockBudgetExhausted
from experiments.ffn_quant_compression import reuse
from experiments.ffn_quant_compression.config import Config
from experiments.ffn_quant_compression.evidence import Progress
from experiments.ffn_quant_compression.reuse import History
from experiments.ffn_quant_compression.sweep import InferenceBudget, complete_rows
from experiments.ffn_quant_compression.tests.test_design_statistics import design
from experiments.ffn_quant_rootcause.evidence import EvidenceProgress
from experiments.ffn_quant_rootcause.interventions import Intervention
from experiments.infinitevl_quant.progress import file_sha256, safe_id


def source_fixture(tmp_path, monkeypatch):
    config = Config(output_dir=tmp_path / "new", rootcause_output=tmp_path / "old")
    data = design()
    audit = {"models": {"qwen": {"content_sha256": "model"}}, "rootcause_audit_sha256": "old-audit"}
    progress = Progress(config, "sweep", audit, data)
    old = EvidenceProgress(config.rootcause_output, "confirm", config_digest="old", inputs={"audit": "old"})
    ref = config.rootcause_output / "references/confirm/qwen/confirm192" / f"{safe_id('id')}.pt"
    ref.parent.mkdir(parents=True)
    torch.save({"audit_digest": "old", "id": "id", "input_digest": "inputs", "logits": torch.zeros(1, 4)}, ref)
    source = {
        "dataset": "test",
        "model_kind": "qwen",
        "split": "confirm192",
        "input_digest": "inputs",
        "intervention": Intervention(bits=None).payload(),
        "reference_sha256": file_sha256(ref),
        "score": 1.0,
        "prediction": "A",
        "target_text": {"tokens": 1, "total_nll": 1.0, "nll": 1.0},
        "elapsed_seconds": 4.0,
        "teacher_seconds": 1.0,
        "generation_seconds": 3.0,
    }
    old.commit("qwen__confirm192__B", "id", {}, source)
    old.finish({"complete": True})
    sample = SimpleNamespace(id="id", dataset="test")
    engine = SimpleNamespace(
        bundle=SimpleNamespace(kind="qwen"),
        progress=progress,
        reference_path=lambda split, sample: config.output_dir / "refs" / "reference.pt",
    )
    monkeypatch.setattr(reuse, "score_prediction", lambda *args: SimpleNamespace(score=1.0))
    return config, data, audit, progress, engine, sample, old


def test_verified_import_rebinds_references_and_preserves_source(tmp_path, monkeypatch):
    config, _, audit, progress, engine, sample, old = source_fixture(tmp_path, monkeypatch)
    path = old.sample_path("qwen__confirm192__B", "id")
    digest = file_sha256(path)
    history = History(config, audit)
    assert (
        history.try_import(
            engine,
            sample,
            (None,) * 4 + ("wrong",),
            "confirm192",
            "B",
            Intervention(bits=None),
            generate=True,
            permitted=True,
        )
        is None
    )
    assert (
        history.try_import(
            engine,
            sample,
            (None,) * 4 + ("inputs",),
            "confirm192",
            "B",
            Intervention(bits=None),
            generate=True,
            permitted=False,
        )
        is None
    )
    row = history.try_import(
        engine,
        sample,
        (None,) * 4 + ("inputs",),
        "confirm192",
        "B",
        Intervention(bits=None),
        generate=True,
        permitted=True,
    )
    assert row["historical_reuse"]["source_file_sha256"] == digest
    assert row["compression"]["q"] == 0 and row["elapsed_seconds"] == 0
    assert file_sha256(path) == digest
    reference = torch.load(engine.reference_path(None, None), weights_only=True)
    assert reference["audit_digest"] == progress.inputs["audit"]
    assert reference["input_digest"] == "inputs"
    assert file_sha256(engine.reference_path(None, None)) == row["reference_sha256"]
    unit = {"spec": Intervention(bits=None).payload(), "generate": True, "input_digest": "inputs"}
    assert progress.completed("qwen__confirm192__B", "id", unit) == row
    with pytest.raises(RuntimeError, match="different intervention"):
        progress.completed("qwen__confirm192__B", "id", {**unit, "generate": False})
    path.write_text("{}")
    with pytest.raises(RuntimeError, match="changed"):
        history.try_import(
            engine,
            sample,
            (None,) * 4 + ("inputs",),
            "confirm192",
            "B",
            Intervention(bits=None),
            generate=True,
            permitted=True,
        )


def test_stage_completion_rejects_corruption(tmp_path):
    config = Config(output_dir=tmp_path)
    progress = Progress(config, "sweep", {"models": {"qwen": {"content_sha256": "model"}}}, design())
    progress.commit("qwen__confirm192__B", "id", {}, {"dataset": "d"})
    progress.finish({"done": True})
    assert progress.verified_complete()
    path = progress.sample_path("qwen__confirm192__B", "id")
    row = json.loads(path.read_text())
    row["compression"]["q"] = 1
    path.write_text(json.dumps(row))
    with pytest.raises(RuntimeError, match="checksum"):
        progress.verified_complete()


def test_budget_resume_keeps_deadline_and_reserve_does_not_create_second_round(tmp_path):
    config = Config(output_dir=tmp_path)
    first = WallclockBudget.open_or_create(
        tmp_path / "budget", wallclock_hours=120, config_digest=config.digest(), create=True
    )
    again = WallclockBudget.open_or_create(
        tmp_path / "budget", wallclock_hours=120, config_digest=config.digest(), create=False
    )
    assert first.started_unix == again.started_unix and first.deadline_unix == again.deadline_unix
    guarded = InferenceBudget(first, 120 * 3600)
    with pytest.raises(WallclockBudgetExhausted, match="reserve"):
        guarded.check(stage="sweep", boundary="test")
    assert first.remaining_seconds() > 0


def test_incomplete_conditions_do_not_enter_curves(tmp_path):
    config = Config(output_dir=tmp_path)
    progress = Progress(config, "sweep", {"models": {"qwen": {"content_sha256": "model"}}}, design())
    progress.commit("qwen__confirm192__B", "id", {}, {"dataset": "d"})
    samples = [SimpleNamespace(id="id"), SimpleNamespace(id="missing")]
    assert complete_rows(progress, "qwen__confirm192__B", samples, required=False) is None
    with pytest.raises(RuntimeError, match="incomplete"):
        complete_rows(progress, "qwen__confirm192__B", samples)
