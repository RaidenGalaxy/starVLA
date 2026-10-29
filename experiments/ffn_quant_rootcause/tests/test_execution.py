import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.ffn_quant_causal.budget import WallclockBudget
from experiments.ffn_quant_mechanism.tests.test_shadow import ToyLayer
from experiments.ffn_quant_rootcause import evaluation, stages
from experiments.ffn_quant_rootcause.config import Config
from experiments.ffn_quant_rootcause.evaluation import Engine
from experiments.ffn_quant_rootcause.evidence import EvidenceProgress
from experiments.ffn_quant_rootcause.interventions import Intervention
from experiments.ffn_quant_rootcause.run_cpu_protocol import entry_budget


class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([ToyLayer()])
        self.calls = 0

    def forward(self, input_ids, **kwargs):
        self.calls += 1
        values = torch.nn.functional.one_hot(input_ids, 128).float()
        return SimpleNamespace(logits=self.model.layers[0](values))

    def generate(self, input_ids, **kwargs):
        return torch.cat((input_ids, torch.tensor([[5]])), -1)


def test_engine_teacher_generation_and_resume_reject_wrong_mode(tmp_path, monkeypatch):
    model = LanguageModel()
    processor = SimpleNamespace(batch_decode=lambda *a, **kw: ["A"])
    bundle = SimpleNamespace(kind="qwen", model=model, processor=processor)
    config = Config(output_dir=tmp_path)
    progress = EvidenceProgress(tmp_path, "confirm", config_digest=config.digest(), inputs={"audit": "test"})
    budget = WallclockBudget.open_or_create(
        tmp_path / "budget", wallclock_hours=1, config_digest=config.digest(), create=True
    )
    prompt = {"input_ids": torch.tensor([[2, 3]])}
    teacher = {"input_ids": torch.tensor([[2, 3, 5, 6]]), "attention_mask": torch.ones(1, 4, dtype=torch.long)}
    labels = torch.tensor([[-100, -100, 5, 6]])
    monkeypatch.setattr(evaluation, "_teacher_inputs", lambda *a: (prompt, teacher, labels, [2, 3]))
    monkeypatch.setattr(
        evaluation, "score_prediction", lambda *a: SimpleNamespace(score=1.0, rule="test", extracted="A")
    )
    sample = SimpleNamespace(id="id", dataset="dataset")
    engine = Engine(bundle, config, progress, budget)
    inputs = engine.inputs(sample)
    baseline = engine.evaluate(sample, inputs, "confirm192", "B", Intervention(bits=None), generate=True)
    assert baseline["score"] == 1
    candidate = engine.evaluate(sample, inputs, "confirm192", "W", Intervention(), generate=True)
    assert candidate["target_text"]["tokens"] == 2
    assert "logits_kl" in candidate["comparison_to_bf16"]
    calls = model.calls
    assert engine.evaluate(sample, inputs, "confirm192", "W", Intervention(), generate=True) == candidate
    assert model.calls == calls
    with pytest.raises(RuntimeError, match="different intervention"):
        engine.evaluate(sample, inputs, "confirm192", "W", Intervention(), generate=False)
    assert engine.inputs(sample) is inputs


def test_missing_sample_invalidates_completed_stage(tmp_path):
    progress = EvidenceProgress(tmp_path, "ratio", config_digest="config", inputs={"audit": "test"})
    progress.commit("a", "id", {}, {"dataset": "dataset"})
    progress.finish({"complete": True})
    progress.sample_path("a", "id").unlink()
    with pytest.raises(RuntimeError, match="checksum"):
        progress.verified_complete()


def test_second_round_can_start_after_round1_deadline(tmp_path):
    config = Config(output_dir=tmp_path)
    p = EvidenceProgress(tmp_path, "dose", config_digest=config.digest(), inputs={"audit": "test"})
    p.finish({"complete": True})
    expired = WallclockBudget.open_or_create(
        tmp_path / "budgets/round1", wallclock_hours=120, config_digest=config.digest(), create=True, now=1.0
    )
    assert expired.remaining_seconds() < 0
    second = entry_budget(config, "all", expired)
    assert second.remaining_seconds() > 0
    again = entry_budget(config, "confirm", expired)
    assert again.deadline_unix == second.deadline_unix


def test_confirmation_lock_uses_screen_only_and_rejects_reselection(tmp_path, monkeypatch):
    config = Config(output_dir=tmp_path)
    payload = {"models": {kind: {"top3_layers": [1, 2, 3]} for kind in ("infinitevl", "qwen")}}
    for stage in ("ratio", "local", "dose", "propagation"):
        path = tmp_path / stage / "summary.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(payload))
    folder = tmp_path / "dose/infinitevl__screen32__W/samples"
    folder.mkdir(parents=True)
    (folder / "sample.json").write_text(json.dumps({"target_text": {"total_nll": 2.0, "tokens": 1}}))
    candidates = {
        kind: {
            "restore_top1": {"nll": 1.1, "spec": Intervention(layer_bits={1: None}).payload(), "stage": "local"},
            "dose_0.5": {"nll": 1.3, "spec": Intervention(gamma=0.5).payload(), "stage": "dose"},
            "dose_0.75": {"nll": 1.5, "spec": Intervention(gamma=0.75).payload(), "stage": "dose"},
        }
        for kind in ("infinitevl", "qwen")
    }
    monkeypatch.setattr(stages, "load_summary", lambda *a: payload)
    monkeypatch.setattr(stages, "screen_candidates", lambda *a: candidates)

    class Forbidden:
        def __iter__(self):
            raise AssertionError("selection accessed confirm samples")

    locked = stages.lock_confirmation(config, {}, {"screen": [], "confirm": Forbidden()})
    assert locked["selected"] == ["restore_top1", "dose_0.5"]
    assert stages.lock_confirmation(config, {}, {"screen": []}) == locked
    candidates["infinitevl"]["dose_0.75"]["nll"] = 0.5
    with pytest.raises(RuntimeError, match="lock changed"):
        stages.lock_confirmation(config, {}, {"screen": []})
