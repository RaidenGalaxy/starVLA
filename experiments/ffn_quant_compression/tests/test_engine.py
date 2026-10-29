from types import SimpleNamespace

import torch

from experiments.ffn_quant_causal.budget import WallclockBudget
from experiments.ffn_quant_compression.config import Config
from experiments.ffn_quant_compression.evidence import Progress
from experiments.ffn_quant_compression.reuse import History
from experiments.ffn_quant_compression.sweep import evaluate_condition
from experiments.ffn_quant_compression.tests.test_design_statistics import design
from experiments.ffn_quant_rootcause import evaluation
from experiments.ffn_quant_rootcause.evaluation import Engine
from experiments.ffn_quant_rootcause.tests.test_execution import LanguageModel


def test_engine_mask_evaluation_and_resume_has_bound_compression_metadata(tmp_path, monkeypatch):
    data = design()
    payload = {"models": {"qwen": {"content_sha256": "toy"}}}
    config = Config(output_dir=tmp_path)
    progress = Progress(config, "sweep", payload, data)
    model = LanguageModel()
    original = model.model.layers[0].mlp
    bundle = SimpleNamespace(kind="qwen", model=model, processor=SimpleNamespace(batch_decode=lambda *a, **k: ["A"]))
    prompt = {"input_ids": torch.tensor([[2, 3]])}
    teacher = {"input_ids": torch.tensor([[2, 3, 5, 6]]), "attention_mask": torch.ones(1, 4, dtype=torch.long)}
    labels = torch.tensor([[-100, -100, 5, 6]])
    monkeypatch.setattr(evaluation, "_teacher_inputs", lambda *a: (prompt, teacher, labels, [2, 3]))
    monkeypatch.setattr(
        evaluation, "score_prediction", lambda *a: SimpleNamespace(score=1.0, rule="test", extracted="A")
    )
    budget = WallclockBudget.open_or_create(
        tmp_path / "budget", wallclock_hours=120, config_digest=config.digest(), create=True
    )
    engine = Engine(bundle, config, progress, budget)
    sample = SimpleNamespace(id="id", dataset="dataset")
    history = History(config, {})
    record = data["models"]["qwen"]["conditions"]["B"]
    baseline = evaluate_condition(engine, [sample], "confirm192", "B", record, history, False, True)
    assert baseline[0]["compression"]["q"] == 0
    label = data["models"]["qwen"]["aliases"]["sensitive_first"]["1"]
    record = data["models"]["qwen"]["conditions"][label]
    candidate = evaluate_condition(engine, [sample], "confirm192", label, record, history, False, True)
    assert candidate[0]["compression"]["quantized_layers"] == [0]
    assert candidate[0]["model_content_sha256"] == "toy"
    assert model.model.layers[0].mlp is original
    calls = model.calls
    assert evaluate_condition(engine, [sample], "confirm192", label, record, history, False, True) == candidate
    assert model.calls == calls
