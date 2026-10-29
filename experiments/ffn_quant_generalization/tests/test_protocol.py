from __future__ import annotations

import contextlib
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from experiments.ffn_quant_generalization import runtime
from experiments.ffn_quant_generalization.config import MODELS, POLICIES, Config
from experiments.ffn_quant_generalization.design import build_design, condition, counts
from experiments.ffn_quant_generalization.prepare import assert_disjoint_splits, select_groups
from experiments.ffn_quant_generalization.runtime import Budget, frozen_json, progress, rows
from experiments.ffn_quant_generalization.scoring import score_prediction
from experiments.ffn_quant_rootcause.interventions import Intervention, QuantBank, intervene


def orders():
    return {key: list(reversed(range(dims[0]))) for key, dims in MODELS.items()}


def test_full_layerwise_matrix_and_nested_masks():
    design = build_design(orders())
    total = 0
    for kind, (size, hidden, intermediate) in MODELS.items():
        model = design["models"][kind]
        assert len(model["main"]) == size * 2
        assert (len(model["confirm"]), len(model["qa"]), len(model["bridge"])) == (10, 5, 4)
        assert len(model["random"]) <= 9
        for policy in POLICIES:
            previous = set()
            for q in range(size + 1):
                row = model["conditions"][model["aliases"][policy][str(q)]]
                current = set(row["quantized_layers"])
                assert len(current) == q and len(row["mask_bits"]) == size
                assert previous <= current
                assert row["coverage_percent"] == pytest.approx(100 * q / size)
                assert row["bf16_saving_percent"] == pytest.approx(75 * q / size)
                assert row["mixed_weight_bits"] == 3 * hidden * intermediate * (16 * (size - q) + 4 * q)
                previous = current
        assert model["aliases"][POLICIES[0]]["0"] == model["aliases"][POLICIES[1]]["0"] == "B"
        assert model["aliases"][POLICIES[0]][str(size)] == model["aliases"][POLICIES[1]][str(size)] == "W"
        total += len(model["main"]) * 32
    assert total == 6400


@pytest.mark.parametrize("kind", list(MODELS))
def test_random_orders_and_precision_roundtrip(kind):
    model = build_design(orders())["models"][kind]
    size = MODELS[kind][0]
    for seed in (42, 43, 44):
        permutation = np.random.default_rng(seed).permutation(size).tolist()
        assert model["random_orders"][str(seed)] == permutation
        for q in counts(size)[1:4]:
            record = model["conditions"][model["aliases"][f"random_{seed}"][str(q)]]
            assert record["quantized_layers"] == sorted(permutation[:q])
            spec = Intervention.from_payload(json.loads(json.dumps(record["spec"])))
            assert set(spec.layer_bits) == set(permutation[:q])
            assert all(v == 4 for v in spec.layer_bits.values())


@pytest.mark.parametrize("layers", [[-1], [36], [1, 1]])
def test_bad_masks_are_rejected(layers):
    with pytest.raises(ValueError):
        condition("infinitevl", layers)


def test_selection_is_image_disjoint_and_does_not_exclude_unused_group_images():
    # Shared candidate images are allowed across source pools; only selected images are reserved.
    pools = {}
    for group in ("human", "machine", "gqa"):
        pools[group] = [
            {"id": f"{group}:{i}", "pixel_sha256": f"{'chart' if group != 'gqa' else 'gqa'}:{i}", "target": "1"}
            for i in range(500)
        ]
    chosen, _ = select_groups(pools, {"chart:0", "gqa:0"}, 20261026)
    fingerprints = []
    for group in chosen.values():
        for values in group.values():
            fingerprints += [r["pixel_sha256"] for r in values]
    assert len(fingerprints) == len(set(fingerprints)) == 168
    assert "chart:0" not in fingerprints
    again, _ = select_groups(pools, {"chart:0", "gqa:0"}, 20261026)
    assert chosen == again


def test_split_leakage_is_rejected():
    sample = SimpleNamespace(id="x")
    split = {"pilot8": [sample], "screen32": [sample], "confirm128": [], "qa32": []}
    with pytest.raises(RuntimeError, match="leakage"):
        assert_disjoint_splits(split, {"x": {"pixel_sha256": "same"}}, set())


@pytest.mark.parametrize(
    "dataset,target,prediction,expected",
    [
        ("ChartQA_TEST", "100", "105", 1),
        ("ChartQA_TEST", "100", "106", 0),
        ("ChartQA_TEST", "0", "0.0", 0),
        ("ChartQA_TEST", "0", "0.001", 0),
        ("ChartQA_TEST", "50%", "0.5", 1),
        ("ChartQA_TEST", "Blue", " blue ", 1),
        ("GQA_TESTDEV_BALANCED", "red", " Red ", 1),
        ("GQA_TESTDEV_BALANCED", "red", "It is red", 0),
    ],
)
def test_frozen_scoring(dataset, target, prediction, expected):
    assert score_prediction(SimpleNamespace(dataset=dataset, target=target), prediction) == expected


def test_budget_restarts_do_not_reset_preparation_start(tmp_path):
    cfg = Config(output_dir=tmp_path / "run")
    first = Budget(cfg)
    second = Budget(cfg)
    assert second.deadline_unix == first.deadline_unix
    started = json.loads((cfg.output_dir / "budget/started.json").read_text())["started_unix"]
    assert first.deadline_unix - started == 240 * 3600
    first.check(stage="test", boundary="first")
    assert json.loads((cfg.output_dir / "budget/round.json").read_text())["round"] == 1
    with pytest.raises(ValueError):
        replace(cfg, total_hours=480)


def test_evidence_rejects_mask_input_and_version_changes(tmp_path):
    cfg = Config(output_dir=tmp_path / "run")
    store = progress(cfg, "sweep", {"version": 1}, {"mask": "01"})
    unit = {"mask": "01", "input": "a"}
    store.commit("model__screen32__m1", "sample", unit, {"target_text": {"tokens": 1}, "dataset": "test"})
    assert store.completed("model__screen32__m1", "sample", unit)
    for changed in ({"mask": "10", "input": "a"}, {"mask": "01", "input": "b"}):
        with pytest.raises(RuntimeError):
            store.completed("model__screen32__m1", "sample", changed)
    with pytest.raises(RuntimeError):
        progress(cfg, "sweep", {"version": 2}, {"mask": "01"})


def test_partial_conditions_are_not_promoted(tmp_path):
    cfg = Config(output_dir=tmp_path / "run")
    store = progress(cfg, "sweep", {}, {})
    store.commit("model__screen32__m1", "a", {}, {"dataset": "test"})
    samples = [SimpleNamespace(id="a"), SimpleNamespace(id="b")]
    assert rows(store, "model", "screen32", "m1", samples, required=False) is None
    with pytest.raises(RuntimeError, match="incomplete"):
        rows(store, "model", "screen32", "m1", samples)


def test_frozen_identity_refuses_changed_payload(tmp_path):
    path = tmp_path / "identity.json"
    frozen_json(path, {"version": 1})
    with pytest.raises(RuntimeError, match="frozen identity"):
        frozen_json(path, {"version": 2})


def test_generation_reuses_teacher_and_rejects_changed_input(tmp_path, monkeypatch):
    cfg = Config(output_dir=tmp_path / "run")
    evidence = progress(cfg, "qa", {}, {})
    monkeypatch.setattr(runtime, "intervene", lambda *args, **kwargs: contextlib.nullcontext())
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return torch.tensor([[1, 2]])

    fake = SimpleNamespace(
        inputs=lambda sample: ({"input_ids": torch.tensor([[1]])}, None, None, None, "input"),
        bundle=SimpleNamespace(
            kind="qwen25vl_7b",
            model=SimpleNamespace(generate=generate),
            processor=SimpleNamespace(batch_decode=lambda *args, **kwargs: ["yes"]),
        ),
        progress=evidence,
        budget=SimpleNamespace(check=lambda **kwargs: None),
        layers=[],
        bank=None,
    )
    sample = SimpleNamespace(id="GQA:1", dataset="GQA_TESTDEV_BALANCED", target="yes")
    record = condition("qwen25vl_7b", [])
    teacher = {
        "id": sample.id,
        "input_digest": "input",
        "row_sha256": "teacher",
        "target_text": {"tokens": 1, "total_nll": 0.1},
    }
    row = runtime.Engine.generate(fake, sample, "B", record, teacher)
    assert row["score"] == 1 and row["target_text"] == teacher["target_text"]
    assert runtime.Engine.generate(fake, sample, "B", record, teacher) == row
    assert len(calls) == 1
    with pytest.raises(RuntimeError, match="teacher record"):
        runtime.Engine.generate(fake, sample, "B", record, {**teacher, "input_digest": "changed"})


class ToyFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 8, bias=False)
        self.up_proj = nn.Linear(4, 8, bias=False)
        self.down_proj = nn.Linear(8, 4, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class ToyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.post_attention_layernorm = nn.LayerNorm(4)
        self.mlp = ToyFFN()

    def forward(self, x):
        return x + self.mlp(self.post_attention_layernorm(x))


def test_intervention_restores_original_modules_and_weights_after_exception():
    torch.manual_seed(42)
    layers = nn.ModuleList([ToyLayer(), ToyLayer()])
    bank = QuantBank(list(layers), 128)
    original = [layer.mlp for layer in layers]
    weights = [p.detach().clone() for p in layers.parameters()]
    x = torch.randn(2, 4)
    baseline = layers[1](layers[0](x))
    with pytest.raises(RuntimeError, match="injected"):
        with intervene(list(layers), bank, Intervention(bits=4)):
            result = layers[1](layers[0](x))
            assert torch.isfinite(result).all()
            raise RuntimeError("injected")
    assert all(layer.mlp is ref for layer, ref in zip(layers, original, strict=True))
    assert all(torch.equal(p, ref) for p, ref in zip(layers.parameters(), weights, strict=True))
    torch.testing.assert_close(layers[1](layers[0](x)), baseline, rtol=0, atol=0)
