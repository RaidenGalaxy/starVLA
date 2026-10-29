from __future__ import annotations

import math

import torch
from torch import nn

from experiments.ffn_quant_causal.adapters import build_ffn_targets
from experiments.ffn_quant_causal.evaluation import aggregate_rows


class ToyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(4, 6, bias=False)
        self.up_proj = nn.Linear(4, 6, bias=False)
        self.down_proj = nn.Linear(6, 4, bias=False)


class ToyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = ToyMLP()


class ToyInner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([ToyLayer(), ToyLayer()])


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = ToyInner()


def test_build_ffn_targets_generic_decoder() -> None:
    targets, audit = build_ffn_targets(ToyModel(), strict=False)
    assert len(targets) == 6
    assert audit["decoder_prefix"] == "model.layers"
    assert [target.name.rsplit(".", 1)[-1] for target in targets[:3]] == [
        "gate_proj", "up_proj", "down_proj",
    ]


def test_aggregate_rows_keeps_sample_and_token_nll() -> None:
    rows = [
        {
            "id": "a", "dataset": "d", "score": 1.0,
            "target_text": {"tokens": 1, "total_nll": 1.0, "nll": 1.0},
            "comparison_to_bf16": {"logits_kl": 0.2, "target_argmax_consistency": 1.0},
            "generation_truncated": False, "teacher_seconds": 1.0,
            "generation_seconds": 2.0,
        },
        {
            "id": "b", "dataset": "d", "score": 0.0,
            "target_text": {"tokens": 3, "total_nll": 6.0, "nll": 2.0},
            "comparison_to_bf16": {"logits_kl": 0.4, "target_argmax_consistency": 0.0},
            "generation_truncated": True, "teacher_seconds": 1.5,
            "generation_seconds": 2.5,
        },
    ]
    result = aggregate_rows(rows)
    assert result["mean_score"] == 50.0
    assert math.isclose(result["target_nll"], 1.75)
    assert math.isclose(result["target_nll_sample_mean"], 1.5)
    assert math.isclose(result["mean_logits_kl"], 0.3)
    assert result["truncated_samples"] == 1

