import pytest
import torch
import torch.nn.functional as F
from torch import nn

from experiments.ffn_quant_rootcause.interventions import QuantBank
from experiments.ffn_quant_rootcause.numerics import (
    decompose_ffn,
    geometry,
    matching_noise,
    scalar_quantize,
    token_groups,
)
from experiments.infinitevl_quant.quantization import quantize_weight


def test_ratio_identity_zero_denominators_and_energy_pooling():
    r = torch.tensor([[[3.0, 4.0], [0.0, 0.0]]])
    f = torch.tensor([[[2.0, -1.0], [1.0, 1.0]]])
    error = torch.tensor([[[0.2, 0.1], [0.0, 0.0]]])
    result = geometry(r, f, error)
    assert result["eta_r"] == pytest.approx(result["eta_f"] * result["rho"], rel=1e-13)
    assert result["error_ffn_nmse"] == pytest.approx(result["eta_f"] ** 2)
    assert result["token_ratios"]["eta_r"]["undefined_tokens"] == 1
    assert result["sum_squared"]["residual"] == 25
    zero = geometry(torch.zeros_like(r), f, error)
    assert zero["rho"] is None and zero["eta_r"] is None
    assert zero["identity_absolute_error"] is None
    assert geometry(r, f, error, torch.zeros(1, 2, dtype=torch.bool))["status"] == "empty_group"
    with pytest.raises(FloatingPointError):
        geometry(r, f, error * float("nan"))


def test_token_masks_align_prediction_rows_exclude_padding_and_allow_overlap():
    teacher = {
        "input_ids": torch.tensor([[8, 99, 99, 7, 3, 4, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 0]]),
    }
    masks = token_groups(teacher, 4, [4, 5], visual_ids={99})
    assert masks["visual"].nonzero()[:, 1].tolist() == [1, 2]
    assert masks["text_prompt"].nonzero()[:, 1].tolist() == [0, 3]
    assert masks["answer_prediction"].nonzero()[:, 1].tolist() == [3, 4]
    assert masks["all_valid"].sum() == 6
    with pytest.raises(ValueError):
        token_groups(teacher, 4, [6], visual_ids={99})


@pytest.mark.parametrize("bits", [3, 4, 8])
@pytest.mark.parametrize("group_size", [3, 128])
def test_independent_scalar_quantizer_matches_affine_edge_cases(bits, group_size):
    torch.manual_seed(8)
    data = torch.cat(
        [
            torch.randn(2, 133),
            torch.ones(1, 133) * 0.25,
            torch.linspace(0.1, 3, 133)[None],
            -torch.linspace(0.1, 3, 133)[None],
        ]
    )
    expected, records = scalar_quantize(data, bits=bits, group_size=group_size)
    actual = quantize_weight(data, bits=bits, group_size=group_size, zero_point=True).dequantized
    assert torch.equal(expected, actual)
    assert any(r["constant"] for r in records)


@pytest.mark.parametrize("direction", ["real", "permuted", "gaussian"])
def test_noise_matches_each_token_energy_including_zero_error(direction):
    error = torch.randn(2, 7, 19)
    error[0, 0] = 0
    actual = matching_noise(error, direction=direction, seed=42)
    assert torch.allclose(actual.double().square().sum(-1), error.double().square().sum(-1), atol=1e-6, rtol=1e-6)
    assert torch.equal(actual, matching_noise(error, direction=direction, seed=42))
    if direction != "real":
        assert not torch.equal(actual, error)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(8, 12)
        self.up_proj = nn.Linear(8, 12)
        self.down_proj = nn.Linear(12, 8)
        self.act_fn = F.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def test_exact_swiglu_decomposition_retains_cross_terms_and_all_combinations():
    torch.manual_seed(7)
    mlp = MLP()
    layer = nn.Module()
    layer.mlp = mlp
    bank = QuantBank([layer])
    result = decompose_ffn(mlp, torch.randn(2, 5, 8), bank, 0, 3, chunk_tokens=3)
    terms = result["terms"]
    assert len(result["combinations_gate_up_down"]) == 8
    assert result["combinations_gate_up_down"]["000"]["nmse"] == 0
    reconstructed = sum(
        terms[k]
        for k in ("gate", "up", "interaction", "gate_up_cross", "gate_interaction_cross", "up_interaction_cross")
    )
    assert reconstructed == pytest.approx(terms["product_error"], abs=1e-7, rel=2e-6)
    assert terms["reconstruction_error"] < 1e-12
    assert terms["down_reconstruction_error"] < 1e-11
    assert 0 <= result["top_1pct_energy_fraction"] <= 1
    assert len(result["channel_error_energy"]) == 12
