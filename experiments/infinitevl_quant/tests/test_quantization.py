from __future__ import annotations

import pytest
import torch
from torch import nn

from experiments.infinitevl_quant.config import SCHEMES
from experiments.infinitevl_quant.inventory import LinearTarget
from experiments.infinitevl_quant.quantization import (
    FakeQuantLinear,
    activation_qdq,
    fp16_rounded_parameters,
    quantize_weight,
    quantized_linears,
)


@pytest.mark.parametrize("bits", (2, 3, 4, 8))
def test_quantizer_tail_group_and_finite(bits):
    value = torch.linspace(-2, 3, 5 * 131).reshape(5, 131)
    result = quantize_weight(value, bits=bits, group_size=128, zero_point=True)
    assert result.dequantized.shape == value.shape
    assert result.groups == 2
    assert torch.isfinite(result.dequantized).all()


@pytest.mark.parametrize("bits", (2, 3, 4, 8))
def test_tail_group_does_not_include_implicit_zero_padding(bits):
    value = torch.linspace(-3, 3, 2 * 131).reshape(2, 131)
    full = quantize_weight(value, bits=bits, group_size=128, zero_point=True)
    tail = quantize_weight(value[:, 128:], bits=bits, group_size=128, zero_point=True)
    assert torch.equal(full.dequantized[:, 128:], tail.dequantized)


@pytest.mark.parametrize("value", (torch.zeros(3, 17), torch.full((3, 17), 2.5)))
def test_quantizer_zero_and_constant_weights(value):
    result = quantize_weight(value, bits=4, group_size=128, zero_point=True)
    assert torch.isfinite(result.dequantized).all()
    assert torch.equal(result.dequantized, value)


def test_quantizer_rejects_nonfinite():
    value = torch.zeros(2, 3)
    value[0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        quantize_weight(value, bits=4, group_size=128, zero_point=True)
    with pytest.raises(ValueError, match="NaN or Inf"):
        activation_qdq(value)


def test_quantization_context_exactly_restores_module_and_type():
    model = nn.Sequential(nn.Linear(7, 5), nn.ReLU(), nn.Linear(5, 3))
    target = LinearTarget("0", model[0], "decoder_ffn", 0)
    original = model[0]
    weight = original.weight.detach().clone()
    with quantized_linears(model, [target], SCHEMES["w4a8"]):
        assert isinstance(model[0], FakeQuantLinear)
        assert model[0].activation_bits == 8
        assert model[2].__class__ is nn.Linear
        assert torch.isfinite(model(torch.randn(2, 7))).all()
    assert model[0] is original
    assert type(model[0]) is nn.Linear
    assert torch.equal(model[0].weight, weight)


def test_w8a16_quantization_has_no_activation_qdq():
    model = nn.Sequential(nn.Linear(7, 5))
    target = LinearTarget("0", model[0], "swa_attention", 0)
    with quantized_linears(model, [target], SCHEMES["w8a16"]) as audit:
        assert isinstance(model[0], FakeQuantLinear)
        assert model[0].activation_bits is None
        assert audit["scheme"]["weight_bits"] == 8
        assert audit["modules"]["0"]["group_size"] == 128


def test_scaled_quantization_and_fp16_rounding_restore_exactly():
    model = nn.Sequential(nn.Linear(11, 6, bias=False))
    target = LinearTarget("0", model[0], "gdn_main", 0)
    original_parameter = model[0].weight.data
    scale = torch.linspace(0.5, 1.5, 11)
    with quantized_linears(
        model, [target], SCHEMES["w3a16"], scales={"0": scale},
    ):
        assert model[0].input_scale.shape == (11,)
    with fp16_rounded_parameters(model):
        assert model[0].weight.dtype == torch.float32
    assert model[0].weight.data.data_ptr() == original_parameter.data_ptr()
