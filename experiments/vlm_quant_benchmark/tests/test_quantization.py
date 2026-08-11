import math

import pytest
import torch

from experiments.vlm_quant_benchmark.metrics import TokenNLL
from experiments.vlm_quant_benchmark.quantization import (
    dynamic_activation_qdq,
    quantize_weight_groupwise,
)


@pytest.mark.parametrize("bits,qmax", [(4, 15), (8, 255)])
def test_asymmetric_ranges_tail_and_determinism(bits, qmax):
    weight = torch.linspace(-3, 2, 2 * 131).reshape(2, 131)
    first = quantize_weight_groupwise(weight, bits=bits, group_size=128, zero_point=True)
    second = quantize_weight_groupwise(weight, bits=bits, group_size=128, zero_point=True)
    assert first.scales.shape == (2, 2)
    assert int(first.qvalues.min()) >= 0
    assert int(first.qvalues.max()) <= qmax
    torch.testing.assert_close(first.dequantized, second.dequantized, rtol=0, atol=0)


@pytest.mark.parametrize("bits,qmax", [(4, 7), (8, 127)])
def test_symmetric_ranges(bits, qmax):
    weight = torch.randn(3, 129)
    result = quantize_weight_groupwise(weight, bits=bits, group_size=128, zero_point=False)
    assert int(result.qvalues.min()) >= -qmax
    assert int(result.qvalues.max()) <= qmax


def test_zero_constant_and_nonfinite():
    zero = quantize_weight_groupwise(torch.zeros(2, 7), bits=4, group_size=4)
    assert torch.isfinite(zero.dequantized).all()
    assert zero.mse == 0
    constant = quantize_weight_groupwise(torch.full((2, 7), 3.25), bits=8, group_size=4)
    assert torch.isfinite(constant.dequantized).all()
    with pytest.raises(ValueError):
        quantize_weight_groupwise(torch.tensor([[float("nan")]]), bits=4)


def test_dynamic_activation_zero_and_range():
    zero = dynamic_activation_qdq(torch.zeros(2, 3, 5), 8)
    assert torch.equal(zero, torch.zeros_like(zero))
    value = dynamic_activation_qdq(torch.randn(2, 3, 5), 4)
    assert torch.isfinite(value).all()


def test_token_nll_uses_causal_shift_and_mask():
    torch.manual_seed(0)
    logits = torch.randn(1, 5, 11)
    labels = torch.tensor([[-100, -100, 3, 4, 5]])
    metric = TokenNLL()
    metric.update(logits, labels)
    expected = torch.nn.functional.cross_entropy(logits[:, 1:4].reshape(-1, 11),
                                                  torch.tensor([3, 4, 5]), reduction="sum")
    assert metric.tokens == 3
    assert math.isclose(metric.total_nll, float(expected), rel_tol=1e-6)
    assert math.isclose(metric.perplexity, math.exp(float(expected) / 3), rel_tol=1e-6)
