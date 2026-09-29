from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest
from torch import nn

from experiments.infinitevl_quant.cpu_backend import (
    CPUFusedRMSNormGated,
    CPUShortConvolution,
    capture_gdn_states,
    chunk_gated_delta_rule,
    qig_recurrent_reference,
    recurrent_gated_delta_rule,
)
from experiments.infinitevl_quant.loader import sliding_cache_mask_sizes


def _inputs(dtype=torch.float32):
    generator = torch.Generator().manual_seed(3)
    q = torch.randn(2, 13, 3, 5, generator=generator).to(dtype)
    k = torch.randn(2, 13, 3, 5, generator=generator).to(dtype)
    v = torch.randn(2, 13, 3, 7, generator=generator).to(dtype)
    g = -torch.rand(2, 13, 3, generator=generator).to(dtype)
    beta = torch.rand(2, 13, 3, generator=generator).to(dtype)
    return q, k, v, g, beta


def test_recurrent_chunk_parity_and_fp32_state():
    values = _inputs(torch.float16)
    recurrent, left_state = recurrent_gated_delta_rule(*values, output_final_state=True)
    chunk, right_state = chunk_gated_delta_rule(
        *values, output_final_state=True, chunk_size=4,
    )
    assert recurrent.dtype == torch.float32
    assert left_state.dtype == torch.float32
    assert torch.max(torch.abs(recurrent - chunk)).item() <= 1e-4
    assert torch.max(torch.abs(left_state - right_state)).item() <= 1e-4


def test_recurrent_applies_fla_default_query_scale_after_l2_norm():
    q = torch.tensor([[[[3.0, 4.0]]]])
    k = torch.tensor([[[[0.0, 2.0]]]])
    v = torch.tensor([[[[6.0, -2.0, 1.0]]]])
    g = torch.zeros(1, 1, 1)
    beta = torch.ones(1, 1, 1)
    output, _ = recurrent_gated_delta_rule(q, k, v, g, beta)
    expected_dot = (4.0 / 5.0) * (2.0 ** -0.5)
    assert torch.allclose(output[0, 0, 0], v[0, 0, 0] * expected_dot)


def test_recurrent_is_differentiable():
    values = list(_inputs())
    values[0].requires_grad_(True)
    output, _ = recurrent_gated_delta_rule(*values)
    output.square().mean().backward()
    assert values[0].grad is not None
    assert torch.isfinite(values[0].grad).all()


def test_qig_context_forces_differentiable_recurrent_dispatch():
    values = list(_inputs())
    values[0].requires_grad_(True)
    with qig_recurrent_reference():
        output, _ = chunk_gated_delta_rule(*values, chunk_size=4)
    output.square().mean().backward()
    assert values[0].grad is not None
    assert torch.isfinite(values[0].grad).all()


@pytest.mark.parametrize(
    ("cumulative", "window", "positions", "expected"),
    (
        (5, 16, [5], (6, 0)),       # before update
        (6, 16, [5], (6, 0)),       # after update
        (5, 4, [5], (4, 2)),        # full window before update
        (6, 4, [5], (4, 2)),        # full window after update
        (0, 32, list(range(65)), (65, 0)),
        (65, 32, list(range(65)), (65, 0)),
    ),
)
def test_sliding_cache_mask_sizes_are_phase_aware(
    cumulative, window, positions, expected,
):
    assert sliding_cache_mask_sizes(
        cumulative, window, torch.tensor(positions),
    ) == expected


def test_short_convolution_cached_parity():
    generator = torch.Generator().manual_seed(5)
    module = CPUShortConvolution(9, 4, activation="silu")
    value = torch.randn(2, 21, 9, generator=generator)
    expected, _ = module(value)
    chunks, state = [], None
    for start, stop in ((0, 3), (3, 11), (11, 12), (12, 21)):
        output, state = module(
            value[:, start:stop], cache=state, output_final_state=True,
        )
        chunks.append(output)
    assert state.shape == (2, 9, 4)
    assert module.weight.shape == (9, 1, 4)
    assert torch.max(torch.abs(expected - torch.cat(chunks, dim=1))).item() <= 1e-5


def test_short_convolution_cache_stays_fp32():
    module = CPUShortConvolution(5, 3).half()
    _output, state = module(
        torch.randn(1, 4, 5, dtype=torch.float16), output_final_state=True,
    )
    assert state.dtype == torch.float32


def test_fused_rmsnorm_gate_formula():
    generator = torch.Generator().manual_seed(7)
    module = CPUFusedRMSNormGated(17, eps=1e-6)
    value = torch.randn(2, 4, 3, 17, generator=generator)
    gate = torch.randn(2, 4, 3, 17, generator=generator)
    actual = module(value, gate)
    expected = (
        value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + 1e-6)
        * module.weight * F.silu(gate)
    )
    assert torch.max(torch.abs(actual - expected)).item() <= 1e-6


def test_state_probe_records_global_prefixes_through_chunk_wrapper():
    class GatedDeltaNet(nn.Module):
        def forward(self, values):
            return chunk_gated_delta_rule(*values, chunk_size=3)[0]

    model = nn.Sequential(GatedDeltaNet())
    values = _inputs()
    with capture_gdn_states(model, (2, 7, 13)) as states:
        model(values)
    assert set(states["0"]) == {2, 7, 13}
    assert all(value.dtype == torch.float32 for value in states["0"].values())
