from __future__ import annotations

import sys
import types

import pytest
import torch
from torch import nn

from experiments.infinitevl_quant.cpu_backend import recurrent_gated_delta_rule

from ..config import MEDIATOR_GROUPS
from ..gdn import GDNInterventionContext, recurrent_gdn_intervention


def _mediators(steps: int = 4) -> dict[str, torch.Tensor]:
    torch.manual_seed(17)
    return {
        "q": torch.randn(1, steps, 2, 3),
        "k": torch.randn(1, steps, 2, 3),
        "v": torch.randn(1, steps, 2, 4),
        "g": -torch.rand(1, steps, 2),
        "beta": torch.sigmoid(torch.randn(1, steps, 2)),
    }


def test_own_state_clamp_is_identity_and_records_final_separately() -> None:
    reference = _mediators()
    expected, _ = recurrent_gated_delta_rule(**reference)
    output, _state, audit = recurrent_gdn_intervention(
        **reference,
        reference=reference,
        carry_clamp=True,
        state_prefixes=(2, 4),
    )
    assert torch.allclose(output, expected, atol=1e-6, rtol=1e-6)
    assert set(audit["states"]) == {"2", "4", "final"}
    assert all(row["carried"]["normalized_frobenius"] == 0.0 for row in audit["states"].values())


def test_carry_clamp_preserves_current_perturbation_but_blocks_history() -> None:
    reference = _mediators()
    candidate = {key: value.clone() for key, value in reference.items()}
    candidate["v"][:, 0] += 1.0
    baseline, _ = recurrent_gated_delta_rule(**reference)
    raw, _state, _audit = recurrent_gdn_intervention(
        **candidate,
        reference=reference,
        carry_clamp=False,
    )
    clamped, _state, audit = recurrent_gdn_intervention(
        **candidate,
        reference=reference,
        carry_clamp=True,
    )
    assert not torch.allclose(clamped[:, 0], baseline[:, 0])
    assert torch.allclose(clamped[:, 0], raw[:, 0], atol=1e-6, rtol=1e-6)
    assert torch.allclose(clamped[:, 1:], baseline[:, 1:], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(raw[:, 1:], baseline[:, 1:])
    assert audit["states"]["final"]["carried"]["normalized_frobenius"] == 0.0


def test_all_recurrence_mediator_repairs_reproduce_reference() -> None:
    reference = _mediators()
    candidate = {key: value + (0.2 if key in {"q", "k", "v"} else 0.05) for key, value in reference.items()}
    expected, _ = recurrent_gated_delta_rule(**reference)
    repaired, _state, _audit = recurrent_gdn_intervention(
        **candidate,
        reference=reference,
        repair_groups=MEDIATOR_GROUPS,
    )
    repeated, _state, _audit = recurrent_gdn_intervention(
        **candidate,
        reference=reference,
        repair_groups=MEDIATOR_GROUPS,
    )
    assert torch.allclose(repaired, expected, atol=1e-6, rtol=1e-6)
    assert torch.equal(repaired, repeated)


def test_gdn_context_restores_kernels_and_hooks_after_exception() -> None:
    module_name = "_ffn_quant_mechanism_fake_remote"
    remote = types.ModuleType(module_name)

    def chunk(*_args, **_kwargs):
        return None, None

    def recurrent(*_args, **_kwargs):
        return None, None

    remote.chunk_gated_delta_rule = chunk
    remote.fused_recurrent_gated_delta_rule = recurrent
    sys.modules[module_name] = remote
    try:
        GatedDeltaNet = type(
            "GatedDeltaNet",
            (nn.Module,),
            {
                "__module__": module_name,
                "__init__": lambda self, index: (
                    nn.Module.__init__(self),
                    setattr(self, "layer_idx", index),
                    setattr(self, "g_proj", nn.Identity()),
                )[-1],
            },
        )

        class FakeModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = nn.ModuleList([GatedDeltaNet(index) for index in range(27)])

        FakeModel.__module__ = module_name
        model = FakeModel()
        original_hook_counts = [(len(layer._forward_pre_hooks), len(layer._forward_hooks)) for layer in model.layers]
        with pytest.raises(RuntimeError, match="injected failure"):
            with GDNInterventionContext(model, capture_reference=True):
                assert remote.chunk_gated_delta_rule is not chunk
                raise RuntimeError("injected failure")
        assert remote.chunk_gated_delta_rule is chunk
        assert remote.fused_recurrent_gated_delta_rule is recurrent
        assert [
            (len(layer._forward_pre_hooks), len(layer._forward_hooks)) for layer in model.layers
        ] == original_hook_counts
    finally:
        sys.modules.pop(module_name, None)
