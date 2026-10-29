from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from experiments.infinitevl_quant.config import SCHEMES
from experiments.infinitevl_quant.inventory import LinearTarget
from experiments.infinitevl_quant.quantization import quantized_linears

from ..shadow import ShadowObservation, shadow_ffns


class ToyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(128, 128, bias=False)
        self.up_proj = nn.Linear(128, 128, bias=False)
        self.down_proj = nn.Linear(128, 128, bias=False)
        self.act_fn = F.silu

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(value)) * self.up_proj(value))


class ToyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.post_attention_layernorm = nn.Identity()
        self.mlp = ToyMLP()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        return residual + self.mlp(self.post_attention_layernorm(value))


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([ToyLayer()])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers[0](value)


def _targets(model: ToyModel) -> list[LinearTarget]:
    return [
        LinearTarget(
            name=f"layers.0.mlp.{projection}_proj",
            module=getattr(model.layers[0].mlp, f"{projection}_proj"),
            scope="decoder_ffn",
            layer=0,
        )
        for projection in ("gate", "up", "down")
    ]


def _nll(logits: torch.Tensor) -> torch.Tensor:
    labels = torch.arange(logits.shape[0]) % logits.shape[-1]
    return F.cross_entropy(logits[:, 0], labels)


def test_shadow_endpoints_match_bf16_and_existing_w4_path() -> None:
    torch.manual_seed(7)
    model = ToyModel().eval()
    value = torch.randn(4, 3, 128)
    source = model.layers[0].mlp
    bf16_path = model(value)
    observations: list[ShadowObservation] = []
    with shadow_ffns(
        model.layers,
        decoder_prefix="layers",
        gammas=0.0,
        observer=observations.append,
    ):
        shadow_b = model(value)
    assert torch.equal(shadow_b, bf16_path)
    assert abs(float(_nll(shadow_b) - _nll(bf16_path))) <= 1e-6
    assert observations[0].residual.data_ptr() == value.data_ptr()

    with shadow_ffns(model.layers, decoder_prefix="layers", gammas=1.0):
        shadow_w = model(value)
    with quantized_linears(model, _targets(model), SCHEMES["w4a16"]):
        existing_w = model(value)
    assert torch.equal(shadow_w, existing_w)
    assert abs(float(_nll(shadow_w) - _nll(existing_w))) <= 1e-6
    assert model.layers[0].mlp is source


def test_shadow_restores_modules_hooks_and_parameters_after_exception() -> None:
    torch.manual_seed(11)
    model = ToyModel().eval()
    original = model.layers[0].mlp
    norm = model.layers[0].post_attention_layernorm
    parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
    hooks_before = dict(norm._forward_pre_hooks)
    with pytest.raises(RuntimeError, match="injected failure"):
        with shadow_ffns(model.layers, decoder_prefix="layers", gammas=1.0):
            assert model.layers[0].mlp is not original
            raise RuntimeError("injected failure")
    assert model.layers[0].mlp is original
    assert dict(norm._forward_pre_hooks) == hooks_before
    for name, value in model.named_parameters():
        assert torch.equal(value, parameters[name])
