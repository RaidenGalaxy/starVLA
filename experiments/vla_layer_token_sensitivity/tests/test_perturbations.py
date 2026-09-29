import torch
from torch import nn

from experiments.vla_layer_token_sensitivity.perturbations import (
    temporary_groupwise_weight_quantization,
    temporary_layer_skip,
)


class ResidualBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)

    def forward(self, hidden):
        return (hidden + self.linear(hidden), "cache")


def test_temporary_layer_skip_replaces_only_hidden_output() -> None:
    block = ResidualBlock()
    hidden = torch.ones(2, 4)
    normal = block(hidden)
    with temporary_layer_skip(block):
        skipped = block(hidden)
    restored = block(hidden)

    assert torch.equal(skipped[0], hidden)
    assert skipped[1] == "cache"
    assert torch.equal(restored[0], normal[0])


def test_temporary_quantization_restores_exact_weight() -> None:
    torch.manual_seed(7)
    block = ResidualBlock()
    original = block.linear.weight.detach().clone()
    with temporary_groupwise_weight_quantization(block, bits=3, group_size=2) as audit:
        assert audit.linears == 1
        assert audit.parameters == original.numel()
        assert not torch.equal(block.linear.weight, original)
    assert torch.equal(block.linear.weight, original)
