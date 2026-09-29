import pytest
import torch
from torch import nn

from experiments.infinitevl_quant.inventory import LinearTarget
from experiments.infinitevl_quant.quantization import FakeQuantLinear
from experiments.infinitevl_quant_diagnosis.audit import audit_quantizer
from experiments.infinitevl_quant_diagnosis.config import SCHEMES
from experiments.infinitevl_quant_diagnosis.quantization import (
    diagnostic_qdq_linears,
    select_targets,
)


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.left = nn.Linear(7, 5, bias=False)
        self.right = nn.Linear(5, 3, bias=False)


def _targets(model):
    return [
        LinearTarget("left", model.left, "decoder_ffn", 1),
        LinearTarget("right", model.right, "gdn_main", 2),
    ]


@pytest.mark.parametrize(
    "scheme_name",
    ("a8_only", "w4_asym_a16", "w4_asym_a8", "w4_sym_channel_a16", "w4_sym_channel_a8"),
)
def test_diagnostic_context_restores_exact_modules(scheme_name):
    model = Toy()
    targets = _targets(model)
    originals = (model.left, model.right)
    with diagnostic_qdq_linears(model, targets, SCHEMES[scheme_name]) as audit:
        assert isinstance(model.left, FakeQuantLinear)
        assert isinstance(model.right, FakeQuantLinear)
        assert audit["module_count"] == 2
        value = model.right(model.left(torch.randn(2, 7)))
        assert torch.isfinite(value).all()
    assert model.left is originals[0]
    assert model.right is originals[1]


def test_select_targets_combines_filters():
    model = Toy()
    selected = select_targets(
        _targets(model), scopes={"decoder_ffn"}, layers={1}, projections={"left"},
    )
    assert [target.name for target in selected] == ["left"]


def test_independent_quantizer_cross_check():
    result = audit_quantizer()
    assert result["passed"]
    assert result["max_reference_error"] <= 1e-6
    assert all(result["nonfinite_rejected"].values())

