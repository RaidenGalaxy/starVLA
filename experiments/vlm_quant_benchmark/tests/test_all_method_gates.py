import pytest
import torch
from torch import nn

from experiments.vlm_quant_benchmark.all_methods import apply_method
from experiments.vlm_quant_benchmark.calibration import ActivationStats
from experiments.vlm_quant_benchmark.config import ExperimentConfig, MethodName, QuantScope
from experiments.vlm_quant_benchmark.selection import select_linears


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(8, 8, bias=True)


class OneLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([DecoderLayer()])


def config_for(method):
    kwargs = {
        "scope": QuantScope.DECODER_CANONICAL,
        "group_size": 4,
        "awq_alpha_grid": (0.0, 0.5, 1.0),
        "awq_clip_grid": (1.0, 0.9),
        "omniquant_steps": 1,
        "squeezellm_kmeans_steps": 1,
    }
    if method in {MethodName.SMOOTHQUANT, MethodName.LLM_INT8, MethodName.VECTORWISE_W8, MethodName.TENDER}:
        kwargs.update(weight_bits=8, activation_bits=8, zero_point=False)
    return ExperimentConfig(method, **kwargs)


@pytest.mark.parametrize(
    "method",
    [
        MethodName.AWQ,
        MethodName.GPTQ,
        MethodName.SMOOTHQUANT,
        MethodName.LLM_INT8,
        MethodName.VECTORWISE_W8,
        MethodName.TENDER,
        MethodName.OMNIQUANT,
        MethodName.SPQR,
        MethodName.SQUEEZELLM,
    ],
)
def test_all_local_method_smokes_are_finite_and_restore(method):
    torch.manual_seed(42)
    model = OneLayerModel()
    selected = select_linears(
        model, QuantScope.DECODER_CANONICAL, strict_qwen_3b=False,
    )
    assert len(selected) == 1
    source = selected[0].module
    reservoir = torch.randn(12, 8)
    stats = ActivationStats(
        abs_sum=reservoir.abs().sum(0),
        abs_max=reservoir.abs().amax(0),
        rows=reservoir.shape[0],
        reservoir=reservoir,
    )
    calibration = {selected[0].name: stats}
    config = config_for(method)
    applied = apply_method(model, selected, config, calibration)
    replacement = model.model.layers[0].self_attn.q_proj
    output = replacement(torch.randn(2, 3, 8))
    assert applied.audit["module_count"] == 1
    assert torch.isfinite(output).all()
    assert torch.isfinite(replacement.weight).all()
    if method in {MethodName.SPQR, MethodName.SQUEEZELLM}:
        assert applied.audit["quality_eval_storage"] == "dense_fp32_materialized"
        assert not applied.audit["deployment_kernel_claim"]
    applied.restore()
    assert model.model.layers[0].self_attn.q_proj is source
