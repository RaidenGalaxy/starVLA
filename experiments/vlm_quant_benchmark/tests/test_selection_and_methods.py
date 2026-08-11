import torch
from torch import nn

from experiments.vlm_quant_benchmark.config import ExperimentConfig, MethodName, QuantScope
from experiments.vlm_quant_benchmark.methods import ScaledQuantLinear
from experiments.vlm_quant_benchmark.safe_methods import apply_method
from experiments.vlm_quant_benchmark.selection import select_linears


class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(8, 8, bias=False)
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(8, 16, bias=False)


class VisionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(8, 24, bias=True)
        self.mlp = nn.Module()
        self.mlp.fc = nn.Linear(8, 16, bias=True)


class ToyVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([DecoderBlock(), DecoderBlock()])
        self.visual = nn.Module()
        self.visual.blocks = nn.ModuleList([VisionBlock()])
        self.visual.merger = nn.Linear(8, 8)
        self.lm_head = nn.Linear(8, 32, bias=False)
        self.patch_embed = nn.Conv3d(3, 8, 2)


def test_selector_scopes_and_exclusions():
    model = ToyVLM()
    decoder = select_linears(model, QuantScope.DECODER_CANONICAL, strict_qwen_3b=False)
    full = select_linears(model, QuantScope.FULL_VLM_EXTENDED, strict_qwen_3b=False)
    assert len(decoder) == 4
    assert len(full) == 7
    assert not any(item.name == "lm_head" for item in full)
    assert not any("patch_embed" in item.name for item in full)


def test_equivalent_scale_without_quantization():
    torch.manual_seed(1)
    source = nn.Linear(7, 5, bias=True)
    scale = torch.rand(7) + 0.5
    wrapper = ScaledQuantLinear(source, source.weight.detach() * scale, scale)
    value = torch.randn(3, 4, 7)
    torch.testing.assert_close(source(value), wrapper(value), atol=1e-6, rtol=1e-6)


def test_rtn_replacement_and_restore():
    model = ToyVLM()
    selected = select_linears(model, QuantScope.DECODER_CANONICAL, strict_qwen_3b=False)
    original = model.model.layers[0].self_attn.q_proj
    config = ExperimentConfig(MethodName.RTN, scope=QuantScope.DECODER_CANONICAL)
    applied = apply_method(model, selected, config)
    assert model.model.layers[0].self_attn.q_proj is not original
    assert applied.audit["module_count"] == 4
    applied.restore()
    assert model.model.layers[0].self_attn.q_proj is original
