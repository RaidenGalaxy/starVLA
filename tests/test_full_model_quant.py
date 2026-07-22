import unittest

import torch
import torch.nn as nn

from experiments.full_model_quant.awq import ActivationCache, apply_input_scale_, apply_internal_scale_, search_awq
from experiments.full_model_quant.evaluation import generation_metrics
from experiments.full_model_quant.modeling import FFNTarget, config_assignment, discover_full_model, make_fp16_weight_reference_
from experiments.full_model_quant.quantization import fake_quantize_weight


class Norm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))


class MLP(nn.Module):
    def __init__(self, width=8, intermediate=12, bias=False):
        super().__init__()
        self.gate_proj = nn.Linear(width, intermediate, bias=bias)
        self.up_proj = nn.Linear(width, intermediate, bias=bias)
        self.down_proj = nn.Linear(intermediate, width, bias=bias)

    def forward(self, value):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(value)) * self.up_proj(value))


class Attention(nn.Module):
    def __init__(self, vision=False):
        super().__init__()
        if vision:
            self.qkv, self.proj = nn.Linear(8, 24), nn.Linear(8, 8)
        else:
            self.q_proj, self.k_proj = nn.Linear(8, 8), nn.Linear(8, 8)
            self.v_proj, self.o_proj = nn.Linear(8, 8), nn.Linear(8, 8)


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.post_attention_layernorm, self.mlp, self.self_attn = Norm(8), MLP(), Attention()


class VisionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm2, self.mlp, self.attn = Norm(8), MLP(bias=True), Attention(vision=True)


class ToyVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList([DecoderLayer(), DecoderLayer()])
        self.model.visual = nn.Module()
        self.model.visual.blocks = nn.ModuleList([VisionBlock() for _ in range(3)])
        self.model.visual.merger = nn.Module()
        self.model.visual.merger.mlp = nn.Sequential(nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 8))
        self.embedding = nn.Embedding(20, 8)
        self.lm_head = nn.Linear(8, 20, bias=False)
        self.lm_head.weight = self.embedding.weight
        self.patch = nn.Conv3d(3, 8, (2, 2, 2))

    def get_input_embeddings(self):
        return self.embedding


class FullModelSelectorTest(unittest.TestCase):
    def setUp(self):
        self.model = ToyVLM()
        self.targets, self.ffns = discover_full_model(self.model, strict_qwen3b=False)

    def test_structural_inventory_and_exclusions(self):
        counts = {group: sum(target.group == group for target in self.targets) for group in (
            "decoder_ffn", "decoder_attention", "vision_ffn", "vision_attention", "vision_merger"
        )}
        self.assertEqual(counts, {"decoder_ffn": 6, "decoder_attention": 8, "vision_ffn": 9, "vision_attention": 6, "vision_merger": 2})
        self.assertEqual(len(self.targets), 31)
        self.assertTrue(all(target.module is not self.model.lm_head for target in self.targets))

    def test_fixed_config_assignments(self):
        hybrid = config_assignment("hybrid_awq_w4_all_linear", self.targets)
        self.assertEqual(sum(value == "awq_w4" for value in hybrid.values()), 15)
        self.assertEqual(sum(value == "rtn_w4" for value in hybrid.values()), 16)
        vision = config_assignment("rtn_w4_vision_only", self.targets)
        self.assertEqual(sum(value == "rtn_w4" for value in vision.values()), 17)


class ReferenceAndQuantizerTest(unittest.TestCase):
    def test_fp16_rounding_and_overflow(self):
        first, second = nn.Linear(3, 2), nn.Linear(3, 2)
        second.load_state_dict(first.state_dict())
        self.assertEqual(make_fp16_weight_reference_(first), make_fp16_weight_reference_(second))
        self.assertTrue(all(parameter.dtype == torch.float32 for parameter in first.parameters()))
        overflow = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            overflow.weight.fill_(100000.0)
        with self.assertRaises(ValueError):
            make_fp16_weight_reference_(overflow)

    def test_fp32_group_quantizer(self):
        value = torch.tensor([[0.0, 0.25, -0.5, 1.0, 0.2]], dtype=torch.float16)
        q4, stats4 = fake_quantize_weight(value, bits=4, group_size=4)
        q8, stats8 = fake_quantize_weight(value, bits=8, group_size=4)
        self.assertEqual(q4.dtype, torch.float32)
        self.assertLessEqual(stats8.mse, stats4.mse)


class VisionAWQTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.norm, self.mlp = Norm(8), MLP(bias=True)
        self.ffn = FFNTarget("vision", 0, "visual.mlp", "visual.norm2", self.mlp, self.norm)

    def test_equivalent_scales_with_bias(self):
        value = torch.randn(5, 8)
        reference = self.mlp(value)
        input_scale = torch.linspace(0.7, 1.3, 8)
        apply_input_scale_(self.ffn, input_scale)
        self.assertLess((self.mlp(value / input_scale) - reference).abs().max().item(), 1e-5)
        apply_internal_scale_(self.ffn, torch.linspace(0.75, 1.25, 12))
        self.assertLess((self.mlp(value / input_scale) - reference).abs().max().item(), 1e-5)

    def test_cache_search_and_restore(self):
        cache = ActivationCache([self.ffn], max_tokens=16)
        with cache:
            self.mlp(torch.randn(3, 8))
        original = {name: value.detach().clone() for name, value in self.mlp.named_parameters()}
        result = search_awq(self.ffn, cache.reconstruction_inputs(self.ffn), cache.mean_abs(self.ffn, "input"), cache.mean_abs(self.ffn, "down"), group_size=4)
        self.assertLessEqual(result.awq_nmse, result.rtn_nmse + 1e-12)
        for name, value in self.mlp.named_parameters():
            self.assertTrue(torch.equal(value, original[name]))


class GenerationMetricTest(unittest.TestCase):
    def test_reference_and_target_are_distinct(self):
        rows = [
            {"candidate": "The red circle.", "reference": "the red circle", "target": "The red circle."},
            {"candidate": "blue square", "reference": "green square", "target": "blue square"},
        ]
        metrics = generation_metrics(rows)
        self.assertEqual(metrics["reference_exact_agreement"], 0.5)
        self.assertEqual(metrics["target_exact_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
