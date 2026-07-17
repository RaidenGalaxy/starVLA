import json
import math
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from experiments.ffn_quant.awq import apply_input_scale_, apply_internal_scale_, search_awq_layer
from experiments.ffn_quant.data import generate_tiny_dataset, load_manifest
from experiments.ffn_quant.metrics import LogitsAccumulator, TensorErrorAccumulator
from experiments.ffn_quant.modeling import ActivationCache, build_module_inventory, discover_decoder_ffns, resolve_scope
from experiments.ffn_quant.quantization import ParameterSnapshot, fake_quantize_weight


class ToyRMSNorm(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))

    def forward(self, value):
        normalized = value.float() * torch.rsqrt(value.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        return normalized.to(value.dtype) * self.weight


class ToyMLP(nn.Module):
    def __init__(self, hidden=8, intermediate=12):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, value):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(value)) * self.up_proj(value))


class ToyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.post_attention_layernorm = ToyRMSNorm(8)
        self.mlp = ToyMLP()

    def forward(self, value):
        return value + self.mlp(self.post_attention_layernorm(value))


class VisionLikeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 8))


class ToyModel(nn.Module):
    def __init__(self, layers=3):
        super().__init__()
        self.visual = VisionLikeBlock()
        self.layers = nn.ModuleList([ToyLayer() for _ in range(layers)])

    def forward(self, value):
        for layer in self.layers:
            value = layer(value)
        return value


class FakeQuantizationTest(unittest.TestCase):
    def test_zero_constant_tail_and_determinism(self):
        weight = torch.zeros(3, 7)
        weight[1] = 0.25
        weight[2] = torch.linspace(-1, 1, 7)
        first, stats = fake_quantize_weight(weight, bits=4, group_size=4)
        second, _ = fake_quantize_weight(weight, bits=4, group_size=4)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first[0], weight[0]))
        self.assertTrue(torch.equal(first[1], weight[1]))
        self.assertEqual(stats.num_groups, 6)

    def test_w8_is_no_worse_than_w4(self):
        torch.manual_seed(1)
        weight = torch.randn(5, 13)
        _w4, stats4 = fake_quantize_weight(weight, bits=4, group_size=8)
        _w8, stats8 = fake_quantize_weight(weight, bits=8, group_size=8)
        self.assertLessEqual(stats8.weight_mse, stats4.weight_mse)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            fake_quantize_weight(torch.tensor([[math.nan]]), bits=4)
        with self.assertRaises(ValueError):
            fake_quantize_weight(torch.ones(2, 2), bits=3)
        with self.assertRaises(ValueError):
            fake_quantize_weight(torch.ones(2), bits=4)

    def test_snapshot_restores_exact_values(self):
        layer = nn.Linear(3, 2, bias=False)
        snapshot = ParameterSnapshot([("weight", layer.weight)])
        original = layer.weight.detach().clone()
        with torch.no_grad():
            layer.weight.add_(1)
        snapshot.restore()
        snapshot.assert_restored()
        self.assertTrue(torch.equal(layer.weight, original))


class ModuleSelectionAndAWQTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = ToyModel()
        self.entries = discover_decoder_ffns(self.model, strict_qwen3b=False)

    def test_selector_excludes_vision_and_resolves_scopes(self):
        self.assertEqual(len(self.entries), 3)
        self.assertEqual(len(resolve_scope(self.entries, "last1")), 1)
        self.assertEqual(len(resolve_scope(self.entries, "last4")), 3)
        inventory = build_module_inventory(self.model, self.entries)
        self.assertEqual(inventory["linear_modules"], 9)
        self.assertTrue(all("visual" not in item["name"] for item in inventory["modules"]))

    def test_equivalent_scaling(self):
        entry = self.entries[0]
        value = torch.randn(2, 5, 8)
        normalized = entry.norm(value)
        reference = entry.mlp(normalized)
        input_scale = torch.linspace(0.7, 1.3, 8)
        apply_input_scale_(entry, input_scale)
        candidate = entry.mlp(normalized / input_scale)
        self.assertLess((candidate - reference).abs().max().item(), 1e-5)
        internal_scale = torch.linspace(0.75, 1.25, 12)
        apply_internal_scale_(entry, internal_scale)
        candidate = entry.mlp(normalized / input_scale)
        self.assertLess((candidate - reference).abs().max().item(), 1e-5)

    def test_activation_cache_and_awq_search(self):
        cache = ActivationCache(self.entries, max_tokens_per_layer=16)
        with cache:
            self.model(torch.randn(2, 5, 8))
        entry = self.entries[-1]
        result = search_awq_layer(
            entry,
            cache.reconstruction_inputs(entry.index),
            cache.mean_abs(entry.index),
            cache.mean_abs(entry.index, down=True),
            group_size=4,
            equivalence_tolerance=1e-5,
        )
        self.assertLessEqual(result.awq_normalized_mse, result.rtn_normalized_mse + 1e-12)
        self.assertEqual(result.input_scale.shape, (8,))
        self.assertEqual(result.internal_scale.shape, (12,))


class DataAndMetricsTest(unittest.TestCase):
    def test_dataset_is_reproducible_and_valid(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_manifest = generate_tiny_dataset(first, seed=42)
            second_manifest = generate_tiny_dataset(second, seed=42)
            first_rows = [json.loads(line) for line in first_manifest.read_text().splitlines()]
            second_rows = [json.loads(line) for line in second_manifest.read_text().splitlines()]
            for row in first_rows:
                row["image"] = Path(row["image"]).name
            for row in second_rows:
                row["image"] = Path(row["image"]).name
            self.assertEqual(first_rows, second_rows)
            samples = load_manifest(first_manifest)
            self.assertEqual(sum(sample.split == "calibration" for sample in samples), 8)
            self.assertEqual(sum(sample.split == "evaluation" for sample in samples), 16)

    def test_identity_metrics(self):
        value = torch.randn(4, 9)
        tensor_metrics = TensorErrorAccumulator()
        tensor_metrics.update(value, value)
        computed = tensor_metrics.compute()
        self.assertEqual(computed["normalized_mse"], 0.0)
        self.assertAlmostEqual(computed["cosine_similarity"], 1.0, places=6)

        logits = torch.randn(3, 11)
        target = torch.tensor([1, 2, 3])
        logits_metrics = LogitsAccumulator()
        logits_metrics.update(logits, logits, target)
        computed_logits = logits_metrics.compute()
        self.assertAlmostEqual(computed_logits["kl_divergence"], 0.0, places=6)
        self.assertEqual(computed_logits["top1_agreement"], 1.0)


if __name__ == "__main__":
    unittest.main()
