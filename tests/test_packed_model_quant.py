import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from experiments.full_model_quant.quantization import fake_quantize_weight
from experiments.packed_model_quant.checkpoint import _write_shards
from experiments.packed_model_quant.linear import PackedWeightOnlyLinear
from experiments.packed_model_quant.packing import (
    dequantize_packed,
    pack_int4,
    quantize_packed_weight,
    unpack_int4,
)


class Int4PackingTest(unittest.TestCase):
    def test_boundaries_nibble_order_and_odd_tail(self):
        codes = torch.tensor([[-7, -1, 0, 1, 7], [7, 6, -6, -7, 0]], dtype=torch.int8)
        packed = pack_int4(codes)
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(tuple(packed.shape), (2, 3))
        self.assertEqual(int(packed[0, 0]), 0xF9)
        self.assertTrue(torch.equal(unpack_int4(packed, 5), codes))
        self.assertEqual(int(packed[0, -1] >> 4), 0)

    def test_rejects_out_of_range_and_wrong_shape(self):
        with self.assertRaises(ValueError):
            pack_int4(torch.tensor([[8]], dtype=torch.int8))
        with self.assertRaises(ValueError):
            unpack_int4(torch.zeros(1, 1, dtype=torch.uint8), 3)


class PackedQuantizerTest(unittest.TestCase):
    def test_w4_fp32_scale_matches_existing_fake_quant(self):
        torch.manual_seed(2)
        weight = torch.randn(5, 11)
        fake, _stats = fake_quantize_weight(weight, bits=4, group_size=4)
        packed = quantize_packed_weight(weight, bits=4, group_size=4, scale_dtype="fp32")
        restored = dequantize_packed(packed.qweight, packed.scales, bits=4, group_size=4, in_features=11)
        self.assertLessEqual(float((fake - restored).abs().max()), 1e-6)

    def test_zero_tail_w8_and_determinism(self):
        value = torch.zeros(3, 7)
        first = quantize_packed_weight(value, bits=8, group_size=4, scale_dtype="fp16")
        second = quantize_packed_weight(value, bits=8, group_size=4, scale_dtype="fp16")
        self.assertTrue(torch.equal(first.qweight, second.qweight))
        self.assertTrue(torch.equal(first.scales, second.scales))
        self.assertEqual(first.stats.groups, 6)
        self.assertEqual(first.stats.mse, 0.0)

    def test_fp16_scale_error_is_explicit(self):
        weight = torch.tensor([[0.0, 0.12345, -0.98765, 0.33333]])
        packed = quantize_packed_weight(weight, bits=4, group_size=3, scale_dtype="fp16")
        self.assertEqual(packed.scales.dtype, torch.float16)
        self.assertGreaterEqual(packed.stats.scale_rounding_mse, 0.0)

    def test_nonfinite_rejected(self):
        for value in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                quantize_packed_weight(torch.tensor([[value]]), bits=4, group_size=4)


class PackedLinearTest(unittest.TestCase):
    def test_fp32_forward_and_no_weight_parameter(self):
        torch.manual_seed(3)
        weight, bias, value = torch.randn(4, 7), torch.randn(4), torch.randn(2, 7)
        packed = quantize_packed_weight(weight, bits=4, group_size=4, scale_dtype="fp32")
        module = PackedWeightOnlyLinear.from_packed(packed, bias=bias)
        restored = dequantize_packed(packed.qweight, packed.scales, bits=4, group_size=4, in_features=7)
        self.assertTrue(torch.equal(module(value), F.linear(value, restored, bias)))
        self.assertFalse(hasattr(module, "weight"))
        self.assertEqual(module.runtime.calls, 1)
        self.assertEqual(module.runtime.max_temporary_bytes, restored.numel() * 4)

    def test_a16_boundary_rounding(self):
        torch.manual_seed(5)
        weight, bias, value = torch.randn(3, 5), torch.randn(3), torch.randn(2, 5)
        packed = quantize_packed_weight(weight, bits=8, group_size=4, scale_dtype="fp16")
        module = PackedWeightOnlyLinear.from_packed(packed, bias=bias, compute_mode="a16_round_fp32_accum")
        restored = dequantize_packed(packed.qweight, packed.scales, bits=8, group_size=4, in_features=5)
        expected = F.linear(value.half().float(), restored.half().float(), bias.half().float()).half().float()
        self.assertTrue(torch.equal(module(value), expected))


class ShardTest(unittest.TestCase):
    def test_safetensors_shards_round_trip(self):
        tensors = {"a": torch.arange(8), "b": torch.randn(3, 4), "c": torch.ones(2)}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            index = _write_shards(tensors, output, max_shard_bytes=64)
            restored = {}
            for name in set(index["weight_map"].values()):
                restored.update(load_file(output / name))
            self.assertEqual(set(restored), set(tensors))
            for key in tensors:
                self.assertTrue(torch.equal(restored[key], tensors[key]))


if __name__ == "__main__":
    unittest.main()
