import unittest

import torch
import torch.nn.functional as F

from experiments.native_model_quant.extension import load_native_extension
from experiments.native_model_quant.linear import (
    NativePackedLinear,
    quantize_activation_reference,
    w4a8_reference,
)
from experiments.packed_model_quant.packing import dequantize_packed, quantize_packed_weight


class StreamingPackedLinearTest(unittest.TestCase):
    def _case(self, bits, bias_enabled):
        torch.manual_seed(11 + bits)
        weight = torch.randn(9, 259)
        bias = torch.randn(9) if bias_enabled else None
        value = torch.randn(2, 3, 259)
        packed = quantize_packed_weight(weight, bits=bits, group_size=128, scale_dtype="fp16")
        module = NativePackedLinear(
            259,
            9,
            bits=bits,
            group_size=128,
            qweight=packed.qweight,
            scales=packed.scales,
            bias=bias,
            backend="streaming_dequant_fp32",
            output_tile_size=4,
        )
        restored = dequantize_packed(
            packed.qweight,
            packed.scales,
            bits=bits,
            group_size=128,
            in_features=259,
        )
        expected = F.linear(value, restored, bias)
        actual = module(value)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-6)
        self.assertFalse(hasattr(module, "weight"))
        self.assertLess(module.runtime.max_float_weight_tile_bytes, module.runtime.full_weight_bytes)
        self.assertLessEqual(module.runtime.max_float_weight_tile_bytes, 4 * 259 * 4)
        self.assertEqual(module.runtime.tiles, 3)

    def test_w4_bias_tail_group_odd_width_3d(self):
        self._case(4, True)

    def test_w8_no_bias_tail_group_odd_width_3d(self):
        self._case(8, False)

    def test_a16_simulation(self):
        torch.manual_seed(22)
        weight, bias, value = torch.randn(7, 131), torch.randn(7), torch.randn(2, 131)
        packed = quantize_packed_weight(weight, bits=4, group_size=128, scale_dtype="fp16")
        module = NativePackedLinear(
            131,
            7,
            bits=4,
            group_size=128,
            qweight=packed.qweight,
            scales=packed.scales,
            bias=bias,
            backend="streaming_a16_sim",
            output_tile_size=3,
        )
        restored = dequantize_packed(
            packed.qweight, packed.scales, bits=4, group_size=128, in_features=131
        )
        expected = F.linear(
            value.half().float(), restored.half().float(), bias.half().float()
        ).half().float()
        torch.testing.assert_close(module(value), expected, atol=1e-5, rtol=0)

    def test_nonfinite_rejected(self):
        packed = quantize_packed_weight(torch.zeros(2, 8), bits=4, group_size=128)
        module = NativePackedLinear(
            8, 2, bits=4, group_size=128, qweight=packed.qweight, scales=packed.scales,
            bias=None, backend="streaming_dequant_fp32",
        )
        with self.assertRaises(FloatingPointError):
            module(torch.tensor([[float("nan")] + [0.0] * 7]))

    def test_w4a8_rejects_wrong_format(self):
        packed = quantize_packed_weight(torch.zeros(2, 8), bits=8, group_size=128)
        with self.assertRaises(ValueError):
            NativePackedLinear(
                8, 2, bits=8, group_size=128, qweight=packed.qweight, scales=packed.scales,
                bias=None, backend="avx512_w4a8",
            )


class NativeExtensionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.extension = load_native_extension()
        except Exception as exc:
            raise unittest.SkipTest(f"native extension unavailable: {exc}") from exc

    def test_activation_quantization_matches_reference(self):
        torch.manual_seed(31)
        value = torch.cat((torch.zeros(1, 259), torch.randn(3, 259)))
        expected_codes, expected_scales = quantize_activation_reference(value)
        codes, scales = self.extension.quantize_activation_groupwise(value.contiguous(), 128)
        self.assertTrue(torch.equal(codes, expected_codes))
        self.assertTrue(torch.equal(scales, expected_scales))
        self.assertEqual(int(codes.min()), -127)
        self.assertLessEqual(int(codes.max()), 127)

    def test_scalar_vnni_and_python_reference(self):
        if not self.extension.supports_avx512_vnni():
            self.skipTest("CPU has no AVX-512 VNNI")
        torch.manual_seed(41)
        for scale_dtype in ("fp16", "fp32"):
            weight, bias, value = torch.randn(11, 259), torch.randn(11), torch.randn(5, 259)
            packed = quantize_packed_weight(
                weight, bits=4, group_size=128, scale_dtype=scale_dtype
            )
            codes, scales = self.extension.quantize_activation_groupwise(value.contiguous(), 128)
            reference = w4a8_reference(
                codes, scales, packed.qweight, packed.scales, in_features=259, bias=bias
            )
            scalar = self.extension.packed_w4a8_linear_scalar(
                codes, scales, packed.qweight, packed.scales, bias, 259
            )
            vnni = self.extension.packed_w4a8_linear_vnni(
                codes, scales, packed.qweight, packed.scales, bias, 259
            )
            torch.testing.assert_close(scalar, reference, atol=1e-4, rtol=1e-5)
            torch.testing.assert_close(vnni, reference, atol=1e-4, rtol=1e-5)
            self.assertTrue(torch.equal(scalar, vnni))

    def test_extension_rejects_nonfinite(self):
        value = torch.zeros(1, 128)
        value[0, 0] = float("inf")
        with self.assertRaises(RuntimeError):
            self.extension.quantize_activation_groupwise(value, 128)


if __name__ == "__main__":
    unittest.main()
