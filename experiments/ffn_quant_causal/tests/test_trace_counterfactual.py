from __future__ import annotations

import math
import unittest

import torch
import torch.nn.functional as F
from torch import nn

from experiments.ffn_quant_causal.counterfactual import (
    FFNWeights,
    replay_counterfactual_pair,
    replay_ffn,
    run_weight_trace_matrix,
    tensor_error_summary,
)
from experiments.ffn_quant_causal.counterfactual_stage import (
    _validate_capture_parity,
)
from experiments.ffn_quant_causal.trace import (
    QwenFFNTracer,
    tensor_distribution_summary,
    trace_summary,
)


class ToyRMSNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + 1e-6) * self.weight


class ToyMLP(nn.Module):
    def __init__(self, hidden: int = 8, intermediate: int = 13) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=True)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(value)) * self.up_proj(value))


class ToyQwenLayer(nn.Module):
    def __init__(self, hidden: int = 8) -> None:
        super().__init__()
        self.self_attn = nn.Linear(hidden, hidden, bias=False)
        self.post_attention_layernorm = ToyRMSNorm(hidden)
        self.mlp = ToyMLP(hidden=hidden)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value + self.self_attn(value)
        ffn_input = self.post_attention_layernorm(residual)
        return residual + self.mlp(ffn_input)


class TraceTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.layer = ToyQwenLayer().eval()
        self.value = torch.randn(2, 5, 8)

    def test_captures_exact_qwen_ffn_stages_without_changing_output(self) -> None:
        reference = self.layer(self.value)
        tracer = QwenFFNTracer({"layer0": self.layer})
        with tracer:
            observed = self.layer(self.value)
        self.assertTrue(torch.equal(reference, observed))
        self.assertEqual(len(tracer.traces["layer0"]), 1)
        trace = tracer.traces["layer0"][0]

        residual = self.value + self.layer.self_attn(self.value)
        ffn_input = self.layer.post_attention_layernorm(residual)
        gate = self.layer.mlp.gate_proj(ffn_input)
        up = self.layer.mlp.up_proj(ffn_input)
        product = F.silu(gate) * up
        down = self.layer.mlp.down_proj(product)
        for expected, actual in (
            (residual, trace.residual),
            (ffn_input, trace.ffn_input),
            (gate, trace.gate),
            (up, trace.up),
            (product, trace.product),
            (down, trace.down_output),
        ):
            self.assertTrue(torch.equal(expected, actual))

        summary = trace_summary(trace)
        self.assertEqual(summary["layer_name"], "layer0")
        self.assertGreater(summary["stages"]["gate"]["abs_p999"], 0.0)
        self.assertTrue(math.isfinite(summary["branch"]["ffn_to_residual_rms"]))

        # Hooks must be removed at context exit: no second record is appended.
        self.layer(self.value)
        self.assertEqual(len(tracer.traces["layer0"]), 1)

    def test_multiple_invocations_and_aligned_token_bound(self) -> None:
        streamed = []
        tracer = QwenFFNTracer(
            {"layer0": self.layer},
            max_tokens=3,
            max_calls_per_layer=1,
            on_trace=streamed.append,
        )
        with tracer:
            self.layer(self.value)
            self.layer(self.value + 1)
        self.assertEqual([trace.call_index for trace in streamed], [0, 1])
        self.assertEqual([trace.call_index for trace in tracer.flattened()], [0])
        self.assertEqual(
            tracer.retention_audit()["layer0"],
            {"calls": 2, "retained": 1, "not_retained": 1},
        )
        for trace in streamed:
            self.assertEqual(trace.ffn_input.shape, (3, 8))
            self.assertEqual(trace.residual.shape, (3, 8))
            self.assertEqual(trace.product.shape, (3, 13))

    def test_rejects_non_qwen_layer_and_nonfinite_summary_is_explicit(self) -> None:
        with self.assertRaises(TypeError):
            QwenFFNTracer({"bad": nn.Linear(2, 2)})
        summary = tensor_distribution_summary(torch.tensor([1.0, float("nan")]))
        self.assertFalse(summary["finite"])
        self.assertEqual(summary["finite_fraction"], 0.5)


class CounterfactualReplayTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(23)
        self.layer = ToyQwenLayer().eval()
        self.value = torch.randn(2, 4, 8)
        tracer = QwenFFNTracer({"source": self.layer})
        with tracer:
            self.layer(self.value)
        self.trace = tracer.traces["source"][0]
        self.weights = FFNWeights.from_mlp(self.layer.mlp)

    def test_reference_replay_matches_every_captured_stage(self) -> None:
        replay = replay_ffn(self.trace.ffn_input, self.weights)
        self.assertEqual(replay.mode, "checkpoint_values_fp32_compute")
        for name, value in replay.stage_tensors().items():
            self.assertLess(
                (value - getattr(self.trace, name)).abs().max().item(),
                1e-6,
            )

    def test_rtn_pair_is_deterministic_and_reports_residual_geometry(self) -> None:
        first = replay_counterfactual_pair(
            self.trace, self.weights, bits=4, group_size=5, compare_capture=True
        )
        second = replay_counterfactual_pair(
            self.trace, self.weights, bits=4, group_size=5, compare_capture=True
        )
        self.assertTrue(torch.equal(first.rtn.down_output, second.rtn.down_output))
        self.assertGreater(
            first.summary["stage_error"]["down_output"]["normalized_mse"], 0.0
        )
        self.assertLess(
            first.summary["reference_vs_captured"]["down_output"]["max_abs_error"],
            1e-6,
        )
        self.assertTrue(math.isfinite(
            first.summary["residual_geometry"]["quant_error_to_residual_rms"]
        ))
        self.assertEqual(first.rtn.weight_audit["gate"]["groups_per_row"], 2)

    def test_weight_by_trace_matrix_covers_cross_source_pairs(self) -> None:
        other = ToyQwenLayer().eval()
        matrix = run_weight_trace_matrix(
            {"source": self.trace, "alternate": self.trace},
            {
                "source": self.weights,
                "alternate": FFNWeights.from_mlp(other.mlp),
            },
            bits=(4, 3),
            group_size=5,
        )
        self.assertEqual(len(matrix["rows"]), 8)
        self.assertEqual(matrix["analysis_scope"], "offline_ffn_local_replay")
        self.assertFalse(matrix["end_to_end_model_score"])
        matched = [
            row for row in matrix["rows"]
            if row["trace_source"] == row["weight_source"]
        ]
        crossed = [
            row for row in matrix["rows"]
            if row["trace_source"] != row["weight_source"]
        ]
        self.assertTrue(all("reference_vs_captured" in row["summary"] for row in matched))
        self.assertTrue(all("reference_vs_captured" not in row["summary"] for row in crossed))

    def test_validation_and_zero_error_metrics(self) -> None:
        identity = tensor_error_summary(self.trace.gate, self.trace.gate)
        self.assertEqual(identity["normalized_mse"], 0.0)
        self.assertEqual(identity["cosine_similarity"], 1.0)
        with self.assertRaises(ValueError):
            replay_ffn(self.trace.ffn_input[..., :-1], self.weights)
        with self.assertRaises(ValueError):
            replay_ffn(self.trace.ffn_input, self.weights, bits=5)

    def test_capture_parity_uses_absolute_and_relative_guardrails(self) -> None:
        passing = {
            "reference_vs_captured": {
                "up": {"max_abs_error": 1.9e-4, "relative_rms_error": 4e-6},
                "down": {"max_abs_error": 8e-6, "relative_rms_error": 3e-6},
            },
            "reference_distributions": {
                "up": {"abs_max": 2.0},
                "down": {"abs_max": 1.0},
            },
        }
        self.assertTrue(_validate_capture_parity(passing, context="toy")["passed"])
        hidden_regression = {
            "reference_vs_captured": {
                "up": {"max_abs_error": 2e-4, "relative_rms_error": 3e-5},
            },
            "reference_distributions": {"up": {"abs_max": 2.0}},
        }
        with self.assertRaises(RuntimeError):
            _validate_capture_parity(hidden_regression, context="toy")

        large_signal_roundoff = {
            "reference_vs_captured": {
                "up": {"max_abs_error": 8.5e-4, "relative_rms_error": 4e-7},
            },
            "reference_distributions": {"up": {"abs_max": 1_100.0}},
        }
        result = _validate_capture_parity(
            large_signal_roundoff, context="large-signal-toy",
        )
        self.assertTrue(result["passed"])
        self.assertEqual(
            result["criterion"], "signal_scaled_max_abs_and_relative_rms",
        )

        large_signal_spike = {
            "reference_vs_captured": {
                "up": {"max_abs_error": 2e-3, "relative_rms_error": 4e-7},
            },
            "reference_distributions": {"up": {"abs_max": 1_100.0}},
        }
        with self.assertRaises(RuntimeError):
            _validate_capture_parity(large_signal_spike, context="spike-toy")


if __name__ == "__main__":
    unittest.main()
