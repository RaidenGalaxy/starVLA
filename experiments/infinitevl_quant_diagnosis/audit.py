from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

import torch

from experiments.infinitevl_quant.config import EXPECTED_COUNTS, MODEL_REVISION
from experiments.infinitevl_quant.evaluation import aggregate_rows
from experiments.infinitevl_quant.inventory import build_inventory, select_scope
from experiments.infinitevl_quant.progress import file_sha256
from experiments.infinitevl_quant.quantization import FakeQuantLinear, quantize_weight

from .config import SCHEMES
from .quantization import diagnostic_qdq_linears


def _rows(root: Path, stage: str, variant: str) -> list[dict]:
    directory = root / stage / variant / "samples"
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return [json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))]


def _close(left: float, right: float, *, tolerance: float = 1e-9) -> bool:
    if math.isnan(left) and math.isnan(right):
        return True
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def _metric_check(published: dict, recomputed: dict) -> dict:
    keys = (
        "samples", "mean_score", "target_nll", "target_ppl", "target_tokens",
        "mean_logits_kl", "target_argmax_consistency", "output_consistency",
        "teacher_seconds", "generation_seconds",
    )
    differences = {}
    for key in keys:
        if key not in published or key not in recomputed:
            continue
        left, right = published[key], recomputed[key]
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if not _close(float(left), float(right)):
                differences[key] = {"published": left, "recomputed": right}
    for dataset, values in published.get("datasets", {}).items():
        for key in ("samples", "score", "nll"):
            left, right = values[key], recomputed["datasets"][dataset][key]
            if not _close(float(left), float(right)):
                differences[f"datasets.{dataset}.{key}"] = {
                    "published": left, "recomputed": right,
                }
    return {"passed": not differences, "differences": differences}


def audit_existing_results(source_output: Path) -> dict:
    required = {
        "preflight": "preflight.json",
        "smoke": "smoke_summary.json",
        "scope": "scope_sensitivity.json",
        "screen": "screen_summary.json",
        "qig": "qig_summary.json",
        "confirm": "confirm_summary.json",
        "report": "report_summary.json",
    }
    completion = {}
    for stage, summary in required.items():
        marker, result = source_output / stage / ".complete", source_output / stage / summary
        completion[stage] = {
            "marker": marker.is_file(), "summary": result.is_file(),
            "marker_sha256": file_sha256(marker) if marker.is_file() else None,
            "summary_sha256": file_sha256(result) if result.is_file() else None,
        }
    if not all(row["marker"] and row["summary"] for row in completion.values()):
        raise RuntimeError(f"source v2 protocol is incomplete: {completion}")

    protocol_identity = json.loads((source_output / "protocol_identity.json").read_text())
    config = json.loads((source_output / "protocol_config.json").read_text())
    if config.get("model_revision") != MODEL_REVISION:
        raise RuntimeError("source protocol used a different model revision")

    stage_checks = {}
    published_screen = json.loads((source_output / "screen/screen_summary.json").read_text())
    recomputed_screen = {}
    for variant, metrics in published_screen["metrics"].items():
        rows = _rows(source_output, "screen", "fp16_rounded" if variant == "fp16" else variant)
        recomputed = aggregate_rows(rows)
        stage_checks[f"screen.{variant}"] = _metric_check(metrics, recomputed)
        recomputed_screen[variant] = recomputed

    published_scope = json.loads((source_output / "scope/scope_sensitivity.json").read_text())
    baseline = aggregate_rows(_rows(source_output, "scope", "bf16"))
    stage_checks["scope.bf16"] = _metric_check(published_scope["baseline"], baseline)
    recomputed_scope = {"bf16": baseline}
    for scope, metrics in published_scope["variants"].items():
        recomputed = aggregate_rows(_rows(source_output, "scope", f"{scope}_w4a16"))
        stage_checks[f"scope.{scope}"] = _metric_check(metrics, recomputed)
        recomputed_scope[scope] = recomputed

    published_confirm = json.loads((source_output / "confirm/confirm_summary.json").read_text())
    recomputed_confirm = {}
    for variant, metrics in published_confirm["metrics"].items():
        recomputed = aggregate_rows(_rows(source_output, "confirm", variant))
        stage_checks[f"confirm.{variant}"] = _metric_check(metrics, recomputed)
        recomputed_confirm[variant] = recomputed
    failures = {key: value for key, value in stage_checks.items() if not value["passed"]}
    if failures:
        raise RuntimeError(f"published metrics do not reproduce: {failures}")

    screen_values = {
        key: {
            "mean_score": value["mean_score"], "target_nll": value["target_nll"],
            "datasets": {name: row["score"] for name, row in value["datasets"].items()},
        }
        for key, value in recomputed_screen.items()
    }
    scope_values = {
        key: {
            "target_nll": value["target_nll"],
            "nll_relative_increase": (
                value["target_nll"] / baseline["target_nll"] - 1.0
            ),
            "mean_logits_kl": value["mean_logits_kl"],
        }
        for key, value in recomputed_scope.items()
    }
    return {
        "passed": True,
        "source_output": str(source_output),
        "model_revision": config["model_revision"],
        "protocol_identity_sha256": hashlib.sha256(
            json.dumps(protocol_identity, sort_keys=True).encode()
        ).hexdigest(),
        "completion": completion,
        "metric_checks": stage_checks,
        "screen": screen_values,
        "scope": scope_values,
        "confirm": {
            key: {"mean_score": value["mean_score"], "target_nll": value["target_nll"]}
            for key, value in recomputed_confirm.items()
        },
    }


def independent_quantize(
    weight: torch.Tensor, *, bits: int, group_size: int, zero_point: bool,
) -> torch.Tensor:
    value = weight.detach().to(torch.float64)
    restored = torch.empty_like(value)
    for start in range(0, value.shape[1], group_size):
        stop = min(start + group_size, value.shape[1])
        group = value[:, start:stop]
        if zero_point:
            qmin, qmax = 0, (1 << bits) - 1
            lower, upper = group.min(dim=1, keepdim=True).values, group.max(dim=1, keepdim=True).values
            constant = upper == lower
            scale = (upper - lower) / (qmax - qmin)
            scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            zero = torch.round(qmin - lower / scale).clamp(qmin, qmax)
            current = (torch.round(group / scale + zero).clamp(qmin, qmax) - zero) * scale
            current = torch.where(constant, group, current)
        else:
            qmax = (1 << (bits - 1)) - 1
            maximum = group.abs().max(dim=1, keepdim=True).values
            scale = torch.where(maximum > 0, maximum / qmax, torch.ones_like(maximum))
            current = torch.round(group / scale).clamp(-qmax, qmax) * scale
        restored[:, start:stop] = current
    return restored.float()


def audit_quantizer() -> dict:
    generator = torch.Generator().manual_seed(20260923)
    cases = {
        "random_tail": torch.randn(7, 259, generator=generator),
        "zeros": torch.zeros(5, 131),
        "constant": torch.full((3, 129), -0.375),
        "mixed_constant": torch.stack((
            torch.zeros(129), torch.ones(129), torch.linspace(-2.0, 3.0, 129),
        )),
    }
    comparisons = {}
    for name, weight in cases.items():
        for bits, zero_point in itertools.product((2, 3, 4), (False, True)):
            actual = quantize_weight(
                weight, bits=bits, group_size=128, zero_point=zero_point,
            ).dequantized
            expected = independent_quantize(
                weight, bits=bits, group_size=128, zero_point=zero_point,
            )
            error = float((actual - expected).abs().max())
            comparisons[f"{name}.w{bits}.zp{int(zero_point)}"] = error
            if error > 1e-6:
                raise RuntimeError(f"quantizer cross-check failed: {name} {bits} {zero_point}: {error}")
    rejected = {}
    for name, value in {
        "nan": torch.tensor([[float("nan")]]),
        "positive_inf": torch.tensor([[float("inf")]]),
        "negative_inf": torch.tensor([[float("-inf")]]),
    }.items():
        try:
            quantize_weight(value, bits=4, group_size=128, zero_point=True)
        except ValueError:
            rejected[name] = True
        else:
            raise RuntimeError(f"quantizer accepted non-finite case {name}")
    return {
        "passed": True, "max_reference_error": max(comparisons.values()),
        "cases": comparisons, "nonfinite_rejected": rejected,
    }


def audit_loaded_model(model) -> dict:
    targets, inventory = build_inventory(model)
    if any(inventory["counts"].get(key) != value for key, value in EXPECTED_COUNTS.items()):
        raise RuntimeError(f"inventory mismatch: {inventory['counts']}")
    safe = select_scope(targets, "decoder_safe")
    original = safe[0].module
    original_weight = original.weight.detach().clone()
    with diagnostic_qdq_linears(model, [safe[0]], SCHEMES["a8_only"]):
        parent_path, key = safe[0].name.rsplit(".", 1)
        replacement = getattr(model.get_submodule(parent_path), key)
        if not isinstance(replacement, FakeQuantLinear):
            raise RuntimeError("diagnostic context did not install FakeQuantLinear")
    parent_path, key = safe[0].name.rsplit(".", 1)
    restored = getattr(model.get_submodule(parent_path), key)
    if restored is not original or not torch.equal(restored.weight, original_weight):
        raise RuntimeError("diagnostic context did not exactly restore the source Linear")
    return {
        "passed": True,
        "inventory": inventory,
        "context_restore": {"target": safe[0].name, "same_object": True, "weight_equal": True},
    }

