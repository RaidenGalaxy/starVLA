from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from safetensors import safe_open

from experiments.infinitevl_quant.progress import atomic_json

from .config import EXPECTED_HIDDEN_LAYERS, FFN_PROJECTIONS


MAX_DISTRIBUTION_SAMPLE = 1_000_000


def _weight_map(root: Path) -> dict[str, str]:
    path = root / "model.safetensors.index.json"
    payload = json.loads(path.read_text())
    mapping = payload.get("weight_map")
    if not isinstance(mapping, dict):
        raise ValueError(f"invalid weight map: {path}")
    return {str(name): str(shard) for name, shard in mapping.items()}


def _read_tensor(root: Path, shard: str, name: str) -> torch.Tensor:
    with safe_open(root / shard, framework="pt", device="cpu") as handle:
        value = handle.get_tensor(name)
    if value.ndim != 2 or not value.is_floating_point():
        raise TypeError(f"{name} is not a floating matrix")
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains NaN/Inf")
    return value.float()


def _distribution(value: torch.Tensor) -> dict:
    flat = value.reshape(-1)
    stride = max(1, math.ceil(flat.numel() / MAX_DISTRIBUTION_SAMPLE))
    sample = flat[::stride][:MAX_DISTRIBUTION_SAMPLE]
    absolute = sample.abs()
    mean = sample.mean()
    centered = sample - mean
    variance = centered.square().mean()
    group = value.reshape(value.shape[0], -1, 128)
    spans = group.amax(dim=-1) - group.amin(dim=-1)
    rms = math.sqrt(float(value.square().mean()))
    return {
        "rms": rms,
        "abs_max": float(value.abs().max()),
        "sample_abs_p99": float(torch.quantile(absolute, 0.99)),
        "sample_abs_p999": float(torch.quantile(absolute, 0.999)),
        "sample_kurtosis": (
            float(centered.pow(4).mean() / variance.square())
            if float(variance) > 0 else 0.0
        ),
        "max_to_rms": float(value.abs().max()) / max(rms, torch.finfo(torch.float32).tiny),
        "g128_span_mean": float(spans.mean()),
        "g128_span_p99": float(torch.quantile(spans.flatten(), 0.99)),
        "sample_elements": sample.numel(),
        "sample_stride": stride,
    }


@torch.inference_mode()
def analyze_ffn_weight_pair(
    infinitevl_root: Path,
    qwen_root: Path,
    *,
    output_path: Path | None = None,
) -> dict:
    """Compare matched FFN checkpoint values one tensor at a time on CPU."""

    roots = {
        "infinitevl": infinitevl_root.resolve(),
        "qwen": qwen_root.resolve(),
    }
    maps = {kind: _weight_map(root) for kind, root in roots.items()}
    rows = []
    global_dot = global_left_power = global_right_power = global_delta_power = 0.0
    for layer in range(EXPECTED_HIDDEN_LAYERS):
        for projection in FFN_PROJECTIONS:
            name = f"model.layers.{layer}.mlp.{projection}.weight"
            missing = [kind for kind in roots if name not in maps[kind]]
            if missing:
                raise KeyError(f"{name} missing from {missing}")
            left = _read_tensor(roots["infinitevl"], maps["infinitevl"][name], name)
            right = _read_tensor(roots["qwen"], maps["qwen"][name], name)
            if left.shape != right.shape:
                raise RuntimeError(f"matched FFN shape mismatch at {name}")
            left_flat, right_flat = left.reshape(-1), right.reshape(-1)
            dot = float(torch.dot(left_flat, right_flat))
            left_power = float(torch.dot(left_flat, left_flat))
            right_power = float(torch.dot(right_flat, right_flat))
            delta_power = float((left_flat - right_flat).square().sum())
            denominator = math.sqrt(left_power * right_power)
            rows.append({
                "name": name,
                "layer": layer,
                "projection": projection,
                "shape": list(left.shape),
                "parameters": left.numel(),
                "cosine_similarity": dot / denominator if denominator else 1.0,
                "relative_l2_infinitevl_minus_qwen": math.sqrt(
                    delta_power / max(right_power, torch.finfo(torch.float32).tiny)
                ),
                "infinitevl": _distribution(left),
                "qwen": _distribution(right),
                "_dot": dot,
                "_left_power": left_power,
                "_right_power": right_power,
                "_delta_power": delta_power,
            })
            global_dot += dot
            global_left_power += left_power
            global_right_power += right_power
            global_delta_power += delta_power
            del left, right, left_flat, right_flat
    if len(rows) != 108:
        raise RuntimeError(f"expected 108 matched FFN tensors, got {len(rows)}")
    by_projection = {}
    for projection in FFN_PROJECTIONS:
        subset = [row for row in rows if row["projection"] == projection]
        by_projection[projection] = {
            "tensors": len(subset),
            "mean_cosine_similarity": sum(row["cosine_similarity"] for row in subset) / len(subset),
            "mean_relative_l2": sum(
                row["relative_l2_infinitevl_minus_qwen"] for row in subset
            ) / len(subset),
        }
    clean_rows = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in rows
    ]
    result = {
        "analysis": "matched_ffn_checkpoint_weight_pair",
        "compute_dtype": "float32_cpu",
        "linears": len(rows),
        "parameters_per_model": sum(row["parameters"] for row in rows),
        "global_cosine_similarity": global_dot / math.sqrt(
            global_left_power * global_right_power
        ),
        "global_relative_l2_infinitevl_minus_qwen": math.sqrt(
            global_delta_power / global_right_power
        ),
        "by_projection": by_projection,
        "rows": clean_rows,
    }
    if output_path is not None:
        atomic_json(output_path, result)
    return result


def summarize_rtn_weight_audits(rtn_root: Path) -> dict:
    """Aggregate already-written RTN weight NMSE without re-quantizing."""

    result = {}
    for kind in ("infinitevl", "qwen"):
        kind_rows = {}
        for bits in (8, 4, 3, 2):
            split = "smoke4" if bits == 2 else "screen32"
            variant = f"{kind}__{split}__rtn_w{bits}a16_ffn"
            path = rtn_root / "quantization_audits" / kind / f"{variant}.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            audit = json.loads(path.read_text())
            modules = audit.get("modules", {})
            if len(modules) != 108:
                raise RuntimeError(f"{variant} has {len(modules)} modules, expected 108")
            values = [float(row["weight_nmse"]) for row in modules.values()]
            by_projection = {}
            for projection in FFN_PROJECTIONS:
                selected = [
                    float(row["weight_nmse"])
                    for name, row in modules.items()
                    if name.endswith(f".{projection}")
                ]
                by_projection[projection] = sum(selected) / len(selected)
            kind_rows[f"w{bits}a16"] = {
                "modules": len(values),
                "mean_weight_nmse": sum(values) / len(values),
                "min_weight_nmse": min(values),
                "max_weight_nmse": max(values),
                "by_projection_mean_nmse": by_projection,
            }
        result[kind] = kind_rows
    return {
        "analysis": "matched_rtn_static_weight_error",
        "source": "formal_rtn_quantization_audits",
        "models": result,
    }
