"""Modality-separated reconstruction residuals for formal MBQ diagnostics."""
from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import nn

from .calibration import release_cpu_memory, temporary_attention_implementation
from .mbq import _decoder_layers
from .sensitivity import causal_predictor_mask


RESIDUAL_FORMAT_VERSION = 1


def residual_summary(reference: torch.Tensor, candidate: torch.Tensor, mask: torch.Tensor) -> dict:
    if reference.shape != candidate.shape:
        raise ValueError(f"residual shape mismatch: {reference.shape} != {candidate.shape}")
    if reference.ndim != 3 or mask.shape != reference.shape[:2]:
        raise ValueError("residual tensors must be [batch, sequence, hidden]")
    values = (candidate.float() - reference.float())[mask.bool()]
    if not values.numel():
        return {"count": 0, "abs_sum": 0.0, "sq_sum": 0.0, "mae": math.nan, "rmse": math.nan}
    flat = values.reshape(-1)
    abs_sum = float(flat.abs().sum().item())
    sq_sum = float(flat.square().sum().item())
    count = flat.numel()
    return {
        "count": count,
        "abs_sum": abs_sum,
        "sq_sum": sq_sum,
        "mae": abs_sum / count,
        "rmse": math.sqrt(sq_sum / count),
    }


@torch.inference_mode()
def capture_reconstruction_outputs(
    model: nn.Module,
    batch: dict,
    *,
    layers_to_test: tuple[int, ...],
    anchor_names: Iterable[str],
    attention_implementation: str = "sdpa",
) -> dict[str, torch.Tensor]:
    """Capture decoder block and selected linear outputs for one sample."""
    named = dict(model.named_modules())
    decoder_layers = _decoder_layers(model)
    modules: dict[str, nn.Module] = {}
    for layer in layers_to_test:
        if layer < 0 or layer >= len(decoder_layers):
            raise ValueError(f"decoder layer {layer} is out of range")
        modules[f"block:{layer}"] = decoder_layers[layer]
    for name in anchor_names:
        if name not in named:
            raise ValueError(f"anchor module is absent: {name}")
        modules[f"linear:{name}"] = named[name]
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def hook(key: str):
        def capture(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captured[key] = tensor.detach().float().cpu().clone()
        return capture

    for key, module in modules.items():
        handles.append(module.register_forward_hook(hook(key)))
    try:
        with temporary_attention_implementation(model, attention_implementation):
            model(
                **batch["model_inputs"], use_cache=False, return_dict=True,
                output_attentions=False, output_hidden_states=False,
            )
    finally:
        for handle in handles:
            handle.remove()
    missing = set(modules) - set(captured)
    if missing:
        raise RuntimeError(f"reconstruction hooks did not fire: {sorted(missing)}")
    release_cpu_memory()
    return captured


def reconstruction_rows(
    batch: dict,
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
    *,
    method: str,
) -> list[dict]:
    masks = {
        "all_tokens": torch.ones_like(batch["vision_mask"], dtype=torch.bool),
        "vision": batch["vision_mask"].bool(),
        "answer_language": causal_predictor_mask(batch["language_mask"].bool()),
        "prompt_language": batch["prompt_language_mask"].bool(),
    }
    rows = []
    for key in sorted(reference):
        if key not in candidate:
            raise ValueError(f"candidate capture is missing {key}")
        location, value = key.split(":", 1)
        layer = int(value) if location == "block" else int(value.split(".layers.", 1)[1].split(".", 1)[0])
        projection = "block_output"
        if location == "linear":
            projection = "o_proj" if value.endswith(".self_attn.o_proj") else "down_proj"
        for modality, mask in masks.items():
            summary = residual_summary(reference[key], candidate[key], mask)
            rows.append({
                "format_version": RESIDUAL_FORMAT_VERSION,
                "id": str(batch["id"]),
                "dataset": str(batch.get("source", "unknown")),
                "method": method,
                "location": "decoder_block_output" if location == "block" else "linear_output",
                "module": value,
                "layer": layer,
                "projection": projection,
                "modality": modality,
                **summary,
            })
    return rows


def aggregate_mbq_block_rms(rows: Iterable[dict]) -> dict[int, float]:
    totals: dict[int, list[float]] = {}
    for row in rows:
        if (
            row.get("method") == "mbq_current"
            and row.get("location") == "decoder_block_output"
            and row.get("modality") == "all_tokens"
        ):
            value = totals.setdefault(int(row["layer"]), [0.0, 0.0])
            value[0] += float(row["sq_sum"])
            value[1] += int(row["count"])
    if not totals or any(count <= 0 for _sq, count in totals.values()):
        raise ValueError("no finite MBQ block residuals are available")
    return {layer: math.sqrt(sq / count) for layer, (sq, count) in sorted(totals.items())}
