"""Numerically safe adapter for strict AWQ equivalence checks."""

from __future__ import annotations

import time

import torch

from .awq import AWQLayerResult, _normalized_mse, _quantize_mlp_, search_awq_layer as _search_awq_layer
from .modeling import DecoderFFN


@torch.no_grad()
def search_awq_layer(
    entry: DecoderFFN,
    reconstruction_inputs: torch.Tensor,
    input_mean_abs: torch.Tensor,
    down_mean_abs: torch.Tensor,
    *,
    group_size: int = 128,
    equivalence_tolerance: float = 1e-5,
) -> AWQLayerResult:
    """Run AWQ and fall back to identity only for strict equivalence failure.

    The base implementation restores the layer's original parameters before
    raising either equivalence error. Other RuntimeErrors are propagated.
    """

    started = time.monotonic()
    try:
        return _search_awq_layer(
            entry,
            reconstruction_inputs,
            input_mean_abs,
            down_mean_abs,
            group_size=group_size,
            equivalence_tolerance=equivalence_tolerance,
        )
    except RuntimeError as exc:
        if "scaling is not equivalent" not in str(exc):
            raise

    device = entry.linears[0].weight.device
    dtype = entry.linears[0].weight.dtype
    inputs = reconstruction_inputs.to(device=device, dtype=dtype)
    reference = entry.mlp(inputs).detach().float()
    _quantize_mlp_(entry, group_size)
    rtn_error = _normalized_mse(reference, entry.mlp(inputs))
    return AWQLayerResult(
        layer=entry.index,
        input_scale=torch.ones_like(input_mean_abs).cpu(),
        internal_scale=torch.ones_like(down_mean_abs).cpu(),
        rtn_normalized_mse=rtn_error,
        awq_normalized_mse=rtn_error,
        equivalence_max_error=0.0,
        input_search=[
            {
                "candidate": "identity_fallback",
                "normalized_mse": rtn_error,
                "reason": "strict_fp32_equivalence_tolerance",
            }
        ],
        internal_search=[{"candidate": "identity_fallback", "normalized_mse": rtn_error}],
        elapsed_seconds=time.monotonic() - started,
    )

