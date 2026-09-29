from __future__ import annotations

import math
from collections import defaultdict

import torch

from experiments.vlm_quant_benchmark.evaluation import processor_inputs

from .cpu_backend import capture_gdn_states


PREFIXES = (64, 128, 256, 512)
EXPECTED_GDN_LAYERS = 27


def state_probe_token_count(
    processor, sample, *, min_pixels: int, max_pixels: int,
) -> int:
    inputs = processor_inputs(
        processor, sample, False, min_pixels=min_pixels, max_pixels=max_pixels,
    )
    return int(inputs["input_ids"].shape[-1])


def _validate_capture(states: dict, sequence_tokens: int) -> None:
    if sequence_tokens < max(PREFIXES):
        raise ValueError(
            f"state-probe prompt has {sequence_tokens} tokens; need at least {max(PREFIXES)}"
        )
    if len(states) != EXPECTED_GDN_LAYERS:
        raise RuntimeError(
            f"captured {len(states)} GDN layers, expected {EXPECTED_GDN_LAYERS}"
        )
    expected = set(PREFIXES)
    incomplete = {
        layer: sorted(set(prefixes) ^ expected)
        for layer, prefixes in states.items()
        if set(prefixes) != expected
    }
    if incomplete:
        raise RuntimeError(f"incomplete GDN prefix capture: {incomplete}")


def collect_states(model, processor, sample, *, min_pixels: int, max_pixels: int):
    inputs = processor_inputs(
        processor, sample, False, min_pixels=min_pixels, max_pixels=max_pixels,
    )
    with capture_gdn_states(model, PREFIXES) as states:
        with torch.inference_mode():
            model(**inputs, use_cache=False, return_dict=True)
    sequence_tokens = int(inputs["input_ids"].shape[-1])
    _validate_capture(states, sequence_tokens)
    return states, sequence_tokens


def compare_states(reference: dict, candidate: dict) -> dict:
    rows = []
    if set(reference) != set(candidate):
        raise ValueError("GDN state layer set differs from BF16 reference")
    if len(reference) != EXPECTED_GDN_LAYERS:
        raise ValueError(
            f"state comparison has {len(reference)} layers, expected {EXPECTED_GDN_LAYERS}"
        )
    for layer in sorted(reference):
        if set(reference[layer]) != set(PREFIXES):
            raise ValueError(f"BF16 layer {layer} does not contain all required prefixes")
        if set(candidate[layer]) != set(PREFIXES):
            raise ValueError(f"candidate layer {layer} does not contain all required prefixes")
        for prefix in PREFIXES:
            left, right = reference[layer][prefix].float(), candidate[layer][prefix].float()
            delta = right - left
            norm = torch.linalg.vector_norm(left).clamp_min(torch.finfo(torch.float32).tiny)
            cosine = torch.nn.functional.cosine_similarity(
                left.flatten(), right.flatten(), dim=0, eps=1e-12,
            )
            rows.append({
                "layer": layer, "prefix": prefix,
                "normalized_frobenius": float(torch.linalg.vector_norm(delta) / norm),
                "cosine_drift": float(1.0 - cosine),
                "max_abs_error": float(delta.abs().max()),
            })
    by_prefix = defaultdict(list)
    for row in rows:
        by_prefix[row["prefix"]].append(row)
    if set(by_prefix) != set(PREFIXES) or any(
        len(values) != EXPECTED_GDN_LAYERS for values in by_prefix.values()
    ):
        raise RuntimeError("state comparison did not produce the exact 27x4 matrix")
    summary = {
        str(prefix): {
            key: sum(row[key] for row in values) / len(values)
            for key in ("normalized_frobenius", "cosine_drift", "max_abs_error")
        } for prefix, values in sorted(by_prefix.items())
    }
    x = [math.log2(prefix) for prefix in sorted(by_prefix)]
    y = [summary[str(prefix)]["normalized_frobenius"] for prefix in sorted(by_prefix)]
    auc = sum(
        (x[index] - x[index - 1]) * (y[index] + y[index - 1]) * 0.5
        for index in range(1, len(x))
    )
    return {"rows": rows, "by_prefix": summary, "state_drift_auc": auc}
