from __future__ import annotations

import dataclasses
import gc
import math
from collections import defaultdict
from pathlib import Path

import torch

from experiments.infinitevl_quant.inventory import decoder_layers
from experiments.infinitevl_quant.progress import StageProgress, atomic_json
from experiments.infinitevl_quant.quantization import quantize_weight
from experiments.vlm_quant_benchmark.data import MultimodalSample

from .adapters import ModelBundle
from .counterfactual import (
    FFNReplay,
    FFNWeights,
    replay_ffn,
    summarize_replays,
)
from .tracing_stage import (
    COUNTERFACTUAL_LAYERS,
    load_trace_artifact,
    trace_artifact_path,
)


COUNTERFACTUAL_BITS = (8, 4, 3, 2)
CAPTURE_PARITY_MAX_ABS_FLOOR = 5e-4
CAPTURE_PARITY_MAX_ABS_SIGNAL_FRACTION = 1e-6
CAPTURE_PARITY_RELATIVE_RMS = 2e-5


def _validate_capture_parity(summary: dict, *, context: str) -> dict:
    """Accept only FP32 GEMM-order noise, not a materially different replay.

    Captured activations are produced by one full-sequence GEMM whereas the
    retained-token replay is a smaller GEMM.  MKL/oneDNN may accumulate those
    products in a different order, so a single absolute threshold is not
    shape or signal-scale invariant.  We require both a tight relative-RMS
    bound and a finite absolute cap scaled from the reference maximum, with a
    small floor for ordinary-magnitude tensors.
    """

    rows = summary.get("reference_vs_captured")
    if not isinstance(rows, dict) or not rows:
        raise RuntimeError(f"missing capture parity statistics for {context}")
    distributions = summary.get("reference_distributions")
    if not isinstance(distributions, dict) or not distributions:
        raise RuntimeError(f"missing reference distributions for {context}")
    failures = {}
    for stage, values in rows.items():
        maximum = float(values["max_abs_error"])
        relative = float(values["relative_rms_error"])
        reference = distributions.get(stage)
        if not isinstance(reference, dict):
            raise RuntimeError(
                f"missing reference distribution for {context}/{stage}"
            )
        reference_maximum = float(reference["abs_max"])
        absolute_limit = max(
            CAPTURE_PARITY_MAX_ABS_FLOOR,
            CAPTURE_PARITY_MAX_ABS_SIGNAL_FRACTION * reference_maximum,
        )
        if (
            not math.isfinite(maximum)
            or not math.isfinite(relative)
            or not math.isfinite(reference_maximum)
            or reference_maximum < 0
            or maximum > absolute_limit
            or relative > CAPTURE_PARITY_RELATIVE_RMS
        ):
            failures[stage] = {
                "max_abs_error": maximum,
                "max_abs_limit": absolute_limit,
                "reference_abs_max": reference_maximum,
                "relative_rms_error": relative,
            }
    if failures:
        raise RuntimeError(
            f"offline replay parity failed for {context}: {failures}; "
            "limits=max_abs<=max("
            f"{CAPTURE_PARITY_MAX_ABS_FLOOR}, "
            f"{CAPTURE_PARITY_MAX_ABS_SIGNAL_FRACTION}*reference_abs_max), "
            f"relative_rms<={CAPTURE_PARITY_RELATIVE_RMS}"
        )
    return {
        "passed": True,
        "criterion": "signal_scaled_max_abs_and_relative_rms",
        "max_abs_floor": CAPTURE_PARITY_MAX_ABS_FLOOR,
        "max_abs_signal_fraction": CAPTURE_PARITY_MAX_ABS_SIGNAL_FRACTION,
        "relative_rms_limit": CAPTURE_PARITY_RELATIVE_RMS,
    }


def _prepared_rtn_weights(
    source: FFNWeights, *, bits: int, group_size: int,
) -> tuple[FFNWeights, dict]:
    values = {}
    audit = {}
    for projection in ("gate", "up", "down"):
        weight = getattr(source, f"{projection}_weight")
        quantized = quantize_weight(
            weight,
            bits=bits,
            group_size=group_size,
            zero_point=True,
        )
        values[f"{projection}_weight"] = quantized.dequantized
        values[f"{projection}_bias"] = getattr(source, f"{projection}_bias")
        audit[projection] = {
            "mse": quantized.mse,
            "nmse": quantized.nmse,
            "max_abs_error": quantized.max_abs_error,
            "groups_per_row": quantized.groups,
            "parameters": weight.numel(),
        }
    return FFNWeights(**values), audit


def _candidate_replay(
    ffn_input: torch.Tensor,
    prepared: FFNWeights,
    audit: dict,
    *,
    bits: int,
    group_size: int,
) -> FFNReplay:
    replay = replay_ffn(ffn_input, prepared, bits=None)
    return dataclasses.replace(
        replay,
        mode=f"asymmetric_groupwise_rtn_w{bits}_fp32_compute",
        bits=bits,
        group_size=group_size,
        weight_audit=audit,
    )


def _aggregate(rows: list[dict]) -> dict:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(
            row["weight_source"], row["trace_source"], row["bits"], row["layer"],
        )].append(row)
    result = []
    for key, values in sorted(groups.items()):
        weight_source, trace_source, bits, layer = key

        def mean(path: tuple[str, ...]) -> float:
            numbers = []
            for row in values:
                value = row
                for part in path:
                    value = value[part]
                numbers.append(float(value))
            return sum(numbers) / len(numbers)

        result.append({
            "weight_source": weight_source,
            "trace_source": trace_source,
            "bits": bits,
            "layer": layer,
            "samples": len(values),
            "product_nmse": mean(("summary", "stage_error", "product", "normalized_mse")),
            "down_nmse": mean(("summary", "stage_error", "down_output", "normalized_mse")),
            "quant_error_to_residual_rms": mean((
                "summary", "residual_geometry", "quant_error_to_residual_rms",
            )),
        })
    return {
        "analysis_scope": "offline_ffn_local_replay",
        "end_to_end_model_score": False,
        "rows": result,
    }


def run_model_counterfactual(
    bundle: ModelBundle,
    samples: list[MultimodalSample],
    *,
    progress: StageProgress,
    trace_root: Path,
    group_size: int,
) -> dict:
    """Replay both models' captured activations through one model's FFN weights."""

    weight_source = bundle.kind
    layers = list(decoder_layers(bundle.model))
    variant = f"weights_{weight_source}"
    for layer_index in COUNTERFACTUAL_LAYERS:
        weights = FFNWeights.from_mlp(layers[layer_index].mlp)
        trace_records = []
        for trace_source in ("infinitevl", "qwen"):
            for sample in samples:
                path = trace_artifact_path(
                    trace_root, trace_source, sample.id, layer_index,
                )
                metadata, trace = load_trace_artifact(path)
                if (
                    metadata["model_kind"] != trace_source
                    or metadata["sample_id"] != sample.id
                    or int(metadata["layer"]) != layer_index
                ):
                    raise RuntimeError(f"trace identity mismatch: {path}")
                trace_records.append((trace_source, sample, trace))

        references = {
            (trace_source, sample.id): replay_ffn(trace.ffn_input, weights)
            for trace_source, sample, trace in trace_records
        }
        for bits in COUNTERFACTUAL_BITS:
            prepared, weight_audit = _prepared_rtn_weights(
                weights, bits=bits, group_size=group_size,
            )
            for trace_source, sample, trace in trace_records:
                row_id = (
                    f"{weight_source}|{trace_source}|{sample.id}|"
                    f"layer{layer_index:02d}|w{bits}"
                )
                if progress.completed_sample(variant, row_id) is not None:
                    continue
                reference = references[(trace_source, sample.id)]
                candidate = _candidate_replay(
                    trace.ffn_input,
                    prepared,
                    weight_audit,
                    bits=bits,
                    group_size=group_size,
                )
                summary = summarize_replays(
                    reference,
                    candidate,
                    residual=trace.residual,
                    captured=trace if trace_source == weight_source else None,
                )
                if trace_source == weight_source:
                    summary["capture_parity"] = _validate_capture_parity(
                        summary,
                        context=(
                            f"{weight_source}/{sample.id}/layer{layer_index:02d}"
                        ),
                    )
                row = {
                    "id": row_id,
                    "dataset": sample.dataset,
                    "sample_id": sample.id,
                    "layer": layer_index,
                    "bits": bits,
                    "group_size": group_size,
                    "weight_source": weight_source,
                    "trace_source": trace_source,
                    "summary": summary,
                    "analysis_scope": "offline_ffn_local_replay",
                    "end_to_end_model_score": False,
                }
                progress.complete_sample(variant, row_id, row)
            del prepared
            gc.collect()
        del references, trace_records, weights
        gc.collect()
        print(
            f"[counterfactual/{weight_source}] layer {layer_index:02d} complete",
            flush=True,
        )

    rows = progress.rows(variant)
    summary = _aggregate(rows)
    summary.update({
        "weight_source": weight_source,
        "raw_rows": len(rows),
        "sentinel_layers": list(COUNTERFACTUAL_LAYERS),
        "bits": list(COUNTERFACTUAL_BITS),
    })
    atomic_json(progress.root / variant / "metrics.json", summary)
    return summary


def combine_counterfactual_summaries(
    progress: StageProgress,
) -> dict:
    rows = []
    for source in ("infinitevl", "qwen"):
        rows.extend(progress.rows(f"weights_{source}"))
    combined = _aggregate(rows)
    # The four source combinations separate weight-distribution sensitivity
    # from activation-distribution sensitivity.  This is a local replay and
    # therefore never promoted to an end-to-end quality result.
    atomic_json(progress.root / "summary.json", combined)
    return combined
