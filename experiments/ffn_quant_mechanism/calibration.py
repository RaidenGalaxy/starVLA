from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .shadow import ShadowObservation, observation_summary

EXPECTED_LAYERS = 36
ABSOLUTE_KEY = "absolute_error_rms"
RELATIVE_KEY = "relative_error_to_residual_rms"


class SampleObservationCollector:
    """Immediately reduce live tensors to JSON-ready layer summaries."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def __call__(self, observation: ShadowObservation) -> None:
        self.rows.append(observation_summary(observation))

    def finalize(self) -> list[dict[str, Any]]:
        layers = [int(row["layer"]) for row in self.rows]
        if layers != list(range(EXPECTED_LAYERS)):
            raise RuntimeError(
                "shadow calibration must observe exactly one ordered call per layer; "
                f"got {layers[:8]}... ({len(layers)} rows)"
            )
        return list(self.rows)


def _validate_rows(rows: Sequence[Mapping[str, Any]], *, model_kind: str) -> None:
    if not rows:
        raise ValueError(f"{model_kind} calibration has no rows")
    seen: set[tuple[str, int]] = set()
    per_sample: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        identifier = row.get("id")
        dataset = row.get("dataset")
        layer_rows = row.get("layers")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("calibration row has no sample id")
        if not isinstance(dataset, str) or not dataset:
            raise ValueError(f"calibration row {identifier} has no dataset")
        if not isinstance(layer_rows, list) or len(layer_rows) != EXPECTED_LAYERS:
            raise ValueError(f"calibration row {identifier} must contain 36 layers")
        for layer_row in layer_rows:
            layer = int(layer_row["layer"])
            key = (identifier, layer)
            if key in seen:
                raise ValueError(f"duplicate calibration cell {key}")
            seen.add(key)
            per_sample[identifier].add(layer)
            for metric in (ABSOLUTE_KEY, RELATIVE_KEY):
                value = float(layer_row[metric])
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"invalid {metric} for {model_kind}/{key}")
    expected = set(range(EXPECTED_LAYERS))
    if any(layers != expected for layers in per_sample.values()):
        raise ValueError(f"{model_kind} calibration has incomplete samples")


def _macro_layer_metric(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
) -> dict[int, dict[str, Any]]:
    cells: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        dataset = str(row["dataset"])
        for layer_row in row["layers"]:
            cells[(int(layer_row["layer"]), dataset)].append(float(layer_row[metric]))
    result = {}
    for layer in range(EXPECTED_LAYERS):
        by_dataset = {
            dataset: statistics.median(values)
            for (current, dataset), values in sorted(cells.items())
            if current == layer
        }
        if len(by_dataset) != 4:
            raise ValueError(f"layer {layer} metric {metric} must contain exactly four datasets")
        result[layer] = {
            "dataset_medians": by_dataset,
            "macro_dataset_median": statistics.mean(by_dataset.values()),
        }
    return result


def _ratio_record(numerator: float, denominator: float, clip: tuple[float, float]) -> dict:
    if numerator <= 0 or denominator <= 0:
        raise ValueError("transport ratios require positive calibration metrics")
    raw = numerator / denominator
    clipped = min(max(raw, clip[0]), clip[1])
    return {"raw": raw, "gamma": clipped, "clipped": clipped != raw}


def derive_transport_scales(
    infinitevl_rows: Sequence[Mapping[str, Any]],
    qwen_rows: Sequence[Mapping[str, Any]],
    *,
    clip: tuple[float, float] = (0.25, 4.0),
) -> dict[str, Any]:
    """Derive layer-only transport coefficients without evaluation leakage."""

    _validate_rows(infinitevl_rows, model_kind="infinitevl")
    _validate_rows(qwen_rows, model_kind="qwen")
    source_rows = {"infinitevl": infinitevl_rows, "qwen": qwen_rows}
    metrics = {
        model: {
            "absolute": _macro_layer_metric(rows, ABSOLUTE_KEY),
            "relative": _macro_layer_metric(rows, RELATIVE_KEY),
        }
        for model, rows in source_rows.items()
    }
    directions = {}
    for source, target in (("infinitevl", "qwen"), ("qwen", "infinitevl")):
        key = f"{source}_to_{target}"
        directions[key] = {}
        for mode in ("absolute", "relative"):
            layers = {}
            for layer in range(EXPECTED_LAYERS):
                numerator = float(metrics[target][mode][layer]["macro_dataset_median"])
                denominator = float(metrics[source][mode][layer]["macro_dataset_median"])
                layers[str(layer)] = _ratio_record(numerator, denominator, clip)
            directions[key][mode] = {
                "layers": layers,
                "clipped_layers": sum(row["clipped"] for row in layers.values()),
            }
    return {
        "format_version": 1,
        "definition": "fixed_layerwise_cross_model_error_transport",
        "calibration": "qig32_only",
        "aggregation": "within_dataset_median_then_four_dataset_macro_mean",
        "clip": list(clip),
        "metrics": {
            model: {
                mode: {str(layer): value for layer, value in layers.items()} for mode, layers in model_metrics.items()
            }
            for model, model_metrics in metrics.items()
        },
        "directions": directions,
        "confirm_data_used": False,
    }


def gamma_map(
    scale_artifact: Mapping[str, Any],
    model_kind: str,
    condition: str,
) -> dict[int, float]:
    if model_kind not in {"infinitevl", "qwen"}:
        raise ValueError(f"unknown model kind: {model_kind}")
    if condition == "B":
        return {layer: 0.0 for layer in range(EXPECTED_LAYERS)}
    if condition in {"W", "G", "M"}:
        return {layer: 1.0 for layer in range(EXPECTED_LAYERS)}
    if condition == "RG":
        condition = "R"
    mode = {"A": "absolute", "R": "relative"}.get(condition)
    if mode is None:
        raise ValueError(f"condition {condition!r} has no shadow gamma mapping")
    target = "qwen" if model_kind == "infinitevl" else "infinitevl"
    direction = scale_artifact["directions"][f"{model_kind}_to_{target}"][mode]
    return {layer: float(direction["layers"][str(layer)]["gamma"]) for layer in range(EXPECTED_LAYERS)}
