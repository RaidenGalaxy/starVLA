from __future__ import annotations

import os
import time
from pathlib import Path

import torch

from experiments.infinitevl_quant.evaluation import _teacher_inputs
from experiments.infinitevl_quant.inventory import decoder_layers
from experiments.infinitevl_quant.progress import StageProgress, atomic_json, safe_id
from experiments.vlm_quant_benchmark.data import MultimodalSample

from .adapters import ModelBundle
from .trace import FFNTrace, QwenFFNTracer, trace_summary


COUNTERFACTUAL_LAYERS = (0, 1, 8, 17, 26, 35)
TRACE_MAX_TOKENS = 256


def trace_artifact_path(
    progress: StageProgress | Path, kind: str, sample_id: str, layer: int,
) -> Path:
    root = progress.root if isinstance(progress, StageProgress) else progress
    return (
        root / "artifacts" / kind / safe_id(sample_id)
        / f"layer_{layer:02d}.pt"
    )


def _atomic_torch_save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def save_trace_artifact(
    path: Path,
    trace: FFNTrace,
    *,
    model_kind: str,
    sample_id: str,
    layer: int,
) -> None:
    _atomic_torch_save(path, {
        "format_version": 1,
        "model_kind": model_kind,
        "sample_id": sample_id,
        "layer": layer,
        "layer_name": trace.layer_name,
        "call_index": trace.call_index,
        **{
            name: value.detach().float().cpu()
            for name, value in trace.stage_tensors().items()
        },
    })


def load_trace_artifact(path: Path) -> tuple[dict, FFNTrace]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise RuntimeError(f"unsupported trace artifact: {path}")
    trace = FFNTrace(
        layer_name=str(payload["layer_name"]),
        call_index=int(payload["call_index"]),
        **{
            name: payload[name].float()
            for name in (
                "ffn_input", "gate", "up", "product", "down_output", "residual",
            )
        },
    )
    return payload, trace


def _sample_artifacts_complete(
    progress: StageProgress, kind: str, sample_id: str,
) -> bool:
    return all(
        trace_artifact_path(progress, kind, sample_id, layer).is_file()
        for layer in COUNTERFACTUAL_LAYERS
    )


def run_model_trace(
    bundle: ModelBundle,
    samples: list[MultimodalSample],
    *,
    progress: StageProgress,
    min_pixels: int,
    max_pixels: int,
) -> dict:
    """Trace all 36 FFNs and persist bounded tensors for six sentinel layers."""

    layers = list(decoder_layers(bundle.model))
    if len(layers) != 36:
        raise RuntimeError(f"{bundle.kind} expected 36 decoder layers, got {len(layers)}")
    named_layers = [
        (f"{bundle.decoder_prefix}.{index}", layer)
        for index, layer in enumerate(layers)
    ]
    name_to_layer = {name: index for index, (name, _layer) in enumerate(named_layers)}
    variant = f"{bundle.kind}__trace_smoke4"
    bundle.model.eval()
    for sample in samples:
        completed = progress.completed_sample(variant, sample.id)
        if completed is not None and _sample_artifacts_complete(
            progress, bundle.kind, sample.id,
        ):
            continue
        summaries: list[dict] = []
        artifact_rows: list[dict] = []

        def consume(trace: FFNTrace) -> None:
            layer = name_to_layer[trace.layer_name]
            summaries.append({"layer": layer, **trace_summary(trace)})
            if layer in COUNTERFACTUAL_LAYERS:
                path = trace_artifact_path(
                    progress, bundle.kind, sample.id, layer,
                )
                save_trace_artifact(
                    path,
                    trace,
                    model_kind=bundle.kind,
                    sample_id=sample.id,
                    layer=layer,
                )
                artifact_rows.append({
                    "layer": layer,
                    "path": str(path),
                    "tokens": int(trace.ffn_input.reshape(-1, trace.ffn_input.shape[-1]).shape[0]),
                })

        _prompt, teacher, _labels, _positions = _teacher_inputs(
            bundle.processor, sample, min_pixels, max_pixels,
        )
        started = time.perf_counter()
        tracer = QwenFFNTracer(
            named_layers,
            max_tokens=TRACE_MAX_TOKENS,
            max_calls_per_layer=0,
            on_trace=consume,
        )
        with tracer, torch.inference_mode():
            output = bundle.model(**teacher, use_cache=False, return_dict=True)
        elapsed = time.perf_counter() - started
        if len(summaries) != 36 or any(
            row["calls"] != 1 for row in tracer.retention_audit().values()
        ):
            raise RuntimeError(
                f"{bundle.kind}/{sample.id} incomplete trace: "
                f"summaries={len(summaries)}, calls={tracer.retention_audit()}"
            )
        if not _sample_artifacts_complete(progress, bundle.kind, sample.id):
            raise RuntimeError(f"missing sentinel trace artifacts for {sample.id}")
        row = {
            "id": sample.id,
            "dataset": sample.dataset,
            "model_kind": bundle.kind,
            "sequence_tokens": int(teacher["input_ids"].shape[-1]),
            "retained_tokens_per_layer": TRACE_MAX_TOKENS,
            "teacher_seconds": elapsed,
            "layer_summaries": sorted(summaries, key=lambda value: value["layer"]),
            "artifacts": sorted(artifact_rows, key=lambda value: value["layer"]),
            "analysis_scope": "observed_ffn_activations",
            "end_to_end_model_score": False,
        }
        progress.complete_sample(variant, sample.id, row)
        print(f"[trace/{bundle.kind}] {sample.id} {elapsed:.1f}s", flush=True)
        del output, teacher, _prompt, _labels

    rows = progress.rows(variant)
    result = {
        "model_kind": bundle.kind,
        "samples": len(rows),
        "layers_per_sample": 36,
        "sentinel_layers": list(COUNTERFACTUAL_LAYERS),
        "max_tokens_per_trace": TRACE_MAX_TOKENS,
        "analysis_scope": "observed_ffn_activations",
        "rows_path": str(progress.root / variant / "samples"),
    }
    atomic_json(progress.root / variant / "metrics.json", result)
    return result
