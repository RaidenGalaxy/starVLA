from __future__ import annotations

import gc
import json
import platform
import sys
from collections.abc import Iterable
from pathlib import Path

import torch
import transformers
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from .all_methods import apply_method
from .calibration import (
    calibration_cache_key,
    collect_calibration_cached,
)
from .config import ExperimentConfig, MethodName
from .data import MultimodalSample, sample_manifest_digest
from .evaluation import evaluate_multimodal, processor_inputs
from .mbq import (
    collect_mbq_search_cached,
    mbq_cache_key,
    mbq_calibration_batches,
)
from .provenance import completed_run, mark_complete
from .reporting import atomic_json, write_report
from .selection import inventory, select_linears


CALIBRATION_METHODS = {
    MethodName.MBQ,
    MethodName.AWQ,
    MethodName.GPTQ,
    MethodName.SMOOTHQUANT,
    MethodName.TENDER,
    MethodName.OMNIQUANT,
    MethodName.SPQR,
    MethodName.SQUEEZELLM,
}


def environment_record() -> dict:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "threads": torch.get_num_threads(),
    }


def load_reference(model_path: str):
    """Load BF16 checkpoint values into FP32 CPU storage and compute."""
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.float32,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
    nonfinite = []
    dtypes: dict[str, int] = {}
    for name, parameter in model.named_parameters():
        dtypes[str(parameter.dtype)] = dtypes.get(str(parameter.dtype), 0) + parameter.numel()
        if not torch.isfinite(parameter).all():
            nonfinite.append(name)
    if nonfinite:
        raise ValueError(f"reference contains non-finite parameters: {nonfinite[:8]}")
    conversion = {
        "baseline_id": "bf16_value_fp32_compute",
        "definition": "BF16 checkpoint values widened to FP32 for CPU storage and compute",
        "fp16_rounding_applied": False,
        "processor_use_fast": True,
        "loaded_parameter_dtypes": dtypes,
        "nonfinite_parameters": nonfinite,
    }
    return model, processor, conversion


def require_finite_quantized_state(model, selected) -> None:
    invalid = []
    for item in selected:
        module = model.get_submodule(item.name)
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor) or not torch.isfinite(weight).all():
            invalid.append(item.name)
    if invalid:
        raise ValueError(f"quantized modules have missing/non-finite weights: {invalid[:8]}")


def require_finite_metrics(value, path: str = "metrics") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            require_finite_metrics(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            require_finite_metrics(item, f"{path}[{index}]")
    elif isinstance(value, float) and not torch.isfinite(torch.tensor(value)):
        raise ValueError(f"non-finite metric at {path}")


def calibration_batches(
    processor,
    samples: Iterable[MultimodalSample],
    *,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
):
    for sample in samples:
        yield dict(processor_inputs(
            processor,
            sample,
            include_target=True,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        ))


def run_config(
    model,
    processor,
    config: ExperimentConfig,
    calibration_samples: list[MultimodalSample],
    evaluation_samples: list[MultimodalSample],
    output_dir: Path,
    *,
    strict_qwen_3b: bool = True,
    resume: bool = False,
    input_digests: dict[str, str] | None = None,
    calibration_cache_dir: Path | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    metrics_path = output_dir / "metrics.json"
    input_digests = input_digests or {
        "model": config.model_revision,
        "calibration_manifest": (
            sample_manifest_digest(calibration_samples) if calibration_samples else ""
        ),
        "evaluation_manifest": sample_manifest_digest(evaluation_samples),
        "processor_revision": config.processor_revision,
    }
    if resume and completed_run(
        output_dir,
        config_digest=config.digest,
        input_digests=input_digests,
    ):
        return json.loads(metrics_path.read_text())

    for stale_name in (
        ".complete", "metrics.json", "error.json", "generations.jsonl", "report.md",
    ):
        (output_dir / stale_name).unlink(missing_ok=True)

    selected = select_linears(model, config.scope, strict_qwen_3b=strict_qwen_3b)
    module_inventory = inventory(selected)
    atomic_json(output_dir / "module_inventory.json", module_inventory)
    atomic_json(config_path, config.to_dict() | {
        "config_hash": config.digest,
        "input_digests": input_digests,
        "environment": environment_record(),
    })

    stats, cache_hit = None, False
    if config.method in CALIBRATION_METHODS:
        if not calibration_samples:
            raise ValueError(f"{config.method.value} requires a disjoint calibration manifest")
        calibration_subset_digest = sample_manifest_digest(
            calibration_samples[:config.calibration_samples]
        )
        cache_root = calibration_cache_dir or output_dir.parent / "_calibration_cache"
        if config.method == MethodName.MBQ:
            calibration_attention = "sdpa"
            cache_key = mbq_cache_key(
                model_revision=input_digests["model"],
                calibration_digest=calibration_subset_digest,
                selected_names=[item.name for item in selected],
                config=config,
                reservoir_rows=32,
                attention_implementation=calibration_attention,
            )
            stats, cache_hit = collect_mbq_search_cached(
                model,
                selected,
                mbq_calibration_batches(
                    processor,
                    calibration_samples[:config.calibration_samples],
                    min_pixels=config.min_pixels,
                    max_pixels=config.max_pixels,
                ),
                config,
                cache_dir=cache_root,
                key=cache_key,
                reservoir_rows=32,
                attention_implementation=calibration_attention,
            )
        else:
            cache_key = calibration_cache_key(
                model_revision=input_digests["model"],
                calibration_digest=calibration_subset_digest,
                selected_names=[item.name for item in selected],
                min_pixels=config.min_pixels,
                max_pixels=config.max_pixels,
                reservoir_rows=32,
                seed=config.seed,
            )
            stats, cache_hit = collect_calibration_cached(
                model,
                selected,
                calibration_batches(
                    processor,
                    calibration_samples[:config.calibration_samples],
                    min_pixels=config.min_pixels,
                    max_pixels=config.max_pixels,
                ),
                cache_dir=cache_root,
                key=cache_key,
                reservoir_rows=32,
                seed=config.seed,
            )

    applied = apply_method(model, selected, config, stats)
    try:
        require_finite_quantized_state(model, selected)
        applied.audit["implementation_tier"] = str(config.implementation_tier)
        applied.audit["calibration_cache_hit"] = cache_hit
        atomic_json(output_dir / "quantization_audit.json", applied.audit)
        metrics = evaluate_multimodal(
            model,
            processor,
            evaluation_samples,
            max_new_tokens=config.max_new_tokens,
            output_jsonl=output_dir / "generations.jsonl",
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
        )
        require_finite_metrics(metrics)
        atomic_json(metrics_path, metrics)
        write_report(
            output_dir / "report.md",
            config.to_dict(),
            module_inventory,
            metrics,
            applied.audit,
        )
        mark_complete(
            output_dir,
            config_digest=config.digest,
            input_digests=input_digests,
        )
        return metrics
    finally:
        applied.restore()
        gc.collect()
