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

from .calibration import collect_calibration
from .config import ExperimentConfig, MethodName
from .data import MultimodalSample
from .evaluation import evaluate_multimodal, processor_inputs
from .methods import apply_method
from .reporting import atomic_json, write_report
from .selection import inventory, select_linears


CALIBRATION_METHODS = {
    MethodName.AWQ,
    MethodName.GPTQ,
    MethodName.SMOOTHQUANT,
    MethodName.TENDER,
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
    """Load BF16 checkpoint values into FP32 storage/compute without FP16 rounding."""
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(model_path)
    nonfinite = []
    dtypes: dict[str, int] = {}
    for name, parameter in model.named_parameters():
        dtypes[str(parameter.dtype)] = dtypes.get(str(parameter.dtype), 0) + parameter.numel()
        if not torch.isfinite(parameter).all():
            nonfinite.append(name)
    if nonfinite:
        raise ValueError(f"reference contains non-finite parameters: {nonfinite[:8]}")
    conversion = {
        "definition": "BF16 checkpoint values widened to FP32 for CPU storage and compute",
        "fp16_rounding_applied": False,
        "loaded_parameter_dtypes": dtypes,
        "nonfinite_parameters": nonfinite,
    }
    return model, processor, conversion


def calibration_batches(processor, samples: Iterable[MultimodalSample]):
    for sample in samples:
        # Teacher forcing deliberately includes the complete calibration answer.
        yield dict(processor_inputs(processor, sample, include_target=True))


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
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path, metrics_path = output_dir / "config.json", output_dir / "metrics.json"
    if resume and config_path.is_file() and metrics_path.is_file():
        previous = json.loads(config_path.read_text())
        if previous.get("config_hash") == config.digest:
            return json.loads(metrics_path.read_text())

    selected = select_linears(model, config.scope, strict_qwen_3b=strict_qwen_3b)
    module_inventory = inventory(selected)
    atomic_json(output_dir / "module_inventory.json", module_inventory)
    atomic_json(config_path, config.to_dict() | {
        "config_hash": config.digest,
        "environment": environment_record(),
    })

    stats = None
    if config.method in CALIBRATION_METHODS:
        if not calibration_samples:
            raise ValueError(f"{config.method.value} requires a disjoint calibration manifest")
        stats = collect_calibration(
            model,
            selected,
            calibration_batches(processor, calibration_samples[:config.calibration_samples]),
            reservoir_rows=32,
        )

    applied = apply_method(model, selected, config, stats)
    try:
        atomic_json(output_dir / "quantization_audit.json", applied.audit)
        metrics = evaluate_multimodal(
            model,
            processor,
            evaluation_samples,
            max_new_tokens=config.max_new_tokens,
            output_jsonl=output_dir / "generations.jsonl",
        )
        atomic_json(metrics_path, metrics)
        write_report(output_dir / "report.md", config.to_dict(), module_inventory, metrics, applied.audit)
        return metrics
    finally:
        applied.restore()
        gc.collect()
