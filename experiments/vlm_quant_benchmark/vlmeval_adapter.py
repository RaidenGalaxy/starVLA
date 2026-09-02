"""VLMEvalKit adapter for official generation-based benchmark scoring.

Register ``Qwen25VLQuantized`` in VLMEvalKit's supported_VLM/config system, then
run its normal ``run.py --data ...`` command.  Keeping the adapter here avoids
forking VLMEvalKit or replacing its dataset-specific prompts and scorers.
"""
from __future__ import annotations

from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info

try:
    from vlmeval.vlm.qwen2_vl.prompt import Qwen2VLPromptMixin
    from vlmeval.vlm.base import BaseModel
except ImportError as error:  # pragma: no cover - optional dependency
    raise ImportError("Install the pinned open-compass/VLMEvalKit package to use this adapter") from error

from .all_methods import apply_method
from .calibration import calibration_cache_key, collect_calibration_cached
from .config import ExperimentConfig, MethodName, QuantScope
from .data import load_manifest, sample_manifest_digest
from .mbq import (
    collect_mbq_search_cached,
    mbq_cache_key,
    mbq_calibration_batches,
    mbq_statistics_cache_key,
)
from .official_awq import load_official_awq_dequantized
from .provenance import source_digest
from .runner import CALIBRATION_METHODS, calibration_batches, load_reference
from .selection import select_linears


class Qwen25VLQuantized(Qwen2VLPromptMixin, BaseModel):
    INTERLEAVE = True

    def __init__(
        self,
        model_path: str,
        method: str = "bf16_value_fp32_compute",
        scope: str = "decoder_canonical",
        calibration_manifest: str | None = None,
        group_size: int = 128,
        max_new_tokens: int = 128,
        weight_bits: int | None = None,
        activation_bits: int | None = None,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1280 * 28 * 28,
        seed: int = 42,
        strict_qwen_3b: bool = True,
        official_awq_path: str | None = None,
        calibration_cache_dir: str | None = None,
        model_revision_digest: str | None = None,
        calibration_samples: int = 128,
        omniquant_steps: int = 20,
        scheme_id: str = "",
        weight_granularity: str | None = None,
        experiment_config: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(use_custom_prompt=True, **kwargs)
        self.model, self.processor, self.reference_conversion = load_reference(model_path)
        method_name, quant_scope = MethodName(method), QuantScope(scope)
        activation = activation_bits
        if activation is None and method_name in {
            MethodName.SMOOTHQUANT, MethodName.LLM_INT8, MethodName.VECTORWISE_W8, MethodName.TENDER
        }:
            activation = 8
        bits = weight_bits if weight_bits is not None else (8 if activation else 4)
        if method_name == MethodName.BF16_VALUE_FP32_COMPUTE:
            bits, activation = None, None
        if experiment_config is not None:
            self.quant_config = ExperimentConfig.from_dict(experiment_config)
            method_name, quant_scope = (
                self.quant_config.method, self.quant_config.scope,
            )
            min_pixels = self.quant_config.min_pixels
            max_pixels = self.quant_config.max_pixels
            seed = self.quant_config.seed
        else:
            self.quant_config = ExperimentConfig(
                method_name, scope=quant_scope, weight_bits=bits,
                activation_bits=activation, group_size=group_size,
                zero_point=bits is not None and activation is None,
                weight_granularity=(
                    weight_granularity or (
                        "per_output_channel"
                        if method_name == MethodName.MBQ and activation == 8
                        else "groupwise"
                    )
                ),
                calibration_samples=calibration_samples,
                omniquant_steps=omniquant_steps,
                scheme_id=scheme_id,
                max_new_tokens=max_new_tokens,
                min_pixels=min_pixels, max_pixels=max_pixels, seed=seed,
            )
        if official_awq_path is not None:
            if experiment_config is not None:
                raise ValueError("official AWQ cannot use a local experiment config")
            if method_name != MethodName.BF16_VALUE_FP32_COMPUTE:
                raise ValueError("official_awq_path cannot be combined with a local method")
            self.official_awq_audit = load_official_awq_dequantized(
                self.model, Path(official_awq_path), group_size=group_size,
            )
            if self.official_awq_audit["packed_modules"] != 252:
                raise ValueError(
                    "official AWQ must contain exactly 252 decoder packed modules"
                )
            self.quant_config = ExperimentConfig(
                MethodName.BF16_VALUE_FP32_COMPUTE,
                scope=QuantScope.DECODER_CANONICAL,
                weight_bits=None,
                activation_bits=None,
                scheme_id="official_w4a16_g128_asym",
                max_new_tokens=max_new_tokens,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                seed=seed,
            )
            self.applied = None
            return
        selected = select_linears(self.model, quant_scope, strict_qwen_3b=strict_qwen_3b)
        stats = None
        if method_name in CALIBRATION_METHODS:
            if not calibration_manifest:
                raise ValueError(f"{method} requires calibration_manifest")
            calibration_attention = "sdpa"
            calibration = load_manifest(Path(calibration_manifest))
            calibration_subset = calibration[:self.quant_config.calibration_samples]
            cache_root = (
                Path(calibration_cache_dir)
                if calibration_cache_dir is not None
                else Path(calibration_manifest).parent / "_calibration_cache"
            )
            if method_name == MethodName.MBQ:
                cache_key = mbq_cache_key(
                    model_revision=model_revision_digest or source_digest(model_path),
                    calibration_digest=sample_manifest_digest(calibration_subset),
                    selected_names=[item.name for item in selected],
                    config=self.quant_config,
                    reservoir_rows=32,
                    attention_implementation=calibration_attention,
                )
                stats, self.calibration_cache_hit = collect_mbq_search_cached(
                    self.model,
                    selected,
                    mbq_calibration_batches(
                        self.processor,
                        calibration_subset,
                        min_pixels=min_pixels,
                        max_pixels=max_pixels,
                    ),
                    self.quant_config,
                    cache_dir=cache_root,
                    key=cache_key,
                    raw_key=mbq_statistics_cache_key(
                        model_revision=model_revision_digest or source_digest(model_path),
                        calibration_digest=sample_manifest_digest(calibration_subset),
                        selected_names=[item.name for item in selected],
                        reservoir_rows=32,
                        seed=self.quant_config.seed,
                        min_pixels=min_pixels,
                        max_pixels=max_pixels,
                        attention_implementation=calibration_attention,
                    ),
                    reservoir_rows=32,
                    attention_implementation=calibration_attention,
                )
            else:
                cache_key = calibration_cache_key(
                    model_revision=model_revision_digest or source_digest(model_path),
                    calibration_digest=sample_manifest_digest(calibration_subset),
                    selected_names=[item.name for item in selected],
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                    reservoir_rows=32,
                    seed=seed,
                    attention_implementation=calibration_attention,
                )
                stats, self.calibration_cache_hit = collect_calibration_cached(
                    self.model,
                    selected,
                    calibration_batches(
                        self.processor,
                        calibration_subset,
                        min_pixels=min_pixels,
                        max_pixels=max_pixels,
                    ),
                    cache_dir=cache_root,
                    key=cache_key,
                    reservoir_rows=32,
                    seed=seed,
                    attention_implementation=calibration_attention,
                )
        self.applied = apply_method(self.model, selected, self.quant_config, stats)
        if stats is not None:
            self.applied.audit["calibration_cache_hit"] = self.calibration_cache_hit
            self.applied.audit["calibration_attention_implementation"] = calibration_attention

    def generate_inner(self, message, dataset=None) -> str:
        content = []
        for item in message:
            if item["type"] == "image":
                content.append({
                    "type": "image",
                    "image": item["value"],
                    "min_pixels": self.quant_config.min_pixels,
                    "max_pixels": self.quant_config.max_pixels,
                })
            elif item["type"] == "text":
                content.append({"type": "text", "text": item["value"]})
            else:
                raise ValueError(f"unsupported VLMEvalKit message type: {item['type']}")
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        )
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, do_sample=False,
                max_new_tokens=self.quant_config.max_new_tokens,
            )
        trimmed = output[:, inputs.input_ids.shape[-1]:]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
