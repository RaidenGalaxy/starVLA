"""Decoder FFN discovery, activation capture and Qwen2.5-VL input building."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
from PIL import Image

from .data import QuantSample

PROJECTION_NAMES = ("gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class DecoderFFN:
    index: int
    layer_name: str
    mlp_name: str
    norm_name: str
    layer: nn.Module
    mlp: nn.Module
    norm: nn.Module

    @property
    def linears(self) -> tuple[nn.Linear, nn.Linear, nn.Linear]:
        return tuple(getattr(self.mlp, name) for name in PROJECTION_NAMES)  # type: ignore[return-value]

    def named_mutable_parameters(self) -> Iterable[tuple[str, nn.Parameter]]:
        yield f"{self.norm_name}.weight", self.norm.weight
        for projection_name, linear in zip(PROJECTION_NAMES, self.linears):
            yield f"{self.mlp_name}.{projection_name}.weight", linear.weight


def discover_decoder_ffns(model: nn.Module, *, strict_qwen3b: bool = True) -> list[DecoderFFN]:
    """Discover text decoder layers by structure, excluding vision MLPs."""

    module_names = {id(module): name for name, module in model.named_modules()}
    found: list[DecoderFFN] = []
    seen_mlps: set[int] = set()
    for layer_name, layer in model.named_modules():
        mlp = getattr(layer, "mlp", None)
        norm = getattr(layer, "post_attention_layernorm", None)
        if not isinstance(mlp, nn.Module) or not isinstance(norm, nn.Module):
            continue
        if id(mlp) in seen_mlps or not hasattr(norm, "weight"):
            continue
        projections = [getattr(mlp, name, None) for name in PROJECTION_NAMES]
        if not all(isinstance(projection, nn.Linear) for projection in projections):
            continue
        seen_mlps.add(id(mlp))
        found.append(
            DecoderFFN(
                index=len(found),
                layer_name=layer_name,
                mlp_name=module_names[id(mlp)],
                norm_name=module_names[id(norm)],
                layer=layer,
                mlp=mlp,
                norm=norm,
            )
        )

    if not found:
        raise RuntimeError("no decoder FFNs with post_attention_layernorm were found")
    if strict_qwen3b:
        if len(found) != 36:
            raise RuntimeError(f"expected 36 Qwen2.5-VL-3B decoder layers, found {len(found)}")
        expected_shapes = ((11008, 2048), (11008, 2048), (2048, 11008))
        for entry in found:
            actual = tuple(tuple(linear.weight.shape) for linear in entry.linears)
            if actual != expected_shapes:
                raise RuntimeError(f"unexpected FFN shapes in layer {entry.index}: {actual}")
    return found


def resolve_scope(entries: list[DecoderFFN], scope: str) -> list[DecoderFFN]:
    if scope == "last1":
        return entries[-1:]
    if scope == "last4":
        return entries[-4:]
    if scope == "all":
        return entries
    raise ValueError(f"unsupported scope: {scope}")


def build_module_inventory(model: nn.Module, entries: list[DecoderFFN]) -> dict[str, Any]:
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    modules: list[dict[str, Any]] = []
    target_parameters = 0
    for entry in entries:
        for projection_name, linear in zip(PROJECTION_NAMES, entry.linears):
            count = linear.weight.numel()
            target_parameters += count
            modules.append(
                {
                    "layer": entry.index,
                    "name": f"{entry.mlp_name}.{projection_name}",
                    "shape": list(linear.weight.shape),
                    "parameters": count,
                }
            )
    return {
        "decoder_layers": len(entries),
        "linear_modules": len(modules),
        "target_parameters": target_parameters,
        "total_parameters": total_parameters,
        "target_fraction": target_parameters / total_parameters,
        "modules": modules,
    }


class ActivationCache:
    """Streaming channel statistics plus bounded reconstruction tokens."""

    def __init__(self, entries: list[DecoderFFN], max_tokens_per_layer: int = 256) -> None:
        self.entries = entries
        self.max_tokens_per_layer = max_tokens_per_layer
        self.mlp_samples: dict[int, list[torch.Tensor]] = {entry.index: [] for entry in entries}
        self.down_samples: dict[int, list[torch.Tensor]] = {entry.index: [] for entry in entries}
        self.mlp_abs_sum: dict[int, torch.Tensor] = {}
        self.down_abs_sum: dict[int, torch.Tensor] = {}
        self.mlp_count: dict[int, int] = {entry.index: 0 for entry in entries}
        self.down_count: dict[int, int] = {entry.index: 0 for entry in entries}
        self._handles: list[Any] = []

    @staticmethod
    def _flatten(value: torch.Tensor) -> torch.Tensor:
        return value.detach().float().reshape(-1, value.shape[-1]).cpu()

    def _record(self, index: int, value: torch.Tensor, *, down: bool) -> None:
        flat = self._flatten(value)
        sums = self.down_abs_sum if down else self.mlp_abs_sum
        counts = self.down_count if down else self.mlp_count
        samples = self.down_samples if down else self.mlp_samples
        sums[index] = sums.get(index, torch.zeros(flat.shape[-1], dtype=torch.float64)) + flat.abs().double().sum(0)
        counts[index] += flat.shape[0]
        used = sum(chunk.shape[0] for chunk in samples[index])
        if used < self.max_tokens_per_layer:
            samples[index].append(flat[: self.max_tokens_per_layer - used].clone())

    def __enter__(self) -> "ActivationCache":
        for entry in self.entries:
            self._handles.append(
                entry.mlp.register_forward_pre_hook(
                    lambda _module, args, index=entry.index: self._record(index, args[0], down=False)
                )
            )
            self._handles.append(
                entry.linears[2].register_forward_pre_hook(
                    lambda _module, args, index=entry.index: self._record(index, args[0], down=True)
                )
            )
        return self

    def __exit__(self, *_args: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def mean_abs(self, index: int, *, down: bool = False) -> torch.Tensor:
        sums = self.down_abs_sum if down else self.mlp_abs_sum
        counts = self.down_count if down else self.mlp_count
        if counts[index] == 0:
            raise RuntimeError(f"no activation statistics captured for layer {index}")
        return (sums[index] / counts[index]).float()

    def reconstruction_inputs(self, index: int) -> torch.Tensor:
        chunks = self.mlp_samples[index]
        if not chunks:
            raise RuntimeError(f"no reconstruction inputs captured for layer {index}")
        return torch.cat(chunks, dim=0)


def load_qwen_cpu(model_path: str) -> tuple[nn.Module, Any]:
    """Load Qwen2.5-VL directly, deliberately bypassing CUDA-only wrappers."""

    try:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Transformers with Qwen2.5-VL support is required; install requirements-cpu.txt first"
        ) from exc

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=56 * 56,
        max_pixels=112 * 112,
    )
    model.to("cpu").eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, processor


def _messages(sample: QuantSample, image: Image.Image, *, include_target: bool) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": sample.prompt},
            ],
        }
    ]
    if include_target:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": sample.target}]})
    return messages


def build_teacher_forced_inputs(processor: Any, sample: QuantSample) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Build one multimodal sample and identify assistant target positions."""

    with Image.open(Path(sample.image)) as source:
        image = source.convert("RGB")
        prompt_text = processor.apply_chat_template(
            _messages(sample, image, include_target=False), tokenize=False, add_generation_prompt=True
        )
        full_text = processor.apply_chat_template(
            _messages(sample, image, include_target=True), tokenize=False, add_generation_prompt=False
        )
        prompt_inputs = processor(text=[prompt_text], images=[image], padding=True, return_tensors="pt")
        full_inputs = processor(text=[full_text], images=[image], padding=True, return_tensors="pt")

    prompt_length = int(prompt_inputs["attention_mask"][0].sum().item())
    full_length = int(full_inputs["attention_mask"][0].sum().item())
    if full_length <= prompt_length:
        raise RuntimeError(f"assistant target produced no tokens for sample {sample.id}")
    target_mask = torch.zeros_like(full_inputs["input_ids"], dtype=torch.bool)
    target_mask[:, prompt_length:full_length] = True
    model_inputs = {key: value.to("cpu") for key, value in full_inputs.items() if torch.is_tensor(value)}
    return model_inputs, target_mask
