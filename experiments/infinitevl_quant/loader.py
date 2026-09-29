from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import torch
import transformers
from huggingface_hub import snapshot_download
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

from .config import MODEL_ID, MODEL_REVISION
from .cpu_backend import (
    CPUFusedRMSNormGated,
    CPURMSNorm,
    CPUShortConvolution,
    install_fla_cpu_shims,
)
from .progress import file_sha256


REQUIRED_MODEL_FILES = (
    "config.json",
    "configuration_infinitevl.py",
    "modeling_infinitevl.py",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
)
PRE_ROUTED_SLIDING_MASK = "_infinitevl_cpu_pre_routed_sliding_mask"


def sliding_cache_mask_sizes(
    cumulative_length: int,
    sliding_window: int,
    cache_position: torch.Tensor,
) -> tuple[int, int]:
    """Return SWA mask dimensions both before and after cache ``update``.

    Transformers asks for mask dimensions before the attention layer updates
    its cache, while InfiniteVL's eager attention asks again after that update.
    The remote implementation assumed only the latter phase.
    """
    if cache_position is None or cache_position.ndim != 1:
        raise ValueError("cache_position must be a one-dimensional tensor")
    query_length = int(cache_position.numel())
    if query_length == 0:
        return min(int(cumulative_length), int(sliding_window) - 1), max(
            int(cumulative_length) - int(sliding_window) + 1, 0,
        )
    expected_post_update = int(cache_position[-1].item()) + 1
    already_updated = int(cumulative_length) >= expected_post_update
    past_length = max(
        int(cumulative_length) - (query_length if already_updated else 0), 0,
    )
    offset = max(past_length - int(sliding_window) + 1, 0)
    cached_past = min(past_length, max(int(sliding_window) - 1, 0))
    return cached_past + query_length, offset


def ensure_snapshot(
    model_id: str = MODEL_ID,
    revision: str = MODEL_REVISION,
    *,
    local_files_only: bool = False,
) -> Path:
    path = Path(snapshot_download(
        model_id,
        revision=revision,
        local_files_only=local_files_only,
        max_workers=2,
        etag_timeout=60,
    ))
    missing = [name for name in REQUIRED_MODEL_FILES if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"InfiniteVL snapshot is incomplete: {missing}")
    index = json.loads((path / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    missing_shards = [name for name in shards if not (path / name).is_file()]
    if missing_shards:
        raise FileNotFoundError(f"InfiniteVL weight shards are incomplete: {missing_shards}")
    return path


def snapshot_provenance(path: Path, *, hash_weights: bool = True) -> dict:
    index = json.loads((path / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    source_names = [
        "config.json", "configuration_infinitevl.py", "modeling_infinitevl.py",
        "modular_infinitevl.py", "processing_infinitevl.py",
        "preprocessor_config.json", "tokenizer_config.json",
        "model.safetensors.index.json",
    ]
    source_hashes = {
        name: file_sha256(path / name) for name in source_names if (path / name).is_file()
    }
    weight_hashes = {
        name: file_sha256(path / name) if hash_weights else "not_computed"
        for name in shards
    }
    combined = hashlib.sha256()
    for name, digest in sorted(source_hashes.items()):
        combined.update(name.encode())
        combined.update(digest.encode())
    for name, digest in sorted(weight_hashes.items()):
        combined.update(name.encode())
        combined.update(digest.encode())
    return {
        "snapshot": str(path),
        "source_sha256": source_hashes,
        "weight_sha256": weight_hashes,
        "combined_sha256": combined.hexdigest(),
        "weight_bytes": sum((path / name).stat().st_size for name in shards),
        "weight_shards": shards,
    }


def _set_eager(config) -> None:
    for current in (
        config,
        getattr(config, "text_config", None),
        getattr(config, "vision_config", None),
    ):
        if current is not None:
            current._attn_implementation = "eager"
            current._attn_implementation_internal = "eager"


def _install_sliding_attention_masks(model: nn.Module) -> int:
    """Route the decoder-created sliding mask to eager SWA layers.

    The pinned source creates both masks but passes ``full_attention`` to every
    decoder layer. FA2 applies the window inside its kernel, while eager
    attention needs the already-created sliding mask. Capture that exact mask
    at creation time and substitute it only at the nine SWA layer boundaries;
    the immutable snapshot remains untouched.
    """
    remote_module = sys.modules.get(model.__class__.__module__)
    if remote_module is None or not hasattr(remote_module, "create_sliding_window_causal_mask"):
        raise RuntimeError("cannot locate InfiniteVL sliding-mask factory")
    original_factory = getattr(
        remote_module,
        "_infinitevl_cpu_original_sliding_mask_factory",
        remote_module.create_sliding_window_causal_mask,
    )
    remote_module._infinitevl_cpu_original_sliding_mask_factory = original_factory
    cache_class = getattr(remote_module, "StaticSlidingWindowLayerPrealloc", None)
    if cache_class is None:
        raise RuntimeError("cannot locate InfiniteVL sliding-window cache class")
    if not hasattr(cache_class, "_infinitevl_cpu_original_get_mask_sizes"):
        cache_class._infinitevl_cpu_original_get_mask_sizes = cache_class.get_mask_sizes

        def patched_get_mask_sizes(self, cache_position):
            return sliding_cache_mask_sizes(
                self.cumulative_length, self.sliding_window, cache_position,
            )

        cache_class.get_mask_sizes = patched_get_mask_sizes
    holder: dict[str, torch.Tensor | None] = {"mask": None}

    def capture_factory(*args, **kwargs):
        mask = original_factory(*args, **kwargs)
        holder["mask"] = mask
        return mask

    remote_module.create_sliding_window_causal_mask = capture_factory
    handles = []
    for module in model.modules():
        if getattr(module, "layer_type", None) != "sliding_attention":
            continue
        attention = getattr(module, "self_attn", None)
        if not isinstance(getattr(attention, "sliding_window", None), int):
            raise RuntimeError("sliding attention layer is missing its window size")

        def hook(_module, args, kwargs):
            if kwargs.pop(PRE_ROUTED_SLIDING_MASK, False):
                return args, kwargs
            sliding_mask = holder["mask"]
            if sliding_mask is None:
                raise RuntimeError("decoder did not create a sliding causal mask")
            kwargs["attention_mask"] = sliding_mask
            return args, kwargs

        handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))
    model._infinitevl_cpu_sliding_mask_handles = handles
    model._infinitevl_cpu_sliding_mask_factory = original_factory
    model._infinitevl_cpu_sliding_cache_patch = (
        "phase_aware_pre_and_post_update_mask_sizes"
    )
    return len(handles)


def _verify_cpu_modules(model: nn.Module) -> dict:
    counts = {"short_convolution": 0, "fused_rmsnorm_gated": 0, "rmsnorm": 0}
    unexpected = []
    for name, module in model.named_modules():
        if isinstance(module, CPUShortConvolution):
            counts["short_convolution"] += 1
        elif isinstance(module, CPUFusedRMSNormGated):
            counts["fused_rmsnorm_gated"] += 1
        elif isinstance(module, CPURMSNorm):
            counts["rmsnorm"] += 1
        elif module.__class__.__module__.startswith("fla"):
            unexpected.append(f"{name}:{module.__class__.__qualname__}")
    if counts["short_convolution"] != 81 or counts["fused_rmsnorm_gated"] != 27:
        raise RuntimeError(f"CPU FLA replacement count mismatch: {counts}")
    if unexpected:
        raise RuntimeError(f"unexpected FLA modules remain: {unexpected[:8]}")
    return counts


def load_infinitevl_cpu(
    model_path: Path,
    *,
    validate_finite: bool = True,
) -> tuple[nn.Module, object, dict]:
    """Load BF16 checkpoint values widened to FP32 and force the CPU backend."""
    install_fla_cpu_shims()
    config = AutoConfig.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
    )
    _set_eager(config)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval()
    _set_eager(model.config)
    for module in model.modules():
        if hasattr(module, "config"):
            _set_eager(module.config)
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True, use_fast=True,
    )
    sliding_mask_layers = _install_sliding_attention_masks(model)
    if sliding_mask_layers != 9:
        raise RuntimeError(
            f"expected 9 eager sliding-mask patches, installed {sliding_mask_layers}"
        )
    replacement_counts = _verify_cpu_modules(model)
    nonfinite = []
    dtypes: dict[str, int] = {}
    for name, parameter in model.named_parameters():
        dtype = str(parameter.dtype)
        dtypes[dtype] = dtypes.get(dtype, 0) + parameter.numel()
        if validate_finite and not torch.isfinite(parameter).all():
            nonfinite.append(name)
    if nonfinite:
        raise ValueError(f"InfiniteVL contains non-finite parameters: {nonfinite[:8]}")
    audit = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "baseline_id": "bf16_value_fp32_compute",
        "definition": "official BF16 checkpoint values widened to FP32 storage and compute",
        "loaded_parameter_dtypes": dtypes,
        "nonfinite_parameters": nonfinite,
        "attn_implementation": "eager",
        "cpu_backend": replacement_counts,
        "eager_sliding_mask_layers": sliding_mask_layers,
        "eager_sliding_mask_source": "decoder_created_sliding_attention",
        "sliding_cache_mask_patch": model._infinitevl_cpu_sliding_cache_patch,
        "processor_use_fast": True,
    }
    return model, processor, audit


def environment_record() -> dict:
    packages = {}
    for name in (
        "transformers", "huggingface-hub", "accelerate", "qwen-vl-utils",
        "einops", "safetensors", "Pillow",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not_installed"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "packages": packages,
    }
