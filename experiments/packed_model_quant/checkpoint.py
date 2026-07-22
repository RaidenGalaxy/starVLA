"""Export and load self-contained packed Qwen2.5-VL checkpoints."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration

from experiments.ffn_quant.modeling import load_qwen_cpu
from experiments.full_model_quant.awq import apply_input_scale_, apply_internal_scale_
from experiments.full_model_quant.modeling import (
    FFNTarget,
    LinearTarget,
    build_inventory,
    config_assignment,
    discover_full_model,
    make_fp16_weight_reference_,
)
from experiments.full_model_quant.run_ablation import _deserialise_awq

from . import FORMAT_VERSION
from .linear import COMPUTE_MODES, PackedWeightOnlyLinear
from .packing import quantize_packed_weight

EXPORT_CONFIGS = ("rtn_w8_all_linear", "rtn_w4_all_linear", "hybrid_awq_w4_all_linear")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _parent_and_leaf(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    parent_name, leaf = name.rsplit(".", 1)
    return model.get_submodule(parent_name), leaf


def _set_module(model: nn.Module, name: str, module: nn.Module) -> None:
    parent, leaf = _parent_and_leaf(model, name)
    setattr(parent, leaf, module)


def _storage_key(value: torch.Tensor) -> tuple[int, int, int, str]:
    return (value.untyped_storage().data_ptr(), value.storage_offset(), value.numel(), str(value.dtype))


def _load_awq(source_run: Path) -> dict[tuple[str, int], Any]:
    path = source_run / "awq_scales.pt"
    raw = torch.load(path, map_location="cpu", weights_only=False)
    results = {}
    for key, value in raw.items():
        domain, index = key.split(":")
        results[(domain, int(index))] = _deserialise_awq(value)
    return results


def _apply_hybrid_scales(ffns: list[FFNTarget], source_run: Path) -> dict[str, int]:
    results = _load_awq(source_run)
    accepted = fallback = 0
    for ffn in ffns:
        result = results.get((ffn.domain, ffn.index))
        if result is None:
            raise RuntimeError(f"missing AWQ scale for {ffn.domain}.{ffn.index}")
        apply_input_scale_(ffn, result.input_scale)
        apply_internal_scale_(ffn, result.internal_scale)
        if result.fallback_reason is None:
            accepted += 1
        else:
            fallback += 1
    return {
        "ffn_blocks": len(ffns),
        "accepted_blocks": accepted,
        "fallback_blocks": fallback,
        "nonidentity_linears": accepted * 3,
        "fallback_linears": fallback * 3,
    }


def _write_shards(tensors: dict[str, torch.Tensor], output: Path, max_shard_bytes: int) -> dict[str, Any]:
    shards: list[dict[str, torch.Tensor]] = []
    current: dict[str, torch.Tensor] = {}
    current_bytes = 0
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().contiguous()
        size = tensor.numel() * tensor.element_size()
        if current and current_bytes + size > max_shard_bytes:
            shards.append(current)
            current, current_bytes = {}, 0
        current[name] = tensor
        current_bytes += size
    if current:
        shards.append(current)
    weight_map: dict[str, str] = {}
    total = len(shards)
    for index, shard in enumerate(shards, 1):
        filename = f"model-{index:05d}-of-{total:05d}.safetensors"
        save_file(shard, output / filename, metadata={"format": "pt", "packed_format": FORMAT_VERSION})
        weight_map.update({name: filename for name in shard})
    index = {
        "metadata": {
            "total_size": sum(value.numel() * value.element_size() for value in tensors.values()),
            "format": FORMAT_VERSION,
        },
        "weight_map": weight_map,
    }
    atomic_json(output / "model.safetensors.index.json", index)
    return index


@torch.no_grad()
def export_checkpoint(
    *,
    model_path: str,
    source_run: Path,
    config_name: str,
    output_dir: Path,
    group_size: int = 128,
    scale_dtype: str = "fp16",
    max_shard_bytes: int = 2_000_000_000,
    resume: bool = False,
) -> dict[str, Any]:
    if config_name not in EXPORT_CONFIGS:
        raise ValueError(f"unsupported export config {config_name!r}")
    source_config = json.loads((source_run / "config.json").read_text(encoding="utf-8"))
    source_inventory = json.loads((source_run / "module_inventory.json").read_text(encoding="utf-8"))
    if source_config["group_size"] != group_size:
        raise RuntimeError("source run group size does not match export group size")
    if source_inventory["linear_modules"] != 414 or source_inventory["target_parameters"] != 3_441_197_056:
        raise RuntimeError("source run is not the expected Qwen2.5-VL-3B inventory")
    identity = {
        "format_version": FORMAT_VERSION,
        "model_path": str(model_path),
        "source_run_hash": source_config["run_hash"],
        "source_inventory_hash": sha256_file(source_run / "module_inventory.json"),
        "awq_scales_hash": sha256_file(source_run / "awq_scales.pt") if "awq" in config_name else None,
        "config": config_name,
        "group_size": group_size,
        "scale_dtype": scale_dtype,
    }
    export_hash = stable_hash(identity)
    metadata_path = output_dir / "quantization_config.json"
    if output_dir.exists():
        if resume and metadata_path.is_file():
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            if existing.get("export_hash") == export_hash:
                return existing
        raise RuntimeError(f"output directory already exists: {output_dir}")
    temporary = output_dir.with_name(output_dir.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        model, processor = load_qwen_cpu(model_path)
        conversion = make_fp16_weight_reference_(model)
        targets, ffns = discover_full_model(model, strict_qwen3b=True)
        inventory = build_inventory(model, targets, ffns)
        awq_summary = {
            "ffn_blocks": 0,
            "accepted_blocks": 0,
            "fallback_blocks": 0,
            "nonidentity_linears": 0,
            "fallback_linears": 0,
        }
        if config_name == "hybrid_awq_w4_all_linear":
            awq_summary = _apply_hybrid_scales(ffns, source_run)
        assignment = config_assignment(config_name, targets)
        target_by_weight = {f"{target.name}.weight": target for target in targets}
        tensors: dict[str, torch.Tensor] = {}
        aliases: dict[str, str] = {}
        seen_storage: dict[tuple[int, int, int, str], str] = {}
        module_rows: list[dict[str, Any]] = []
        qweight_bytes = scale_bytes = quantized_parameters = 0
        for position, target in enumerate(targets, 1):
            if position == 1 or position % 10 == 0 or position == len(targets):
                print(f"Packing Linear {position}/{len(targets)}: {target.name}", flush=True)
            mode = assignment[target.name]
            bits = 8 if mode == "rtn_w8" else 4
            packed = quantize_packed_weight(
                target.module.weight, bits=bits, group_size=group_size, scale_dtype=scale_dtype
            )
            qkey, skey = f"{target.name}.qweight", f"{target.name}.scales"
            tensors[qkey], tensors[skey] = packed.qweight, packed.scales
            qweight_bytes += packed.stats.qweight_bytes
            scale_bytes += packed.stats.scale_bytes
            quantized_parameters += packed.stats.parameters
            module_rows.append({
                "name": target.name,
                "group": target.group,
                "layer": target.layer,
                "mode": mode,
                "shape": [packed.out_features, packed.in_features],
                "bits": bits,
                "group_size": group_size,
                "groups": packed.stats.groups,
                "qweight_shape": list(packed.qweight.shape),
                "qweight_dtype": str(packed.qweight.dtype),
                "scale_shape": list(packed.scales.shape),
                "scale_dtype": str(packed.scales.dtype),
                **packed.stats.to_dict(),
            })
        for name, value in model.state_dict().items():
            if name in target_by_weight:
                continue
            key = _storage_key(value)
            if key in seen_storage:
                aliases[name] = seen_storage[key]
                continue
            seen_storage[key] = name
            tensors[name] = value.detach().half().cpu() if value.is_floating_point() else value.detach().cpu()
        model.config.save_pretrained(temporary)
        processor.save_pretrained(temporary)
        index = _write_shards(tensors, temporary, max_shard_bytes)
        inventory_payload = {
            **inventory,
            "format_version": FORMAT_VERSION,
            "modules": module_rows,
            "awq": awq_summary,
            "aliases": aliases,
        }
        atomic_json(temporary / "packed_inventory.json", inventory_payload)
        non_quantized_bytes = int(index["metadata"]["total_size"]) - qweight_bytes - scale_bytes
        quant_config = {
            **identity,
            "export_hash": export_hash,
            "reference": conversion,
            "packing": {
                "w4": "signed_twos_complement_low_nibble_even_v1",
                "qrange_w4": [-7, 7],
                "qrange_w8": [-127, 127],
                "odd_tail_padding": "zero_high_nibble",
            },
            "linear_modules": len(module_rows),
            "quantized_parameters": quantized_parameters,
            "aliases": aliases,
            "awq": awq_summary,
            "tensor_bytes": int(index["metadata"]["total_size"]),
            "qweight_bytes": qweight_bytes,
            "scale_bytes": scale_bytes,
            "non_quantized_bytes": non_quantized_bytes,
        }
        atomic_json(temporary / "quantization_config.json", quant_config)
        temporary.replace(output_dir)
        return quant_config
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _assign_alias(model: nn.Module, alias: str, canonical: str) -> None:
    try:
        value = model.get_parameter(canonical)
        parent, leaf = _parent_and_leaf(model, alias)
        parent._parameters[leaf] = value
        return
    except AttributeError:
        value = model.get_buffer(canonical)
        parent, leaf = _parent_and_leaf(model, alias)
        parent._buffers[leaf] = value


def _replace_from_inventory(
    model: nn.Module, inventory: dict[str, Any], *, compute_mode: str
) -> list[str]:
    targets, _ffns = discover_full_model(model, strict_qwen3b=True)
    discovered = {target.name: target for target in targets}
    replaced = []
    for row in inventory["modules"]:
        name = row["name"]
        target = discovered.get(name)
        if target is None:
            raise RuntimeError(f"packed target was not found in model config: {name}")
        qshape = tuple(row["qweight_shape"])
        sdtype = torch.float16 if row["scale_dtype"] == "torch.float16" else torch.float32
        qdtype = torch.uint8 if row["bits"] == 4 else torch.int8
        qweight = torch.empty(qshape, dtype=qdtype, device="meta")
        scales = torch.empty(tuple(row["scale_shape"]), dtype=sdtype, device="meta")
        bias = target.module.bias
        replacement = PackedWeightOnlyLinear(
            row["shape"][1], row["shape"][0], bits=row["bits"], group_size=row["group_size"],
            qweight=qweight, scales=scales, bias=bias, compute_mode=compute_mode,
        )
        _set_module(model, name, replacement)
        replaced.append(name)
    return replaced


def load_packed_checkpoint(
    checkpoint: Path, *, compute_mode: str = "dequant_fp32"
) -> tuple[nn.Module, Any, dict[str, Any], dict[str, Any]]:
    if compute_mode not in COMPUTE_MODES:
        raise ValueError(f"unsupported compute mode {compute_mode!r}")
    quant_config = json.loads((checkpoint / "quantization_config.json").read_text(encoding="utf-8"))
    inventory = json.loads((checkpoint / "packed_inventory.json").read_text(encoding="utf-8"))
    if quant_config["format_version"] != FORMAT_VERSION:
        raise RuntimeError("unsupported packed checkpoint format")
    config = AutoConfig.from_pretrained(checkpoint)
    with init_empty_weights(include_buffers=False):
        model = Qwen2_5_VLForConditionalGeneration(config)
        replaced = _replace_from_inventory(model, inventory, compute_mode=compute_mode)
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shard_names = sorted(set(index["weight_map"].values()))
    loaded: set[str] = set()
    for shard_name in shard_names:
        shard = load_file(checkpoint / shard_name, device="cpu")
        model.load_state_dict(shard, strict=False, assign=True)
        loaded.update(shard)
        del shard
        gc.collect()
    for alias, canonical in quant_config.get("aliases", {}).items():
        _assign_alias(model, alias, canonical)
    missing = sorted(set(index["weight_map"]) - loaded)
    if missing:
        raise RuntimeError(f"checkpoint tensors were not loaded: {missing[:3]}")
    meta = [name for name, value in list(model.named_parameters()) + list(model.named_buffers()) if value.device.type == "meta"]
    if meta:
        raise RuntimeError(f"model still contains meta tensors after loading: {meta[:5]}")
    packed_modules = [(name, module) for name, module in model.named_modules() if isinstance(module, PackedWeightOnlyLinear)]
    if len(packed_modules) != 414 or len(replaced) != 414:
        raise RuntimeError(f"expected 414 packed Linear modules, found {len(packed_modules)}")
    if any(hasattr(module, "weight") for _name, module in packed_modules):
        raise RuntimeError("a packed target exposes a floating weight")
    # CPU execution remains FP32. Preserve serialized group-scale dtype so its
    # storage error remains observable, while all other values follow the
    # existing FP32-compute reference protocol.
    for parameter in model.parameters():
        if parameter.is_floating_point() and parameter.dtype != torch.float32:
            parameter.data = parameter.data.float()
    packed_scale_ids = {id(module.scales) for _name, module in packed_modules}
    for module in model.modules():
        for key, value in module._buffers.items():
            if value is not None and value.is_floating_point() and id(value) not in packed_scale_ids:
                module._buffers[key] = value.float()
    processor = AutoProcessor.from_pretrained(checkpoint)
    model.eval()
    return model, processor, quant_config, inventory

