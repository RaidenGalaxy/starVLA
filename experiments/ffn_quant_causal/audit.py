from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import transformers
from torch import nn

from experiments.infinitevl_quant.progress import (
    StageProgress,
    atomic_json,
    file_sha256,
    json_digest,
)

from .config import (
    EXPECTED_FFN_LINEAR_COUNT,
    EXPECTED_HIDDEN_LAYERS,
    FFN_PROJECTIONS,
    STAGES,
    CausalConfig,
    ModelSpec,
)


SCORER_RESOURCE_RELATIVE = Path(
    "data/vlm_quant_benchmark/LMUData/TextVQA_VAL_local.tsv"
)


@dataclasses.dataclass(frozen=True)
class FFNTarget:
    name: str
    module: nn.Linear
    layer: int
    projection: str

    @property
    def parameter_count(self) -> int:
        return self.module.weight.numel() + (
            0 if self.module.bias is None else self.module.bias.numel()
        )


def _config_payload(source: Path) -> dict[str, Any]:
    path = source / "config.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text())
    text = value.get("text_config", value)
    if not isinstance(text, dict):
        raise ValueError(f"invalid text model config in {path}")
    return text


def _validate_config_against_spec(config: dict[str, Any], spec: ModelSpec) -> None:
    actual = {
        "model_type": config.get("model_type"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "hidden_size": config.get("hidden_size"),
        "intermediate_size": config.get("intermediate_size"),
    }
    expected = {
        "model_type": spec.model_type,
        "num_hidden_layers": spec.hidden_layers,
        "hidden_size": spec.hidden_size,
        "intermediate_size": spec.intermediate_size,
    }
    mismatches = {
        key: {"actual": actual[key], "expected": expected[key]}
        for key in expected if actual[key] != expected[key]
    }
    if mismatches:
        raise RuntimeError(f"{spec.key} config mismatch: {mismatches}")


def _weight_map(source: Path) -> dict[str, str]:
    index_path = source / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    value = json.loads(index_path.read_text())
    weights = value.get("weight_map")
    if not isinstance(weights, dict):
        raise ValueError(f"invalid safetensors weight map: {index_path}")
    return {str(key): str(item) for key, item in weights.items()}


def _expected_weight_names() -> list[str]:
    return [
        f"model.layers.{layer}.mlp.{projection}.weight"
        for layer in range(EXPECTED_HIDDEN_LAYERS)
        for projection in FFN_PROJECTIONS
    ]


def static_ffn_inventory(source: Path, spec: ModelSpec) -> dict:
    """Validate config/index without allocating checkpoint tensors."""

    source = source.resolve()
    config = _config_payload(source)
    _validate_config_against_spec(config, spec)
    weights = _weight_map(source)
    expected_names = _expected_weight_names()
    missing = [name for name in expected_names if name not in weights]
    if missing:
        raise RuntimeError(
            f"{spec.key} static FFN inventory is missing {len(missing)} weights: "
            f"{missing[:8]}"
        )
    decoder_mlp_parameters = sorted(
        name for name in weights
        if name.startswith("model.layers.")
        and ".mlp." in name
    )
    unexpected = sorted(set(decoder_mlp_parameters) - set(expected_names))
    if len(expected_names) != EXPECTED_FFN_LINEAR_COUNT or unexpected:
        raise RuntimeError(
            f"{spec.key} static FFN inventory mismatch: "
            f"expected={EXPECTED_FFN_LINEAR_COUNT}, unexpected={unexpected[:8]}"
        )
    shard_counts = Counter(weights[name] for name in expected_names)
    return {
        "passed": True,
        "model_key": spec.key,
        "model_type": config["model_type"],
        "decoder_layers": config["num_hidden_layers"],
        "hidden_size": config["hidden_size"],
        "intermediate_size": config["intermediate_size"],
        "linear_count": len(expected_names),
        "by_projection": {projection: EXPECTED_HIDDEN_LAYERS for projection in FFN_PROJECTIONS},
        "expected_shapes": {
            "gate_proj": [spec.intermediate_size, spec.hidden_size],
            "up_proj": [spec.intermediate_size, spec.hidden_size],
            "down_proj": [spec.hidden_size, spec.intermediate_size],
        },
        "shards": dict(sorted(shard_counts.items())),
        "weights": expected_names,
    }


def _decoder_layers(model: nn.Module) -> tuple[str, nn.ModuleList]:
    for path in (
        "model.language_model.layers",
        "language_model.layers",
        "model.layers",
    ):
        try:
            value = model.get_submodule(path)
        except AttributeError:
            continue
        if isinstance(value, nn.ModuleList):
            return path, value
    raise TypeError("could not locate decoder ModuleList")


def runtime_ffn_inventory(
    model: nn.Module, spec: ModelSpec,
) -> tuple[list[FFNTarget], dict]:
    decoder_path, layers = _decoder_layers(model)
    if len(layers) != spec.hidden_layers:
        raise RuntimeError(
            f"{spec.key} runtime decoder has {len(layers)} layers, "
            f"expected {spec.hidden_layers}"
        )
    targets = []
    for layer_index, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if not isinstance(mlp, nn.Module):
            raise RuntimeError(f"{spec.key} layer {layer_index} has no MLP module")
        local_linears = {
            name: module for name, module in mlp.named_modules()
            if name and isinstance(module, nn.Linear)
        }
        if set(local_linears) != set(FFN_PROJECTIONS):
            raise RuntimeError(
                f"{spec.key} layer {layer_index} FFN Linear set mismatch: "
                f"{sorted(local_linears)}"
            )
        for projection in FFN_PROJECTIONS:
            module = local_linears[projection]
            expected_shape = (
                (spec.hidden_size, spec.intermediate_size)
                if projection == "down_proj"
                else (spec.intermediate_size, spec.hidden_size)
            )
            if tuple(module.weight.shape) != expected_shape:
                raise RuntimeError(
                    f"{spec.key} layer {layer_index} {projection} has shape "
                    f"{tuple(module.weight.shape)}, expected {expected_shape}"
                )
            targets.append(FFNTarget(
                name=f"{decoder_path}.{layer_index}.mlp.{projection}",
                module=module,
                layer=layer_index,
                projection=projection,
            ))
    if len(targets) != EXPECTED_FFN_LINEAR_COUNT:
        raise RuntimeError(
            f"{spec.key} runtime FFN count {len(targets)} != "
            f"{EXPECTED_FFN_LINEAR_COUNT}"
        )
    by_projection = Counter(target.projection for target in targets)
    audit = {
        "passed": True,
        "model_key": spec.key,
        "decoder_path": decoder_path,
        "linear_count": len(targets),
        "by_projection": dict(by_projection),
        "parameters": sum(target.parameter_count for target in targets),
        "modules": [{
            "name": target.name,
            "layer": target.layer,
            "projection": target.projection,
            "weight_shape": list(target.module.weight.shape),
            "weight_dtype": str(target.module.weight.dtype),
            "parameters": target.parameter_count,
        } for target in targets],
    }
    return targets, audit


def audit_model_identity(
    source: Path,
    spec: ModelSpec,
    *,
    hash_weights: bool = True,
) -> dict:
    """Content-address a complete local model source and validate its config."""

    source = source.resolve()
    config = _config_payload(source)
    _validate_config_against_spec(config, spec)
    weight_map = _weight_map(source)
    shard_names = sorted(set(weight_map.values()))
    missing_shards = [name for name in shard_names if not (source / name).is_file()]
    if missing_shards:
        raise FileNotFoundError(
            f"{spec.key} source is missing weight shards: {missing_shards}"
        )
    metadata_names = sorted(
        item.name for item in source.iterdir()
        if item.is_file() and not item.name.endswith(".safetensors")
    )
    file_records = {}
    combined = hashlib.sha256()
    for name in metadata_names + shard_names:
        path = source / name
        is_weight = name in shard_names
        digest = file_sha256(path) if hash_weights or not is_weight else "not_computed"
        record = {"bytes": path.stat().st_size, "sha256": digest}
        file_records[name] = record
        combined.update(name.encode())
        combined.update(str(record["bytes"]).encode())
        combined.update(digest.encode())
    content_digest = combined.hexdigest() if hash_weights else None
    return {
        "passed": True,
        "model": spec.to_dict(),
        "source": str(source),
        "content_sha256": content_digest,
        "weights_fully_hashed": hash_weights,
        "weight_bytes": sum((source / name).stat().st_size for name in shard_names),
        "weight_shards": shard_names,
        "files": file_records,
        "static_ffn_inventory": static_ffn_inventory(source, spec),
    }


def audit_model_pair(
    sources: dict[str, Path], specs: dict[str, ModelSpec], *, hash_weights: bool = True,
) -> dict:
    if set(sources) != {"infinitevl", "qwen"} or set(specs) != set(sources):
        raise ValueError("model-pair audit requires exactly infinitevl and qwen")
    models = {
        key: audit_model_identity(sources[key], specs[key], hash_weights=hash_weights)
        for key in ("infinitevl", "qwen")
    }
    if hash_weights and models["infinitevl"]["content_sha256"] == models["qwen"]["content_sha256"]:
        raise RuntimeError("comparison model sources unexpectedly have the same content digest")
    return {
        "passed": True,
        "models": models,
        "pair_digest": json_digest({
            key: models[key]["content_sha256"] for key in sorted(models)
        }),
    }


def require_cpu_environment() -> dict:
    if torch.cuda.is_available():
        raise RuntimeError("the protocol is CPU-only but PyTorch reports CUDA available")
    return {
        "cpu_only": True,
        "torch_cuda_available": False,
        "torch_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "default_dtype": str(torch.get_default_dtype()),
    }


def protocol_input_digests(
    model_pair_audit: dict,
    split_audit: dict,
    *,
    calibration_manifest: Path,
    evaluation_manifest: Path,
) -> dict[str, str]:
    models = model_pair_audit.get("models", {})
    result = {
        f"model_{key}": str(models[key].get("content_sha256"))
        for key in ("infinitevl", "qwen")
    }
    if any(value in {"", "None"} for value in result.values()):
        raise ValueError("fully hashed model identities are required for resumable stages")
    repository = Path(__file__).resolve().parents[2]
    scorer_resource = repository / SCORER_RESOURCE_RELATIVE
    if not scorer_resource.is_file():
        raise FileNotFoundError(
            f"deterministic scorer resource is missing: {scorer_resource}"
        )
    runtime_packages = {}
    for distribution in (
        "accelerate", "einops", "huggingface-hub", "numpy", "Pillow",
        "qwen-vl-utils", "safetensors", "tokenizers", "torchvision",
    ):
        try:
            runtime_packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            runtime_packages[distribution] = "not_installed"
    result.update({
        "model_pair": str(model_pair_audit["pair_digest"]),
        "calibration_manifest": file_sha256(calibration_manifest),
        "evaluation_manifest": file_sha256(evaluation_manifest),
        "scorer_textvqa_annotations": file_sha256(scorer_resource),
        "data_splits": json_digest(split_audit["digests"]),
        "implementation": implementation_audit()["digest"],
        "runtime_stack": json_digest({
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "packages": runtime_packages,
            "torch_build": torch.__config__.show(),
        }),
    })
    return result


def scorer_resource_audit() -> dict[str, dict[str, str | int]]:
    repository = Path(__file__).resolve().parents[2]
    path = repository / SCORER_RESOURCE_RELATIVE
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "textvqa_annotations": {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        },
    }


def implementation_audit() -> dict:
    """Hash all local code that can change protocol values or selection.

    Model repository Python/config files are already covered by the complete
    model-source hash.  This manifest covers the new orchestration package and
    the reused local quantization/evaluation primitives.  Tests are excluded
    because they cannot affect execution.
    """

    repository = Path(__file__).resolve().parents[2]
    package = repository / "experiments" / "ffn_quant_causal"
    files = sorted(
        path for path in package.glob("*.py") if path.name != "__pycache__"
    )
    for relative in (
        "experiments/infinitevl_quant",
        "experiments/vlm_quant_benchmark",
    ):
        files.extend(sorted((repository / relative).glob("*.py")))
    files.append(
        repository / "experiments/infinitevl_quant_ffn_bf16_4x64/scoring.py"
    )
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"implementation audit files missing: {missing}")
    records = {}
    combined = hashlib.sha256()
    for path in sorted(set(files)):
        relative = str(path.relative_to(repository))
        digest = file_sha256(path)
        records[relative] = digest
        combined.update(relative.encode())
        combined.update(digest.encode())
    return {
        "digest": combined.hexdigest(),
        "files": records,
        "file_count": len(records),
    }


def build_protocol_identity(
    config: CausalConfig, input_digests: dict[str, str],
) -> dict:
    return {
        "format_version": 1,
        "protocol": "ffn_quant_causal",
        "config_digest": config.digest(),
        "input_digests": dict(sorted(input_digests.items())),
        "models": {
            "infinitevl_revision": config.to_dict()["models"]["infinitevl"]["revision"],
            "qwen_revision": config.to_dict()["models"]["qwen"]["revision"],
        },
    }


def ensure_protocol_identity(output_dir: Path, identity: dict) -> Path:
    path = output_dir / "protocol_identity.json"
    if path.is_file():
        existing = json.loads(path.read_text())
        if existing != identity:
            raise RuntimeError(
                "output directory belongs to a different causal protocol identity"
            )
    else:
        atomic_json(path, identity)
    return path


def stage_progress(
    config: CausalConfig, stage: str, input_digests: dict[str, str],
) -> StageProgress:
    if stage not in STAGES:
        raise ValueError(f"unknown causal protocol stage: {stage}")
    return StageProgress(
        config.output_dir,
        stage,
        config_digest=config.digest(),
        inputs=dict(sorted(input_digests.items())),
    )
