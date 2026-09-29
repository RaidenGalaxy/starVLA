from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from experiments.vlm_quant_benchmark.config import (
    ExperimentConfig,
    ImplementationTier,
    MethodName,
    QuantScope,
)
from experiments.vlm_quant_benchmark.mbq import MBQSearchResult, apply_mbq
from experiments.vlm_quant_benchmark.sensitivity import causal_predictor_mask
from experiments.vlm_quant_benchmark.selection import SelectedLinear

from . import SCHEMA_VERSION


TOKEN_GROUPS = ("vision", "prompt_language", "answer_predictor")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def decoder_layers(model: nn.Module) -> Sequence[nn.Module]:
    candidates = (
        ("model", "language_model", "layers"),
        ("language_model", "layers"),
        ("model", "layers"),
    )
    for path in candidates:
        current: object = model
        for component in path:
            current = getattr(current, component, None)
            if current is None:
                break
        if isinstance(current, (nn.ModuleList, list, tuple)):
            return current
    raise TypeError("cannot locate decoder layers")


def _identity_layer_output(_module: nn.Module, args: tuple, output):
    if not args or not isinstance(args[0], torch.Tensor):
        raise TypeError("cannot identify skipped layer hidden-state input")
    hidden = args[0]
    if isinstance(output, torch.Tensor):
        return hidden
    if isinstance(output, tuple):
        if not output:
            raise TypeError("cannot replace an empty tuple layer output")
        return (hidden, *output[1:])
    if isinstance(output, list):
        if not output:
            raise TypeError("cannot replace an empty list layer output")
        return [hidden, *output[1:]]
    raise TypeError(f"unsupported layer output type: {type(output).__name__}")


def _replace_output_hidden(output, hidden: torch.Tensor):
    if isinstance(output, torch.Tensor):
        return hidden
    if isinstance(output, tuple):
        if not output:
            raise TypeError("cannot replace an empty tuple layer output")
        return (hidden, *output[1:])
    if isinstance(output, list):
        if not output:
            raise TypeError("cannot replace an empty list layer output")
        return [hidden, *output[1:]]
    raise TypeError(f"unsupported layer output type: {type(output).__name__}")


def _token_freeze_hook(mask: torch.Tensor):
    if mask.ndim != 2 or mask.dtype is not torch.bool:
        raise ValueError("token freeze mask must be a rank-2 boolean tensor")

    def hook(_module: nn.Module, args: tuple, output):
        if not args or not isinstance(args[0], torch.Tensor):
            raise TypeError("cannot identify frozen layer hidden-state input")
        layer_input = args[0]
        layer_output = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(layer_output, torch.Tensor):
            raise TypeError("cannot identify frozen layer hidden-state output")
        if layer_input.shape != layer_output.shape:
            raise ValueError("layer input/output shapes differ during token freeze")
        if tuple(mask.shape) != tuple(layer_output.shape[:2]):
            raise ValueError(
                "token freeze mask shape does not match layer hidden states: "
                f"{tuple(mask.shape)} != {tuple(layer_output.shape[:2])}"
            )
        current_mask = mask.to(device=layer_output.device).unsqueeze(-1)
        frozen = torch.where(current_mask, layer_input, layer_output)
        return _replace_output_hidden(output, frozen)

    return hook


def _generation_token_freeze_hook(mask: torch.Tensor, audit: dict[str, int]):
    """Freeze visual rows on prefill and leave one-token decode calls intact."""
    if mask.ndim != 2 or mask.dtype is not torch.bool:
        raise ValueError("token freeze mask must be a rank-2 boolean tensor")

    def hook(_module: nn.Module, args: tuple, output):
        if not args or not isinstance(args[0], torch.Tensor):
            raise TypeError("cannot identify frozen layer hidden-state input")
        layer_input = args[0]
        layer_output = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(layer_output, torch.Tensor):
            raise TypeError("cannot identify frozen layer hidden-state output")
        if layer_input.shape != layer_output.shape:
            raise ValueError("layer input/output shapes differ during token freeze")
        current_shape = tuple(layer_output.shape[:2])
        if current_shape == tuple(mask.shape):
            current_mask = mask.to(device=layer_output.device).unsqueeze(-1)
            audit["prefill_calls"] += 1
            return _replace_output_hidden(
                output, torch.where(current_mask, layer_input, layer_output),
            )
        if layer_output.shape[1] == 1:
            audit["decode_calls"] += 1
            return output
        raise ValueError(
            "generation token freeze saw an unexpected non-decode shape: "
            f"{current_shape} != {tuple(mask.shape)}"
        )

    return hook


@contextmanager
def temporary_layer_set_skip(
    layers: Sequence[nn.Module], indices: Iterable[int],
) -> Iterator[None]:
    normalized = tuple(sorted(set(int(index) for index in indices)))
    if not normalized:
        yield
        return
    invalid = [index for index in normalized if index < 0 or index >= len(layers)]
    if invalid:
        raise IndexError(f"layer indices out of range: {invalid}")
    handles = [
        layers[index].register_forward_hook(_identity_layer_output)
        for index in normalized
    ]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def temporary_token_group_freeze(
    layers: Sequence[nn.Module],
    indices: Iterable[int],
    mask: torch.Tensor,
) -> Iterator[None]:
    """Freeze selected token rows across selected decoder layers.

    The layer still runs normally, so non-selected rows receive the original
    layer output. Selected rows are replaced by the layer input. This matches
    the causal behavior of ShortV-style token-group freezing, but does not
    claim wall-clock savings because the dense operators are still executed.
    """
    normalized = tuple(sorted(set(int(index) for index in indices)))
    if not normalized:
        yield
        return
    invalid = [index for index in normalized if index < 0 or index >= len(layers)]
    if invalid:
        raise IndexError(f"layer indices out of range: {invalid}")
    if mask.ndim != 2 or mask.dtype is not torch.bool:
        raise ValueError("token freeze mask must be a rank-2 boolean tensor")
    if not mask.any():
        raise ValueError("token freeze mask cannot be empty")
    handles = [
        layers[index].register_forward_hook(_token_freeze_hook(mask))
        for index in normalized
    ]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def temporary_generation_token_group_freeze(
    layers: Sequence[nn.Module],
    indices: Iterable[int],
    mask: torch.Tensor,
) -> Iterator[dict[str, int]]:
    """Freeze selected visual rows only during autoregressive prefill.

    Cached decoding subsequently invokes each decoder layer with one token.
    Those calls must remain untouched because the original visual rows are no
    longer present in the hidden-state sequence.
    """
    normalized = tuple(sorted(set(int(index) for index in indices)))
    invalid = [index for index in normalized if index < 0 or index >= len(layers)]
    if invalid:
        raise IndexError(f"layer indices out of range: {invalid}")
    if mask.ndim != 2 or mask.dtype is not torch.bool:
        raise ValueError("token freeze mask must be a rank-2 boolean tensor")
    if normalized and not mask.any():
        raise ValueError("token freeze mask cannot be empty")
    audit = {"prefill_calls": 0, "decode_calls": 0}
    handles = [
        layers[index].register_forward_hook(
            _generation_token_freeze_hook(mask, audit),
        )
        for index in normalized
    ]
    try:
        yield audit
    finally:
        for handle in handles:
            handle.remove()


def layer_set_id(indices: Iterable[int]) -> str:
    normalized = tuple(sorted(set(int(index) for index in indices)))
    return "baseline" if not normalized else "+".join(map(str, normalized))


def result_key(row: dict) -> tuple[str, str, str, str]:
    return (
        str(row["sample_id"]),
        str(row["quant_scheme"]),
        str(row["layer_set_id"]),
        str(row["operation"]),
    )


def load_existing_rows(path: Path, *, config_hash: str) -> tuple[list[dict], set[tuple]]:
    if not path.is_file():
        return [], set()
    rows: list[dict] = []
    keys: set[tuple] = set()
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("config_hash") != config_hash:
                raise ValueError(
                    f"resume config hash mismatch at {path}:{line_number}"
                )
            key = result_key(row)
            if key in keys:
                raise ValueError(f"duplicate result key at {path}:{line_number}: {key}")
            keys.add(key)
            rows.append(row)
    return rows, keys


def append_jsonl_fsync(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    temporary.replace(path)


def token_masks(batch: dict) -> dict[str, torch.Tensor]:
    answer_predictor = causal_predictor_mask(batch["language_mask"].bool())
    masks = {
        "vision": batch["vision_mask"].bool(),
        # The final prompt token predicts the first answer token.  Assign this
        # causal boundary to the answer group so equal-weight group summaries
        # never count one hidden position twice.
        "prompt_language": batch["prompt_language_mask"].bool() & ~answer_predictor,
        "answer_predictor": answer_predictor,
    }
    shape = batch["model_inputs"]["input_ids"].shape
    if any(mask.shape != shape for mask in masks.values()):
        raise ValueError("token masks do not match input_ids shape")
    names = list(masks)
    for position, left in enumerate(names):
        if not masks[left].any():
            raise ValueError(f"empty token mask: {left}")
        for right in names[position + 1:]:
            if (masks[left] & masks[right]).any():
                raise ValueError(f"overlapping token masks: {left}/{right}")
    return masks


@dataclass(frozen=True)
class TeacherReference:
    target_nll: float
    final_hidden: torch.Tensor
    answer_logits: torch.Tensor
    masks: dict[str, torch.Tensor]
    token_counts: dict[str, int]


@dataclass(frozen=True)
class PreparedTeacherInput:
    decoder_inputs: dict[str, torch.Tensor]
    labels: torch.Tensor
    masks: dict[str, torch.Tensor]


@torch.inference_mode()
def prepare_teacher_input(model: nn.Module, batch: dict) -> PreparedTeacherInput:
    """Compute the unchanged vision/input embedding path once per sample."""
    masks = token_masks(batch)
    model_inputs = batch["model_inputs"]
    input_ids = model_inputs["input_ids"]
    inputs_embeds = model.model.get_input_embeddings()(input_ids)
    pixel_values = model_inputs.get("pixel_values")
    image_grid_thw = model_inputs.get("image_grid_thw")
    if pixel_values is not None:
        image_embeds = model.model.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0).to(
            inputs_embeds.device, inputs_embeds.dtype,
        )
        image_mask, _ = model.model.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds,
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
    pixel_values_videos = model_inputs.get("pixel_values_videos")
    video_grid_thw = model_inputs.get("video_grid_thw")
    if pixel_values_videos is not None:
        video_embeds = model.model.get_video_features(
            pixel_values_videos, video_grid_thw,
        )
        video_embeds = torch.cat(video_embeds, dim=0).to(
            inputs_embeds.device, inputs_embeds.dtype,
        )
        _, video_mask = model.model.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds,
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
    position_ids, _rope_deltas = model.model.get_rope_index(
        input_ids,
        image_grid_thw,
        video_grid_thw,
        second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
        attention_mask=model_inputs.get("attention_mask"),
    )
    return PreparedTeacherInput(
        decoder_inputs={
            "inputs_embeds": inputs_embeds.detach(),
            "position_ids": position_ids,
            "attention_mask": model_inputs.get("attention_mask"),
        },
        labels=model_inputs["labels"],
        masks=masks,
    )


def _finite_scalar(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"non-finite {name}: {value}")
    return value


@torch.inference_mode()
def teacher_reference(
    model: nn.Module,
    batch: dict,
    *,
    prepared: PreparedTeacherInput | None = None,
) -> TeacherReference:
    prepared = prepared or prepare_teacher_input(model, batch)
    masks = prepared.masks
    output = model.model(
        **prepared.decoder_inputs,
        use_cache=False,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=False,
    )
    hidden_device = output.last_hidden_state
    answer_hidden = hidden_device[masks["answer_predictor"]]
    targets = prepared.labels[batch["language_mask"].bool()]
    if answer_hidden.shape[0] != targets.numel():
        raise ValueError(
            "answer predictor/target count mismatch: "
            f"{answer_hidden.shape[0]} != {targets.numel()}"
        )
    answer_logits_device = model.lm_head(answer_hidden).float()
    nll = _finite_scalar(
        float(F.cross_entropy(answer_logits_device, targets).item()), "target NLL",
    )
    hidden = hidden_device.detach().float().cpu()
    answer_logits = answer_logits_device.detach().cpu()
    if not torch.isfinite(hidden).all() or not torch.isfinite(answer_logits).all():
        raise ValueError("teacher forward returned non-finite tensors")
    return TeacherReference(
        target_nll=nll,
        final_hidden=hidden,
        answer_logits=answer_logits,
        masks={name: mask.cpu() for name, mask in masks.items()},
        token_counts={name: int(mask.sum().item()) for name, mask in masks.items()},
    )


def compare_teacher(candidate: TeacherReference, reference: TeacherReference) -> dict:
    if candidate.final_hidden.shape != reference.final_hidden.shape:
        raise ValueError("candidate/reference hidden-state shape mismatch")
    if candidate.answer_logits.shape != reference.answer_logits.shape:
        raise ValueError("candidate/reference answer-logit shape mismatch")
    hidden_drift: dict[str, float] = {}
    for name in TOKEN_GROUPS:
        mask = reference.masks[name]
        if not torch.equal(mask, candidate.masks[name]):
            raise ValueError(f"candidate/reference {name} masks differ")
        left = reference.final_hidden[mask]
        right = candidate.final_hidden[mask]
        value = float((1.0 - F.cosine_similarity(left, right, dim=-1)).mean().item())
        hidden_drift[name] = _finite_scalar(value, f"{name} hidden drift")
    ref_log_probs = F.log_softmax(reference.answer_logits, dim=-1)
    candidate_log_probs = F.log_softmax(candidate.answer_logits, dim=-1)
    ref_probs = ref_log_probs.exp()
    answer_kl = float(
        (ref_probs * (ref_log_probs - candidate_log_probs)).sum(dim=-1).mean().item()
    )
    return {
        "target_nll": candidate.target_nll,
        "delta_target_nll": _finite_scalar(
            candidate.target_nll - reference.target_nll, "delta target NLL",
        ),
        "answer_logit_kl": _finite_scalar(answer_kl, "answer logit KL"),
        "final_hidden_cosine_drift": hidden_drift,
        "token_counts": dict(reference.token_counts),
    }


def mbq_config(weight_bits: int) -> ExperimentConfig:
    if weight_bits not in (3, 4):
        raise ValueError("Q-ShortV MBQ supports W3 or W4")
    return ExperimentConfig(
        MethodName.MBQ,
        scope=QuantScope.DECODER_CANONICAL,
        weight_bits=weight_bits,
        activation_bits=None,
        group_size=128,
        zero_point=True,
        weight_granularity="groupwise",
        calibration_samples=128,
        scheme_id=f"mbq_w{weight_bits}a16_g128_asym",
        implementation_tier=ImplementationTier.LOCAL_VALIDATED,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        seed=42,
    )


def mbq_w4_config() -> ExperimentConfig:
    """Backward-compatible W4 configuration used by the original probes."""
    return mbq_config(4)


def load_validated_mbq(
    artifact_path: Path,
    audit_path: Path,
    selected: Sequence[SelectedLinear],
    *,
    weight_bits: int,
) -> tuple[MBQSearchResult, dict]:
    config = mbq_config(weight_bits)
    payload = torch.load(artifact_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("scales"), dict):
        raise ValueError("invalid MBQ search artifact")
    scales = payload["scales"]
    expected = {item.name for item in selected}
    if set(scales) != expected:
        raise ValueError("MBQ artifact module set does not match selected decoder linears")
    audit = json.loads(audit_path.read_text())
    if (
        audit.get("method") != "mbq"
        or audit.get("weight_bits") != weight_bits
        or audit.get("activation_bits") is not None
        or audit.get("scope") not in (None, config.scope.value)
    ):
        raise ValueError(
            f"quantization audit is not the validated MBQ W{weight_bits} configuration"
        )
    modules = audit.get("modules", {})
    if set(modules) != expected:
        raise ValueError("quantization audit module set does not match selected linears")
    for name, scale in scales.items():
        recorded = modules[name]
        actual_min = float(scale.min().item())
        actual_max = float(scale.max().item())
        if not math.isclose(actual_min, float(recorded["scale_min"]), abs_tol=1e-6):
            raise ValueError(f"MBQ scale minimum mismatch for {name}")
        if not math.isclose(actual_max, float(recorded["scale_max"]), abs_tol=1e-6):
            raise ValueError(f"MBQ scale maximum mismatch for {name}")
    search = MBQSearchResult(scales=scales, audit=dict(payload.get("audit", {})))
    provenance = {
        "artifact": str(artifact_path.resolve()),
        "artifact_sha256": file_sha256(artifact_path),
        "artifact_key": str(payload.get("key", "")),
        "audit": str(audit_path.resolve()),
        "audit_sha256": file_sha256(audit_path),
        "weight_bits": weight_bits,
        "modules": len(scales),
    }
    return search, provenance


def load_validated_mbq_w4(
    artifact_path: Path,
    audit_path: Path,
    selected: Sequence[SelectedLinear],
) -> tuple[MBQSearchResult, dict]:
    """Backward-compatible W4 loader used by the original probes."""
    return load_validated_mbq(
        artifact_path, audit_path, selected, weight_bits=4,
    )


@contextmanager
def applied_mbq(
    model: nn.Module,
    selected: list[SelectedLinear],
    search: MBQSearchResult,
    *,
    weight_bits: int,
) -> Iterator[dict]:
    applied = apply_mbq(model, selected, mbq_config(weight_bits), search)
    try:
        yield applied.audit
    finally:
        applied.restore()


@contextmanager
def applied_mbq_w4(
    model: nn.Module,
    selected: list[SelectedLinear],
    search: MBQSearchResult,
) -> Iterator[dict]:
    """Backward-compatible W4 context manager used by the original probes."""
    with applied_mbq(
        model, selected, search, weight_bits=4,
    ) as audit:
        yield audit
