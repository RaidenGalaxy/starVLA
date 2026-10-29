"""Matched MBQ/VVS/QIG search for the InfiniteVL--Qwen FFN study.

This module deliberately contains orchestration rather than a third quantizer.
Both models use the same MBQ collectors/search objective and the same
block-search implementation.  The only backend-specific QIG behavior is:

* InfiniteVL replays Gated DeltaNet blocks under the differentiable recurrent
  reference implementation; and
* an already-routed sliding mask marker is added only when an InfiniteVL SWA
  block is replayed in isolation.

All produced weights remain floating fake-quant/QDQ weights.  Nothing in this
module is evidence for packed-kernel memory, latency, or throughput gains.
"""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import hashlib
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from torch import nn

from experiments.infinitevl_quant import mbq_qig as _legacy
from experiments.infinitevl_quant.config import QuantScheme, SCHEMES
from experiments.infinitevl_quant.cpu_backend import qig_recurrent_reference
from experiments.infinitevl_quant.inventory import LinearTarget, decoder_layers
from experiments.infinitevl_quant.loader import PRE_ROUTED_SLIDING_MASK
from experiments.infinitevl_quant.progress import json_digest
from experiments.vlm_quant_benchmark.mbq import MBQSearchResult, aggregate_gradient_ratios
from experiments.vlm_quant_benchmark import mbq as _mbq
from experiments.vlm_quant_benchmark.vg_qig_mbq import (
    capture_block_traces,
    make_quantized_block,
    qig_token_importance,
    search_vulnerable_block,
)


FORMAT_VERSION = 1
MODEL_KINDS = ("infinitevl", "qwen")
EXPECTED_FFN_LINEARS = 108
EXPECTED_LAYERS = 36
EXPECTED_SPLIT_SIZES = {
    "mbq_calibration": 128,
    "vvs": 16,
    "qig": 32,
    "qig_acceptance": 32,
}
DEFAULT_WEIGHT_BITS = 4
DEFAULT_STEP_GRID = (8, 16, 32)
DEFAULT_SCALE_GRID = tuple(index / 20 for index in range(20))
FFN_QIG_GROUPS = 2
TOP_FRACTION = 0.10


@dataclasses.dataclass(frozen=True)
class MatchedSearchSettings:
    """Frozen search budget shared by both model backends."""

    min_pixels: int
    max_pixels: int
    seed: int = 42
    weight_bits: int = DEFAULT_WEIGHT_BITS
    group_size: int = 128
    scale_grid: tuple[float, ...] = DEFAULT_SCALE_GRID
    step_grid: tuple[int, ...] = DEFAULT_STEP_GRID
    coordinate_sweeps: int = 2
    spearman_threshold: float = 0.98
    jaccard_threshold: float = 0.90
    vvs_relative_threshold: float = 0.05
    vvs_max_layers: int = 4
    vvs_bootstrap_samples: int = 2_000
    budget_hours: float = 18.0

    def __post_init__(self) -> None:
        if self.weight_bits not in (2, 3, 4):
            raise ValueError("matched MBQ/QIG supports W2/W3/W4 A16 only")
        if self.group_size != 128:
            raise ValueError("matched search fixes asymmetric group size to 128")
        if self.min_pixels <= 0 or self.max_pixels < self.min_pixels:
            raise ValueError("invalid pixel bounds")
        if self.step_grid != DEFAULT_STEP_GRID:
            raise ValueError("matched QIG step grid is fixed to (8, 16, 32)")
        if self.coordinate_sweeps != 2:
            raise ValueError("matched QIG uses exactly two coordinate sweeps")
        if self.budget_hours <= 0:
            raise ValueError("QIG budget must be positive")
        if not self.scale_grid or any(alpha < 0 or alpha >= 1 for alpha in self.scale_grid):
            raise ValueError("scale exponents must satisfy 0 <= alpha < 1")

    @property
    def scheme(self) -> QuantScheme:
        scheme = SCHEMES[f"w{self.weight_bits}a16"]
        if scheme.group_size != self.group_size or not scheme.zero_point:
            raise RuntimeError("unexpected shared quantizer configuration")
        return scheme

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self) | {
            "quantization": "asymmetric_groupwise_fake_quant",
            "activation_quantization": None,
            "compute": "float32_cpu",
            "no_speed_or_memory_claim": True,
        }


@dataclasses.dataclass
class MatchedSearchResult:
    """Explicitly separates the three scale states used by the protocol."""

    mbq_scales: dict[str, torch.Tensor]
    qig_candidate_scales: dict[str, torch.Tensor]
    accepted_scales: dict[str, torch.Tensor]
    audit: dict[str, Any]
    cache_hit: bool = False


def resolve_weight_bits(
    rtn_evidence: dict[str, Any] | None = None,
    *,
    requested_bits: int | None = None,
) -> tuple[int, dict[str, Any]]:
    """Resolve the expensive-search format without inventing an RTN decision.

    The RTN/report layer may provide an explicit ``selected_weight_bits`` field.
    In its absence the preregistered recovery target is W4.  We intentionally do
    not infer a winner from arbitrary summary field names or a tiny QA sample.
    """

    if requested_bits is not None:
        bits, source = requested_bits, "explicit_cli_or_runner_choice"
    elif rtn_evidence is not None and "selected_weight_bits" in rtn_evidence:
        bits, source = int(rtn_evidence["selected_weight_bits"]), "rtn_selection_record"
    else:
        bits, source = DEFAULT_WEIGHT_BITS, "preregistered_default_w4"
    if bits not in (2, 3, 4):
        raise ValueError(f"unsupported MBQ/QIG frontier W{bits}A16")
    return bits, {
        "weight_bits": bits,
        "source": source,
        "default_is_not_an_empirical_claim": source == "preregistered_default_w4",
    }


def _sample_ids(samples: Iterable[Any]) -> list[str]:
    ids = [str(sample.id) for sample in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("split contains duplicate sample IDs")
    return ids


def validate_search_contract(
    model_kind: str,
    targets: list[LinearTarget],
    splits: dict[str, list[Any]],
) -> dict[str, Any]:
    if model_kind not in MODEL_KINDS:
        raise ValueError(f"unsupported model kind: {model_kind}")
    counts = {name: len(splits.get(name, ())) for name in EXPECTED_SPLIT_SIZES}
    if counts != EXPECTED_SPLIT_SIZES:
        raise ValueError(
            f"MBQ/QIG split contract mismatch: actual={counts}, "
            f"expected={EXPECTED_SPLIT_SIZES}"
        )
    if len(targets) != EXPECTED_FFN_LINEARS:
        raise ValueError(
            f"matched search requires {EXPECTED_FFN_LINEARS} FFN Linears, "
            f"got {len(targets)}"
        )
    names = [target.name for target in targets]
    if len(names) != len(set(names)):
        raise ValueError("FFN target names are not unique")
    layer_projection = defaultdict(set)
    for target in targets:
        leaf = target.name.rsplit(".", 1)[-1]
        if leaf not in {"gate_proj", "up_proj", "down_proj"}:
            raise ValueError(f"non-FFN target in matched search: {target.name}")
        layer_projection[int(target.layer)].add(leaf)
    expected_projections = {"gate_proj", "up_proj", "down_proj"}
    if set(layer_projection) != set(range(EXPECTED_LAYERS)) or any(
        values != expected_projections for values in layer_projection.values()
    ):
        raise ValueError("targets are not exactly gate/up/down for decoder layers 0..35")

    ids = {name: set(_sample_ids(splits[name])) for name in EXPECTED_SPLIT_SIZES}
    if not ids["vvs"] <= ids["qig"] <= ids["mbq_calibration"]:
        raise ValueError("VVS16 and QIG32 must be nested in calibration128")
    overlap = ids["mbq_calibration"] & ids["qig_acceptance"]
    if overlap:
        raise ValueError(
            f"acceptance32 is not independent of calibration128: {sorted(overlap)[:8]}"
        )
    return {
        "model_kind": model_kind,
        "target_count": len(targets),
        "layers": EXPECTED_LAYERS,
        "split_counts": counts,
        "vvs_within_qig_within_calibration": True,
        "acceptance_disjoint_from_calibration_by_id": True,
    }


def build_search_identity(
    *,
    model_kind: str,
    protocol_identity: str,
    targets: list[LinearTarget],
    splits: dict[str, list[Any]],
    settings: MatchedSearchSettings,
) -> tuple[str, dict[str, Any]]:
    if not protocol_identity:
        raise ValueError("protocol_identity must be non-empty")
    payload = {
        "format_version": FORMAT_VERSION,
        "stage": "matched_ffn_mbq_vvs_qig",
        "model_kind": model_kind,
        "protocol_identity": protocol_identity,
        "targets": [{
            "name": target.name,
            "layer": int(target.layer),
            "weight_shape": list(target.module.weight.shape),
        } for target in targets],
        "samples": {
            name: _sample_ids(splits[name]) for name in EXPECTED_SPLIT_SIZES
        },
        "settings": settings.to_dict(),
        "algorithms": {
            "mbq": "infinitevl_quant.mbq_qig.run_mbq+ffn_only_compat_search",
            "vvs": "exact_per_sample_replay+unchanged_auto_gate_layers",
            "qig": "backend_aware_recurrent_or_standard_ig",
            "acceptance": "global_0.2pct_and_each_source_2pct",
        },
    }
    return json_digest(payload), payload


def _atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_torch_cache(path: Path, *, identity: str, kind: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("format_version") != FORMAT_VERSION
        or payload.get("identity") != identity
        or payload.get("kind") != kind
    ):
        return None
    return payload


def _validate_scales(
    scales: dict[str, torch.Tensor], targets: list[LinearTarget], *, label: str,
) -> dict[str, torch.Tensor]:
    expected = {target.name: target.module.in_features for target in targets}
    if set(scales) != set(expected):
        raise RuntimeError(
            f"{label} scale names mismatch: missing={len(set(expected) - set(scales))}, "
            f"extra={len(set(scales) - set(expected))}"
        )
    result = {}
    for name, width in expected.items():
        value = scales[name].detach().float().cpu()
        if value.ndim != 1 or value.numel() != width:
            raise RuntimeError(f"{label} invalid scale shape at {name}")
        if not torch.isfinite(value).all() or (value <= 0).any():
            raise RuntimeError(f"{label} non-positive/non-finite scale at {name}")
        result[name] = value.clone()
    return result


def _scale_digest(scales: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(scales):
        value = scales[name].detach().float().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _mbq_resume_identity(
    *,
    samples: list[Any],
    targets: list[LinearTarget],
    settings: MatchedSearchSettings,
    reservoir_rows: int,
    modality_mode: str,
    algorithm_version: int,
    attention_implementation: str,
    parent_search_identity: str,
) -> str:
    """Bind a raw-collector checkpoint to every input that affects its state."""

    return json_digest({
        "format_version": FORMAT_VERSION,
        "kind": "matched_ffn_mbq_raw_reservoir_resume",
        "parent_search_identity": parent_search_identity,
        "samples": [{
            "id": str(sample.id),
            "source": str(getattr(sample, "dataset", "unknown")),
        } for sample in samples],
        "targets": [{
            "name": target.name,
            "shape": list(target.module.weight.shape),
        } for target in targets],
        "reservoir_rows": int(reservoir_rows),
        "seed": int(settings.seed),
        "min_pixels": int(settings.min_pixels),
        "max_pixels": int(settings.max_pixels),
        "attention_implementation": attention_implementation,
        "modality_mode": modality_mode,
        "algorithm_version": int(algorithm_version),
    })


def _serialize_mbq_reservoir(reservoir) -> dict[str, Any]:
    """Serialize the complete global priority-reservoir state losslessly."""

    keys = sorted(reservoir.generators)
    return {
        "rows": int(reservoir.rows),
        "modalities": list(reservoir.modalities),
        "values": {
            key: value.detach().cpu().clone()
            for key, value in reservoir.values.items()
        },
        "priorities": {
            key: value.detach().cpu().clone()
            for key, value in reservoir.priorities.items()
        },
        "counts": {key: int(reservoir.counts[key]) for key in keys},
        "generator_states": {
            key: reservoir.generators[key].get_state().cpu().clone()
            for key in keys
        },
    }


def _restore_mbq_reservoir(
    payload: dict[str, Any],
    selected,
    *,
    rows: int,
    seed: int,
    modalities: tuple[str, ...],
):
    """Restore and validate every mutable field of ``_ModalReservoir``."""

    if int(payload.get("rows", -1)) != rows:
        raise RuntimeError("MBQ resume reservoir row count mismatch")
    if tuple(payload.get("modalities", ())) != modalities:
        raise RuntimeError("MBQ resume reservoir modality mismatch")
    reservoir = _mbq._ModalReservoir(selected, rows, seed, modalities)
    expected = set(reservoir.generators)
    values = payload.get("values", {})
    priorities = payload.get("priorities", {})
    counts = payload.get("counts", {})
    generator_states = payload.get("generator_states", {})
    if set(generator_states) != expected or set(counts) != expected:
        raise RuntimeError("MBQ resume reservoir key mismatch")
    if set(values) != set(priorities) or not set(values) <= expected:
        raise RuntimeError("MBQ resume reservoir value/priority key mismatch")
    widths = {item.name: int(item.module.in_features) for item in selected}
    for key in expected:
        name, _modality = key
        count = int(counts[key])
        if count < 0:
            raise RuntimeError(f"negative MBQ resume count at {key}")
        state = generator_states[key]
        if not isinstance(state, torch.Tensor) or state.device.type != "cpu":
            raise RuntimeError(f"invalid MBQ generator state at {key}")
        try:
            reservoir.generators[key].set_state(state)
        except RuntimeError as error:
            raise RuntimeError(f"invalid MBQ generator state at {key}") from error
        reservoir.counts[key] = count
        if key not in values:
            if count:
                raise RuntimeError(f"missing non-empty MBQ reservoir at {key}")
            continue
        value, priority = values[key], priorities[key]
        if (
            not isinstance(value, torch.Tensor)
            or not isinstance(priority, torch.Tensor)
            or value.device.type != "cpu"
            or priority.device.type != "cpu"
            or value.ndim != 2
            or value.shape[1] != widths[name]
            or priority.ndim != 1
            or priority.numel() != value.shape[0]
            or value.shape[0] != min(rows, count)
            or not torch.isfinite(value).all()
            or not torch.isfinite(priority).all()
            or bool((priority < 0).any())
            or bool((priority >= 1).any())
        ):
            raise RuntimeError(f"invalid MBQ reservoir tensor at {key}")
        reservoir.values[key] = value.detach().float().cpu().clone()
        reservoir.priorities[key] = priority.detach().float().cpu().clone()
    return reservoir


def _write_mbq_resume_status(
    path: Path,
    *,
    status: str,
    identity: str,
    expected_ids: list[str],
    completed_ids: list[str],
    state_path: Path,
    detail: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps({
        "format_version": FORMAT_VERSION,
        "identity": identity,
        "status": status,
        "resume_granularity": "one_completed_sample_global_reservoir",
        "sample_count": len(expected_ids),
        "sample_ids": expected_ids,
        "completed_sample_count": len(completed_ids),
        "completed_ordered_sample_ids": completed_ids,
        "state_artifact": str(state_path),
        "mathematical_equivalence": (
            "The checkpoint contains the global reservoir values, priorities, "
            "counts, every per-target/modality torch.Generator state, gradient "
            "records, and token counts; per-sample reservoirs are never concatenated."
        ),
        "detail": detail,
    }, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def _save_mbq_resume_state(
    path: Path,
    *,
    identity: str,
    reservoir,
    gradient_records: dict[str, list[dict]],
    token_counts: dict[str, int],
    completed_ids: list[str],
    modality_mode: str,
    algorithm_version: int,
) -> None:
    """Atomically commit one sample boundary of the global MBQ collector."""

    _atomic_torch(path, {
        "format_version": FORMAT_VERSION,
        "identity": identity,
        "kind": "mbq_raw_global_reservoir_resume",
        "reservoir": _serialize_mbq_reservoir(reservoir),
        "gradient_records": {
            name: list(rows) for name, rows in gradient_records.items()
        },
        "token_counts": {name: int(value) for name, value in token_counts.items()},
        "completed_ordered_sample_ids": list(completed_ids),
        "modality_mode": modality_mode,
        "algorithm_version": int(algorithm_version),
    })


def _load_mbq_resume_state(
    path: Path,
    *,
    identity: str,
    expected_ids: list[str],
    selected,
    rows: int,
    seed: int,
    modalities: tuple[str, ...],
    modality_mode: str,
    algorithm_version: int,
):
    if not path.is_file():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise RuntimeError(f"cannot load MBQ resume state: {path}") from error
    if (
        payload.get("format_version") != FORMAT_VERSION
        or payload.get("identity") != identity
        or payload.get("kind") != "mbq_raw_global_reservoir_resume"
        or payload.get("modality_mode") != modality_mode
        or int(payload.get("algorithm_version", -1)) != algorithm_version
    ):
        raise RuntimeError("MBQ resume state identity/schema mismatch")
    completed_ids = [
        str(value) for value in payload.get("completed_ordered_sample_ids", ())
    ]
    if completed_ids != expected_ids[:len(completed_ids)]:
        raise RuntimeError("MBQ resume completed IDs are not the expected ordered prefix")
    reservoir = _restore_mbq_reservoir(
        payload.get("reservoir", {}), selected,
        rows=rows, seed=seed, modalities=modalities,
    )
    expected_anchors = {
        item.name for item in selected
        if item.name.endswith((".self_attn.o_proj", ".mlp.down_proj"))
    }
    gradient_records = payload.get("gradient_records", {})
    if set(gradient_records) != expected_anchors and completed_ids:
        raise RuntimeError("MBQ resume gradient anchor mismatch")
    for name, records in gradient_records.items():
        if not isinstance(records, list) or len(records) != len(completed_ids):
            raise RuntimeError(f"MBQ resume gradient record count mismatch at {name}")
    token_counts = payload.get("token_counts", {})
    expected_token_keys = (
        {"vision", "language", "samples"}
        if modality_mode == "vision_answer"
        else set(modalities) | {"samples"}
    )
    if set(token_counts) != expected_token_keys:
        raise RuntimeError("MBQ resume token-count schema mismatch")
    if (
        any(int(value) < 0 for value in token_counts.values())
        or int(token_counts["samples"]) != len(completed_ids)
    ):
        raise RuntimeError("MBQ resume token counts are inconsistent")
    return reservoir, defaultdict(list, {
        name: list(records) for name, records in gradient_records.items()
    }), {name: int(value) for name, value in token_counts.items()}, completed_ids


def _collect_mbq_raw_statistics_resumable(
    model: nn.Module,
    selected,
    batches: Iterable[dict],
    *,
    reservoir_rows: int,
    seed: int,
    attention_implementation: str = "sdpa",
    modality_mode: str = "vision_answer",
    algorithm_version: int = 4,
    resume_path: Path,
    resume_status_path: Path,
    resume_identity: str,
    expected_ids: list[str],
    cooperative_check: Callable[[str], None] | None = None,
):
    """Exact legacy MBQ collector with atomic global-state sample checkpoints."""

    if not selected:
        raise ValueError("MBQ requires at least one selected decoder Linear")
    modalities = _mbq.enabled_modalities(modality_mode)
    if modality_mode == "vision_prompt_answer" and algorithm_version != 5:
        raise ValueError("three-token-class MBQ requires algorithm version 5")
    loaded = _load_mbq_resume_state(
        resume_path, identity=resume_identity, expected_ids=expected_ids,
        selected=selected, rows=reservoir_rows, seed=seed,
        modalities=modalities, modality_mode=modality_mode,
        algorithm_version=algorithm_version,
    )
    if loaded is None:
        reservoirs = _mbq._ModalReservoir(
            selected, reservoir_rows, seed, modalities,
        )
        gradient_records: dict[str, list[dict]] = defaultdict(list)
        token_counts = (
            {"vision": 0, "language": 0, "samples": 0}
            if modality_mode == "vision_answer"
            else {modality: 0 for modality in modalities} | {"samples": 0}
        )
        completed_ids: list[str] = []
    else:
        reservoirs, gradient_records, token_counts, completed_ids = loaded

    _write_mbq_resume_status(
        resume_status_path, status="running", identity=resume_identity,
        expected_ids=expected_ids, completed_ids=completed_ids,
        state_path=resume_path,
        detail="restored exact sample boundary" if loaded else "starting empty collector",
    )
    model.eval()
    current_masks: dict[str, torch.Tensor | None] = {
        modality: None for modality in modalities
    }
    activation_handles = []
    captured: dict[str, torch.Tensor] = {}

    def activation_hook(name: str):
        def hook(_module, args):
            if not args or not isinstance(args[0], torch.Tensor):
                return
            value = args[0]
            for modality in modalities:
                mask = current_masks[modality]
                if mask is None or value.shape[:2] != mask.shape:
                    raise RuntimeError(f"MBQ mask/activation shape mismatch at {name}")
                reservoirs.add(name, modality, value[mask])
        return hook

    def gradient_hook(name: str):
        def hook(_module, _args, output):
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"MBQ anchor {name} did not return a tensor")
            captured[name] = output
        return hook

    for item in selected:
        activation_handles.append(
            item.module.register_forward_pre_hook(activation_hook(item.name))
        )
        if item.name.endswith((".self_attn.o_proj", ".mlp.down_proj")):
            activation_handles.append(
                item.module.register_forward_hook(gradient_hook(item.name))
            )
    layers = _mbq._decoder_layers(model)

    def enable_decoder_grad(_module, args):
        if not args or not isinstance(args[0], torch.Tensor):
            raise RuntimeError("MBQ could not intercept first decoder hidden state")
        return (args[0].detach().requires_grad_(True), *args[1:])

    decoder_handle = layers[0].register_forward_pre_hook(enable_decoder_grad)
    parameters = list(model.parameters())
    original_requires_grad = [parameter.requires_grad for parameter in parameters]
    for parameter in parameters:
        parameter.requires_grad_(False)
    seen_batch_ids: list[str] = []
    try:
        with _mbq.temporary_attention_implementation(
            model, attention_implementation,
        ):
            for batch in batches:
                sample_id = str(batch["id"])
                position = len(seen_batch_ids)
                if position >= len(expected_ids) or sample_id != expected_ids[position]:
                    raise RuntimeError(
                        "MBQ batch order/identity differs from the frozen sample order"
                    )
                seen_batch_ids.append(sample_id)
                if position < len(completed_ids):
                    continue
                if cooperative_check is not None:
                    cooperative_check(f"mbq_before_sample:{sample_id}")
                batch_masks = {
                    "vision": batch["vision_mask"].bool(),
                    "answer_language": batch.get(
                        "answer_language_mask", batch["language_mask"],
                    ).bool(),
                }
                if "prompt_language" in modalities:
                    batch_masks["prompt_language"] = batch["prompt_language_mask"].bool()
                shapes = {tuple(mask.shape) for mask in batch_masks.values()}
                if len(shapes) != 1:
                    raise RuntimeError(f"MBQ modality mask shape mismatch for {sample_id}")
                for modality in modalities:
                    mask = batch_masks[modality]
                    if not mask.any():
                        raise ValueError(f"MBQ sample {sample_id} has empty {modality} mask")
                    current_masks[modality] = mask
                for left_index, left in enumerate(modalities):
                    for right in modalities[left_index + 1:]:
                        if (batch_masks[left] & batch_masks[right]).any():
                            raise ValueError(
                                f"MBQ sample {sample_id} has overlapping {left}/{right} masks"
                            )
                if modality_mode == "vision_answer":
                    token_counts["vision"] += int(batch_masks["vision"].sum().item())
                    token_counts["language"] += int(
                        batch_masks["answer_language"].sum().item()
                    )
                else:
                    for modality in modalities:
                        token_counts[modality] += int(batch_masks[modality].sum().item())
                token_counts["samples"] += 1
                captured.clear()
                with torch.enable_grad():
                    output = model(
                        **batch["model_inputs"], use_cache=False, return_dict=True,
                        output_attentions=False, output_hidden_states=False,
                    )
                    if output.loss is None or not torch.isfinite(output.loss):
                        raise ValueError(f"non-finite MBQ SFT loss for {sample_id}")
                    names = list(captured)
                    gradients = torch.autograd.grad(
                        output.loss, [captured[name] for name in names],
                        allow_unused=False,
                    )
                for name, gradient in zip(names, gradients):
                    modality_records = {}
                    for modality in modalities:
                        values = gradient[batch_masks[modality]]
                        if not values.numel():
                            raise RuntimeError(
                                f"MBQ has an empty {modality} gradient at {name}"
                            )
                        absolute_sum = float(values.abs().sum().item())
                        if not torch.isfinite(torch.tensor(absolute_sum)):
                            raise ValueError(
                                f"non-finite MBQ {modality} gradient at {name}"
                            )
                        modality_records[modality] = {
                            "sum": absolute_sum,
                            "count": values.numel(),
                            "mean": absolute_sum / values.numel(),
                        }
                    if modality_mode == "vision_answer":
                        vision = modality_records["vision"]
                        language = modality_records["answer_language"]
                        record = {
                            "source": str(batch.get("source", "unknown")),
                            "vision_sum": vision["sum"],
                            "vision_count": vision["count"],
                            "vision_mean": vision["mean"],
                            "language_sum": language["sum"],
                            "language_count": language["count"],
                            "language_mean": language["mean"],
                        }
                    else:
                        record = {
                            "source": str(batch.get("source", "unknown")),
                            "modalities": modality_records,
                        }
                    gradient_records[name].append(record)
                del output, gradients
                captured.clear()
                _mbq.release_cpu_memory()
                completed_ids.append(sample_id)
                _save_mbq_resume_state(
                    resume_path, identity=resume_identity, reservoir=reservoirs,
                    gradient_records=gradient_records, token_counts=token_counts,
                    completed_ids=completed_ids, modality_mode=modality_mode,
                    algorithm_version=algorithm_version,
                )
                _write_mbq_resume_status(
                    resume_status_path, status="running", identity=resume_identity,
                    expected_ids=expected_ids, completed_ids=completed_ids,
                    state_path=resume_path,
                    detail=f"atomically committed sample {sample_id}",
                )
                if cooperative_check is not None:
                    cooperative_check(f"mbq_after_sample:{sample_id}")
    except BaseException as error:
        _write_mbq_resume_status(
            resume_status_path, status="interrupted_or_failed",
            identity=resume_identity, expected_ids=expected_ids,
            completed_ids=completed_ids, state_path=resume_path,
            detail=f"{type(error).__name__}: {error}",
        )
        raise
    finally:
        decoder_handle.remove()
        for handle in activation_handles:
            handle.remove()
        for parameter, required in zip(parameters, original_requires_grad):
            parameter.requires_grad_(required)
        for modality in modalities:
            current_masks[modality] = None

    if seen_batch_ids != expected_ids:
        raise RuntimeError(
            f"MBQ batch IDs incomplete: saw {len(seen_batch_ids)}, "
            f"expected {len(expected_ids)}"
        )
    raw = _mbq.MBQRawStatistics(
        stats=reservoirs.finish(selected),
        gradient_records=dict(gradient_records),
        token_counts=token_counts,
        modality_mode=modality_mode,
        algorithm_version=algorithm_version,
    )
    _write_mbq_resume_status(
        resume_status_path, status="raw_collection_completed",
        identity=resume_identity, expected_ids=expected_ids,
        completed_ids=completed_ids, state_path=resume_path,
        detail="all samples restored or collected; global scale search may proceed",
    )
    return raw


def _annotate_vvs_audit(audit: dict[str, Any]) -> dict[str, Any]:
    """Make the inherited VVS gate's one-sided semantics impossible to miss."""

    return dict(audit) | {
        "score_definition": "clip(1 - d_quant / d_fp, 0, 1)",
        "directionality": "one_sided_loss_of_vision_dependence",
        "measurement_contains_d_fp_and_d_quant": True,
        "limitation": (
            "A zero/unselected VVS score does not show FFN insensitivity. "
            "Quantization that increases vision-ablation distance or causes a "
            "different global failure is clipped to zero by this gate."
        ),
        "unselected_layers_are_not_evidence_of_ffn_insensitivity": True,
    }


def _matched_ffn_scale_search(
    raw,
    selected,
    scheme: QuantScheme,
    *,
    scale_grid: tuple[float, ...],
) -> MBQSearchResult:
    """FFN-only form of the existing InfiniteVL MBQ scale search.

    The original helper computes an unconditional attention median and raises
    ``StatisticsError`` for an FFN-only target set.  This keeps its exact
    gradient-ratio, reconstruction, tie-break, and group-sharing rules while
    calculating medians only for the selected module kind.
    """

    ratios = aggregate_gradient_ratios(raw.gradient_records, "source_balanced")
    if not ratios:
        raise RuntimeError("MBQ produced no gradient-ratio observations")
    if any(_legacy._kind(name) != "mlp" for name in ratios):
        raise ValueError("matched FFN search received non-MLP gradient records")
    median = statistics.median(float(value) for value in ratios.values())
    anchors = {
        (_legacy._block_name(name), _legacy._kind(name)): float(value)
        for name, value in ratios.items()
    }
    groups = defaultdict(list)
    for item in selected:
        if _legacy._kind(item.name) != "mlp":
            raise ValueError(f"non-FFN selected module: {item.name}")
        groups[(_legacy._block_name(item.name), _legacy._group(item.name))].append(item)

    scales, group_audit = {}, {}
    for (block, group_name), modules in sorted(groups.items()):
        modules = sorted(modules, key=lambda item: item.name)
        ratio = max(anchors[(block, "mlp")], median)
        representative = modules[0]
        stats = raw.stats[representative.name]
        activation = (
            stats.vision.abs().mean(dim=0) * stats.vision_rows
            + stats.language.abs().mean(dim=0) * stats.language_rows
        ) / (stats.vision_rows + stats.language_rows)
        activation = activation.clamp_min(1e-4)
        best, history = None, []
        for alpha in scale_grid:
            scale = activation.pow(alpha).clamp_min(1e-4)
            scale = scale / torch.sqrt(scale.max() * scale.min())
            loss, details = _legacy._candidate_error(
                modules, stats, scale, ratio, scheme,
            )
            row = {
                "alpha": float(alpha),
                "weighted_mae": float(loss),
                "modules": details,
            }
            history.append(row)
            rank = (float(loss), float(alpha))
            if best is None or rank < best[0]:
                best = (rank, scale.clone())
        if best is None:
            raise RuntimeError(f"empty MBQ scale grid for {block}:{group_name}")
        (loss, alpha), scale = best
        for item in modules:
            if item.module.in_features != scale.numel():
                raise RuntimeError(f"shared MBQ scale shape mismatch at {item.name}")
            scales[item.name] = scale.clone()
        group_audit[f"{block}:{group_name}"] = {
            "modules": [item.name for item in modules],
            "selected_alpha": alpha,
            "weighted_mae": loss,
            "raw_vision_to_answer_gradient_ratio": anchors[(block, "mlp")],
            "effective_ratio": ratio,
            "history": history,
        }
    return MBQSearchResult(scales, {
        "algorithm": "matched_ffn_only_modality_balanced_linear_group_search",
        "compatibility_note": (
            "same InfiniteVL MBQ collector/objective; FFN-only median avoids the "
            "legacy helper's empty-attention StatisticsError"
        ),
        "gradient_aggregation": "source_balanced",
        "scale_grid": list(scale_grid),
        "gradient_ratio_median": median,
        "token_counts": raw.token_counts,
        "groups": group_audit,
        "fake_quantized": True,
        "deployment_kernel_claim": False,
    })


def run_mbq_ffn(
    model,
    processor,
    targets: list[LinearTarget],
    samples,
    settings: MatchedSearchSettings,
    *,
    artifact: Path,
    search_identity: str,
    cooperative_check: Callable[[str], None] | None = None,
) -> tuple[MBQSearchResult, bool]:
    """Run exact FFN-only MBQ with global-reservoir sample checkpoints."""

    if not search_identity:
        raise ValueError("search_identity must be non-empty")
    expected_ids = _sample_ids(samples)
    key = json_digest({
        "samples": expected_ids,
        "targets": [target.name for target in targets],
        "scheme": settings.scheme.to_dict(),
        "seed": settings.seed,
        "search_identity": search_identity,
        "collector": "resumable_global_priority_reservoir_v1",
    })
    if artifact.is_file():
        payload = torch.load(artifact, map_location="cpu", weights_only=True)
        if (
            payload.get("format_version") != FORMAT_VERSION
            or payload.get("key") != key
            or not isinstance(payload.get("scales"), dict)
            or not isinstance(payload.get("audit"), dict)
        ):
            raise RuntimeError(f"invalid completed MBQ artifact: {artifact}")
        return MBQSearchResult(payload["scales"], payload["audit"]), True

    selected = _legacy.selected_linears(targets)
    reservoir_rows = 32
    modality_mode = "vision_answer"
    algorithm_version = 4
    attention_implementation = "eager"
    resume_path = artifact.with_suffix(artifact.suffix + ".raw_resume.pt")
    resume_status = artifact.with_suffix(artifact.suffix + ".resume.json")
    resume_identity = _mbq_resume_identity(
        samples=list(samples),
        targets=targets,
        settings=settings,
        reservoir_rows=reservoir_rows,
        modality_mode=modality_mode,
        algorithm_version=algorithm_version,
        attention_implementation=attention_implementation,
        parent_search_identity=search_identity,
    )
    raw = _collect_mbq_raw_statistics_resumable(
        model,
        selected,
        _legacy.mbq_calibration_batches(
            processor,
            samples,
            min_pixels=settings.min_pixels,
            max_pixels=settings.max_pixels,
        ),
        reservoir_rows=reservoir_rows,
        seed=settings.seed,
        attention_implementation=attention_implementation,
        modality_mode=modality_mode,
        algorithm_version=algorithm_version,
        resume_path=resume_path,
        resume_status_path=resume_status,
        resume_identity=resume_identity,
        expected_ids=expected_ids,
        cooperative_check=cooperative_check,
    )
    if cooperative_check is not None:
        cooperative_check("mbq_before_scale_search")
    result = _matched_ffn_scale_search(
        raw,
        selected,
        settings.scheme,
        scale_grid=settings.scale_grid,
    )
    result = MBQSearchResult(result.scales, dict(result.audit) | {
        "resume_granularity": "one_completed_sample_global_reservoir",
        "per_sample_resume_supported": True,
        "resume_state_artifact": str(resume_path),
        "resume_status_artifact": str(resume_status),
        "resume_identity": resume_identity,
        "completed_samples": len(expected_ids),
        "mathematically_equivalent_to_uninterrupted_collector": True,
        "non_equivalent_per_sample_concatenation_used": False,
    })
    artifact.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch(artifact, {
        "format_version": FORMAT_VERSION,
        "key": key,
        "scales": result.scales,
        "audit": result.audit,
    })
    _write_mbq_resume_status(
        resume_status,
        status="completed",
        identity=resume_identity,
        expected_ids=expected_ids,
        completed_ids=expected_ids,
        state_path=resume_path,
        detail="global scale search completed and final artifact committed",
    )
    if cooperative_check is not None:
        cooperative_check("mbq_after_scale_search")
    return result, False


def run_vvs_resumable(
    model,
    processor,
    targets: list[LinearTarget],
    samples,
    scheme: QuantScheme,
    base: MBQSearchResult,
    *,
    cache_root: Path,
    identity: str,
    min_pixels: int,
    max_pixels: int,
    relative_threshold: float,
    max_layers: int,
    bootstrap_samples: int,
    seed: int,
    cooperative_check: Callable[[str], None] | None = None,
) -> tuple[list[int], dict[str, Any]]:
    """Exact sample-wise VVS replay with persistent per-sample outputs.

    ``measure_vulnerability`` is separable by sample: its full-precision and
    one-layer-quantized representations do not aggregate across samples. We
    therefore persist each sample's complete 36-layer rows and apply the
    unchanged global source-stratified gate only after all rows are restored.
    This is not an approximation or a concatenation of fitted reservoirs.
    """

    config = _legacy.experiment_config(scheme, seed=seed)
    selected = _legacy.selected_linears(targets)
    scores: dict[int, list[tuple[str, str, float]]] = defaultdict(list)
    measurement_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total_seconds = 0.0
    cache_hits = 0
    sample_audit = []
    for sample in samples:
        if cooperative_check is not None:
            cooperative_check(f"vvs_before_sample:{sample.id}")
        sample_identity = json_digest({
            "identity": identity,
            "stage": "vvs_sample",
            "sample_id": str(sample.id),
        })
        token = hashlib.sha256(str(sample.id).encode()).hexdigest()[:20]
        path = cache_root / f"vvs_{token}.pt"
        cached = _load_torch_cache(
            path, identity=sample_identity, kind="vvs_sample",
        )
        if cached is None:
            sample_scores, measurement = _legacy.measure_vulnerability(
                model,
                selected,
                base,
                _legacy.mbq_calibration_batches(
                    processor,
                    [sample],
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                ),
                config,
            )
            if set(sample_scores) != set(range(EXPECTED_LAYERS)):
                raise RuntimeError(
                    f"VVS sample {sample.id} did not produce all decoder layers"
                )
            _atomic_torch(path, {
                "format_version": FORMAT_VERSION,
                "identity": sample_identity,
                "kind": "vvs_sample",
                "sample_id": str(sample.id),
                "scores": sample_scores,
                "measurement": measurement,
            })
            hit = False
        else:
            if cached.get("sample_id") != str(sample.id):
                raise RuntimeError(f"VVS sample cache identity mismatch at {path}")
            sample_scores = cached["scores"]
            measurement = cached["measurement"]
            hit = True
            cache_hits += 1
        for layer, rows in sample_scores.items():
            if len(rows) != 1 or str(rows[0][0]) != str(sample.id):
                raise RuntimeError(
                    f"VVS cache must contain one row for {sample.id}/layer {layer}"
                )
            scores[int(layer)].extend(rows)
        per_layer = measurement.get("per_layer", {})
        if set(map(int, per_layer)) != set(range(EXPECTED_LAYERS)):
            raise RuntimeError(f"VVS measurement is incomplete for {sample.id}")
        for layer, rows in per_layer.items():
            if len(rows) != 1 or str(rows[0].get("id")) != str(sample.id):
                raise RuntimeError(
                    f"VVS measurement cache must contain one row for "
                    f"{sample.id}/layer {layer}"
                )
            measurement_rows[str(int(layer))].extend(rows)
        total_seconds += float(measurement.get("seconds", 0.0))
        sample_audit.append({
            "id": str(sample.id),
            "source": str(getattr(sample, "dataset", "unknown")),
            "cache_hit": hit,
            "artifact": str(path),
        })
        if cooperative_check is not None:
            cooperative_check(f"vvs_after_sample:{sample.id}")

    selected_layers, gate = _legacy.auto_gate_layers(
        dict(scores),
        relative_threshold=relative_threshold,
        max_layers=max_layers,
        bootstrap_samples=bootstrap_samples,
        inclusion_probability=0.5,
        seed=seed,
    )
    gate["measurement"] = {
        "per_layer": dict(sorted(measurement_rows.items(), key=lambda item: int(item[0]))),
        "seconds": total_seconds,
        "intervention": "zero_vision_tokens_at_first_decoder_input",
        "measurement": "final_norm_answer_predictor_hidden_mae",
    }
    gate.update({
        "sample_ids": _sample_ids(samples),
        "resume_granularity": "one_sample_all_36_layers",
        "per_sample_resume_supported": True,
        "sample_cache_hits": cache_hits,
        "sample_artifacts": sample_audit,
        "equivalence": (
            "unchanged per-sample vulnerability rows followed by the unchanged "
            "global source-stratified auto_gate_layers"
        ),
    })
    return selected_layers, gate


def _experiment_config(settings: MatchedSearchSettings):
    base = _legacy.experiment_config(
        settings.scheme,
        qig_steps=settings.step_grid,
        qig_alphas=settings.scale_grid,
        seed=settings.seed,
    )
    return dataclasses.replace(
        base,
        calibration_samples=128,
        mbq_scale_grid=settings.scale_grid,
        vg_vvs_samples=16,
        vg_qig_samples=32,
        vg_vvs_threshold=settings.vvs_relative_threshold,
        vg_vvs_max_layers=settings.vvs_max_layers,
        vg_gate_bootstrap_samples=settings.vvs_bootstrap_samples,
        vg_ig_step_grid=settings.step_grid,
        vg_ig_rank_threshold=settings.spearman_threshold,
        vg_ig_topk_jaccard_threshold=settings.jaccard_threshold,
        vg_coordinate_sweeps=settings.coordinate_sweeps,
    )


def _prepare_trace_backend(
    model_kind: str,
    traces: dict[int, list[Any]],
    layers: list[Any],
) -> dict[str, Any]:
    swa_layers = []
    injected = 0
    if model_kind == "infinitevl":
        for layer_index, layer_traces in traces.items():
            is_swa = getattr(layers[layer_index], "layer_type", None) == "sliding_attention"
            if is_swa:
                swa_layers.append(layer_index)
            for trace in layer_traces:
                # Be strict: the marker is meaningful only to the InfiniteVL
                # loader's SWA pre-hook and must never leak into GDN replay.
                trace.kwargs.pop(PRE_ROUTED_SLIDING_MASK, None)
                if is_swa:
                    trace.kwargs[PRE_ROUTED_SLIDING_MASK] = True
                    injected += 1
    elif model_kind == "qwen":
        for layer_traces in traces.values():
            for trace in layer_traces:
                if PRE_ROUTED_SLIDING_MASK in trace.kwargs:
                    raise RuntimeError(
                        "InfiniteVL private sliding-mask kwarg reached a Qwen trace"
                    )
    else:
        raise ValueError(f"unsupported model kind: {model_kind}")
    return {
        "model_kind": model_kind,
        "infinitevl_swa_layers": swa_layers,
        "private_mask_markers_injected": injected,
        "qwen_private_kwarg_count": 0,
    }


def _backend_qig_importance(
    model_kind: str,
    fp_block,
    quant_block,
    trace,
    steps: int,
) -> dict[str, torch.Tensor]:
    backend = qig_recurrent_reference() if model_kind == "infinitevl" else contextlib.nullcontext()
    with backend:
        return qig_token_importance(
            fp_block, quant_block, trace, steps, iqr_multiplier=1.5,
        )


def _validate_importance(value: dict[str, torch.Tensor], *, sample_id: str) -> None:
    if set(value) != {"vision", "answer_language"}:
        raise RuntimeError(f"invalid QIG modalities for {sample_id}: {sorted(value)}")
    for modality, weights in value.items():
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise FloatingPointError(f"invalid QIG weights for {sample_id}/{modality}")


def _average_ranks(value: torch.Tensor) -> torch.Tensor:
    """Return one-dimensional average ranks, including exact tie handling."""

    flattened = value.detach().float().cpu().flatten()
    if not torch.isfinite(flattened).all():
        raise FloatingPointError("QIG convergence vector contains NaN or Inf")
    if not flattened.numel():
        raise ValueError("QIG convergence vector is empty")
    ordered, indices = torch.sort(flattened, stable=True)
    sorted_ranks = torch.empty_like(ordered)
    start = 0
    while start < ordered.numel():
        end = start + 1
        while end < ordered.numel() and bool(ordered[end] == ordered[start]):
            end += 1
        sorted_ranks[start:end] = 0.5 * (start + end - 1)
        start = end
    ranks = torch.empty_like(sorted_ranks)
    ranks[indices] = sorted_ranks
    return ranks


def _spearman_average_ties(left: torch.Tensor, right: torch.Tensor) -> float:
    left, right = left.flatten(), right.flatten()
    if left.numel() != right.numel() or not left.numel():
        raise ValueError("QIG convergence vectors must have equal non-zero length")
    if left.numel() == 1:
        return 1.0 if torch.equal(left, right) else 0.0
    a, b = _average_ranks(left), _average_ranks(right)
    a, b = a - a.mean(), b - b.mean()
    denominator = torch.sqrt(a.square().sum() * b.square().sum())
    if denominator <= 0:
        return 1.0 if torch.equal(left, right) else 0.0
    return float((a * b).sum() / denominator)


def _top_fraction_jaccard_ties(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    fraction: float = TOP_FRACTION,
) -> float:
    """Tie-inclusive top-fraction Jaccard on already masked valid tokens."""

    left, right = left.detach().float().flatten(), right.detach().float().flatten()
    if left.numel() != right.numel() or not left.numel():
        raise ValueError("QIG convergence vectors must have equal non-zero length")
    if not 0 < fraction <= 1:
        raise ValueError("top fraction must be in (0, 1]")
    count = max(1, math.ceil(left.numel() * fraction))

    def selected(value: torch.Tensor) -> set[int]:
        threshold = torch.topk(value, count).values.min()
        # Include the complete boundary tie instead of relying on arbitrary
        # topk ordering. The vectors contain only valid modality tokens here.
        return set(value.ge(threshold).nonzero(as_tuple=False).flatten().tolist())

    a, b = selected(left), selected(right)
    return len(a & b) / len(a | b)


def _convergence_record(trace, value: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Strip structural zeros before QIG step-convergence comparisons."""

    vision_mask = trace.vision_mask.bool()
    answer_mask = trace.answer_mask.bool()
    vision = value["vision"]
    answer = value["answer_language"]
    if vision.shape != vision_mask.shape or answer.shape != answer_mask.shape:
        raise RuntimeError(
            f"QIG importance/mask shape mismatch for {trace.sample_id}: "
            f"vision={tuple(vision.shape)}/{tuple(vision_mask.shape)} "
            f"answer={tuple(answer.shape)}/{tuple(answer_mask.shape)}"
        )
    active_vision = vision[vision_mask].detach().float().cpu()
    active_answer = answer[answer_mask].detach().float().cpu()
    if not active_vision.numel() or not active_answer.numel():
        raise RuntimeError(f"empty valid QIG modality at {trace.sample_id}")
    vector = torch.cat((active_vision.flatten(), active_answer.flatten()))
    if not torch.isfinite(vector).all():
        raise FloatingPointError(f"non-finite valid QIG weights at {trace.sample_id}")
    return {
        "sample_id": str(trace.sample_id),
        "source": str(trace.source),
        "vector": vector,
        "vision_tokens": int(active_vision.numel()),
        "answer_tokens": int(active_answer.numel()),
    }


def _choose_masked_qig_steps(
    results: dict[int, list[dict[str, Any]]],
    *,
    rank_threshold: float,
    jaccard_threshold: float,
) -> tuple[int, dict[str, Any]]:
    """Choose steps from valid tokens with source-balanced median diagnostics."""

    steps = sorted(results)
    if not steps:
        raise ValueError("QIG convergence requires at least one step")
    comparisons = []
    for low, high in zip(steps, steps[1:]):
        low_rows = {
            (str(row["source"]), str(row["sample_id"])): row
            for row in results[low]
        }
        high_rows = {
            (str(row["source"]), str(row["sample_id"])): row
            for row in results[high]
        }
        if not low_rows or set(low_rows) != set(high_rows):
            raise RuntimeError(
                f"unaligned QIG convergence traces for steps {low}/{high}"
            )
        per_source: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: {
                "spearman": [],
                "jaccard": [],
                "valid_tokens": [],
                "positive_mass_tokens": [],
                "empty_positive_mass": [],
            },
        )
        valid_tokens = 0
        positive_mass_tokens = 0
        empty_positive_mass_traces = 0
        for (source, _sample_id), low_row in sorted(low_rows.items()):
            high_row = high_rows[(source, _sample_id)]
            left, right = low_row["vector"], high_row["vector"]
            if left.numel() != right.numel():
                raise RuntimeError("QIG valid-token count changed across IG steps")
            valid_tokens += int(left.numel())
            positive = left.gt(0) | right.gt(0)
            positive_count = int(positive.sum())
            positive_mass_tokens += positive_count
            per_source[source]["valid_tokens"].append(int(left.numel()))
            per_source[source]["positive_mass_tokens"].append(positive_count)
            per_source[source]["empty_positive_mass"].append(positive_count == 0)
            if positive_count == 0:
                # With no attribution mass there is no ranking evidence.  A
                # pair of all-zero vectors must not count as perfect agreement.
                spearman = jaccard = 0.0
                empty_positive_mass_traces += 1
            else:
                spearman = _spearman_average_ties(left[positive], right[positive])
                jaccard = _top_fraction_jaccard_ties(
                    left[positive], right[positive],
                )
            per_source[source]["spearman"].append(spearman)
            per_source[source]["jaccard"].append(jaccard)
        source_rows = {}
        for source, metrics in sorted(per_source.items()):
            source_rows[source] = {
                "median_spearman": float(statistics.median(metrics["spearman"])),
                "median_top10_jaccard": float(statistics.median(metrics["jaccard"])),
                "traces": len(metrics["spearman"]),
                "valid_tokens": int(sum(metrics["valid_tokens"])),
                "positive_mass_tokens": int(sum(metrics["positive_mass_tokens"])),
                "empty_positive_mass_traces": int(sum(
                    metrics["empty_positive_mass"]
                )),
            }
        rank_median = float(statistics.median(
            row["median_spearman"] for row in source_rows.values()
        ))
        jaccard_median = float(statistics.median(
            row["median_top10_jaccard"] for row in source_rows.values()
        ))
        row = {
            "low": low,
            "high": high,
            "median_spearman": rank_median,
            "median_top10_jaccard": jaccard_median,
            "per_source": source_rows,
            "valid_tokens": valid_tokens,
            "positive_mass_tokens": positive_mass_tokens,
            "empty_positive_mass_traces": empty_positive_mass_traces,
            "aggregation": "median_within_source_then_median_across_sources",
            "structural_zeros_excluded": True,
            "zero_mass_valid_tokens_excluded": True,
            "empty_positive_mass_policy": "score_zero_not_converged_evidence",
            "tie_ranking": "average",
        }
        comparisons.append(row)
        if rank_median >= rank_threshold and jaccard_median >= jaccard_threshold:
            return high, {"comparisons": comparisons, "converged": True}
    return steps[-1], {"comparisons": comparisons, "converged": False}


def _qig_budget_projection(
    *,
    consumed_seconds: float,
    measured_seconds: float,
    measured_integrated_steps: int,
    trace_count: int,
    layer_count: int,
    step_grid: tuple[int, ...],
    scale_candidates: int,
    coordinate_sweeps: int,
) -> dict[str, float | int | list[int]]:
    """Conservative QIG ETA including IG, trace time, and block search reserve."""

    if consumed_seconds < 0 or measured_seconds <= 0 or measured_integrated_steps <= 0:
        raise ValueError("invalid QIG timing observations")
    if trace_count <= 0 or layer_count <= 0 or not step_grid:
        raise ValueError("invalid QIG projected workload")
    seconds_per_integrated_step_trace = measured_seconds / measured_integrated_steps
    ig_upper_seconds = (
        seconds_per_integrated_step_trace
        * trace_count
        * layer_count
        * sum(step_grid)
    )
    # One IG integration step includes two forwards and one backward. Treat it
    # as an upper bound for one two-forward block objective. Every alpha also
    # needs an activation-capture forward, conservatively charged as half a
    # unit. Initial/final objective calls add two units per layer.
    search_units_per_trace_layer = (
        2.0
        + coordinate_sweeps
        * FFN_QIG_GROUPS
        * (1.0 + 1.5 * scale_candidates)
    )
    block_search_reserve_seconds = (
        seconds_per_integrated_step_trace
        * trace_count
        * layer_count
        * search_units_per_trace_layer
    )
    remaining_ig_seconds = max(0.0, ig_upper_seconds - measured_seconds)
    projected_total_seconds = (
        consumed_seconds + remaining_ig_seconds + block_search_reserve_seconds
    )
    return {
        "consumed_seconds_at_projection": float(consumed_seconds),
        "measured_seconds": float(measured_seconds),
        "measured_integrated_steps": int(measured_integrated_steps),
        "seconds_per_integrated_step_trace": float(
            seconds_per_integrated_step_trace
        ),
        "trace_count": int(trace_count),
        "layer_count": int(layer_count),
        "step_grid": list(step_grid),
        "ig_upper_seconds": float(ig_upper_seconds),
        "remaining_ig_seconds": float(remaining_ig_seconds),
        "block_search_units_per_trace_layer": float(
            search_units_per_trace_layer
        ),
        "block_search_reserve_seconds": float(block_search_reserve_seconds),
        "projected_total_seconds": float(projected_total_seconds),
        "projected_total_hours": float(projected_total_seconds / 3600.0),
        "includes_capture_elapsed": True,
        "includes_conservative_block_search_reserve": True,
    }


def _restore_budget_decision(
    cached: dict[str, Any] | None,
    *,
    selected_layers: list[int],
    configured_steps: tuple[int, ...],
) -> tuple[list[int], tuple[int, ...], bool, float, dict[str, Any]]:
    """Validate and restore an immutable QIG budget decision."""

    if cached is None:
        return list(selected_layers), tuple(configured_steps), False, 0.0, {}
    consumed = float(cached.get("consumed_seconds", 0.0))
    if not math.isfinite(consumed) or consumed < 0:
        raise RuntimeError("invalid cumulative QIG budget in cache")
    effective_layers = [int(value) for value in cached.get(
        "effective_layers", selected_layers,
    )]
    effective_steps = tuple(int(value) for value in cached.get(
        "effective_steps", configured_steps,
    ))
    if effective_layers != selected_layers[:len(effective_layers)]:
        raise RuntimeError("cached QIG layers are not a selected-layer prefix")
    if not effective_layers or len(effective_steps) < 2:
        raise RuntimeError("cached QIG budget has an empty effective workload")
    if any(step not in configured_steps for step in effective_steps):
        raise RuntimeError("cached QIG steps are outside the configured grid")
    frozen = bool(cached.get("decision_frozen", False))
    decision = dict(cached.get("decision", {}))
    return effective_layers, effective_steps, frozen, consumed, decision


def _importance_cache_path(root: Path, sample_id: str, steps: int) -> Path:
    token = hashlib.sha256(sample_id.encode()).hexdigest()[:20]
    return root / f"qig_{token}_s{steps}.pt"


def _importance_cached(
    *,
    model_kind: str,
    fp_block,
    quant_block,
    trace,
    steps: int,
    root: Path,
    identity: str,
) -> tuple[dict[str, torch.Tensor], bool, float]:
    path = _importance_cache_path(root, str(trace.sample_id), steps)
    cached = _load_torch_cache(path, identity=identity, kind="qig_importance")
    if cached is not None and cached.get("sample_id") == str(trace.sample_id) and cached.get("steps") == steps:
        value = cached["weights"]
        _validate_importance(value, sample_id=str(trace.sample_id))
        # Keep the original timing estimate so an interrupted run makes the
        # same budget decision when all of its first probes are cache hits.
        return value, True, float(cached.get("compute_seconds", 0.0))
    started = time.perf_counter()
    value = _backend_qig_importance(
        model_kind, fp_block, quant_block, trace, steps,
    )
    elapsed = time.perf_counter() - started
    _validate_importance(value, sample_id=str(trace.sample_id))
    _atomic_torch(path, {
        "format_version": FORMAT_VERSION,
        "identity": identity,
        "kind": "qig_importance",
        "sample_id": str(trace.sample_id),
        "steps": steps,
        "compute_seconds": elapsed,
        "weights": {name: tensor.detach().cpu() for name, tensor in value.items()},
    })
    return value, False, elapsed


def run_qig_backend_aware(
    model_kind: str,
    model,
    processor,
    targets: list[LinearTarget],
    samples,
    settings: MatchedSearchSettings,
    base: MBQSearchResult,
    selected_layers: list[int],
    *,
    cache_root: Path,
    identity: str,
    cooperative_check: Callable[[str], None] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Run resumable QIG with only the required backend-specific behavior."""

    base_scales = _validate_scales(base.scales, targets, label="MBQ")
    if not selected_layers:
        return base_scales, {
            "skipped": True,
            "reason": "empty_vulnerability_gate",
            "candidate_equals_mbq": True,
        }
    if len(selected_layers) != len(set(selected_layers)) or any(
        layer < 0 or layer >= EXPECTED_LAYERS for layer in selected_layers
    ):
        raise ValueError(f"invalid VVS-selected layers: {selected_layers}")
    budget_path = cache_root / "budget_decision.pt"
    budget_cache = _load_torch_cache(
        budget_path, identity=identity, kind="qig_budget_decision",
    )
    (
        effective_layers,
        effective_steps,
        decision_frozen,
        prior_consumed_seconds,
        budget_decision,
    ) = _restore_budget_decision(
        budget_cache,
        selected_layers=selected_layers,
        configured_steps=settings.step_grid,
    )
    if budget_cache is not None and budget_cache.get("skipped"):
        return base_scales, {
            "skipped": True,
            "reason": budget_cache["reason"],
            "candidate_equals_mbq": True,
            "budget_decision_cache_hit": True,
            "budget_decision_frozen": decision_frozen,
            "cumulative_elapsed_seconds": prior_consumed_seconds,
            "effective_layers": effective_layers,
            "effective_steps": list(effective_steps),
            "decision": budget_decision,
        }

    session_started = time.perf_counter()
    budget_seconds = settings.budget_hours * 3600.0

    def cumulative_elapsed() -> float:
        return prior_consumed_seconds + (time.perf_counter() - session_started)

    def persist_budget(
        reason: str,
        *,
        skipped: bool = False,
        completed: bool = False,
    ) -> None:
        _atomic_torch(budget_path, {
            "format_version": FORMAT_VERSION,
            "identity": identity,
            "kind": "qig_budget_decision",
            "effective_layers": list(effective_layers),
            "effective_steps": list(effective_steps),
            "decision_frozen": bool(decision_frozen),
            "decision": dict(budget_decision),
            "consumed_seconds": cumulative_elapsed(),
            "budget_seconds": budget_seconds,
            "skipped": bool(skipped),
            "completed": bool(completed),
            "reason": reason,
        })

    if prior_consumed_seconds >= budget_seconds:
        decision_frozen = True
        persist_budget("qig_cumulative_budget_exhausted_before_resume", skipped=True)
        return base_scales, {
            "skipped": True,
            "reason": "qig_cumulative_budget_exhausted_before_resume",
            "candidate_equals_mbq": True,
            "budget_decision_cache_hit": budget_cache is not None,
            "budget_decision_frozen": True,
            "cumulative_elapsed_seconds": cumulative_elapsed(),
            "effective_layers": effective_layers,
            "effective_steps": list(effective_steps),
        }

    config = _experiment_config(settings)
    selected = _legacy.selected_linears(targets)
    decoder = list(decoder_layers(model))
    if cooperative_check is not None:
        cooperative_check("qig_before_trace_capture")
    capture_started = time.perf_counter()
    traces = capture_block_traces(
        model,
        effective_layers if decision_frozen else selected_layers,
        _legacy.mbq_calibration_batches(
            processor,
            samples,
            min_pixels=settings.min_pixels,
            max_pixels=settings.max_pixels,
        ),
        decoder_layers=decoder,
    )
    capture_seconds = time.perf_counter() - capture_started
    if cooperative_check is not None:
        cooperative_check("qig_after_trace_capture")
    trace_backend = _prepare_trace_backend(model_kind, traces, decoder)
    by_layer = defaultdict(list)
    for item in selected:
        layer = int(item.name.split(".layers.", 1)[1].split(".", 1)[0])
        by_layer[layer].append(item)

    result = {name: value.clone() for name, value in base_scales.items()}
    audit: dict[str, Any] = {
        "algorithm": "matched_backend_aware_vulnerability_gated_qig_mbq",
        "baseline": "zero_input",
        "sample_ids": _sample_ids(samples),
        "requested_layers": list(selected_layers),
        "configured_steps": list(settings.step_grid),
        "coordinate_sweeps": settings.coordinate_sweeps,
        "budget_hours": settings.budget_hours,
        "budget_is_cumulative_across_resume": True,
        "prior_consumed_seconds": prior_consumed_seconds,
        "trace_capture_seconds": capture_seconds,
        "trace_backend": trace_backend,
        "layers": {},
        "fake_quantized": True,
        "deployment_kernel_claim": False,
        "budget_decision_cache_hit": budget_cache is not None,
        "budget_decision_frozen": decision_frozen,
        "decision": dict(budget_decision),
    }

    def stop_for_budget(reason: str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        nonlocal decision_frozen
        decision_frozen = True
        persist_budget(reason, skipped=True)
        audit.update({
            "skipped": True,
            "reason": reason,
            "candidate_equals_mbq": True,
            "budget_decision_frozen": True,
            "effective_layers": list(effective_layers),
            "effective_steps": list(effective_steps),
            "cumulative_elapsed_seconds": cumulative_elapsed(),
        })
        return base_scales, audit

    persist_budget("trace_capture_complete")
    if cumulative_elapsed() >= budget_seconds:
        return stop_for_budget("qig_cumulative_budget_exhausted_during_trace_capture")

    first_pair_seconds = None

    for layer_position, layer in enumerate(list(selected_layers)):
        if layer not in effective_layers:
            break
        input_scale_digest = _scale_digest(result)
        layer_identity = json_digest({
            "identity": identity,
            "layer": layer,
            "input_scales": input_scale_digest,
            # The overall identity already fixes the step grid. Keeping this
            # independent of the ETA reduction lets a partial layer resume
            # after the reduction decision is restored.
            "configured_steps": list(settings.step_grid),
        })
        layer_root = cache_root / f"layer_{layer:02d}"
        layer_cache = layer_root / "search.pt"
        cached = _load_torch_cache(
            layer_cache, identity=layer_identity, kind="qig_layer_search",
        )
        if cached is not None:
            if not decision_frozen:
                raise RuntimeError(
                    "QIG layer cache exists before the budget decision was frozen"
                )
            replacements = _validate_scales(
                cached["scales"],
                [target for target in targets if target.layer == layer],
                label=f"cached QIG layer {layer}",
            )
            result.update(replacements)
            audit["layers"][str(layer)] = cached["audit"] | {"cache_hit": True}
            persist_budget("layer_cache_restored")
            continue

        fp_block = copy.deepcopy(decoder[layer]).float().eval()
        for parameter in fp_block.parameters():
            parameter.requires_grad_(False)
        modules = by_layer[layer]
        base_block = make_quantized_block(fp_block, modules, result, config)
        audit_traces, seen_sources = [], set()
        for trace in traces[layer]:
            if trace.source not in seen_sources:
                audit_traces.append(trace)
                seen_sources.add(trace.source)
        if not audit_traces:
            raise RuntimeError(f"layer {layer} has no source-stratified QIG audit traces")

        steps_for_layer = effective_steps
        initial_steps = steps_for_layer[:2]
        convergence_inputs = {step: [] for step in initial_steps}
        weights_cache: dict[tuple[str, int], dict[str, torch.Tensor]] = {}
        cache_hits = 0
        measured_seconds = 0.0
        for trace in audit_traces:
            for step in initial_steps:
                if cumulative_elapsed() >= budget_seconds:
                    return stop_for_budget(
                        "qig_cumulative_budget_exhausted_before_importance"
                    )
                value, hit, seconds = _importance_cached(
                    model_kind=model_kind,
                    fp_block=fp_block,
                    quant_block=base_block,
                    trace=trace,
                    steps=step,
                    root=layer_root / "importance",
                    identity=layer_identity,
                )
                weights_cache[(str(trace.sample_id), step)] = value
                convergence_inputs[step].append(_convergence_record(trace, value))
                cache_hits += int(hit)
                measured_seconds += seconds
                persist_budget("importance_checkpoint")
                if cooperative_check is not None:
                    cooperative_check(
                        f"qig_after_importance:layer{layer}:s{step}:{trace.sample_id}"
                    )
                if cumulative_elapsed() >= budget_seconds:
                    return stop_for_budget(
                        "qig_cumulative_budget_exhausted_after_importance"
                    )

            if (
                not decision_frozen
                and first_pair_seconds is None
                and measured_seconds > 0
            ):
                first_pair_seconds = measured_seconds
                initial_projection = _qig_budget_projection(
                    consumed_seconds=cumulative_elapsed(),
                    measured_seconds=first_pair_seconds,
                    measured_integrated_steps=sum(initial_steps),
                    trace_count=len(traces[layer]),
                    layer_count=len(effective_layers),
                    step_grid=tuple(settings.step_grid),
                    scale_candidates=len(settings.scale_grid),
                    coordinate_sweeps=settings.coordinate_sweeps,
                )
                audit["initial_budget_projection"] = initial_projection
                if initial_projection["projected_total_seconds"] > budget_seconds:
                    effective_layers = effective_layers[:1]
                    effective_steps = settings.step_grid[:2]
                    steps_for_layer = effective_steps
                    revised_projection = _qig_budget_projection(
                        consumed_seconds=cumulative_elapsed(),
                        measured_seconds=first_pair_seconds,
                        measured_integrated_steps=sum(initial_steps),
                        trace_count=len(traces[layer]),
                        layer_count=1,
                        step_grid=tuple(effective_steps),
                        scale_candidates=len(settings.scale_grid),
                        coordinate_sweeps=settings.coordinate_sweeps,
                    )
                    audit["budget_reduction"] = {
                        "effective_layers": list(effective_layers),
                        "effective_steps": list(effective_steps),
                        "revised_projection": revised_projection,
                    }
                    budget_decision = {
                        "initial_projection": initial_projection,
                        "revised_projection": revised_projection,
                        "outcome": "reduced_to_one_layer_two_steps",
                    }
                    decision_frozen = True
                    audit.update({
                        "budget_decision_frozen": True,
                        "decision": dict(budget_decision),
                    })
                    persist_budget("budget_decision_reduced_and_frozen")
                    if revised_projection["projected_total_seconds"] > budget_seconds:
                        return stop_for_budget(
                            "qig_eta_exceeds_18h_after_conservative_reduction"
                        )
                else:
                    budget_decision = {
                        "initial_projection": initial_projection,
                        "outcome": "full_workload_within_budget",
                    }
                    decision_frozen = True
                    audit.update({
                        "budget_decision_frozen": True,
                        "decision": dict(budget_decision),
                    })
                    persist_budget("budget_decision_full_and_frozen")

        chosen_steps, convergence = _choose_masked_qig_steps(
            convergence_inputs,
            rank_threshold=settings.spearman_threshold,
            jaccard_threshold=settings.jaccard_threshold,
        )
        computed_steps = list(initial_steps)
        for step in steps_for_layer[2:]:
            if convergence["converged"]:
                break
            convergence_inputs[step] = []
            for trace in audit_traces:
                if cumulative_elapsed() >= budget_seconds:
                    return stop_for_budget(
                        "qig_cumulative_budget_exhausted_before_importance"
                    )
                value, hit, seconds = _importance_cached(
                    model_kind=model_kind,
                    fp_block=fp_block,
                    quant_block=base_block,
                    trace=trace,
                    steps=step,
                    root=layer_root / "importance",
                    identity=layer_identity,
                )
                weights_cache[(str(trace.sample_id), step)] = value
                convergence_inputs[step].append(_convergence_record(trace, value))
                cache_hits += int(hit)
                measured_seconds += seconds
                persist_budget("importance_checkpoint")
                if cooperative_check is not None:
                    cooperative_check(
                        f"qig_after_importance:layer{layer}:s{step}:{trace.sample_id}"
                    )
                if cumulative_elapsed() >= budget_seconds:
                    return stop_for_budget(
                        "qig_cumulative_budget_exhausted_after_importance"
                    )
            computed_steps.append(step)
            chosen_steps, convergence = _choose_masked_qig_steps(
                convergence_inputs,
                rank_threshold=settings.spearman_threshold,
                jaccard_threshold=settings.jaccard_threshold,
            )

        weights = {}
        for trace in traces[layer]:
            key = (str(trace.sample_id), chosen_steps)
            if key in weights_cache:
                value = weights_cache[key]
            else:
                if cumulative_elapsed() >= budget_seconds:
                    return stop_for_budget(
                        "qig_cumulative_budget_exhausted_before_importance"
                    )
                value, hit, seconds = _importance_cached(
                    model_kind=model_kind,
                    fp_block=fp_block,
                    quant_block=base_block,
                    trace=trace,
                    steps=chosen_steps,
                    root=layer_root / "importance",
                    identity=layer_identity,
                )
                weights_cache[key] = value
                cache_hits += int(hit)
                measured_seconds += seconds
            weights[str(trace.sample_id)] = value
            persist_budget("importance_checkpoint")
            if cooperative_check is not None:
                cooperative_check(
                    f"qig_after_importance:layer{layer}:s{chosen_steps}:{trace.sample_id}"
                )
            if cumulative_elapsed() >= budget_seconds:
                return stop_for_budget(
                    "qig_cumulative_budget_exhausted_after_importance"
                )

        if cumulative_elapsed() >= budget_seconds:
            return stop_for_budget(
                "qig_cumulative_budget_exhausted_before_block_search"
            )
        if cooperative_check is not None:
            cooperative_check(f"qig_before_block_search:layer{layer}")
        search_started = time.perf_counter()
        replacements, search = search_vulnerable_block(
            fp_block,
            modules,
            result,
            traces[layer],
            weights,
            config,
        )
        search_seconds = time.perf_counter() - search_started
        persist_budget("block_search_complete")
        if cooperative_check is not None:
            cooperative_check(f"qig_after_block_search:layer{layer}")
        if cumulative_elapsed() >= budget_seconds:
            return stop_for_budget(
                "qig_cumulative_budget_exhausted_during_block_search"
            )
        if not search.get("non_regression"):
            raise RuntimeError(f"QIG block search regressed its local objective at layer {layer}")
        result.update(replacements)
        layer_audit = {
            "chosen_steps": chosen_steps,
            "convergence": convergence,
            "computed_steps": computed_steps,
            "search": search,
            "importance_cache_hits": cache_hits,
            "importance_compute_seconds": measured_seconds,
            "block_search_seconds": search_seconds,
            "cache_hit": False,
        }
        audit["layers"][str(layer)] = layer_audit
        _atomic_torch(layer_cache, {
            "format_version": FORMAT_VERSION,
            "identity": layer_identity,
            "kind": "qig_layer_search",
            "scales": {name: value.cpu() for name, value in replacements.items()},
            "audit": layer_audit,
        })
        persist_budget("layer_search_committed")
        del fp_block, base_block
        if layer_position + 1 >= len(effective_layers):
            break

    result = _validate_scales(result, targets, label="QIG candidate")
    persist_budget("qig_search_complete", completed=True)
    audit.update({
        "effective_layers": list(effective_layers),
        "effective_steps": list(effective_steps),
        "cumulative_elapsed_seconds": cumulative_elapsed(),
        "session_elapsed_seconds": time.perf_counter() - session_started,
        "budget_decision_frozen": decision_frozen,
        "decision": dict(budget_decision),
    })
    return result, audit


def _module_identity(model, targets: list[LinearTarget]) -> dict[str, tuple]:
    return {
        target.name: (
            module,
            type(module),
            module.weight,
            module.weight.data_ptr(),
            module.weight._version,
            module.bias,
            None if module.bias is None else module.bias.data_ptr(),
            None if module.bias is None else module.bias._version,
        )
        for target in targets
        for module in (model.get_submodule(target.name),)
    }


def _assert_module_identity(model, before: dict[str, tuple]) -> None:
    failures = []
    for name, signature in before.items():
        (
            original, module_type, weight, pointer, version,
            bias, bias_pointer, bias_version,
        ) = signature
        module = model.get_submodule(name)
        if not (
            module is original
            and type(module) is module_type
            and module.weight is weight
            and module.weight.data_ptr() == pointer
            and module.weight._version == version
            and module.bias is bias
            and (
                bias is None
                or (
                    module.bias.data_ptr() == bias_pointer
                    and module.bias._version == bias_version
                )
            )
        ):
            failures.append(name)
    if failures:
        raise RuntimeError(f"acceptance fake-quant context failed to restore: {failures[:8]}")


def _save_scale_stage(
    path: Path,
    *,
    identity: str,
    kind: str,
    scales: dict[str, torch.Tensor],
    audit: dict[str, Any],
) -> None:
    _atomic_torch(path, {
        "format_version": FORMAT_VERSION,
        "identity": identity,
        "kind": kind,
        "scales": {name: value.detach().float().cpu() for name, value in scales.items()},
        "scale_digest": _scale_digest(scales),
        "audit": audit,
    })


def run_matched_mbq_qig(
    model_kind: str,
    model,
    processor,
    targets: list[LinearTarget],
    splits: dict[str, list[Any]],
    *,
    output_dir: Path,
    protocol_identity: str,
    settings: MatchedSearchSettings,
    cooperative_check: Callable[[str], None] | None = None,
) -> MatchedSearchResult:
    """Execute one model's matched MBQ -> VVS -> QIG -> acceptance pipeline."""

    contract = validate_search_contract(model_kind, targets, splits)
    identity, identity_payload = build_search_identity(
        model_kind=model_kind,
        protocol_identity=protocol_identity,
        targets=targets,
        splits=splits,
        settings=settings,
    )
    root = output_dir / model_kind / f"w{settings.weight_bits}a16"
    root.mkdir(parents=True, exist_ok=True)
    identity_path = root / "identity.json"
    if identity_path.is_file():
        existing = json.loads(identity_path.read_text())
        if existing.get("digest") != identity:
            raise RuntimeError(
                f"search identity mismatch at {root}; use a new output directory"
            )
    else:
        temporary = identity_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(
            {"digest": identity, "payload": identity_payload},
            ensure_ascii=False, sort_keys=True, indent=2,
        ) + "\n")
        temporary.replace(identity_path)

    final_path = root / "accepted_scales.pt"
    cached_final = _load_torch_cache(
        final_path, identity=identity, kind="accepted_qig_scales",
    )
    mbq_path = root / "mbq_scales.pt"
    qig_path = root / "qig_candidate_scales.pt"
    if cached_final is not None and mbq_path.is_file() and qig_path.is_file():
        cached_mbq = _load_torch_cache(mbq_path, identity=identity, kind="mbq_scales")
        cached_qig = _load_torch_cache(
            qig_path, identity=identity, kind="qig_candidate_scales",
        )
        if cached_mbq is not None and cached_qig is not None:
            return MatchedSearchResult(
                mbq_scales=_validate_scales(cached_mbq["scales"], targets, label="cached MBQ"),
                qig_candidate_scales=_validate_scales(
                    cached_qig["scales"], targets, label="cached QIG candidate",
                ),
                accepted_scales=_validate_scales(
                    cached_final["scales"], targets, label="cached accepted QIG",
                ),
                audit=cached_final["audit"],
                cache_hit=True,
            )

    mbq_cache = _load_torch_cache(mbq_path, identity=identity, kind="mbq_scales")
    if mbq_cache is None:
        mbq, legacy_cache_hit = run_mbq_ffn(
            model,
            processor,
            targets,
            splits["mbq_calibration"],
            settings,
            artifact=root / f"legacy_mbq_{identity}.pt",
            search_identity=identity,
            cooperative_check=cooperative_check,
        )
        mbq_scales = _validate_scales(mbq.scales, targets, label="MBQ")
        mbq_audit = dict(mbq.audit) | {
            "legacy_artifact_cache_hit": legacy_cache_hit,
            "calibration_samples": 128,
            "sample_ids": _sample_ids(splits["mbq_calibration"]),
        }
        _save_scale_stage(
            mbq_path,
            identity=identity,
            kind="mbq_scales",
            scales=mbq_scales,
            audit=mbq_audit,
        )
    else:
        mbq_scales = _validate_scales(mbq_cache["scales"], targets, label="cached MBQ")
        mbq_audit = mbq_cache["audit"] | {"stage_cache_hit": True}
    mbq = MBQSearchResult(mbq_scales, mbq_audit)

    vvs_path = root / "vvs.pt"
    vvs_cache = _load_torch_cache(vvs_path, identity=identity, kind="vvs")
    if vvs_cache is None:
        selected_layers, vvs_audit = run_vvs_resumable(
            model,
            processor,
            targets,
            splits["vvs"],
            settings.scheme,
            mbq,
            cache_root=root / "vvs_samples",
            identity=identity,
            min_pixels=settings.min_pixels,
            max_pixels=settings.max_pixels,
            relative_threshold=settings.vvs_relative_threshold,
            max_layers=settings.vvs_max_layers,
            bootstrap_samples=settings.vvs_bootstrap_samples,
            seed=settings.seed,
            cooperative_check=cooperative_check,
        )
        vvs_audit = _annotate_vvs_audit(vvs_audit)
        _atomic_torch(vvs_path, {
            "format_version": FORMAT_VERSION,
            "identity": identity,
            "kind": "vvs",
            "selected_layers": list(selected_layers),
            "audit": vvs_audit,
        })
    else:
        selected_layers = [int(value) for value in vvs_cache["selected_layers"]]
        vvs_audit = _annotate_vvs_audit(vvs_cache["audit"]) | {
            "stage_cache_hit": True,
        }

    qig_cache = _load_torch_cache(
        qig_path, identity=identity, kind="qig_candidate_scales",
    )
    if qig_cache is None:
        qig_scales, qig_audit = run_qig_backend_aware(
            model_kind,
            model,
            processor,
            targets,
            splits["qig"],
            settings,
            mbq,
            selected_layers,
            cache_root=root / "qig_cache",
            identity=identity,
            cooperative_check=cooperative_check,
        )
        _save_scale_stage(
            qig_path,
            identity=identity,
            kind="qig_candidate_scales",
            scales=qig_scales,
            audit=qig_audit,
        )
    else:
        qig_scales = _validate_scales(
            qig_cache["scales"], targets, label="cached QIG candidate",
        )
        qig_audit = qig_cache["audit"] | {"stage_cache_hit": True}

    if cached_final is None:
        if cooperative_check is not None:
            cooperative_check("acceptance_before")
        if qig_audit.get("skipped") or not selected_layers:
            accepted_scales = {name: value.clone() for name, value in mbq_scales.items()}
            acceptance_audit = {
                "skipped": True,
                "reason": qig_audit.get("reason", "empty_vulnerability_gate"),
                "accepted_layers": [],
                "final_equals_mbq": True,
            }
        else:
            acceptance_layers = [
                int(value)
                for value in qig_audit.get("effective_layers", selected_layers)
            ]
            before = _module_identity(model, targets)
            try:
                accepted_scales, acceptance_audit = _legacy.accept_qig_layers(
                    model,
                    processor,
                    targets,
                    splits["qig_acceptance"],
                    settings.scheme,
                    mbq_scales,
                    qig_scales,
                    acceptance_layers,
                    min_pixels=settings.min_pixels,
                    max_pixels=settings.max_pixels,
                )
            finally:
                _assert_module_identity(model, before)
            accepted_scales = _validate_scales(
                accepted_scales, targets, label="accepted QIG",
            )
            acceptance_audit = dict(acceptance_audit) | {
                "acceptance_samples": 32,
                "sample_ids": _sample_ids(splits["qig_acceptance"]),
                "ordered_layers": acceptance_layers,
                "global_relative_nll_limit": 0.002,
                "per_source_relative_nll_limit": 0.02,
                "fallback": "rejecting a layer keeps its current MBQ/accepted scale",
            }
        final_audit = {
            "identity": identity,
            "contract": contract,
            "settings": settings.to_dict(),
            "model_kind": model_kind,
            "scale_states": {
                "mbq": _scale_digest(mbq_scales),
                "qig_candidate": _scale_digest(qig_scales),
                "accepted": _scale_digest(accepted_scales),
            },
            "mbq": mbq_audit,
            "vvs": vvs_audit,
            "qig": qig_audit,
            "acceptance": acceptance_audit,
            "labels": [
                "cpu_reference_backend",
                "fake_quantized",
                "ffn_only",
                "matched_cross_model_protocol",
                "no_speed_or_memory_claim",
            ],
        }
        _save_scale_stage(
            final_path,
            identity=identity,
            kind="accepted_qig_scales",
            scales=accepted_scales,
            audit=final_audit,
        )
        (root / ".complete").write_text(identity + "\n")
        if cooperative_check is not None:
            cooperative_check("acceptance_after")
    else:
        accepted_scales = _validate_scales(
            cached_final["scales"], targets, label="cached accepted QIG",
        )
        final_audit = cached_final["audit"]

    return MatchedSearchResult(
        mbq_scales=mbq_scales,
        qig_candidate_scales=qig_scales,
        accepted_scales=accepted_scales,
        audit=final_audit,
        cache_hit=False,
    )
