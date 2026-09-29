"""CPU research implementation of Vulnerability-Gated QIG-MBQ.

This module intentionally implements fake-quantized W2A16/W3A16/W4A16/W4A8
inference.  It is an experimental quality-evaluation backend, not a
packed-kernel backend.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .config import ExperimentConfig, MethodName
from .mbq import (
    MBQSearchResult, _block_identity, _decoder_layers, _effective_group_size,
    _group_kind, mbq_calibration_batches,
)
from .methods import ScaledQuantLinear, W8A8Linear, _parent_and_key
from .quantization import quantize_weight_groupwise
from .selection import SelectedLinear
from .sensitivity import causal_predictor_mask, stable_hash_subset


VG_FORMAT_VERSION = 1
VG_ALGORITHM = "vulnerability_gated_qig_mbq"
GROUP_ORDER = ("attention_input", "attention_output", "mlp_input", "mlp_output")


@dataclasses.dataclass
class BlockTrace:
    sample_id: str
    source: str
    hidden: torch.Tensor
    kwargs: dict[str, Any]
    vision_mask: torch.Tensor
    answer_mask: torch.Tensor


def vulnerability_score(d_fp: float, d_quant: float, eps: float = 1e-12) -> float:
    """Fraction of FP vision dependence lost after quantization."""
    if d_fp <= eps:
        return 0.0
    return min(1.0, max(0.0, 1.0 - d_quant / d_fp))


def _source_balanced(values: list[tuple[str, float]]) -> float:
    by_source: dict[str, list[float]] = defaultdict(list)
    for source, value in values:
        by_source[source].append(float(value))
    if not by_source:
        return 0.0
    return sum(sum(v) / len(v) for v in by_source.values()) / len(by_source)


def auto_gate_layers(
    per_sample_scores: dict[int, list[tuple[str, str, float]]],
    *,
    relative_threshold: float,
    max_layers: int,
    bootstrap_samples: int,
    inclusion_probability: float,
    seed: int,
) -> tuple[list[int], dict]:
    """Source-stratified bootstrap gate with deterministic top-K tie breaking."""
    if max_layers == 0 or not per_sample_scores:
        return [], {"scores": {}, "inclusion_probability": {}, "selected": []}

    def aggregate(rows: list[tuple[str, str, float]]) -> float:
        return _source_balanced([(source, value) for _id, source, value in rows])

    point = {layer: aggregate(rows) for layer, rows in per_sample_scores.items()}
    maximum = max(point.values(), default=0.0)
    eligible = {
        layer for layer, score in point.items()
        if score > 0 and score >= relative_threshold * maximum
    }
    point_top = sorted(eligible, key=lambda x: (-point[x], x))[:max_layers]

    sources: dict[str, list[str]] = defaultdict(list)
    for rows in per_sample_scores.values():
        for sample_id, source, _value in rows:
            if sample_id not in sources[source]:
                sources[source].append(sample_id)
    generator = torch.Generator().manual_seed(seed)
    included = defaultdict(int)
    for _ in range(bootstrap_samples):
        sampled: dict[str, list[str]] = {}
        for source, ids in sorted(sources.items()):
            indices = torch.randint(len(ids), (len(ids),), generator=generator).tolist()
            sampled[source] = [ids[i] for i in indices]
        boot = {}
        for layer, rows in per_sample_scores.items():
            lookup = {(sample_id, source): value for sample_id, source, value in rows}
            vals = []
            for source, ids in sampled.items():
                for sample_id in ids:
                    key = (sample_id, source)
                    if key in lookup:
                        vals.append((source, lookup[key]))
            boot[layer] = _source_balanced(vals)
        boot_max = max(boot.values(), default=0.0)
        candidates = [
            layer for layer, score in boot.items()
            if score > 0 and score >= relative_threshold * boot_max
        ]
        for layer in sorted(candidates, key=lambda x: (-boot[x], x))[:max_layers]:
            included[layer] += 1
    probabilities = {
        layer: included[layer] / bootstrap_samples for layer in sorted(point)
    }
    selected = [
        layer for layer in point_top
        if probabilities[layer] >= inclusion_probability
    ]
    return selected, {
        "scores": point,
        "relative_threshold": relative_threshold,
        "max_layers": max_layers,
        "bootstrap_samples": bootstrap_samples,
        "inclusion_probability": probabilities,
        "required_inclusion_probability": inclusion_probability,
        "selected": selected,
        "empty_gate_fallback": not selected,
    }


def _first_tensor(output: Any) -> torch.Tensor:
    value = output[0] if isinstance(output, (tuple, list)) else output
    if not isinstance(value, torch.Tensor):
        raise TypeError("decoder block did not return a tensor")
    return value


def _block_forward(block: nn.Module, hidden: torch.Tensor, kwargs: dict[str, Any]) -> torch.Tensor:
    return _first_tensor(block(hidden, **kwargs))


def _clone_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    return value


def capture_block_traces(
    model: nn.Module,
    layers: list[int],
    batches,
    *,
    decoder_layers: list[nn.Module],
) -> dict[int, list[BlockTrace]]:
    """Capture exact layer inputs and tensor kwargs from normal full-model calls."""
    traces: dict[int, list[BlockTrace]] = {layer: [] for layer in layers}
    current: dict[str, Any] = {}
    handles = []

    def prehook(layer: int):
        def hook(_module, args, kwargs):
            hidden = args[0] if args else kwargs["hidden_states"]
            allowed = {
                key: _clone_value(value) for key, value in kwargs.items()
                if key in {"attention_mask", "position_ids", "cache_position", "position_embeddings"}
            }
            traces[layer].append(BlockTrace(
                sample_id=current["id"], source=current["source"],
                hidden=hidden.detach().float().cpu().clone(), kwargs=allowed,
                vision_mask=current["vision"].cpu().clone(),
                answer_mask=current["answer"].cpu().clone(),
            ))
        return hook

    for layer in layers:
        handles.append(decoder_layers[layer].register_forward_pre_hook(prehook(layer), with_kwargs=True))
    try:
        with torch.inference_mode():
            for batch in batches:
                current.update({
                    "id": str(batch["id"]),
                    "source": str(batch.get("source", "unknown")),
                    "vision": batch["vision_mask"].bool(),
                    "answer": causal_predictor_mask(batch["language_mask"].bool()),
                })
                model(**batch["model_inputs"], use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    return traces


def measure_vulnerability(
    model: nn.Module,
    selected: list[SelectedLinear],
    base: MBQSearchResult,
    batches,
    config: ExperimentConfig,
    layer_indices: tuple[int, ...] = (),
) -> tuple[dict[int, list[tuple[str, str, float]]], dict]:
    """Measure per-sample vision-dependence loss from one-block quantization."""
    layers = _decoder_layers(model)
    by_layer: dict[int, list[SelectedLinear]] = defaultdict(list)
    for item in selected:
        index = int(_block_identity(item.name).rsplit(".", 1)[1])
        by_layer[index].append(item)
    materialized = list(batches)
    current = {"ablate": False, "mask": None}
    final_hidden: list[torch.Tensor] = []

    def intervene(_module, args):
        if not current["ablate"]:
            return None
        hidden = args[0].clone()
        hidden[current["mask"]] = 0
        return (hidden, *args[1:])

    def capture(_module, _args, output):
        final_hidden.append(_first_tensor(output).detach().float().cpu())

    first_handle = layers[0].register_forward_pre_hook(intervene)
    language_model = None
    for path in ("model.language_model", "language_model", "model"):
        try:
            language_model = model.get_submodule(path)
        except AttributeError:
            continue
        if hasattr(language_model, "norm"):
            break
    if language_model is None or not hasattr(language_model, "norm"):
        first_handle.remove()
        raise TypeError("could not locate final decoder norm for VVS")
    norm_handle = language_model.norm.register_forward_hook(capture)

    def representations() -> list[tuple[torch.Tensor, torch.Tensor]]:
        result = []
        with torch.inference_mode():
            for batch in materialized:
                pair = []
                current["mask"] = batch["vision_mask"].bool()
                for ablate in (False, True):
                    current["ablate"] = ablate
                    final_hidden.clear()
                    model(**batch["model_inputs"], use_cache=False, return_dict=True)
                    pair.append(final_hidden[-1].clone())
                result.append((pair[0], pair[1]))
        return result

    started = time.perf_counter()
    try:
        fp_pairs = representations()
        scores: dict[int, list[tuple[str, str, float]]] = {}
        layer_audit = {}
        indices = list(layer_indices) if layer_indices else list(range(len(layers)))
        for index in indices:
            layer = layers[index]
            fp_block = layer
            modules = by_layer[index]
            quant = make_quantized_block(fp_block, modules, base.scales, config)
            layers[index] = quant
            try:
                quant_pairs = representations()
            finally:
                layers[index] = fp_block
            rows = []
            details = []
            for batch, fp_pair, quant_pair in zip(materialized, fp_pairs, quant_pairs):
                mask = causal_predictor_mask(batch["language_mask"].bool()).cpu()
                d_fp = float((fp_pair[0] - fp_pair[1]).abs().mean(dim=-1)[mask].mean())
                d_quant = float((quant_pair[0] - quant_pair[1]).abs().mean(dim=-1)[mask].mean())
                score = vulnerability_score(d_fp, d_quant)
                sample_id, source = str(batch["id"]), str(batch.get("source", "unknown"))
                rows.append((sample_id, source, score))
                details.append({"id": sample_id, "source": source, "d_fp": d_fp,
                                "d_quant": d_quant, "vvs": score})
            scores[index] = rows
            layer_audit[str(index)] = details
    finally:
        first_handle.remove()
        norm_handle.remove()
        current["ablate"] = False
    return scores, {"per_layer": layer_audit, "seconds": time.perf_counter() - started,
                    "intervention": "zero_vision_tokens_at_first_decoder_input",
                    "measurement": "final_norm_answer_predictor_hidden_mae"}


def iqr_clip(values: torch.Tensor, multiplier: float) -> torch.Tensor:
    finite = values[torch.isfinite(values)]
    if finite.numel() < 4:
        return torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0)
    q1, q3 = torch.quantile(finite.float(), torch.tensor([0.25, 0.75]))
    upper = q3 + multiplier * (q3 - q1)
    return torch.nan_to_num(values, nan=0.0, posinf=float(upper), neginf=0.0).clamp(0, upper)


def qig_token_importance(
    fp_block: nn.Module,
    quant_block: nn.Module,
    trace: BlockTrace,
    steps: int,
    *,
    iqr_multiplier: float,
) -> dict[str, torch.Tensor]:
    """Integrated gradients of exact block-output quantization discrepancy."""
    x = trace.hidden.detach().float()
    total_gradient = torch.zeros_like(x)
    for alpha in torch.linspace(1.0 / steps, 1.0, steps):
        interpolated = (x * alpha).requires_grad_(True)
        fp = _block_forward(fp_block, interpolated, trace.kwargs)
        quant = _block_forward(quant_block, interpolated, trace.kwargs)
        discrepancy = (fp - quant).abs().mean(dim=-1).sum()
        gradient, = torch.autograd.grad(discrepancy, interpolated)
        total_gradient.add_(gradient.detach())
    token_scores = (x * (total_gradient / steps)).abs().mean(dim=-1)
    result = {}
    for name, mask in (
        ("vision", trace.vision_mask), ("answer_language", trace.answer_mask),
    ):
        weights = torch.zeros_like(token_scores)
        clipped = iqr_clip(token_scores[mask], iqr_multiplier)
        mass = clipped.sum()
        if clipped.numel() and mass > 0:
            weights[mask] = clipped / mass
        elif mask.any():
            weights[mask] = mask[mask].float() / mask.sum()
        result[name] = weights.detach()
    return result


def _rank(value: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(value, stable=True)
    rank = torch.empty_like(order, dtype=torch.float32)
    rank[order] = torch.arange(value.numel(), dtype=torch.float32)
    return rank


def spearman(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() != right.numel() or left.numel() < 2:
        return 1.0
    a, b = _rank(left.flatten()), _rank(right.flatten())
    a, b = a - a.mean(), b - b.mean()
    denominator = torch.sqrt(a.square().sum() * b.square().sum())
    return float((a * b).sum() / denominator) if denominator > 0 else 1.0


def top_fraction_jaccard(left: torch.Tensor, right: torch.Tensor, fraction: float = 0.1) -> float:
    count = max(1, math.ceil(left.numel() * fraction))
    a = set(torch.topk(left.flatten(), count).indices.tolist())
    b = set(torch.topk(right.flatten(), count).indices.tolist())
    return len(a & b) / len(a | b)


def choose_qig_steps(
    results: dict[int, list[torch.Tensor]],
    *, rank_threshold: float,
    jaccard_threshold: float,
) -> tuple[int, dict]:
    """Choose the first adjacent IG step pair stable on median audit metrics."""
    steps = sorted(results)
    comparisons = []
    for low, high in zip(steps, steps[1:]):
        rank = [spearman(a, b) for a, b in zip(results[low], results[high])]
        jac = [top_fraction_jaccard(a, b) for a, b in zip(results[low], results[high])]
        rank_median = float(torch.tensor(rank).median())
        jac_median = float(torch.tensor(jac).median())
        row = {"low": low, "high": high, "median_spearman": rank_median,
               "median_top10_jaccard": jac_median}
        comparisons.append(row)
        if rank_median >= rank_threshold and jac_median >= jaccard_threshold:
            return high, {"comparisons": comparisons, "converged": True}
    return steps[-1], {"comparisons": comparisons, "converged": False}


def _local_name(full_name: str) -> str:
    marker = ".layers."
    tail = full_name.split(marker, 1)[1]
    return tail.split(".", 1)[1]


def _install_scaled(
    target_block: nn.Module,
    fp_block: nn.Module,
    item: SelectedLinear,
    scale: torch.Tensor,
    config: ExperimentConfig,
) -> None:
    local = _local_name(item.name)
    source = fp_block.get_submodule(local)
    quantized = quantize_weight_groupwise(
        source.weight.detach().float() * scale.float().unsqueeze(0),
        bits=config.weight_bits,
        group_size=_effective_group_size(source, config),
        zero_point=config.zero_point,
    )
    parent, key = _parent_and_key(target_block, local)
    replacement = (
        ScaledQuantLinear(source, quantized.dequantized, scale.float())
        if config.activation_bits is None
        else W8A8Linear(
            source,
            quantized.dequantized,
            scale.float(),
            activation_bits=config.activation_bits,
        )
    )
    setattr(parent, key, replacement)


def make_quantized_block(
    fp_block: nn.Module,
    modules: list[SelectedLinear],
    scales: dict[str, torch.Tensor],
    config: ExperimentConfig,
) -> nn.Module:
    block = copy.deepcopy(fp_block).float().eval()
    for item in modules:
        _install_scaled(block, fp_block, item, scales[item.name], config)
    return block


def _weighted_error(error: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor | None) -> float:
    values = error[mask]
    if not values.numel():
        return 0.0
    if weight is None:
        return float(values.mean())
    selected = weight[mask]
    mass = selected.sum()
    return float((values * selected).sum() / mass) if mass > 0 else float(values.mean())


def block_objective(
    fp_block: nn.Module,
    quant_block: nn.Module,
    traces: list[BlockTrace],
    qig_weights: dict[str, dict[str, torch.Tensor]] | None,
) -> tuple[float, dict]:
    rows = []
    with torch.inference_mode():
        for trace in traces:
            reference = _block_forward(fp_block, trace.hidden, trace.kwargs)
            candidate = _block_forward(quant_block, trace.hidden, trace.kwargs)
            error = (reference - candidate).abs().mean(dim=-1)
            weights = None if qig_weights is None else qig_weights[trace.sample_id]
            vision = _weighted_error(error, trace.vision_mask, None if weights is None else weights["vision"])
            answer = _weighted_error(error, trace.answer_mask, None if weights is None else weights["answer_language"])
            rows.append((trace.source, vision, answer))
    vision = _source_balanced([(source, value) for source, value, _ in rows])
    answer = _source_balanced([(source, value) for source, _, value in rows])
    return 0.5 * vision + 0.5 * answer, {"vision_mae": vision, "answer_mae": answer}


def _activation_scale(
    fp_block: nn.Module,
    representative: SelectedLinear,
    traces: list[BlockTrace],
    alpha: float,
) -> torch.Tensor:
    captured = []
    local = _local_name(representative.name)
    module = fp_block.get_submodule(local)

    def hook(_module, args):
        captured.append(args[0].detach().float())

    handle = module.register_forward_pre_hook(hook)
    rows = []
    try:
        with torch.inference_mode():
            for trace in traces:
                captured.clear()
                _block_forward(fp_block, trace.hidden, trace.kwargs)
                value = captured[0]
                rows.append((trace.source, "vision", value[trace.vision_mask].abs().mean(dim=0)))
                rows.append((trace.source, "answer_language", value[trace.answer_mask].abs().mean(dim=0)))
    finally:
        handle.remove()
    modality_means = []
    for modality in ("vision", "answer_language"):
        source_means = []
        for source in sorted({row[0] for row in rows}):
            values = [row[2] for row in rows if row[0] == source and row[1] == modality]
            if values:
                source_means.append(torch.stack(values).mean(dim=0))
        modality_means.append(torch.stack(source_means).mean(dim=0))
    activation = (modality_means[0] + modality_means[1]).mul(0.5).clamp_min(1e-4)
    scale = activation.pow(alpha).clamp_min(1e-4)
    return scale / torch.sqrt(scale.max() * scale.min())


def search_vulnerable_block(
    fp_block: nn.Module,
    modules: list[SelectedLinear],
    base_scales: dict[str, torch.Tensor],
    traces: list[BlockTrace],
    qig_weights: dict[str, dict[str, torch.Tensor]] | None,
    config: ExperimentConfig,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Deterministic group-coordinate search using exact block outputs."""
    groups: dict[str, list[SelectedLinear]] = defaultdict(list)
    for item in modules:
        groups[_group_kind(item.name)].append(item)
    current = {item.name: base_scales[item.name].clone() for item in modules}
    quant_block = make_quantized_block(fp_block, modules, current, config)
    initial, _ = block_objective(fp_block, quant_block, traces, qig_weights)
    history = []
    for sweep in range(config.vg_coordinate_sweeps):
        changed = False
        for kind in GROUP_ORDER:
            members = sorted(groups.get(kind, []), key=lambda item: item.name)
            if not members:
                continue
            representative = members[0]
            candidates = [("base", None, current[representative.name].clone())]
            for alpha in config.mbq_scale_grid:
                candidates.append(("alpha", float(alpha), _activation_scale(
                    fp_block, representative, traces, float(alpha),
                )))
            best = None
            rows = []
            for candidate_type, alpha, scale in candidates:
                for item in members:
                    _install_scaled(quant_block, fp_block, item, scale, config)
                loss, parts = block_objective(fp_block, quant_block, traces, qig_weights)
                rows.append({"candidate": candidate_type, "alpha": alpha, "loss": loss, **parts})
                tie = (0 if candidate_type == "base" else 1, -1.0 if alpha is None else alpha)
                rank = (loss, *tie)
                if best is None or rank < best[0]:
                    best = (rank, candidate_type, alpha, scale.clone())
            assert best is not None
            _rank_value, candidate_type, alpha, scale = best
            for item in members:
                old = current[item.name]
                if not torch.equal(old, scale):
                    changed = True
                current[item.name] = scale.clone()
                _install_scaled(quant_block, fp_block, item, scale, config)
            history.append({"sweep": sweep, "group": kind, "selected": candidate_type,
                            "alpha": alpha, "candidates": rows})
        if not changed:
            break
    final, parts = block_objective(fp_block, quant_block, traces, qig_weights)
    return current, {"initial_loss": initial, "final_loss": final, **parts,
                     "non_regression": final <= initial + 1e-12, "history": history}


def vg_cache_key(
    *, model_revision: str, calibration_digest: str, selected_names: list[str],
    config: ExperimentConfig,
) -> str:
    payload = {
        "format_version": VG_FORMAT_VERSION, "algorithm": VG_ALGORITHM,
        "model_revision": model_revision, "calibration_digest": calibration_digest,
        "selected_names": selected_names, "config": config.to_dict(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def vg_vulnerability_cache_key(
    *, model_revision: str, calibration_digest: str, selected_names: list[str],
    config: ExperimentConfig,
) -> str:
    """Method-independent VVS key shared by QIG and uniform VG variants."""
    payload_config = config.to_dict()
    payload_config["method"] = "vg_shared_vulnerability"
    for field in (
        "vg_ig_step_grid", "vg_ig_rank_threshold",
        "vg_ig_topk_jaccard_threshold", "vg_iqr_multiplier",
        "vg_coordinate_sweeps",
    ):
        payload_config.pop(field, None)
    payload = {
        "format_version": VG_FORMAT_VERSION, "stage": "vulnerability",
        "model_revision": model_revision, "calibration_digest": calibration_digest,
        "selected_names": selected_names, "config": payload_config,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save_vg_search(path: Path, *, key: str, result: MBQSearchResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"format_version": VG_FORMAT_VERSION, "key": key,
                "scales": result.scales, "audit": result.audit}, temporary)
    temporary.replace(path)


def load_vg_search(path: Path, *, key: str) -> MBQSearchResult | None:
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != VG_FORMAT_VERSION or payload.get("key") != key:
        return None
    return MBQSearchResult(payload["scales"], payload["audit"])


def finalize_vg_result(
    base: MBQSearchResult,
    replacements: dict[str, torch.Tensor],
    *, gate_audit: dict, qig_audit: dict, search_audit: dict,
    method: MethodName,
) -> MBQSearchResult:
    scales = {name: value.clone() for name, value in base.scales.items()}
    scales.update({name: value.clone() for name, value in replacements.items()})
    return MBQSearchResult(scales, {
        "format_version": VG_FORMAT_VERSION, "algorithm": VG_ALGORITHM,
        "method": method.value, "base_mbq": base.audit,
        "vulnerability_gate": gate_audit, "qig": qig_audit,
        "block_search": search_audit,
        "changed_module_count": len(replacements),
        "implementation_tier": "cpu_research_approximation",
        "deployment_kernel_claim": False,
    })


def _atomic_torch(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _sample_subset(samples, total: int, salt: str):
    sources = sorted({sample.dataset for sample in samples})
    if not sources or total % len(sources):
        raise ValueError(f"VG sample count {total} must divide {len(sources)} sources")
    return stable_hash_subset(samples, total // len(sources), salt=salt)


def collect_vg_qig_search_cached(
    model: nn.Module,
    selected: list[SelectedLinear],
    processor,
    calibration_samples,
    config: ExperimentConfig,
    base: MBQSearchResult,
    *,
    cache_dir: Path,
    key: str,
    vulnerability_key: str | None = None,
) -> tuple[MBQSearchResult, bool]:
    """Run all VG stages with atomic layer/sample caches on CPU."""
    cache_path = cache_dir / f"vg_{key}.pt"
    cached = load_vg_search(cache_path, key=key)
    if cached is not None:
        return cached, True
    stage_root = cache_dir / f"vg_{key}"
    stage_root.mkdir(parents=True, exist_ok=True)
    vvs_samples = _sample_subset(
        calibration_samples, config.vg_vvs_samples, f"vg-vvs-v{config.vg_algorithm_version}",
    )
    qig_samples = _sample_subset(
        calibration_samples, config.vg_qig_samples, f"vg-qig-v{config.vg_algorithm_version}",
    )

    vvs_path = cache_dir / f"vg_vulnerability_{vulnerability_key or key}.pt"
    if vvs_path.exists():
        payload = torch.load(vvs_path, map_location="cpu", weights_only=True)
        per_sample_scores, measurement_audit = payload["scores"], payload["audit"]
    else:
        per_sample_scores, measurement_audit = measure_vulnerability(
            model, selected, base,
            mbq_calibration_batches(
                processor, vvs_samples, min_pixels=config.min_pixels,
                max_pixels=config.max_pixels,
            ),
            config,
            layer_indices=config.vg_candidate_layers,
        )
        _atomic_torch(vvs_path, {"scores": per_sample_scores, "audit": measurement_audit})
    gated_layers, gate_audit = auto_gate_layers(
        per_sample_scores,
        relative_threshold=config.vg_vvs_threshold,
        max_layers=config.vg_vvs_max_layers,
        bootstrap_samples=config.vg_gate_bootstrap_samples,
        inclusion_probability=config.vg_gate_inclusion_probability,
        seed=config.seed,
    )
    gate_audit["measurement"] = measurement_audit
    gate_audit["sample_ids"] = [sample.id for sample in vvs_samples]
    if not gated_layers:
        result = finalize_vg_result(
            base, {}, gate_audit=gate_audit,
            qig_audit={"skipped": True, "reason": "empty_vulnerability_gate"},
            search_audit={"skipped": True}, method=config.method,
        )
        save_vg_search(cache_path, key=key, result=result)
        return result, False

    traces = capture_block_traces(
        model, gated_layers,
        mbq_calibration_batches(
            processor, qig_samples, min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
        ),
        decoder_layers=list(_decoder_layers(model)),
    )
    by_layer: dict[int, list[SelectedLinear]] = defaultdict(list)
    for item in selected:
        by_layer[int(_block_identity(item.name).rsplit(".", 1)[1])].append(item)

    all_replacements: dict[str, torch.Tensor] = {}
    qig_audit: dict[str, Any] = {
        "enabled": config.method == MethodName.VG_QIG_MBQ,
        "baseline": "zero_input", "sample_ids": [sample.id for sample in qig_samples],
        "layers": {},
    }
    search_audit = {}
    decoder = list(_decoder_layers(model))
    for layer in gated_layers:
        layer_dir = stage_root / f"layer_{layer:02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        layer_cache = layer_dir / "search.pt"
        if layer_cache.exists():
            payload = torch.load(layer_cache, map_location="cpu", weights_only=True)
            all_replacements.update(payload["scales"])
            qig_audit["layers"][str(layer)] = payload["qig_audit"]
            search_audit[str(layer)] = payload["search_audit"]
            continue
        fp_block = copy.deepcopy(decoder[layer]).float().eval()
        for parameter in fp_block.parameters():
            parameter.requires_grad_(False)
        modules = by_layer[layer]
        base_block = make_quantized_block(fp_block, modules, base.scales, config)
        weights_by_sample = None
        layer_qig_audit: dict[str, Any] = {"mode": "uniform"}
        if config.method == MethodName.VG_QIG_MBQ:
            audit_traces = []
            seen_sources = set()
            for trace in traces[layer]:
                if trace.source not in seen_sources:
                    audit_traces.append(trace)
                    seen_sources.add(trace.source)
            convergence_inputs: dict[int, list[torch.Tensor]] = {
                step: [] for step in config.vg_ig_step_grid
            }
            audit_weights: dict[tuple[str, int], dict[str, torch.Tensor]] = {}
            for trace in audit_traces:
                for step in config.vg_ig_step_grid:
                    value = qig_token_importance(
                        fp_block, base_block, trace, step,
                        iqr_multiplier=config.vg_iqr_multiplier,
                    )
                    audit_weights[(trace.sample_id, step)] = value
                    convergence_inputs[step].append(torch.cat([
                        value["vision"].flatten(), value["answer_language"].flatten(),
                    ]))
            chosen_steps, convergence = choose_qig_steps(
                convergence_inputs,
                rank_threshold=config.vg_ig_rank_threshold,
                jaccard_threshold=config.vg_ig_topk_jaccard_threshold,
            )
            weights_by_sample = {}
            for trace in traces[layer]:
                qig_file = layer_dir / (
                    "qig_" + hashlib.sha256(trace.sample_id.encode()).hexdigest()[:16] + ".pt"
                )
                if qig_file.exists():
                    value = torch.load(qig_file, map_location="cpu", weights_only=True)["weights"]
                elif (trace.sample_id, chosen_steps) in audit_weights:
                    value = audit_weights[(trace.sample_id, chosen_steps)]
                    _atomic_torch(qig_file, {"steps": chosen_steps, "weights": value})
                else:
                    value = qig_token_importance(
                        fp_block, base_block, trace, chosen_steps,
                        iqr_multiplier=config.vg_iqr_multiplier,
                    )
                    _atomic_torch(qig_file, {"steps": chosen_steps, "weights": value})
                weights_by_sample[trace.sample_id] = value
            layer_qig_audit = {
                "mode": "qig", "chosen_steps": chosen_steps,
                "convergence": convergence,
                "audit_sample_ids": [trace.sample_id for trace in audit_traces],
                "iqr_multiplier": config.vg_iqr_multiplier,
            }
        layer_scales, layer_search = search_vulnerable_block(
            fp_block, modules, base.scales, traces[layer], weights_by_sample, config,
        )
        # The base candidate is always present, so this check is both an
        # invariant and a guard against numerical/search regressions.
        if not layer_search["non_regression"]:
            raise RuntimeError(f"VG block search regressed layer {layer}")
        all_replacements.update(layer_scales)
        qig_audit["layers"][str(layer)] = layer_qig_audit
        search_audit[str(layer)] = layer_search
        _atomic_torch(layer_cache, {"scales": layer_scales,
                                   "qig_audit": layer_qig_audit,
                                   "search_audit": layer_search})
        del fp_block, base_block
    result = finalize_vg_result(
        base, all_replacements, gate_audit=gate_audit,
        qig_audit=qig_audit, search_audit=search_audit, method=config.method,
    )
    save_vg_search(cache_path, key=key, result=result)
    return result, False
