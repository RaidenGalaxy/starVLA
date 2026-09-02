"""CPU-compatible MBQ adaptation for Qwen2.5-VL decoder quantization.

The implementation follows the modality-balanced objective from thu-nics/MBQ
at commit a4d460dfb4b1c07b5d1f3ddda6e86d1c90d6e7f1.  It deliberately produces
fake-quantized floating-point modules; it does not include the authors' CUDA
kernels or make deployment speed/memory claims.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import statistics
import time
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .calibration import release_cpu_memory, temporary_attention_implementation
from .config import ExperimentConfig, QuantScope
from .evaluation import processor_inputs, target_token_span
from .methods import AppliedQuantization, ScaledQuantLinear, W8A8Linear, _parent_and_key
from .quantization import dynamic_activation_qdq, quantize_weight_groupwise
from .selection import SelectedLinear


UPSTREAM_REPOSITORY = "https://github.com/thu-nics/MBQ"
UPSTREAM_COMMIT = "a4d460dfb4b1c07b5d1f3ddda6e86d1c90d6e7f1"
FORMAT_VERSION = 4


@dataclasses.dataclass
class ModalityActivationStats:
    vision: torch.Tensor
    language: torch.Tensor
    vision_rows: int
    language_rows: int


@dataclasses.dataclass
class MBQSearchResult:
    scales: dict[str, torch.Tensor]
    audit: dict


@dataclasses.dataclass
class MBQRawStatistics:
    stats: dict[str, ModalityActivationStats]
    gradient_records: dict[str, list[dict]]
    token_counts: dict


def mbq_cache_key(
    *,
    model_revision: str,
    calibration_digest: str,
    selected_names: list[str],
    config: ExperimentConfig,
    reservoir_rows: int,
    attention_implementation: str,
) -> str:
    payload = {
        "format_version": FORMAT_VERSION,
        "algorithm": "modality_balanced_decoder_group_reconstruction",
        "upstream_commit": UPSTREAM_COMMIT,
        "model_revision": model_revision,
        "calibration_digest": calibration_digest,
        "selected_names": selected_names,
        "scheme_id": config.scheme_id,
        "weight_bits": config.weight_bits,
        "activation_bits": config.activation_bits,
        "group_size": config.group_size,
        "zero_point": config.zero_point,
        "weight_granularity": config.weight_granularity,
        "scale_grid": config.mbq_scale_grid,
        "loss_mode": config.mbq_loss_mode,
        "reweight": config.mbq_reweight,
        "distort": config.mbq_distort,
        "gradient_aggregation": config.mbq_gradient_aggregation,
        "objective_normalization": config.mbq_objective_normalization,
        "median_floor": config.mbq_median_floor,
        "reconstruction_level": config.mbq_reconstruction_level,
        "min_pixels": config.min_pixels,
        "max_pixels": config.max_pixels,
        "reservoir_rows": reservoir_rows,
        "seed": config.seed,
        "attention_implementation": attention_implementation,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def mbq_statistics_cache_key(
    *,
    model_revision: str,
    calibration_digest: str,
    selected_names: list[str],
    reservoir_rows: int,
    seed: int,
    min_pixels: int,
    max_pixels: int,
    attention_implementation: str,
) -> str:
    """Key expensive raw MBQ observations independently of search variants."""
    payload = {
        "format_version": FORMAT_VERSION,
        "algorithm": "mbq_raw_modality_statistics",
        "upstream_commit": UPSTREAM_COMMIT,
        "model_revision": model_revision,
        "calibration_digest": calibration_digest,
        "selected_names": selected_names,
        "reservoir_rows": reservoir_rows,
        "seed": seed,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "attention_implementation": attention_implementation,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def mbq_calibration_batches(processor, samples, *, min_pixels: int, max_pixels: int):
    """Build teacher-forcing inputs plus exact vision/assistant masks."""
    tokenizer = processor.tokenizer
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    image_token = getattr(processor, "image_token_id", None)
    if image_token is None:
        image_token = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if im_end is None or im_end < 0 or image_token is None or image_token < 0:
        raise ValueError("processor is missing Qwen image/im_end token IDs")
    trailing = set(tokenizer.encode("\n", add_special_tokens=False))
    for sample in samples:
        prompt = processor_inputs(
            processor, sample, False, min_pixels=min_pixels, max_pixels=max_pixels,
        )
        teacher = processor_inputs(
            processor, sample, True, min_pixels=min_pixels, max_pixels=max_pixels,
        )
        labels, _positions = target_token_span(
            prompt["input_ids"], teacher["input_ids"],
            im_end_token_id=im_end, excluded_trailing_token_ids=trailing,
        )
        vision_mask = teacher["input_ids"].eq(int(image_token))
        language_mask = labels.ne(-100)
        prompt_mask = torch.zeros_like(vision_mask)
        prompt_mask[..., :prompt["input_ids"].shape[-1]] = True
        prompt_mask &= teacher.get(
            "attention_mask", torch.ones_like(teacher["input_ids"]),
        ).bool()
        for special_id in tokenizer.all_special_ids:
            prompt_mask &= teacher["input_ids"].ne(int(special_id))
        prompt_mask &= ~vision_mask
        if not vision_mask.any() or not language_mask.any():
            raise ValueError(f"MBQ sample {sample.id} has an empty modality mask")
        if (vision_mask & language_mask).any():
            raise ValueError(f"MBQ sample {sample.id} has overlapping modality masks")
        model_inputs = dict(teacher)
        model_inputs["labels"] = labels
        yield {
            "id": sample.id,
            "source": sample.dataset,
            "model_inputs": model_inputs,
            "vision_mask": vision_mask,
            "language_mask": language_mask,
            "prompt_language_mask": prompt_mask,
        }


class _ModalReservoir:
    def __init__(self, selected: list[SelectedLinear], rows: int, seed: int) -> None:
        self.rows = rows
        self.values: dict[tuple[str, str], torch.Tensor] = {}
        self.priorities: dict[tuple[str, str], torch.Tensor] = {}
        self.counts: dict[tuple[str, str], int] = defaultdict(int)
        self.generators = {}
        for item in selected:
            for modality in ("vision", "language"):
                digest = hashlib.sha256(
                    f"{seed}:{item.name}:{modality}".encode()
                ).digest()
                generator = torch.Generator(device="cpu")
                generator.manual_seed(int.from_bytes(digest[:8], "little") % (2**63 - 1))
                self.generators[(item.name, modality)] = generator

    def add(self, name: str, modality: str, value: torch.Tensor) -> None:
        value = value.detach().float().reshape(-1, value.shape[-1]).cpu()
        if value.numel() == 0:
            return
        if not torch.isfinite(value).all():
            raise ValueError(f"non-finite MBQ activation at {name}/{modality}")
        key = (name, modality)
        self.counts[key] += value.shape[0]
        priority = torch.rand(value.shape[0], generator=self.generators[key])
        if key in self.values:
            value = torch.cat((self.values[key], value), dim=0)
            priority = torch.cat((self.priorities[key], priority), dim=0)
        keep = min(self.rows, value.shape[0])
        chosen = torch.topk(priority, keep, sorted=True).indices
        self.values[key] = value[chosen].clone()
        self.priorities[key] = priority[chosen].clone()

    def finish(self, selected: list[SelectedLinear]) -> dict[str, ModalityActivationStats]:
        result = {}
        for item in selected:
            vision = self.values.get((item.name, "vision"))
            language = self.values.get((item.name, "language"))
            if vision is None or language is None:
                raise RuntimeError(f"MBQ did not observe both modalities at {item.name}")
            result[item.name] = ModalityActivationStats(
                vision=vision,
                language=language,
                vision_rows=self.counts[(item.name, "vision")],
                language_rows=self.counts[(item.name, "language")],
            )
        return result


def _decoder_layers(model: nn.Module):
    candidates = (
        "model.language_model.layers",
        "model.layers",
        "language_model.layers",
    )
    for path in candidates:
        current = model
        try:
            for part in path.split("."):
                current = getattr(current, part)
        except AttributeError:
            continue
        if isinstance(current, nn.ModuleList):
            return current
    raise TypeError("MBQ requires a model with decoder layers")


def _block_identity(name: str) -> str:
    marker = ".layers."
    if marker not in name:
        raise ValueError(f"cannot identify decoder block for {name}")
    prefix, rest = name.split(marker, 1)
    index = rest.split(".", 1)[0]
    return f"{prefix}{marker}{index}"


def _sensitivity_kind(name: str) -> str:
    return "attention" if ".self_attn." in name else "mlp"


def _group_kind(name: str) -> str:
    if name.endswith((".q_proj", ".k_proj", ".v_proj")):
        return "attention_input"
    if name.endswith(".o_proj"):
        return "attention_output"
    if name.endswith((".gate_proj", ".up_proj")):
        return "mlp_input"
    if name.endswith(".down_proj"):
        return "mlp_output"
    raise ValueError(f"unsupported MBQ decoder projection: {name}")


def aggregate_gradient_ratios(
    records: dict[str, list[dict]],
    mode: str,
) -> dict[str, float]:
    """Aggregate per-sample modality gradients using an explicit weighting rule."""
    ratios = {}
    for name, rows in records.items():
        if not rows:
            raise RuntimeError(f"MBQ has no gradient records at {name}")
        if mode == "pooled_elements":
            vision = sum(row["vision_sum"] for row in rows) / sum(
                row["vision_count"] for row in rows
            )
            language = sum(row["language_sum"] for row in rows) / sum(
                row["language_count"] for row in rows
            )
        elif mode == "sample_mean":
            vision = statistics.fmean(row["vision_mean"] for row in rows)
            language = statistics.fmean(row["language_mean"] for row in rows)
        elif mode == "source_balanced":
            sources = sorted({row["source"] for row in rows})
            vision = statistics.fmean(
                statistics.fmean(
                    row["vision_mean"] for row in rows if row["source"] == source
                )
                for source in sources
            )
            language = statistics.fmean(
                statistics.fmean(
                    row["language_mean"] for row in rows if row["source"] == source
                )
                for source in sources
            )
        else:
            raise ValueError(f"unsupported MBQ gradient aggregation: {mode}")
        if language <= 0:
            raise RuntimeError(f"MBQ language gradient is zero at {name}")
        ratios[name] = vision / language
    return ratios


def collect_mbq_raw_statistics(
    model: nn.Module,
    selected: list[SelectedLinear],
    batches: Iterable[dict],
    *,
    reservoir_rows: int,
    seed: int,
    attention_implementation: str = "sdpa",
) -> MBQRawStatistics:
    """Collect reusable modality reservoirs and per-sample gradient records."""
    if not selected:
        raise ValueError("MBQ requires at least one selected decoder Linear")
    model.eval()
    reservoirs = _ModalReservoir(selected, reservoir_rows, seed)
    current_masks: dict[str, torch.Tensor | None] = {"vision": None, "language": None}
    activation_handles = []
    captured: dict[str, torch.Tensor] = {}
    gradient_records: dict[str, list[dict]] = defaultdict(list)

    def activation_hook(name: str):
        def hook(_module, args):
            if not args or not isinstance(args[0], torch.Tensor):
                return
            value = args[0]
            for modality in ("vision", "language"):
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
        activation_handles.append(item.module.register_forward_pre_hook(activation_hook(item.name)))
        if item.name.endswith((".self_attn.o_proj", ".mlp.down_proj")):
            activation_handles.append(item.module.register_forward_hook(gradient_hook(item.name)))

    layers = _decoder_layers(model)

    def enable_decoder_grad(_module, args):
        if not args or not isinstance(args[0], torch.Tensor):
            raise RuntimeError("MBQ could not intercept first decoder hidden state")
        return (args[0].detach().requires_grad_(True), *args[1:])

    decoder_handle = layers[0].register_forward_pre_hook(enable_decoder_grad)
    parameters = list(model.parameters())
    original_requires_grad = [parameter.requires_grad for parameter in parameters]
    for parameter in parameters:
        parameter.requires_grad_(False)
    token_counts = {"vision": 0, "language": 0, "samples": 0}
    try:
        with temporary_attention_implementation(model, attention_implementation):
            for batch in batches:
                vision_mask = batch["vision_mask"].bool()
                language_mask = batch["language_mask"].bool()
                current_masks["vision"] = vision_mask
                current_masks["language"] = language_mask
                token_counts["vision"] += int(vision_mask.sum().item())
                token_counts["language"] += int(language_mask.sum().item())
                token_counts["samples"] += 1
                captured.clear()
                with torch.enable_grad():
                    output = model(
                        **batch["model_inputs"], use_cache=False, return_dict=True,
                        output_attentions=False, output_hidden_states=False,
                    )
                    if output.loss is None or not torch.isfinite(output.loss):
                        raise ValueError(f"non-finite MBQ SFT loss for {batch['id']}")
                    names = list(captured)
                    gradients = torch.autograd.grad(
                        output.loss, [captured[name] for name in names],
                        allow_unused=False,
                    )
                for name, gradient in zip(names, gradients):
                    vision_grad = gradient[vision_mask]
                    language_grad = gradient[language_mask]
                    if not vision_grad.numel() or not language_grad.numel():
                        raise RuntimeError(f"MBQ has an empty gradient modality at {name}")
                    vision_sum = float(vision_grad.abs().sum().item())
                    language_sum = float(language_grad.abs().sum().item())
                    gradient_records[name].append({
                        "source": str(batch.get("source", "unknown")),
                        "vision_sum": vision_sum,
                        "vision_count": vision_grad.numel(),
                        "vision_mean": vision_sum / vision_grad.numel(),
                        "language_sum": language_sum,
                        "language_count": language_grad.numel(),
                        "language_mean": language_sum / language_grad.numel(),
                    })
                del output, gradients, batch
                captured.clear()
                release_cpu_memory()
    finally:
        decoder_handle.remove()
        for handle in activation_handles:
            handle.remove()
        for parameter, required in zip(parameters, original_requires_grad):
            parameter.requires_grad_(required)
        current_masks["vision"] = None
        current_masks["language"] = None

    return MBQRawStatistics(
        stats=reservoirs.finish(selected),
        gradient_records=dict(gradient_records),
        token_counts=token_counts,
    )


def collect_mbq_statistics(
    model: nn.Module,
    selected: list[SelectedLinear],
    batches: Iterable[dict],
    *,
    reservoir_rows: int,
    seed: int,
    attention_implementation: str = "sdpa",
    gradient_aggregation: str = "pooled_elements",
) -> tuple[dict[str, ModalityActivationStats], dict[str, float], dict]:
    """Compatibility wrapper returning aggregated MBQ statistics."""
    raw = collect_mbq_raw_statistics(
        model,
        selected,
        batches,
        reservoir_rows=reservoir_rows,
        seed=seed,
        attention_implementation=attention_implementation,
    )
    ratios = aggregate_gradient_ratios(raw.gradient_records, gradient_aggregation)
    counts = dict(raw.token_counts)
    counts["gradient_aggregation"] = gradient_aggregation
    return raw.stats, ratios, counts


def _effective_group_size(source: nn.Linear, config: ExperimentConfig) -> int:
    return source.in_features if config.weight_granularity == "per_output_channel" else config.group_size


def _candidate_error(
    modules: list[SelectedLinear],
    stats: ModalityActivationStats,
    scale: torch.Tensor,
    ratio: float,
    config: ExperimentConfig,
) -> tuple[float, dict]:
    numerator = 0.0
    denominator = 0.0
    module_errors = {}
    for item in modules:
        source = item.module
        scaled_weight = source.weight.detach().float() * scale.unsqueeze(0)
        quantized = quantize_weight_groupwise(
            scaled_weight,
            bits=config.weight_bits,
            group_size=_effective_group_size(source, config),
            zero_point=config.zero_point,
        )
        errors = {}
        for modality, weight in (("language", 1.0), ("vision", ratio)):
            value = getattr(stats, modality).float()
            reference = F.linear(value, source.weight.detach().float(), source.bias)
            candidate_input = value / scale
            if config.activation_bits is not None:
                candidate_input = dynamic_activation_qdq(
                    candidate_input, config.activation_bits, per_token=True,
                )
            candidate = F.linear(candidate_input, quantized.dequantized, source.bias)
            absolute = (reference - candidate).abs()
            errors[modality] = float(absolute.mean().item())
            if config.mbq_objective_normalization == "token_weighted":
                original_rows = getattr(stats, f"{modality}_rows")
                objective_values = original_rows * absolute.shape[-1]
            else:
                objective_values = 1.0
            numerator += weight * errors[modality] * objective_values
            # Upstream MBQ weights the modality error sum, not its normalizer.
            # Keeping this denominator unweighted also makes audit loss values
            # comparable across reweighting ablations.
            denominator += objective_values
        module_errors[item.name] = errors
    return numerator / max(denominator, 1), module_errors


def search_mbq_scales(
    selected: list[SelectedLinear],
    stats: dict[str, ModalityActivationStats],
    raw_ratios: dict[str, float],
    config: ExperimentConfig,
    token_counts: dict,
) -> MBQSearchResult:
    """Search shared QKV/Gate-Up scales and individual O/Down scales per block."""
    if config.scope != QuantScope.DECODER_CANONICAL:
        raise ValueError("MBQ scale search is decoder-only")
    if config.mbq_reconstruction_level != "linear_group":
        raise NotImplementedError(
            "exact block-output reconstruction requires a separate backend; "
            "refusing to label linear-group reconstruction as block-output"
        )
    anchor_ratios: dict[tuple[str, str], float] = {}
    by_kind: dict[str, list[float]] = {"attention": [], "mlp": []}
    for name, ratio in raw_ratios.items():
        kind = _sensitivity_kind(name)
        anchor_ratios[(_block_identity(name), kind)] = float(ratio)
        by_kind[kind].append(float(ratio))
    medians = {
        kind: statistics.median(values) if values else 1.0
        for kind, values in by_kind.items()
    }
    groups: dict[tuple[str, str], list[SelectedLinear]] = defaultdict(list)
    for item in selected:
        groups[(_block_identity(item.name), _group_kind(item.name))].append(item)

    scales, group_audit = {}, {}
    for (block, group_kind), modules in sorted(groups.items()):
        sensitivity = "attention" if group_kind.startswith("attention") else "mlp"
        raw_ratio = anchor_ratios.get((block, sensitivity))
        if raw_ratio is None:
            raise RuntimeError(f"missing MBQ {sensitivity} sensitivity for {block}")
        if not config.mbq_reweight:
            ratio = 1.0
        elif config.mbq_median_floor:
            ratio = max(raw_ratio, medians[sensitivity])
        else:
            ratio = raw_ratio
        modules = sorted(modules, key=lambda item: item.name)
        representative = next(
            (
                item for item in modules
                if item.name.endswith((".q_proj", ".gate_proj", ".o_proj", ".down_proj"))
            ),
            modules[0],
        )
        modality_stats = stats[representative.name]
        activation = (
            modality_stats.vision.abs().mean(dim=0) * modality_stats.vision_rows
            + modality_stats.language.abs().mean(dim=0) * modality_stats.language_rows
        ) / (modality_stats.vision_rows + modality_stats.language_rows)
        activation = activation.clamp_min(1e-4)
        best = None
        history = []
        for alpha in config.mbq_scale_grid:
            scale = activation.pow(alpha).clamp_min(1e-4)
            scale = scale / torch.sqrt(scale.max() * scale.min())
            loss, module_errors = _candidate_error(
                modules, modality_stats, scale, ratio, config,
            )
            history.append({
                "alpha": alpha,
                "weighted_mae": loss,
                "module_mae": module_errors,
            })
            if best is None or loss < best[0]:
                best = (loss, alpha, scale.clone())
        assert best is not None
        loss, alpha, scale = best
        for item in modules:
            if item.module.in_features != scale.numel():
                raise RuntimeError(f"shared MBQ scale shape mismatch at {item.name}")
            scales[item.name] = scale.clone()
        group_audit[f"{block}:{group_kind}"] = {
            "modules": [item.name for item in modules],
            "sensitivity_kind": sensitivity,
            "raw_vision_to_language_gradient_ratio": raw_ratio,
            "median_floor": medians[sensitivity],
            "effective_ratio": ratio,
            "vision_mass_share": (
                ratio * modality_stats.vision_rows
                / (
                    ratio * modality_stats.vision_rows
                    + modality_stats.language_rows
                )
            ),
            "selected_alpha": alpha,
            "weighted_mae": loss,
            "history": history,
        }

    return MBQSearchResult(scales=scales, audit={
        "format_version": FORMAT_VERSION,
        "algorithm": "modality_balanced_decoder_group_reconstruction",
        "reconstruction_level": "decoder_linear_group",
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "loss_mode": config.mbq_loss_mode,
        "scale_grid": list(config.mbq_scale_grid),
        "reweight": config.mbq_reweight,
        "distort": config.mbq_distort,
        "gradient_aggregation": config.mbq_gradient_aggregation,
        "objective_normalization": config.mbq_objective_normalization,
        "median_floor_enabled": config.mbq_median_floor,
        "token_counts": token_counts,
        "gradient_ratio_medians": medians,
        "raw_gradient_ratios": raw_ratios,
        "groups": group_audit,
        "deployment_kernel_claim": False,
        "compressed_resident_memory_claim": False,
    })


def save_mbq_search(path: Path, *, key: str, result: MBQSearchResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "key": key,
        "scales": result.scales,
        "audit": result.audit,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_mbq_search(
    path: Path, *, key: str, selected: list[SelectedLinear],
) -> MBQSearchResult | None:
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != FORMAT_VERSION or payload.get("key") != key:
        return None
    names = [item.name for item in selected]
    scales = payload.get("scales", {})
    if set(scales) != set(names):
        return None
    for item in selected:
        scale = scales[item.name]
        if scale.numel() != item.module.in_features or not torch.isfinite(scale).all():
            return None
    return MBQSearchResult(scales=scales, audit=payload["audit"])


def save_mbq_raw_statistics(
    path: Path, *, key: str, raw: MBQRawStatistics,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "key": key,
        "stats": {
            name: {
                "vision": value.vision,
                "language": value.language,
                "vision_rows": value.vision_rows,
                "language_rows": value.language_rows,
            }
            for name, value in raw.stats.items()
        },
        "gradient_records": raw.gradient_records,
        "token_counts": raw.token_counts,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_mbq_raw_statistics(
    path: Path, *, key: str, selected: list[SelectedLinear],
) -> MBQRawStatistics | None:
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != FORMAT_VERSION or payload.get("key") != key:
        return None
    names = {item.name for item in selected}
    serialized = payload.get("stats", {})
    records = payload.get("gradient_records", {})
    if set(serialized) != names or set(records) != {
        item.name for item in selected
        if item.name.endswith((".self_attn.o_proj", ".mlp.down_proj"))
    }:
        return None
    stats = {}
    for item in selected:
        value = serialized[item.name]
        vision, language = value["vision"], value["language"]
        if (
            vision.ndim != 2 or language.ndim != 2
            or vision.shape[1] != item.module.in_features
            or language.shape[1] != item.module.in_features
            or not torch.isfinite(vision).all()
            or not torch.isfinite(language).all()
        ):
            return None
        stats[item.name] = ModalityActivationStats(
            vision=vision,
            language=language,
            vision_rows=int(value["vision_rows"]),
            language_rows=int(value["language_rows"]),
        )
    return MBQRawStatistics(
        stats=stats,
        gradient_records=records,
        token_counts=payload["token_counts"],
    )


def collect_mbq_search_cached(
    model: nn.Module,
    selected: list[SelectedLinear],
    batches: Iterable[dict],
    config: ExperimentConfig,
    *,
    cache_dir: Path,
    key: str,
    raw_key: str | None = None,
    reservoir_rows: int = 32,
    attention_implementation: str = "sdpa",
) -> tuple[MBQSearchResult, bool]:
    path = cache_dir / f"mbq_{key}.pt"
    cached = load_mbq_search(path, key=key, selected=selected)
    if cached is not None:
        return cached, True
    raw_cache_hit = False
    raw_path = cache_dir / f"mbq_stats_{raw_key}.pt" if raw_key else None
    raw = (
        load_mbq_raw_statistics(raw_path, key=raw_key, selected=selected)
        if raw_path is not None and raw_key is not None else None
    )
    if raw is None:
        raw_started = time.perf_counter()
        raw = collect_mbq_raw_statistics(
            model, selected, batches,
            reservoir_rows=reservoir_rows,
            seed=config.seed,
            attention_implementation=attention_implementation,
        )
        raw_seconds = time.perf_counter() - raw_started
        if raw_path is not None and raw_key is not None:
            save_mbq_raw_statistics(raw_path, key=raw_key, raw=raw)
    else:
        raw_cache_hit = True
        raw_seconds = 0.0
    ratios = aggregate_gradient_ratios(
        raw.gradient_records, config.mbq_gradient_aggregation,
    )
    token_counts = dict(raw.token_counts)
    token_counts["gradient_aggregation"] = config.mbq_gradient_aggregation
    search_started = time.perf_counter()
    result = search_mbq_scales(
        selected, raw.stats, ratios, config, token_counts,
    )
    result.audit["search_seconds"] = time.perf_counter() - search_started
    result.audit["raw_statistics_seconds"] = raw_seconds
    result.audit["raw_statistics_cache_hit"] = raw_cache_hit
    result.audit["raw_statistics_cache_key"] = raw_key
    save_mbq_search(path, key=key, result=result)
    return result, False


def apply_mbq(
    model: nn.Module,
    selected: list[SelectedLinear],
    config: ExperimentConfig,
    search: MBQSearchResult,
) -> AppliedQuantization:
    if set(search.scales) != {item.name for item in selected}:
        raise ValueError("MBQ scale artifact does not match selected modules")
    originals, module_audit = {}, {}
    started = time.perf_counter()
    try:
        for item in selected:
            source = item.module
            scale = search.scales[item.name].float()
            quantized = quantize_weight_groupwise(
                source.weight.detach().float() * scale.unsqueeze(0),
                bits=config.weight_bits,
                group_size=_effective_group_size(source, config),
                zero_point=config.zero_point,
            )
            if config.activation_bits is None:
                replacement = ScaledQuantLinear(
                    source, quantized.dequantized, scale,
                )
            else:
                replacement = W8A8Linear(
                    source, quantized.dequantized, scale,
                    activation_bits=config.activation_bits,
                )
            parent, key = _parent_and_key(model, item.name)
            originals[item.name] = source
            setattr(parent, key, replacement)
            module_audit[item.name] = {
                "subsystem": item.subsystem,
                "weight_count": source.weight.numel(),
                "weight_mse": quantized.mse,
                "weight_nmse": quantized.nmse,
                "group_size": _effective_group_size(source, config),
                "zero_point": config.zero_point,
                "activation_bits": config.activation_bits,
                "scale_min": float(scale.min().item()),
                "scale_max": float(scale.max().item()),
            }
    except Exception:
        for name, original in originals.items():
            parent, key = _parent_and_key(model, name)
            setattr(parent, key, original)
        raise

    total_weights = sum(value["weight_count"] for value in module_audit.values())
    logical_bits = sum(
        (
            config.weight_bits
            + (16 + (config.weight_bits if config.zero_point else 0))
            / value["group_size"]
        ) * value["weight_count"]
        for value in module_audit.values()
    ) / total_weights
    audit = dict(search.audit)
    audit.update({
        "method": "mbq",
        "scheme_id": config.scheme_id,
        "scope": config.scope.value,
        "weight_bits": config.weight_bits,
        "activation_bits": config.activation_bits,
        "weight_granularity": config.weight_granularity,
        "module_count": len(originals),
        "logical_bits_per_weight": logical_bits,
        "apply_seconds": time.perf_counter() - started,
        "modules": module_audit,
    })
    return AppliedQuantization(model, originals, audit)
