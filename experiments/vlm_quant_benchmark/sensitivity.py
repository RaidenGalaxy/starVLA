"""Sample-level modality sensitivity diagnostics for Qwen-style VLMs."""
from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from .calibration import release_cpu_memory, temporary_attention_implementation
from .data import MultimodalSample
from .mbq import _decoder_layers, _sensitivity_kind
from .selection import SelectedLinear


SENSITIVITY_FORMAT_VERSION = 3


def causal_predictor_mask(language_mask: torch.Tensor) -> torch.Tensor:
    """Map target-label positions to the hidden positions predicting them.

    Hugging Face causal-LM loss compares logits at position ``t - 1`` with the
    label at position ``t``.  MBQ's upstream-compatible mask remains available
    as ``language_mask``; this shifted mask is the physically effective mask
    for hidden-state gradient and perturbation diagnostics.
    """
    if language_mask.ndim != 2:
        raise ValueError("language mask must have shape [batch, sequence]")
    result = torch.zeros_like(language_mask, dtype=torch.bool)
    result[..., :-1] = language_mask[..., 1:].bool()
    return result


def stable_hash_subset(
    samples: Iterable[MultimodalSample],
    per_dataset: int,
    *,
    salt: str,
    exclude_ids: set[str] | None = None,
) -> list[MultimodalSample]:
    """Select a deterministic, outcome-independent subset per dataset."""
    excluded = exclude_ids or set()
    buckets: dict[str, list[MultimodalSample]] = {}
    for sample in samples:
        if sample.id not in excluded:
            buckets.setdefault(sample.dataset, []).append(sample)
    selected = []
    for dataset, values in sorted(buckets.items()):
        ranked = sorted(
            values,
            key=lambda item: hashlib.sha256(
                f"{salt}:{dataset}:{item.id}".encode()
            ).digest(),
        )
        if len(ranked) < per_dataset:
            raise ValueError(
                f"{dataset} has {len(ranked)} available samples; "
                f"{per_dataset} requested"
            )
        selected.extend(ranked[:per_dataset])
    return selected


def vision_mass_share(ratio: float, vision_tokens: int, language_tokens: int) -> float:
    weighted_vision = max(ratio, 0.0) * vision_tokens
    total = weighted_vision + language_tokens
    return weighted_vision / total if total else math.nan


def _layer_index(name: str) -> int:
    marker = ".layers."
    if marker not in name:
        raise ValueError(f"cannot identify decoder layer in {name}")
    return int(name.split(marker, 1)[1].split(".", 1)[0])


def sensitivity_anchors(selected: list[SelectedLinear]) -> list[SelectedLinear]:
    anchors = [
        item for item in selected
        if item.name.endswith((".self_attn.o_proj", ".mlp.down_proj"))
    ]
    if not anchors:
        raise ValueError("no MBQ sensitivity anchors were selected")
    return anchors


def _gradient_summary(gradient: torch.Tensor, mask: torch.Tensor) -> dict:
    values = gradient[mask].detach().float().abs()
    if not values.numel():
        return {
            "sum": 0.0,
            "count": 0,
            "mean": math.nan,
            "zero_fraction": math.nan,
            "p50": math.nan,
            "p95": math.nan,
        }
    flattened = values.reshape(-1)
    return {
        "sum": float(flattened.sum().item()),
        "count": flattened.numel(),
        "mean": float(flattened.mean().item()),
        "zero_fraction": float(flattened.eq(0).float().mean().item()),
        "p50": float(torch.quantile(flattened, 0.5).item()),
        "p95": float(torch.quantile(flattened, 0.95).item()),
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return math.nan
    if denominator <= 0:
        return math.inf if numerator > 0 else math.nan
    return numerator / denominator


def collect_sample_sensitivity(
    model: nn.Module,
    selected: list[SelectedLinear],
    batches: Iterable[dict],
    *,
    attention_implementation: str = "sdpa",
) -> list[dict]:
    """Collect per-sample gradients at attention/MLP output projections."""
    anchors = sensitivity_anchors(selected)
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def capture(name: str):
        def hook(_module, _args, output):
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"sensitivity anchor {name} did not return a tensor")
            captured[name] = output
        return hook

    for item in anchors:
        handles.append(item.module.register_forward_hook(capture(item.name)))

    layers = _decoder_layers(model)

    def enable_decoder_grad(_module, args):
        if not args or not isinstance(args[0], torch.Tensor):
            raise RuntimeError("could not intercept decoder hidden states")
        return (args[0].detach().requires_grad_(True), *args[1:])

    decoder_handle = layers[0].register_forward_pre_hook(enable_decoder_grad)
    parameters = list(model.parameters())
    original_requires_grad = [parameter.requires_grad for parameter in parameters]
    for parameter in parameters:
        parameter.requires_grad_(False)
    rows = []
    model.eval()
    try:
        with temporary_attention_implementation(model, attention_implementation):
            for batch in batches:
                answer_label_mask = batch["language_mask"].bool()
                masks = {
                    "vision": batch["vision_mask"].bool(),
                    # Kept to reproduce the official MBQ labels != -100
                    # aggregation, despite causal-LM loss being shifted.
                    "answer_language": answer_label_mask,
                    "answer_predictor_language": causal_predictor_mask(
                        answer_label_mask
                    ),
                    "prompt_language": batch["prompt_language_mask"].bool(),
                }
                captured.clear()
                with torch.enable_grad():
                    output = model(
                        **batch["model_inputs"],
                        use_cache=False,
                        return_dict=True,
                        output_attentions=False,
                        output_hidden_states=False,
                    )
                    if output.loss is None or not torch.isfinite(output.loss):
                        raise ValueError(f"non-finite SFT loss for {batch['id']}")
                    names = list(captured)
                    gradients = torch.autograd.grad(
                        output.loss,
                        [captured[name] for name in names],
                        allow_unused=False,
                    )
                token_counts = {
                    modality: int(mask.sum().item())
                    for modality, mask in masks.items()
                }
                for name, gradient in zip(names, gradients):
                    summaries = {
                        modality: _gradient_summary(gradient, mask)
                        for modality, mask in masks.items()
                    }
                    answer_ratio = _safe_ratio(
                        summaries["vision"]["mean"],
                        summaries["answer_language"]["mean"],
                    )
                    predictor_ratio = _safe_ratio(
                        summaries["vision"]["mean"],
                        summaries["answer_predictor_language"]["mean"],
                    )
                    prompt_ratio = _safe_ratio(
                        summaries["vision"]["mean"],
                        summaries["prompt_language"]["mean"],
                    )
                    rows.append({
                        "format_version": SENSITIVITY_FORMAT_VERSION,
                        "id": str(batch["id"]),
                        "dataset": str(batch.get("source", "unknown")),
                        "module": name,
                        "layer": _layer_index(name),
                        "kind": _sensitivity_kind(name),
                        "target_nll": float(output.loss.detach().item()),
                        "token_counts": token_counts,
                        "gradients": summaries,
                        "vision_to_answer_ratio": answer_ratio,
                        "vision_to_answer_predictor_ratio": predictor_ratio,
                        "vision_to_prompt_ratio": prompt_ratio,
                        "vision_mass_share": vision_mass_share(
                            predictor_ratio,
                            token_counts["vision"],
                            token_counts["answer_predictor_language"],
                        ),
                        "answer_ratio_basis": "causal_predictor_position",
                    })
                del output, gradients, batch
                captured.clear()
                release_cpu_memory()
    finally:
        decoder_handle.remove()
        for handle in handles:
            handle.remove()
        for parameter, required in zip(parameters, original_requires_grad):
            parameter.requires_grad_(required)
    return rows


@torch.inference_mode()
def evaluate_sample_target_nll(
    model: nn.Module,
    batches: Iterable[dict],
    *,
    variant: str,
    attention_implementation: str = "sdpa",
) -> list[dict]:
    """Evaluate continuous teacher-forced loss without generation."""
    rows = []
    model.eval()
    with temporary_attention_implementation(model, attention_implementation):
        for batch in batches:
            output = model(
                **batch["model_inputs"], use_cache=False, return_dict=True,
                output_attentions=False, output_hidden_states=False,
            )
            if output.loss is None or not torch.isfinite(output.loss):
                raise ValueError(f"non-finite target NLL for {batch['id']}")
            rows.append({
                "format_version": SENSITIVITY_FORMAT_VERSION,
                "variant": variant,
                "dataset": str(batch.get("source", "unknown")),
                "id": str(batch["id"]),
                "target_nll": float(output.loss.item()),
                "answer_tokens": int(batch["language_mask"].sum().item()),
                "vision_tokens": int(batch["vision_mask"].sum().item()),
            })
            del output, batch
            release_cpu_memory()
    return rows


@torch.inference_mode()
def collect_modality_perturbations(
    model: nn.Module,
    batch: dict,
    *,
    layers_to_test: tuple[int, ...],
    noise_rms: dict[int, float] | float,
    seed: int,
    attention_implementation: str = "sdpa",
) -> list[dict]:
    """Measure causal NLL/KL changes after block-output modality noise.

    The hook location deliberately matches the decoder-block residual RMS used
    by the formal experiment. ``output`` may be either a tensor or the tuple
    returned by Qwen decoder blocks; only the hidden-state tensor is changed.
    """
    masks = {
        "vision": batch["vision_mask"].bool(),
        "answer_language": causal_predictor_mask(
            batch["language_mask"].bool()
        ),
        "prompt_language": batch["prompt_language_mask"].bool(),
    }
    decoder_layers = _decoder_layers(model)
    with temporary_attention_implementation(model, attention_implementation):
        baseline = model(
            **batch["model_inputs"], use_cache=False, return_dict=True,
            output_attentions=False, output_hidden_states=False,
        )
    labels = batch["model_inputs"]["labels"]
    target_positions = labels[0].ne(-100).nonzero(as_tuple=False).reshape(-1)
    prediction_positions = target_positions - 1
    valid = prediction_positions.ge(0)
    prediction_positions = prediction_positions[valid]
    baseline_logits = baseline.logits[0, prediction_positions].float().cpu()
    baseline_probs = baseline_logits.softmax(dim=-1)
    baseline_argmax = baseline_logits.argmax(dim=-1)
    baseline_nll = float(baseline.loss.item())
    rows = []
    for layer_index in layers_to_test:
        if layer_index < 0 or layer_index >= len(decoder_layers):
            raise ValueError(f"decoder layer {layer_index} is out of range")
        rms = noise_rms[layer_index] if isinstance(noise_rms, dict) else noise_rms
        if rms <= 0:
            raise ValueError("perturbation RMS must be positive")
        for modality, mask in masks.items():
            if not mask.any():
                continue
            digest = hashlib.sha256(
                f"{seed}:{batch['id']}:{layer_index}:{modality}".encode()
            ).digest()
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int.from_bytes(digest[:8], "little") % (2**63 - 1))

            def perturb(
                _module, _args, output, *, selected_mask=mask, amplitude=rms,
            ):
                hidden = output[0] if isinstance(output, tuple) else output
                noise = torch.randn(
                    hidden.shape,
                    generator=generator,
                    dtype=torch.float32,
                    device="cpu",
                ).to(hidden.device, hidden.dtype)
                changed = hidden + noise * selected_mask.unsqueeze(-1) * amplitude
                if isinstance(output, tuple):
                    return (changed, *output[1:])
                return changed

            handle = decoder_layers[layer_index].register_forward_hook(perturb)
            try:
                with temporary_attention_implementation(
                    model, attention_implementation,
                ):
                    output = model(
                        **batch["model_inputs"], use_cache=False, return_dict=True,
                        output_attentions=False, output_hidden_states=False,
                    )
            finally:
                handle.remove()
            perturbed_logits = output.logits[0, prediction_positions].float().cpu()
            log_probs = perturbed_logits.log_softmax(dim=-1)
            kl = F.kl_div(log_probs, baseline_probs, reduction="batchmean")
            changed = perturbed_logits.argmax(dim=-1).ne(baseline_argmax)
            rows.append({
                "format_version": SENSITIVITY_FORMAT_VERSION,
                "id": str(batch["id"]),
                "dataset": str(batch.get("source", "unknown")),
                "layer": layer_index,
                "modality": modality,
                "perturbation_location": "decoder_block_output",
                "answer_mask_alignment": "causal_predictor_position",
                "noise_rms": float(rms),
                "baseline_nll": baseline_nll,
                "perturbed_nll": float(output.loss.item()),
                "delta_nll": float(output.loss.item()) - baseline_nll,
                "target_kl": float(kl.item()),
                "target_argmax_change_rate": float(changed.float().mean().item()),
            })
            del output
            release_cpu_memory()
    del baseline
    return rows
