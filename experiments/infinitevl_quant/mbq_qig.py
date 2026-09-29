from __future__ import annotations

import copy
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.vlm_quant_benchmark.config import (
    ExperimentConfig,
    ImplementationTier,
    MethodName,
    QuantScope,
)
from experiments.vlm_quant_benchmark.mbq import (
    MBQSearchResult,
    aggregate_gradient_ratios,
    collect_mbq_raw_statistics,
    mbq_calibration_batches,
)
from experiments.vlm_quant_benchmark.selection import SelectedLinear
from experiments.vlm_quant_benchmark.vg_qig_mbq import (
    _activation_scale,
    _install_scaled,
    auto_gate_layers,
    block_objective,
    capture_block_traces,
    choose_qig_steps,
    make_quantized_block,
    measure_vulnerability,
    qig_token_importance,
)

from .config import QuantScheme
from .cpu_backend import qig_recurrent_reference
from .evaluation import teacher_nll_only
from .inventory import LinearTarget, decoder_layers
from .loader import PRE_ROUTED_SLIDING_MASK
from .progress import json_digest
from .quantization import activation_qdq, quantize_weight, quantized_linears


def _qig_token_importance_recurrent(*args, **kwargs):
    """Run IG with the differentiable recurrent GDN reference explicitly."""
    with qig_recurrent_reference():
        return qig_token_importance(*args, **kwargs)


def selected_linears(targets: list[LinearTarget]) -> list[SelectedLinear]:
    return [SelectedLinear(
        target.name, target.module, target.scope, target.parameter_count,
    ) for target in targets]


def experiment_config(
    scheme: QuantScheme,
    *,
    qig_steps: tuple[int, ...] = (4, 8, 16),
    qig_alphas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75),
    seed: int = 42,
) -> ExperimentConfig:
    return ExperimentConfig(
        MethodName.VG_QIG_MBQ,
        scope=QuantScope.DECODER_CANONICAL,
        weight_bits=scheme.weight_bits,
        activation_bits=scheme.activation_bits,
        group_size=scheme.group_size,
        zero_point=scheme.zero_point,
        weight_granularity=(
            "per_output_channel" if scheme.per_output_channel else "groupwise"
        ),
        calibration="multimodal",
        calibration_samples=16,
        seed=seed,
        mbq_scale_grid=qig_alphas,
        vg_vvs_samples=4,
        vg_qig_samples=8,
        vg_vvs_threshold=0.05,
        vg_vvs_max_layers=2,
        vg_gate_bootstrap_samples=500,
        vg_ig_step_grid=qig_steps,
        vg_coordinate_sweeps=1,
        implementation_tier=ImplementationTier.CPU_RESEARCH,
        scheme_id=scheme.name,
    )


def _block_name(name: str) -> str:
    prefix, rest = name.split(".layers.", 1)
    return f"{prefix}.layers.{rest.split('.', 1)[0]}"


def _group(name: str) -> str:
    leaf = name.rsplit(".", 1)[-1]
    if ".self_attn." in name:
        if leaf in {"q_proj", "k_proj", "v_proj", "g_proj"}:
            return "attention_input"
        if leaf == "o_proj":
            return "attention_output"
    if ".mlp." in name:
        if leaf in {"gate_proj", "up_proj"}:
            return "mlp_input"
        if leaf == "down_proj":
            return "mlp_output"
    raise ValueError(f"unsupported decoder-safe projection: {name}")


def _kind(name: str) -> str:
    return "attention" if ".self_attn." in name else "mlp"


def _candidate_error(modules, stats, scale, ratio, scheme: QuantScheme) -> tuple[float, dict]:
    total, mass, audit = 0.0, 0, {}
    for item in modules:
        source = item.module
        weight = source.weight.detach().float() * scale.unsqueeze(0)
        group_size = source.in_features if scheme.per_output_channel else scheme.group_size
        quantized = quantize_weight(
            weight, bits=scheme.weight_bits, group_size=group_size,
            zero_point=scheme.zero_point,
        )
        module_audit = {}
        for modality, values, rows, weight_factor in (
            ("vision", stats.vision.float(), stats.vision_rows, ratio),
            ("answer_language", stats.language.float(), stats.language_rows, 1.0),
        ):
            reference = F.linear(values, source.weight.detach().float(), source.bias)
            candidate_input = values / scale
            if scheme.activation_bits == 8:
                candidate_input = activation_qdq(candidate_input)
            candidate = F.linear(candidate_input, quantized.dequantized, source.bias)
            error = float((reference - candidate).abs().mean())
            module_audit[modality] = error
            elements = rows * reference.shape[-1]
            total += weight_factor * error * elements
            mass += elements
        audit[item.name] = module_audit
    return total / max(mass, 1), audit


def _search_mbq_scales(raw, selected, scheme: QuantScheme) -> MBQSearchResult:
    ratios = aggregate_gradient_ratios(raw.gradient_records, "source_balanced")
    medians = {kind: statistics.median([
        value for name, value in ratios.items() if _kind(name) == kind
    ]) for kind in ("attention", "mlp")}
    anchors = {(_block_name(name), _kind(name)): value for name, value in ratios.items()}
    groups = defaultdict(list)
    for item in selected:
        groups[(_block_name(item.name), _group(item.name))].append(item)
    alpha_grid = tuple(index / 20 for index in range(20))
    scales, group_audit = {}, {}
    for (block, group_name), modules in sorted(groups.items()):
        modules = sorted(modules, key=lambda item: item.name)
        kind = "attention" if group_name.startswith("attention") else "mlp"
        ratio = max(anchors[(block, kind)], medians[kind])
        representative = modules[0]
        stats = raw.stats[representative.name]
        activation = (
            stats.vision.abs().mean(dim=0) * stats.vision_rows
            + stats.language.abs().mean(dim=0) * stats.language_rows
        ) / (stats.vision_rows + stats.language_rows)
        activation = activation.clamp_min(1e-4)
        best, history = None, []
        for alpha in alpha_grid:
            scale = activation.pow(alpha).clamp_min(1e-4)
            scale = scale / torch.sqrt(scale.max() * scale.min())
            loss, details = _candidate_error(modules, stats, scale, ratio, scheme)
            history.append({"alpha": alpha, "weighted_mae": loss, "modules": details})
            if best is None or (loss, alpha) < (best[0], best[1]):
                best = (loss, alpha, scale.clone())
        loss, alpha, scale = best
        for item in modules:
            if item.module.in_features != scale.numel():
                raise RuntimeError(f"MBQ shared scale shape mismatch at {item.name}")
            scales[item.name] = scale.clone()
        group_audit[f"{block}:{group_name}"] = {
            "modules": [item.name for item in modules],
            "selected_alpha": alpha, "weighted_mae": loss,
            "raw_vision_to_answer_gradient_ratio": anchors[(block, kind)],
            "effective_ratio": ratio, "history": history,
        }
    return MBQSearchResult(scales, {
        "algorithm": "infinitevl_modality_balanced_linear_group_search",
        "gradient_aggregation": "source_balanced",
        "scale_grid": list(alpha_grid),
        "gradient_ratio_medians": medians,
        "token_counts": raw.token_counts,
        "groups": group_audit,
        "fake_quantized": True,
        "deployment_kernel_claim": False,
    })


def run_mbq(
    model,
    processor,
    targets: list[LinearTarget],
    samples,
    scheme: QuantScheme,
    *,
    artifact: Path,
    min_pixels: int,
    max_pixels: int,
    seed: int,
) -> tuple[MBQSearchResult, bool]:
    key = json_digest({
        "samples": [sample.id for sample in samples],
        "targets": [target.name for target in targets],
        "scheme": scheme.to_dict(), "seed": seed,
    })
    if artifact.is_file():
        payload = torch.load(artifact, map_location="cpu", weights_only=True)
        if payload.get("key") == key:
            return MBQSearchResult(payload["scales"], payload["audit"]), True
    selected = selected_linears(targets)
    raw = collect_mbq_raw_statistics(
        model, selected,
        mbq_calibration_batches(
            processor, samples, min_pixels=min_pixels, max_pixels=max_pixels,
        ),
        reservoir_rows=32, seed=seed, attention_implementation="eager",
        modality_mode="vision_answer", algorithm_version=4,
    )
    result = _search_mbq_scales(raw, selected, scheme)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact.with_suffix(".tmp")
    torch.save({"format_version": 1, "key": key, "scales": result.scales,
                "audit": result.audit}, temporary)
    temporary.replace(artifact)
    return result, False


def run_vvs(
    model,
    processor,
    targets: list[LinearTarget],
    samples,
    scheme: QuantScheme,
    base: MBQSearchResult,
    *,
    min_pixels: int,
    max_pixels: int,
    relative_threshold: float,
    max_layers: int,
    bootstrap_samples: int,
    seed: int,
) -> tuple[list[int], dict]:
    config = experiment_config(scheme, seed=seed)
    scores, measurement = measure_vulnerability(
        model, selected_linears(targets), base,
        mbq_calibration_batches(
            processor, samples, min_pixels=min_pixels, max_pixels=max_pixels,
        ),
        config,
    )
    selected, gate = auto_gate_layers(
        scores, relative_threshold=relative_threshold, max_layers=max_layers,
        bootstrap_samples=bootstrap_samples, inclusion_probability=0.5, seed=seed,
    )
    gate["measurement"] = measurement
    gate["sample_ids"] = [sample.id for sample in samples]
    return selected, gate


def _search_qig_block(
    fp_block,
    modules,
    base_scales,
    traces,
    weights,
    config,
    alpha_grid,
):
    grouped = defaultdict(list)
    for item in modules:
        grouped[_group(item.name)].append(item)
    current = {item.name: base_scales[item.name].clone() for item in modules}
    quant_block = make_quantized_block(fp_block, modules, current, config)
    initial, initial_parts = block_objective(fp_block, quant_block, traces, weights)
    history = []
    for group_name in ("attention_input", "attention_output", "mlp_input", "mlp_output"):
        members = sorted(grouped.get(group_name, []), key=lambda item: item.name)
        if not members:
            continue
        candidates = [("base", None, current[members[0].name].clone())]
        for alpha in alpha_grid:
            candidates.append((
                "alpha", alpha,
                _activation_scale(fp_block, members[0], traces, alpha),
            ))
        best, rows = None, []
        for kind, alpha, scale in candidates:
            for item in members:
                _install_scaled(quant_block, fp_block, item, scale, config)
            loss, parts = block_objective(fp_block, quant_block, traces, weights)
            rows.append({"candidate": kind, "alpha": alpha, "loss": loss, **parts})
            rank = (loss, 0 if kind == "base" else 1, -1.0 if alpha is None else alpha)
            if best is None or rank < best[0]:
                best = (rank, scale.clone(), kind, alpha)
        _rank, scale, kind, alpha = best
        for item in members:
            current[item.name] = scale.clone()
            _install_scaled(quant_block, fp_block, item, scale, config)
        history.append({
            "group": group_name, "selected": kind, "alpha": alpha,
            "candidates": rows,
        })
    final, final_parts = block_objective(fp_block, quant_block, traces, weights)
    return current, {
        "initial_loss": initial, "initial_parts": initial_parts,
        "final_loss": final, "final_parts": final_parts,
        "non_regression": final <= initial + 1e-12, "history": history,
    }


def run_qig_layers(
    model,
    processor,
    targets: list[LinearTarget],
    samples,
    scheme: QuantScheme,
    base: MBQSearchResult,
    layers: list[int],
    *,
    min_pixels: int,
    max_pixels: int,
    step_grid: tuple[int, ...],
    alpha_grid: tuple[float, ...],
    spearman_threshold: float,
    jaccard_threshold: float,
    budget_hours: float,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict]:
    if not layers:
        return dict(base.scales), {"skipped": True, "reason": "empty_vulnerability_gate"}
    config = experiment_config(
        scheme, qig_steps=step_grid, qig_alphas=alpha_grid, seed=seed,
    )
    selected = selected_linears(targets)
    traces = capture_block_traces(
        model, layers,
        mbq_calibration_batches(
            processor, samples, min_pixels=min_pixels, max_pixels=max_pixels,
        ),
        decoder_layers=list(decoder_layers(model)),
    )
    # Loader hooks have already replaced the remote model's erroneous full
    # mask before these traces were captured. Preserve each trace's own mask
    # when an SWA block (and its copied hook) is replayed in isolation.
    for layer_traces in traces.values():
        for trace in layer_traces:
            trace.kwargs[PRE_ROUTED_SLIDING_MASK] = True
    by_layer = defaultdict(list)
    for item in selected:
        index = int(item.name.split(".layers.", 1)[1].split(".", 1)[0])
        by_layer[index].append(item)
    result = {name: value.clone() for name, value in base.scales.items()}
    audit = {
        "algorithm": "vulnerability_gated_qig_mbq",
        "baseline": "zero_input", "sample_ids": [sample.id for sample in samples],
        "requested_layers": layers, "layers": {}, "budget_hours": budget_hours,
    }
    decoder = list(decoder_layers(model))
    effective_layers, effective_steps = list(layers), tuple(step_grid)
    first_elapsed = None
    for layer_position, layer in enumerate(effective_layers):
        fp_block = copy.deepcopy(decoder[layer]).float().eval()
        for parameter in fp_block.parameters():
            parameter.requires_grad_(False)
        modules = by_layer[layer]
        base_block = make_quantized_block(fp_block, modules, result, config)
        audit_traces, seen = [], set()
        for trace in traces[layer]:
            if trace.source not in seen:
                audit_traces.append(trace)
                seen.add(trace.source)
        steps_for_layer = tuple(effective_steps)
        initial_steps = steps_for_layer[:min(2, len(steps_for_layer))]
        convergence_inputs = {step: [] for step in initial_steps}
        cached = {}
        for trace_index, trace in enumerate(audit_traces):
            started = time.perf_counter()
            for step in initial_steps:
                value = _qig_token_importance_recurrent(
                    fp_block, base_block, trace, step, iqr_multiplier=1.5,
                )
                cached[(trace.sample_id, step)] = value
                convergence_inputs[step].append(torch.cat((
                    value["vision"].flatten(), value["answer_language"].flatten(),
                )))
            if first_elapsed is None:
                first_elapsed = time.perf_counter() - started
                projected = (
                    first_elapsed * len(traces[layer]) * len(effective_layers)
                    * len(steps_for_layer) / len(initial_steps)
                )
                audit["initial_eta_hours"] = projected / 3600.0
                if projected > budget_hours * 3600:
                    effective_layers = effective_layers[:1]
                    effective_steps = tuple(step for step in (4, 8) if step in step_grid)
                    if len(effective_steps) < 2:
                        effective_steps = tuple(sorted(step_grid)[:2])
                    revised = first_elapsed * len(traces[layer])
                    audit["reduced_eta_hours"] = revised / 3600.0
                    audit["budget_reduction"] = {
                        "layers": effective_layers, "steps": effective_steps,
                    }
                    steps_for_layer = effective_steps
                    if revised > budget_hours * 3600:
                        audit.update({"skipped": True, "reason": "qig_eta_exceeds_18h"})
                        return dict(base.scales), audit
        chosen_steps, convergence = choose_qig_steps(
            convergence_inputs, rank_threshold=spearman_threshold,
            jaccard_threshold=jaccard_threshold,
        )
        computed_steps = list(initial_steps)
        # Only pay for larger IG grids if the previous adjacent pair did not
        # meet both convergence thresholds.
        for step in steps_for_layer[len(initial_steps):]:
            if convergence["converged"]:
                break
            convergence_inputs[step] = []
            for trace in audit_traces:
                value = _qig_token_importance_recurrent(
                    fp_block, base_block, trace, step, iqr_multiplier=1.5,
                )
                cached[(trace.sample_id, step)] = value
                convergence_inputs[step].append(torch.cat((
                    value["vision"].flatten(), value["answer_language"].flatten(),
                )))
            computed_steps.append(step)
            chosen_steps, convergence = choose_qig_steps(
                convergence_inputs, rank_threshold=spearman_threshold,
                jaccard_threshold=jaccard_threshold,
            )
        weights = {}
        for trace in traces[layer]:
            value = cached.get((trace.sample_id, chosen_steps))
            if value is None:
                value = _qig_token_importance_recurrent(
                    fp_block, base_block, trace, chosen_steps, iqr_multiplier=1.5,
                )
            weights[trace.sample_id] = value
        replacements, search = _search_qig_block(
            fp_block, modules, result, traces[layer], weights, config, alpha_grid,
        )
        result.update(replacements)
        audit["layers"][str(layer)] = {
            "chosen_steps": chosen_steps, "convergence": convergence,
            "computed_steps": computed_steps,
            "search": search,
        }
        if layer_position + 1 >= len(effective_layers):
            break
    audit["effective_layers"] = effective_layers
    audit["effective_steps"] = list(effective_steps)
    audit["fake_quantized"] = True
    audit["deployment_kernel_claim"] = False
    return result, audit


def accept_qig_layers(
    model,
    processor,
    targets: list[LinearTarget],
    samples,
    scheme: QuantScheme,
    base_scales: dict[str, torch.Tensor],
    candidate_scales: dict[str, torch.Tensor],
    ordered_layers: list[int],
    *,
    min_pixels: int,
    max_pixels: int,
) -> tuple[dict[str, torch.Tensor], dict]:
    current = {name: value.clone() for name, value in base_scales.items()}
    with quantized_linears(model, targets, scheme, scales=current):
        current_metric = teacher_nll_only(
            model, processor, samples, min_pixels=min_pixels, max_pixels=max_pixels,
        )
    audit = {"initial": current_metric, "layers": {}}
    for layer in ordered_layers:
        trial = {name: value.clone() for name, value in current.items()}
        changed = [target.name for target in targets if target.layer == layer]
        for name in changed:
            trial[name] = candidate_scales[name].clone()
        with quantized_linears(model, targets, scheme, scales=trial):
            metric = teacher_nll_only(
                model, processor, samples, min_pixels=min_pixels, max_pixels=max_pixels,
            )
        global_change = metric["nll"] / current_metric["nll"] - 1.0
        source_changes = {
            source: metric["by_source"][source] / current_metric["by_source"][source] - 1.0
            for source in current_metric["by_source"]
        }
        accepted = global_change <= 0.002 and max(source_changes.values()) <= 0.02
        audit["layers"][str(layer)] = {
            "accepted": accepted, "global_relative_change": global_change,
            "source_relative_change": source_changes, "metric": metric,
        }
        if accepted:
            current, current_metric = trial, metric
    audit["final"] = current_metric
    return current, audit
