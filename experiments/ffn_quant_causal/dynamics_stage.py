from __future__ import annotations

import gc
import json
import os
from collections import defaultdict
from pathlib import Path

import torch

from experiments.infinitevl_quant.config import SCHEMES
from experiments.infinitevl_quant.progress import StageProgress, atomic_json, safe_id
from experiments.infinitevl_quant.quantization import quantized_linears
from experiments.infinitevl_quant.state_probe import (
    collect_states,
    compare_states,
    state_probe_token_count,
)
from experiments.vlm_quant_benchmark.data import MultimodalSample
from experiments.vlm_quant_benchmark.mbq import mbq_calibration_batches
from experiments.vlm_quant_benchmark.sensitivity import collect_modality_perturbations

from .adapters import ModelBundle


PROPAGATION_LAYERS = (1, 17, 35)
RELATIVE_NOISE_RMS = 0.01


def _trace_row(trace_root: Path, kind: str, sample_id: str) -> dict:
    from experiments.infinitevl_quant.progress import safe_id as progress_safe_id

    path = (
        trace_root / f"{kind}__trace_smoke4" / "samples"
        / f"{progress_safe_id(sample_id)}.json"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    row = json.loads(path.read_text())
    if row.get("id") != sample_id or row.get("model_kind") != kind:
        raise RuntimeError(f"trace-row identity mismatch: {path}")
    return row


def _aggregate_noise(rows: list[dict]) -> dict:
    grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for sample in rows:
        for result in sample["perturbations"]:
            grouped[(int(result["layer"]), str(result["modality"]))].append(result)
    aggregates = []
    for (layer, modality), values in sorted(grouped.items()):
        aggregates.append({
            "layer": layer,
            "modality": modality,
            "samples": len(values),
            "mean_delta_nll": sum(float(row["delta_nll"]) for row in values) / len(values),
            "mean_target_kl": sum(float(row["target_kl"]) for row in values) / len(values),
            "mean_target_argmax_change_rate": sum(
                float(row["target_argmax_change_rate"]) for row in values
            ) / len(values),
        })
    return {
        "perturbation": "deterministic_gaussian_decoder_block_output",
        "normalization": "one_percent_of_observed_block_output_rms_per_model_sample_layer",
        "cross_model_interpretation": (
            "matched relative noise; absolute amplitudes intentionally differ with signal RMS"
        ),
        "layers": list(PROPAGATION_LAYERS),
        "relative_noise_rms": RELATIVE_NOISE_RMS,
        "aggregates": aggregates,
    }


def run_relative_noise_propagation(
    bundle: ModelBundle,
    samples: list[MultimodalSample],
    *,
    progress: StageProgress,
    trace_root: Path,
    min_pixels: int,
    max_pixels: int,
    seed: int,
) -> dict:
    """Compare downstream amplification after matched relative block noise."""

    variant = f"{bundle.kind}__relative_noise_smoke4"
    for sample in samples:
        if progress.completed_sample(variant, sample.id) is not None:
            continue
        trace = _trace_row(trace_root, bundle.kind, sample.id)
        summaries = {
            int(row["layer"]): row for row in trace["layer_summaries"]
        }
        amplitudes = {
            layer: RELATIVE_NOISE_RMS * float(summaries[layer]["branch"]["merged_rms"])
            for layer in PROPAGATION_LAYERS
        }
        batch = next(mbq_calibration_batches(
            bundle.processor,
            [sample],
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        ))
        perturbations = collect_modality_perturbations(
            bundle.model,
            batch,
            layers_to_test=PROPAGATION_LAYERS,
            noise_rms=amplitudes,
            seed=seed,
            attention_implementation="eager",
        )
        if not perturbations:
            raise RuntimeError(f"no propagation rows for {bundle.kind}/{sample.id}")
        progress.complete_sample(variant, sample.id, {
            "id": sample.id,
            "dataset": sample.dataset,
            "model_kind": bundle.kind,
            "relative_noise_rms": RELATIVE_NOISE_RMS,
            "absolute_noise_rms": {str(key): value for key, value in amplitudes.items()},
            "perturbations": perturbations,
            "analysis_scope": "causal_decoder_block_noise_propagation",
            "end_to_end_model_score": False,
        })
        print(f"[propagation/{bundle.kind}] {sample.id} complete", flush=True)
        gc.collect()
    rows = progress.rows(variant)
    summary = _aggregate_noise(rows)
    summary.update({"model_kind": bundle.kind, "samples": len(rows)})
    atomic_json(progress.root / variant / "metrics.json", summary)
    return summary


def _atomic_torch_save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def select_state_probe_samples(
    processor,
    candidates: list[MultimodalSample],
    *,
    min_pixels: int,
    max_pixels: int,
) -> list[tuple[MultimodalSample, int]]:
    by_dataset: dict[str, list[MultimodalSample]] = defaultdict(list)
    for sample in candidates:
        by_dataset[sample.dataset].append(sample)
    selected = []
    for dataset, values in sorted(by_dataset.items()):
        for sample in values:
            tokens = state_probe_token_count(
                processor, sample, min_pixels=min_pixels, max_pixels=max_pixels,
            )
            if tokens >= 512:
                selected.append((sample, tokens))
                break
        else:
            raise RuntimeError(f"{dataset} has no diagnostic sample with >=512 prompt tokens")
    if len(selected) != 4:
        raise RuntimeError(f"expected four state-probe samples, got {len(selected)}")
    return selected


def run_infinitevl_ffn_state_drift(
    bundle: ModelBundle,
    candidates: list[MultimodalSample],
    *,
    progress: StageProgress,
    min_pixels: int,
    max_pixels: int,
) -> dict:
    """Measure GDN recurrent-state drift caused only by W4 FFN RTN."""

    if bundle.kind != "infinitevl":
        raise ValueError("GDN state drift is defined only for InfiniteVL")
    selected = select_state_probe_samples(
        bundle.processor,
        candidates,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    reference_root = progress.root / "gdn_state_references"
    for sample, sequence_tokens in selected:
        path = reference_root / f"{safe_id(sample.id)}.pt"
        if not path.is_file():
            states, observed_tokens = collect_states(
                bundle.model,
                bundle.processor,
                sample,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            if observed_tokens != sequence_tokens:
                raise RuntimeError("state-probe token count changed between selection and capture")
            _atomic_torch_save(path, {
                "format_version": 1,
                "id": sample.id,
                "sequence_tokens": sequence_tokens,
                "states": states,
            })
    variant = "infinitevl__gdn_state_drift_rtn_w4a16_ffn"
    before = {
        target.name: (
            module,
            module.weight,
            module.weight.data_ptr(),
            module.weight._version,
        )
        for target in bundle.ffn_targets
        for module in (bundle.model.get_submodule(target.name),)
    }
    scheme = SCHEMES["w4a16"]
    try:
        with quantized_linears(
            bundle.model, bundle.ffn_targets, scheme,
        ) as quant_audit:
            atomic_json(progress.root / "gdn_state_quantization.json", quant_audit)
            for sample, sequence_tokens in selected:
                if progress.completed_sample(variant, sample.id) is not None:
                    continue
                reference_path = reference_root / f"{safe_id(sample.id)}.pt"
                reference = torch.load(
                    reference_path, map_location="cpu", weights_only=True,
                )
                if reference.get("id") != sample.id:
                    raise RuntimeError(f"state reference identity mismatch: {reference_path}")
                candidate, observed_tokens = collect_states(
                    bundle.model,
                    bundle.processor,
                    sample,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                )
                if observed_tokens != sequence_tokens:
                    raise RuntimeError("quantized state-probe token count changed")
                comparison = compare_states(reference["states"], candidate)
                progress.complete_sample(variant, sample.id, {
                    "id": sample.id,
                    "dataset": sample.dataset,
                    "sequence_tokens": sequence_tokens,
                    "comparison": comparison,
                    "quantized_scope": "ffn_only",
                    "scheme": "rtn_w4a16_asymmetric_groupwise_g128",
                })
                print(f"[gdn-state] {sample.id} complete", flush=True)
    finally:
        failures = []
        for name, (module, weight, pointer, version) in before.items():
            current = bundle.model.get_submodule(name)
            if not (
                current is module
                and current.weight is weight
                and current.weight.data_ptr() == pointer
                and current.weight._version == version
            ):
                failures.append(name)
        if failures:
            raise RuntimeError(f"GDN state probe failed to restore modules: {failures[:8]}")
    rows = progress.rows(variant)
    aucs = [float(row["comparison"]["state_drift_auc"]) for row in rows]
    summary = {
        "samples": len(rows),
        "state_drift_auc_mean": sum(aucs) / len(aucs),
        "state_drift_auc_by_sample": {
            row["id"]: row["comparison"]["state_drift_auc"] for row in rows
        },
        "quantized_scope": "ffn_only",
        "scheme": "rtn_w4a16_asymmetric_groupwise_g128",
        "interpretation": "downstream GDN state drift induced by upstream FFN fake quantization",
    }
    atomic_json(progress.root / variant / "metrics.json", summary)
    return summary
