from __future__ import annotations

import contextlib
import math
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from experiments.ffn_quant_causal.adapters import ModelBundle
from experiments.ffn_quant_causal.budget import WallclockBudget
from experiments.ffn_quant_causal.evaluation import (
    _reference_path,
    aggregate_rows,
    validate_variant_rows,
)
from experiments.infinitevl_quant.evaluation import (
    _compare_logits,
    _target_metrics,
    _teacher_inputs,
)
from experiments.infinitevl_quant.inventory import decoder_layers
from experiments.infinitevl_quant.progress import StageProgress, atomic_json, safe_id
from experiments.infinitevl_quant_ffn_bf16_4x64.scoring import (
    SCORER_ID,
    score_prediction,
)
from experiments.vlm_quant_benchmark.data import MultimodalSample

from .calibration import SampleObservationCollector
from .gdn import (
    GDNReference,
    capture_gdn_reference,
    intervene_gdn,
)
from .shadow import shadow_ffns


def _atomic_torch_save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def gdn_reference_path(root: Path, sample_id: str) -> Path:
    return root / f"{safe_id(sample_id)}.pt"


def save_gdn_reference(path: Path, reference: GDNReference, *, sample_id: str) -> None:
    _atomic_torch_save(path, reference.payload(sample_id=sample_id))


def load_gdn_reference(path: Path, *, sample_id: str) -> GDNReference:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return GDNReference.from_payload(payload, sample_id=sample_id)


def collect_shadow_calibration(
    bundle: ModelBundle,
    samples: Sequence[MultimodalSample],
    *,
    progress: StageProgress,
    variant: str,
    min_pixels: int,
    max_pixels: int,
    bits: int,
    group_size: int,
    scales: Mapping[str, torch.Tensor] | None = None,
    wallclock_budget: WallclockBudget | None = None,
) -> list[dict[str, Any]]:
    layers = list(decoder_layers(bundle.model))
    for sample in samples:
        if progress.completed_sample(variant, sample.id) is not None:
            continue
        if wallclock_budget is not None:
            wallclock_budget.check(
                stage=progress.root.name,
                boundary="before_calibration_sample",
                variant=variant,
                sample_id=sample.id,
            )
        _prompt, teacher, _labels, _positions = _teacher_inputs(
            bundle.processor,
            sample,
            min_pixels,
            max_pixels,
        )
        collector = SampleObservationCollector()
        started = time.perf_counter()
        with (
            shadow_ffns(
                layers,
                decoder_prefix=bundle.decoder_prefix,
                gammas=0.0,
                bits=bits,
                group_size=group_size,
                scales=scales,
                observer=collector,
            ),
            torch.inference_mode(),
        ):
            bundle.model(**teacher, use_cache=False, return_dict=True)
        elapsed = time.perf_counter() - started
        progress.complete_sample(
            variant,
            sample.id,
            {
                "id": sample.id,
                "dataset": sample.dataset,
                "model_kind": bundle.kind,
                "sequence_tokens": int(teacher["input_ids"].shape[-1]),
                "teacher_seconds": elapsed,
                "layers": collector.finalize(),
                "scope": "all_36_ffn_shadow_calibration",
            },
        )
        if wallclock_budget is not None:
            wallclock_budget.check(
                stage=progress.root.name,
                boundary="after_calibration_sample",
                variant=variant,
                sample_id=sample.id,
            )
    rows = progress.rows(variant)
    if len(rows) != len(samples):
        raise RuntimeError(f"incomplete calibration variant {variant}")
    return rows


def _target_logits(output, positions: Sequence[int]) -> torch.Tensor:
    rows = torch.tensor([position - 1 for position in positions], dtype=torch.long)
    return output.logits[0, rows].detach().float().cpu()


def _eos_ids(model) -> set[int]:
    value = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if value is None:
        value = getattr(getattr(model, "config", None), "eos_token_id", None)
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    return {int(item) for item in value}


def _append_token(inputs: dict[str, torch.Tensor], token: torch.Tensor) -> None:
    inputs["input_ids"] = torch.cat((inputs["input_ids"], token.reshape(1, 1)), dim=-1)
    if "attention_mask" in inputs:
        one = torch.ones_like(inputs["attention_mask"][:, :1])
        inputs["attention_mask"] = torch.cat((inputs["attention_mask"], one), dim=-1)
    for name in ("position_ids", "cache_position"):
        inputs.pop(name, None)


@torch.no_grad()
def greedy_generate_with_gdn_intervention(
    bundle: ModelBundle,
    prompt: Mapping[str, torch.Tensor],
    *,
    gammas: Mapping[int, float],
    repair_groups: Sequence[str],
    carry_clamp: bool,
    max_new_tokens: int,
    bits: int,
    group_size: int,
    scales: Mapping[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Greedy no-cache generation with a same-prefix BF16 GDN reference.

    At each decoding step the reference pass sees the exact prefix generated
    by the candidate.  This avoids applying teacher-forced mediator tensors to
    a diverged autoregressive sequence.
    """

    if bundle.kind != "infinitevl":
        raise ValueError("GDN-mediated generation is InfiniteVL-only")
    layers = list(decoder_layers(bundle.model))
    inputs = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in prompt.items()}
    generated: list[torch.Tensor] = []
    eos = _eos_ids(bundle.model)
    for _ in range(max_new_tokens):
        reference = GDNReference()
        with (
            shadow_ffns(
                layers,
                decoder_prefix=bundle.decoder_prefix,
                gammas=0.0,
                bits=bits,
                group_size=group_size,
                scales=scales,
            ),
            capture_gdn_reference(bundle.model) as captured,
        ):
            bundle.model(**inputs, use_cache=False, return_dict=True)
            reference = captured
        with (
            shadow_ffns(
                layers,
                decoder_prefix=bundle.decoder_prefix,
                gammas=gammas,
                bits=bits,
                group_size=group_size,
                scales=scales,
            ),
            intervene_gdn(
                bundle.model,
                reference,
                repair_groups=repair_groups,
                carry_clamp=carry_clamp,
            ),
        ):
            output = bundle.model(**inputs, use_cache=False, return_dict=True)
        token = output.logits[:, -1].argmax(dim=-1)
        generated.append(token.cpu())
        _append_token(inputs, token)
        if int(token.item()) in eos:
            break
    if not generated:
        return torch.empty((1, 0), dtype=inputs["input_ids"].dtype)
    return torch.stack(generated, dim=1)


def _write_reference_logits(path: Path, sample_id: str, logits: torch.Tensor) -> None:
    _atomic_torch_save(
        path,
        {
            "format_version": 2,
            "id": sample_id,
            "target_logits": logits,
        },
    )


def _read_reference_logits(path: Path, sample_id: str) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"missing FP32 reference logits: {path}")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value.get("format_version") != 2 or value.get("id") != sample_id:
        raise RuntimeError(f"invalid reference logits: {path}")
    return value["target_logits"].float()


def evaluate_shadow_variant(
    bundle: ModelBundle,
    samples: Sequence[MultimodalSample],
    *,
    progress: StageProgress,
    variant: str,
    gammas: Mapping[int, float],
    reference_dir: Path,
    write_reference: bool,
    gdn_reference_dir: Path | None,
    capture_gdn: bool,
    repair_groups: Sequence[str] = (),
    carry_clamp: bool = False,
    force_gdn_intervention: bool = False,
    min_pixels: int,
    max_pixels: int,
    max_new_tokens: int,
    generate: bool,
    bits: int,
    group_size: int,
    scales: Mapping[str, torch.Tensor] | None = None,
    wallclock_budget: WallclockBudget | None = None,
) -> dict[str, Any]:
    if (capture_gdn or repair_groups or carry_clamp or force_gdn_intervention) and bundle.kind != "infinitevl":
        raise ValueError("GDN capture/intervention is InfiniteVL-only")
    if capture_gdn and (repair_groups or carry_clamp):
        raise ValueError("reference capture and GDN intervention are mutually exclusive")
    if (repair_groups or carry_clamp or force_gdn_intervention) and gdn_reference_dir is None:
        raise ValueError("GDN intervention requires a reference directory")
    layers = list(decoder_layers(bundle.model))
    reference_dir.mkdir(parents=True, exist_ok=True)
    if gdn_reference_dir is not None:
        gdn_reference_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        reference_path = _reference_path(reference_dir, sample.id)
        state_path = None if gdn_reference_dir is None else gdn_reference_path(gdn_reference_dir, sample.id)
        complete = progress.completed_sample(variant, sample.id)
        required_reference = write_reference and not reference_path.is_file()
        required_gdn = capture_gdn and (state_path is None or not state_path.is_file())
        if complete is not None and not required_reference and not required_gdn:
            continue
        if wallclock_budget is not None:
            wallclock_budget.check(
                stage=progress.root.name,
                boundary="before_sample",
                variant=variant,
                sample_id=sample.id,
            )
        prompt, teacher, labels, positions = _teacher_inputs(
            bundle.processor,
            sample,
            min_pixels,
            max_pixels,
        )
        gdn_reference = None
        if repair_groups or carry_clamp or force_gdn_intervention:
            assert state_path is not None
            gdn_reference = load_gdn_reference(state_path, sample_id=sample.id)
        intervention = None
        captured_reference = None
        started = time.perf_counter()
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                shadow_ffns(
                    layers,
                    decoder_prefix=bundle.decoder_prefix,
                    gammas=gammas,
                    bits=bits,
                    group_size=group_size,
                    scales=scales,
                )
            )
            if capture_gdn:
                captured_reference = stack.enter_context(capture_gdn_reference(bundle.model))
            elif gdn_reference is not None:
                intervention = stack.enter_context(
                    intervene_gdn(
                        bundle.model,
                        gdn_reference,
                        repair_groups=repair_groups,
                        carry_clamp=carry_clamp,
                    )
                )
            with torch.inference_mode():
                output = bundle.model(**teacher, use_cache=False, return_dict=True)
        teacher_seconds = time.perf_counter() - started
        logits = _target_logits(output, positions)
        target = _target_metrics(output.logits, labels)
        comparison = {}
        if write_reference:
            _write_reference_logits(reference_path, sample.id, logits)
        else:
            comparison = _compare_logits(
                logits,
                _read_reference_logits(reference_path, sample.id),
            )
        if capture_gdn:
            assert state_path is not None and captured_reference is not None
            save_gdn_reference(state_path, captured_reference, sample_id=sample.id)

        prediction = ""
        generation_seconds = 0.0
        generation_tokens = 0
        truncated = False
        if generate:
            started = time.perf_counter()
            if repair_groups or carry_clamp or force_gdn_intervention:
                new_ids = greedy_generate_with_gdn_intervention(
                    bundle,
                    prompt,
                    gammas=gammas,
                    repair_groups=repair_groups,
                    carry_clamp=carry_clamp,
                    max_new_tokens=max_new_tokens,
                    bits=bits,
                    group_size=group_size,
                    scales=scales,
                )
            else:
                with (
                    shadow_ffns(
                        layers,
                        decoder_prefix=bundle.decoder_prefix,
                        gammas=gammas,
                        bits=bits,
                        group_size=group_size,
                        scales=scales,
                    ),
                    torch.inference_mode(),
                ):
                    generated = bundle.model.generate(
                        **prompt,
                        do_sample=False,
                        max_new_tokens=max_new_tokens,
                        use_cache=True,
                    )
                new_ids = generated[:, int(prompt["input_ids"].shape[-1]) :].cpu()
            generation_seconds = time.perf_counter() - started
            generation_tokens = int(new_ids.shape[-1])
            truncated = generation_tokens >= max_new_tokens
            prediction = bundle.processor.batch_decode(
                new_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
        scored = score_prediction(sample, prediction) if generate else None
        row = {
            "id": sample.id,
            "dataset": sample.dataset,
            "target": sample.target,
            "prediction": prediction,
            "score": math.nan if scored is None else float(scored.score),
            "scoring": None
            if scored is None
            else {
                "scorer": SCORER_ID,
                "rule": scored.rule,
                "extracted": scored.extracted,
            },
            "target_text": target,
            "comparison_to_bf16": comparison,
            "teacher_seconds": teacher_seconds,
            "generation_seconds": generation_seconds,
            "generation_tokens": generation_tokens,
            "generation_truncated": truncated,
            "sequence_tokens": int(teacher["input_ids"].shape[-1]),
            "condition": {
                "gammas": {str(key): float(value) for key, value in gammas.items()},
                "repair_groups": sorted(set(repair_groups)),
                "carry_clamp": bool(carry_clamp),
                "gdn_backend": (
                    "paired_recurrent_intervention"
                    if repair_groups or carry_clamp or force_gdn_intervention
                    else "native"
                ),
                "diagnostic_counterfactual": bool(repair_groups or carry_clamp or force_gdn_intervention),
            },
            "state_audits": ({} if intervention is None else intervention.state_audits),
        }
        progress.complete_sample(variant, sample.id, row)
        if wallclock_budget is not None:
            wallclock_budget.check(
                stage=progress.root.name,
                boundary="after_sample_committed",
                variant=variant,
                sample_id=sample.id,
            )
    rows = validate_variant_rows(progress.rows(variant), list(samples), variant=variant)
    summary = aggregate_rows(rows)
    consistency = float(summary["target_argmax_consistency"])
    summary["target_argmax_change_rate"] = 1.0 - consistency if math.isfinite(consistency) else math.nan
    summary["variant"] = variant
    summary["diagnostic_counterfactual"] = bool(repair_groups or carry_clamp or force_gdn_intervention)
    atomic_json(progress.root / variant / "metrics.json", summary)
    return summary
