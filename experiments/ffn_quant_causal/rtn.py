from __future__ import annotations

import gc
from contextlib import nullcontext
from pathlib import Path

import torch

from experiments.infinitevl_quant.config import SCHEMES
from experiments.infinitevl_quant.progress import StageProgress, atomic_json
from experiments.infinitevl_quant.quantization import quantized_linears
from experiments.vlm_quant_benchmark.data import MultimodalSample

from .adapters import ModelBundle, infinitevl_attention_targets
from .budget import WallclockBudget
from .evaluation import evaluate_variant


MATCHED_FFN_BITS = (8, 4, 3)


def _module_identity(model, targets) -> dict[str, tuple]:
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


def _assert_restored(model, before: dict[str, tuple]) -> None:
    failures = []
    for name, signature in before.items():
        (
            original,
            module_type,
            weight,
            weight_pointer,
            weight_version,
            bias,
            bias_pointer,
            bias_version,
        ) = signature
        current = model.get_submodule(name)
        restored = (
            current is original
            and type(current) is module_type
            and getattr(current, "weight", None) is weight
            and current.weight.data_ptr() == weight_pointer
            and current.weight._version == weight_version
            and getattr(current, "bias", None) is bias
            and (
                bias is None
                or (
                    current.bias.data_ptr() == bias_pointer
                    and current.bias._version == bias_version
                )
            )
        )
        if not restored:
            failures.append(name)
    if failures:
        raise RuntimeError(
            "fake-quant context did not exactly restore original modules: "
            f"{failures[:8]}"
        )


def _evaluate(
    bundle: ModelBundle,
    samples: list[MultimodalSample],
    *,
    progress: StageProgress,
    variant: str,
    reference_dir: Path,
    write_reference: bool,
    min_pixels: int,
    max_pixels: int,
    max_new_tokens: int,
    generate: bool,
    wallclock_budget: WallclockBudget | None,
) -> dict:
    summary = evaluate_variant(
        bundle.model,
        bundle.processor,
        samples,
        progress=progress,
        variant=variant,
        reference_dir=reference_dir,
        write_reference=write_reference,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        max_new_tokens=max_new_tokens,
        generate=generate,
        wallclock_budget=wallclock_budget,
    )
    if not torch.isfinite(torch.tensor(summary["target_nll"])):
        raise FloatingPointError(f"non-finite target NLL for {variant}")
    return summary


def _quantized_evaluation(
    bundle: ModelBundle,
    targets,
    bits: int,
    samples: list[MultimodalSample],
    *,
    progress: StageProgress,
    variant: str,
    reference_dir: Path,
    min_pixels: int,
    max_pixels: int,
    max_new_tokens: int,
    generate: bool,
    audit_root: Path,
    wallclock_budget: WallclockBudget | None,
) -> dict:
    scheme = SCHEMES[f"w{bits}a16"]
    before = _module_identity(bundle.model, targets)
    try:
        with quantized_linears(bundle.model, targets, scheme) as quant_audit:
            atomic_json(audit_root / f"{variant}.json", quant_audit)
            return _evaluate(
                bundle,
                samples,
                progress=progress,
                variant=variant,
                reference_dir=reference_dir,
                write_reference=False,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                max_new_tokens=max_new_tokens,
                generate=generate,
                wallclock_budget=wallclock_budget,
            )
    finally:
        _assert_restored(bundle.model, before)
        gc.collect()


def run_matched_rtn(
    bundle: ModelBundle,
    splits: dict[str, list[MultimodalSample]],
    *,
    progress: StageProgress,
    output_dir: Path,
    min_pixels: int,
    max_pixels: int,
    max_new_tokens: int,
    generate: bool = True,
    wallclock_budget: WallclockBudget | None = None,
) -> dict:
    """Run the matched FFN-only RTN matrix and InfiniteVL's W4 factorial.

    W2 is deliberately only a four-sample failure-boundary smoke test.  It is
    not mixed into the 32-sample W8/W4/W3 screen statistics.
    """

    kind = bundle.kind
    reference_root = output_dir / "references" / kind
    audit_root = progress.root / "quantization_audits" / kind
    audit_root.mkdir(parents=True, exist_ok=True)
    result: dict[str, dict] = {}

    screen_reference = reference_root / "screen32"
    baseline_variant = f"{kind}__screen32__bf16"
    result[baseline_variant] = _evaluate(
        bundle,
        splits["screen"],
        progress=progress,
        variant=baseline_variant,
        reference_dir=screen_reference,
        write_reference=True,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        max_new_tokens=max_new_tokens,
        generate=generate,
        wallclock_budget=wallclock_budget,
    )
    for bits in MATCHED_FFN_BITS:
        variant = f"{kind}__screen32__rtn_w{bits}a16_ffn"
        result[variant] = _quantized_evaluation(
            bundle,
            bundle.ffn_targets,
            bits,
            splits["screen"],
            progress=progress,
            variant=variant,
            reference_dir=screen_reference,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            max_new_tokens=max_new_tokens,
            generate=generate,
            audit_root=audit_root,
            wallclock_budget=wallclock_budget,
        )

    smoke_reference = reference_root / "smoke4"
    smoke_baseline = f"{kind}__smoke4__bf16"
    result[smoke_baseline] = _evaluate(
        bundle,
        splits["smoke4"],
        progress=progress,
        variant=smoke_baseline,
        reference_dir=smoke_reference,
        write_reference=True,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        max_new_tokens=max_new_tokens,
        generate=False,
        wallclock_budget=wallclock_budget,
    )
    smoke_w2 = f"{kind}__smoke4__rtn_w2a16_ffn"
    result[smoke_w2] = _quantized_evaluation(
        bundle,
        bundle.ffn_targets,
        2,
        splits["smoke4"],
        progress=progress,
        variant=smoke_w2,
        reference_dir=smoke_reference,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        max_new_tokens=max_new_tokens,
        generate=False,
        audit_root=audit_root,
        wallclock_budget=wallclock_budget,
    )

    if kind == "infinitevl":
        attention = infinitevl_attention_targets(
            bundle.model, include_dynamics=False,
        )
        decoder_safe = [*bundle.ffn_targets, *attention]
        if len(attention) != 171 or len(decoder_safe) != 279:
            raise RuntimeError(
                "InfiniteVL W4 factorial inventory mismatch: "
                f"attention={len(attention)}, decoder_safe={len(decoder_safe)}"
            )
        for suffix, targets in (
            ("attn_main", attention),
            ("decoder_safe", decoder_safe),
        ):
            variant = f"{kind}__screen32__rtn_w4a16_{suffix}"
            result[variant] = _quantized_evaluation(
                bundle,
                targets,
                4,
                splits["screen"],
                progress=progress,
                variant=variant,
                reference_dir=screen_reference,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                max_new_tokens=max_new_tokens,
                generate=generate,
                audit_root=audit_root,
                wallclock_budget=wallclock_budget,
            )

    # Projection and layer-1 interventions are teacher-forced exploratory
    # probes on the already-used screen partition.  The dedicated QIG
    # acceptance partition remains untouched until MBQ/QIG layer rollback.
    diagnostic_reference = reference_root / "diagnostic8"
    diagnostic_baseline = f"{kind}__diagnostic8__bf16"
    result[diagnostic_baseline] = _evaluate(
        bundle,
        splits["diagnostic8"],
        progress=progress,
        variant=diagnostic_baseline,
        reference_dir=diagnostic_reference,
        write_reference=True,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        max_new_tokens=max_new_tokens,
        generate=False,
        wallclock_budget=wallclock_budget,
    )
    projection_targets = {
        projection: [
            target for target in bundle.ffn_targets
            if target.name.endswith(f".{projection}")
        ]
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    layer_one = [target for target in bundle.ffn_targets if target.layer == 1]
    layer_one_gate = [
        target for target in layer_one if target.name.endswith(".gate_proj")
    ]
    if any(len(values) != 36 for values in projection_targets.values()):
        raise RuntimeError(f"{kind} projection intervention inventory mismatch")
    if len(layer_one) != 3 or len(layer_one_gate) != 1:
        raise RuntimeError(f"{kind} layer-1 intervention inventory mismatch")
    interventions = [
        (f"all_{projection}", targets)
        for projection, targets in projection_targets.items()
    ] + [
        ("layer01_gate_proj", layer_one_gate),
        ("layer01_ffn", layer_one),
    ]
    for bits in (4, 3):
        for label, targets in interventions:
            variant = f"{kind}__diagnostic8__rtn_w{bits}a16_{label}"
            result[variant] = _quantized_evaluation(
                bundle,
                targets,
                bits,
                splits["diagnostic8"],
                progress=progress,
                variant=variant,
                reference_dir=diagnostic_reference,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                max_new_tokens=max_new_tokens,
                generate=False,
                audit_root=audit_root,
                wallclock_budget=wallclock_budget,
            )

    return result
