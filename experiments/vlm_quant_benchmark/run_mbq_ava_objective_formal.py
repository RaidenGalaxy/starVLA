"""Formal fixed-A_VA MBQ reconstruction-objective sweep."""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import os
import random
import shutil
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import psutil
import torch

from .config import ExperimentConfig, ImplementationTier, MethodName, QuantScope
from .data import (
    file_digest,
    load_manifest,
    sample_manifest_digest,
    write_manifest,
)
from .mbq import (
    MBQSearchResult,
    apply_mbq,
    collect_mbq_search_cached,
    load_mbq_search,
    mbq_cache_key,
    mbq_calibration_batches,
    mbq_statistics_cache_key,
)
from .provenance import source_digest
from .reporting import atomic_json
from .residuals import (
    capture_reconstruction_outputs,
    prediction_change_summary,
    reconstruction_rows,
)
from .run_mbq_factorial_formal import (
    _atomic_csv,
    _atomic_jsonl,
    _benjamini_hochberg,
    _nll_samples,
    _sample_cache_name,
    _scale_digest,
)
from .run_sensitivity_ablation import _evaluate_samples_cached
from .run_sensitivity_formal import _active_quant_processes, _run, pid_lock
from .runner import environment_record, load_reference
from .selection import select_linears
from .sensitivity import sensitivity_anchors, stable_hash_subset


STAGES = (
    "preflight",
    "search",
    "w4_nll",
    "screen_analysis",
    "residuals",
    "promotion",
    "final",
)
VA = "vision_answer"
VPA = "vision_prompt_answer"
LAYERS = (0, 6, 12, 18, 24, 30, 34)
CONTROL = "ava_ova_control"
P100 = "ava_ovpa_tw_p100"
ENDPOINT_CELLS = {
    CONTROL: "A_VA__O_VA",
    P100: "A_VA__O_VPA",
}
VARIANT_SPECS = {
    CONTROL: ("token_weighted", 0.0, VA),
    "ava_ovpa_tw_p0125": ("token_weighted", 0.125, VPA),
    "ava_ovpa_tw_p025": ("token_weighted", 0.25, VPA),
    "ava_ovpa_tw_p050": ("token_weighted", 0.50, VPA),
    P100: ("token_weighted", 1.00, VPA),
    "ava_ovpa_mm_p025": ("modality_mean", 0.25, VPA),
    "ava_ovpa_mm_p050": ("modality_mean", 0.50, VPA),
    "ava_ovpa_mm_p100": ("modality_mean", 1.00, VPA),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the formal fixed-A_VA MBQ objective sweep",
    )
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/vlm_quant_benchmark/sensitivity_analysis_v1/"
            "ava_objective_v1"
        ),
    )
    parser.add_argument(
        "--formal-root",
        type=Path,
        default=Path(
            "outputs/vlm_quant_benchmark/sensitivity_analysis_v1/formal_128"
        ),
    )
    parser.add_argument(
        "--trimodal-root",
        type=Path,
        default=Path(
            "outputs/vlm_quant_benchmark/sensitivity_analysis_v1/trimodal_v1"
        ),
    )
    parser.add_argument(
        "--factorial-root",
        type=Path,
        default=Path(
            "outputs/vlm_quant_benchmark/sensitivity_analysis_v1/"
            "trimodal_factorial_v1"
        ),
    )
    parser.add_argument("--raw-cache-dir", type=Path)
    parser.add_argument(
        "--model-path",
        default="playground/Pretrained_models/Qwen2.5-VL-3B-Instruct",
    )
    parser.add_argument(
        "--calibration-manifest",
        type=Path,
        default=Path(
            "data/vlm_quant_benchmark/multimodal_calibration/"
            "multimodal_calibration_4x32.jsonl"
        ),
    )
    parser.add_argument(
        "--evaluation-manifest",
        type=Path,
        default=Path(
            "data/vlm_quant_benchmark/top3_protocol_v4/"
            "vlm_confirm_4x256.jsonl"
        ),
    )
    parser.add_argument(
        "--vlmevalkit-root", type=Path, default=Path("third_party/VLMEvalKit"),
    )
    parser.add_argument(
        "--lmu-data-root",
        type=Path,
        default=Path("data/vlm_quant_benchmark/LMUData"),
    )
    parser.add_argument(
        "--squeezellm-result",
        type=Path,
        default=Path(
            "outputs/vlm_quant_benchmark/top3_protocol_v4/confirm/"
            "squeezellm_w4a16_g128_asym/official_result.json"
        ),
    )
    parser.add_argument(
        "--mbq-current-result",
        type=Path,
        default=Path(
            "outputs/vlm_quant_benchmark/top3_protocol_v4/confirm/"
            "mbq_w4a16_g128_asym/official_result.json"
        ),
    )
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--skip-real-smoke", action="store_true")
    parser.add_argument("--allow-nonstandard-model", action="store_true")
    return parser.parse_args()


def objective_configs(
    *, calibration_samples: int = 128, seed: int = 42,
) -> dict[str, ExperimentConfig]:
    common = dict(
        method=MethodName.MBQ,
        scope=QuantScope.DECODER_CANONICAL,
        weight_bits=4,
        activation_bits=None,
        group_size=128,
        zero_point=True,
        weight_granularity="groupwise",
        calibration_samples=calibration_samples,
        calibration_protocol="multimodal_train_4x32",
        seed=seed,
        implementation_tier=ImplementationTier.LOCAL_VALIDATED,
        mbq_modality_mode=VPA,
        mbq_algorithm_version=5,
        mbq_gradient_aggregation="pooled_elements",
        mbq_reconstruction_level="linear_group",
        mbq_median_floor=False,
        mbq_reweight=True,
        mbq_activation_mode=VA,
    )
    configs = {}
    for variant, (normalization, prompt_scale, objective_mode) in VARIANT_SPECS.items():
        if variant == CONTROL:
            scheme = "mbq_factorial_a_va__o_va_w4a16"
            configured_scale = 1.0
        elif variant == P100:
            scheme = "mbq_factorial_a_va__o_vpa_w4a16"
            configured_scale = 1.0
        else:
            scale_tag = str(prompt_scale).replace(".", "")
            norm_tag = "tw" if normalization == "token_weighted" else "mm"
            scheme = f"mbq_ava_{norm_tag}_p{scale_tag}_w4a16"
            configured_scale = prompt_scale
        configs[variant] = ExperimentConfig(
            **common,
            mbq_objective_mode=objective_mode,
            mbq_objective_normalization=normalization,
            mbq_prompt_objective_scale=configured_scale,
            scheme_id=scheme,
        )
    return configs


def _raw_cache_dir(args: argparse.Namespace) -> Path:
    return args.raw_cache_dir or args.trimodal_root / "_calibration_cache"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _search_identity(
    selected,
    calibration,
    config: ExperimentConfig,
    args: argparse.Namespace,
) -> tuple[str, str]:
    model_digest = source_digest(args.model_path)
    calibration_digest = sample_manifest_digest(calibration)
    names = [item.name for item in selected]
    key = mbq_cache_key(
        model_revision=model_digest,
        calibration_digest=calibration_digest,
        selected_names=names,
        config=config,
        reservoir_rows=32,
        attention_implementation="sdpa",
    )
    raw_key = mbq_statistics_cache_key(
        model_revision=model_digest,
        calibration_digest=calibration_digest,
        selected_names=names,
        reservoir_rows=32,
        seed=config.seed,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        attention_implementation="sdpa",
        modality_mode=config.mbq_modality_mode,
        algorithm_version=config.mbq_algorithm_version,
    )
    return key, raw_key


def _new_search(
    model,
    selected,
    processor,
    calibration,
    config: ExperimentConfig,
    args: argparse.Namespace,
    *,
    require_shared_raw: bool = True,
) -> tuple[MBQSearchResult, bool, str, str]:
    key, raw_key = _search_identity(selected, calibration, config, args)
    search_cache = args.output_dir / "_calibration_cache"
    raw_cache = _raw_cache_dir(args) if require_shared_raw else search_cache
    raw_path = raw_cache / f"mbq_stats_{raw_key}.pt"
    if require_shared_raw and not raw_path.is_file():
        raise FileNotFoundError(
            f"required read-only tri-modal raw statistics are absent: {raw_path}"
        )
    result, cache_hit = collect_mbq_search_cached(
        model,
        selected,
        mbq_calibration_batches(
            processor,
            calibration,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
        ),
        config,
        cache_dir=search_cache,
        raw_cache_dir=raw_cache,
        key=key,
        raw_key=raw_key,
        reservoir_rows=32,
        attention_implementation="sdpa",
    )
    return result, cache_hit, key, raw_key


def _endpoint_search(
    variant: str,
    selected,
    calibration,
    config: ExperimentConfig,
    args: argparse.Namespace,
) -> tuple[MBQSearchResult, str, Path]:
    key, _raw_key = _search_identity(selected, calibration, config, args)
    path = args.factorial_root / "_calibration_cache" / f"mbq_{key}.pt"
    result = load_mbq_search(path, key=key, selected=selected)
    if result is None:
        raise FileNotFoundError(f"verified factorial endpoint is unavailable: {path}")
    cell = ENDPOINT_CELLS[variant]
    run_audit = json.loads(
        (args.factorial_root / "search_audits/run_audit.json").read_text()
    )
    source = next(
        (item for item in run_audit["cells"] if item["cell"] == cell),
        None,
    )
    if source is None:
        raise RuntimeError(f"factorial search audit is missing reuse cell: {cell}")
    if (
        source["config_digest"] != config.digest
        or source["search_key"] != key
        or source["scale_digest"] != _scale_digest(result)
        or source["groups"] != 144
        or source["scales"] != 252
    ):
        raise RuntimeError(f"factorial reuse identity mismatch: {cell}")
    verification = json.loads(
        (args.factorial_root / "search_audits/endpoint_verification.json").read_text()
    )
    historical = verification["endpoints"].get(cell)
    if historical is not None and not historical["nll_reuse_allowed"]:
        raise RuntimeError(f"factorial endpoint did not pass exact verification: {cell}")
    return result, key, path


def _load_search(
    variant: str,
    model,
    selected,
    processor,
    calibration,
    config: ExperimentConfig,
    args: argparse.Namespace,
) -> tuple[MBQSearchResult, bool, str]:
    if variant in ENDPOINT_CELLS:
        result, key, _path = _endpoint_search(
            variant, selected, calibration, config, args,
        )
        return result, True, key
    result, hit, key, _raw_key = _new_search(
        model, selected, processor, calibration, config, args,
    )
    return result, hit, key


def _run_search_stage(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    configs = objective_configs(seed=args.seed)
    calibration = load_manifest(args.calibration_manifest)[:128]
    model, processor, _ = load_reference(args.model_path)
    selected = select_linears(
        model,
        QuantScope.DECODER_CANONICAL,
        strict_qwen_3b=not args.allow_nonstandard_model,
    )
    searches = {}
    statuses = []
    try:
        for variant, config in configs.items():
            if variant in ENDPOINT_CELLS:
                search, key, source_path = _endpoint_search(
                    variant, selected, calibration, config, args,
                )
                cache_hit = True
                origin = "verified_factorial_endpoint"
            else:
                search, cache_hit, key, _raw_key = _new_search(
                    model, selected, processor, calibration, config, args,
                )
                source_path = args.output_dir / "_calibration_cache" / f"mbq_{key}.pt"
                origin = "new_or_resumed"
            searches[variant] = search
            audit = dict(search.audit)
            audit.update({
                "variant": variant,
                "configured_prompt_objective_scale": VARIANT_SPECS[variant][1],
                "scale_digest": _scale_digest(search),
                "artifact_origin": origin,
            })
            atomic_json(args.output_dir / "search_audits" / f"{variant}.json", audit)
            statuses.append({
                "variant": variant,
                "normalization": VARIANT_SPECS[variant][0],
                "prompt_objective_scale": VARIANT_SPECS[variant][1],
                "objective_mode": config.effective_mbq_objective_mode,
                "activation_mode": config.effective_mbq_activation_mode,
                "groups": len(search.audit["groups"]),
                "scales": len(search.scales),
                "scale_digest": _scale_digest(search),
                "search_key": key,
                "config": config.to_dict(),
                "config_digest": config.digest,
                "cache_hit": cache_hit,
                "artifact_origin": origin,
                "artifact_path": str(source_path.resolve()),
                "artifact_sha256": file_digest(source_path),
            })
    finally:
        del model, processor
    control_groups = searches[CONTROL].audit["groups"]
    activation_mismatches = {}
    for variant, search in searches.items():
        mismatches = [
            group for group, value in search.audit["groups"].items()
            if (
                group not in control_groups
                or value["activation_digest"]
                != control_groups[group]["activation_digest"]
                or value["activation_modalities"]
                != ["vision", "answer_language"]
            )
        ]
        activation_mismatches[variant] = mismatches
    if any(activation_mismatches.values()):
        raise RuntimeError(f"A_VA activation invariance failed: {activation_mismatches}")
    if any(item["groups"] != 144 or item["scales"] != 252 for item in statuses):
        raise RuntimeError("objective searches do not contain 144 groups / 252 scales")
    atomic_json(args.output_dir / "search_audits/activation_invariance.json", {
        "status": "complete",
        "control": CONTROL,
        "groups": len(control_groups),
        "all_activation_digests_exact": True,
        "activation_modalities": ["vision", "answer_language"],
        "mismatches": activation_mismatches,
    })
    atomic_json(args.output_dir / "search_audits/run_audit.json", {
        "status": "complete",
        "variants": statuses,
        "shared_raw_cache": str(_raw_cache_dir(args).resolve()),
        "shared_raw_cache_read_only": True,
    })


def _endpoint_nll_rows(
    variant: str, args: argparse.Namespace,
) -> tuple[list[dict], Path]:
    path = args.factorial_root / "w4_nll/factorial_nll_results.jsonl"
    cell = ENDPOINT_CELLS[variant]
    rows = [
        row for row in _read_jsonl(path)
        if row.get("variant") == cell
    ]
    return rows, path


def _run_nll_stage(args: argparse.Namespace) -> None:
    configs = objective_configs(seed=args.seed)
    samples = _nll_samples(args)
    expected_ids = {sample.id for sample in samples}
    calibration = load_manifest(args.calibration_manifest)[:128]
    rows = []
    statuses = []
    model = processor = selected = None
    try:
        for variant, config in configs.items():
            if variant in ENDPOINT_CELLS:
                source_rows, source_path = _endpoint_nll_rows(variant, args)
                if (
                    len(source_rows) != 128
                    or {str(row["id"]) for row in source_rows} != expected_ids
                ):
                    raise RuntimeError(f"invalid endpoint NLL rows: {variant}")
                variant_rows = []
                for source in source_rows:
                    row = dict(source)
                    row.update({
                        "source_variant": row.get("variant"),
                        "variant": variant,
                        "config_digest": config.digest,
                        "result_origin": "verified_factorial_endpoint",
                        "source_result_path": str(source_path.resolve()),
                        "source_result_sha256": file_digest(source_path),
                    })
                    variant_rows.append(row)
                cache_hit = True
                search_key = None
            else:
                if model is None:
                    torch.set_num_threads(args.threads)
                    torch.manual_seed(args.seed)
                    model, processor, _ = load_reference(args.model_path)
                    selected = select_linears(
                        model,
                        QuantScope.DECODER_CANONICAL,
                        strict_qwen_3b=not args.allow_nonstandard_model,
                    )
                search, cache_hit, search_key = _load_search(
                    variant,
                    model,
                    selected,
                    processor,
                    calibration,
                    config,
                    args,
                )
                applied = apply_mbq(model, selected, config, search)
                try:
                    evaluated = _evaluate_samples_cached(
                        model,
                        processor,
                        samples,
                        variant=variant,
                        config_digest=config.digest,
                        result_dir=args.output_dir / "w4_nll/_sample_cache",
                        analysis_ids=set(),
                        promotion_ids=set(),
                        min_pixels=config.min_pixels,
                        max_pixels=config.max_pixels,
                        resume=args.resume,
                    )
                finally:
                    applied.restore()
                variant_rows = [
                    dict(row, result_origin="new_or_resumed") for row in evaluated
                ]
            if len(variant_rows) != 128:
                raise RuntimeError(f"{variant} produced {len(variant_rows)} NLL rows")
            rows.extend(variant_rows)
            statuses.append({
                "variant": variant,
                "status": "complete",
                "rows": len(variant_rows),
                "config": config.to_dict(),
                "config_digest": config.digest,
                "search_key": search_key,
                "search_cache_hit": cache_hit,
                "result_origin": variant_rows[0]["result_origin"],
            })
    finally:
        if model is not None:
            del model, processor
    if len(rows) != 1024:
        raise RuntimeError(f"objective sweep requires 1024 NLL rows, found {len(rows)}")
    _atomic_jsonl(args.output_dir / "w4_nll/objective_nll_results.jsonl", rows)
    atomic_json(args.output_dir / "w4_nll/run_audit.json", {
        "status": "complete",
        "rows": len(rows),
        "variants": statuses,
    })


def paired_effect(
    baseline: dict[tuple[str, str], float],
    candidate_rows: list[dict],
    *,
    value_key: str,
    variant: str,
    bootstrap_samples: int,
    seed: int,
    expected_per_dataset: int,
    lower_is_better: bool = True,
) -> dict:
    paired = []
    for row in candidate_rows:
        key = (str(row["dataset"]), str(row["id"]))
        if key not in baseline:
            continue
        candidate = float(row[value_key])
        improvement = baseline[key] - candidate if lower_is_better else candidate - baseline[key]
        paired.append((key[0], key[1], improvement))
    expected = 4 * expected_per_dataset
    if len(paired) != expected:
        raise RuntimeError(f"{variant} requires {expected} paired rows, found {len(paired)}")
    buckets = {
        dataset: [value for name, _identifier, value in paired if name == dataset]
        for dataset in sorted({name for name, _identifier, _value in paired})
    }
    if len(buckets) != 4 or set(map(len, buckets.values())) != {expected_per_dataset}:
        raise RuntimeError(f"{variant} is not balanced across four datasets")
    mean = statistics.fmean(statistics.fmean(values) for values in buckets.values())
    rng = random.Random(f"ava-objective:{seed}:{variant}:{value_key}")
    boot = [
        statistics.fmean(
            statistics.fmean(rng.choices(values, k=len(values)))
            for values in buckets.values()
        )
        for _ in range(bootstrap_samples)
    ]
    low = float(np.quantile(boot, 0.025))
    high = float(np.quantile(boot, 0.975))
    p = min(
        1.0,
        2 * min(
            sum(value <= 0 for value in boot) / len(boot),
            sum(value >= 0 for value in boot) / len(boot),
        ),
    )
    return {
        "variant": variant,
        "paired_samples": len(paired),
        "dataset_balanced_nll_improvement": mean,
        "ci_low": low,
        "ci_high": high,
        "bootstrap_p_two_sided": p,
        "exploratory": low <= 0,
        "per_dataset_improvement": {
            dataset: statistics.fmean(values)
            for dataset, values in buckets.items()
        },
    }


def _search_summary(args: argparse.Namespace) -> list[dict]:
    control = json.loads(
        (args.output_dir / "search_audits" / f"{CONTROL}.json").read_text()
    )
    control_alpha = {
        name: value["selected_alpha"] for name, value in control["groups"].items()
    }
    rows = []
    for variant, (normalization, prompt_scale, _objective) in VARIANT_SPECS.items():
        audit = json.loads(
            (args.output_dir / "search_audits" / f"{variant}.json").read_text()
        )
        groups = audit["groups"]
        prompt_mass = [
            value["modality_mass_share"].get("prompt_language", 0.0)
            for value in groups.values()
        ]
        changed = sum(
            value["selected_alpha"] != control_alpha[name]
            for name, value in groups.items()
        )
        rows.append({
            "variant": variant,
            "normalization": normalization,
            "prompt_objective_scale": prompt_scale,
            "groups": len(groups),
            "alpha_changed_groups_vs_control": changed,
            "alpha_changed_fraction_vs_control": changed / len(groups),
            "median_prompt_mass_share": statistics.median(prompt_mass),
            "mean_prompt_mass_share": statistics.fmean(prompt_mass),
            "selected_alpha_counts": json.dumps(
                dict(sorted(Counter(
                    value["selected_alpha"] for value in groups.values()
                ).items())),
                sort_keys=True,
            ),
            "activation_digest_invariant": True,
        })
    return rows


def _run_screen_analysis(args: argparse.Namespace) -> None:
    rows = _read_jsonl(args.output_dir / "w4_nll/objective_nll_results.jsonl")
    baseline = {
        (str(row["dataset"]), str(row["id"])): float(row["target_nll"])
        for row in rows if row["variant"] == CONTROL
    }
    p100 = {
        (str(row["dataset"]), str(row["id"])): float(row["target_nll"])
        for row in rows if row["variant"] == P100
    }
    if len(baseline) != 128 or len(p100) != 128:
        raise RuntimeError("control and p100 NLL endpoints must each contain 128 rows")
    effects = []
    p100_effects = {}
    for variant in VARIANT_SPECS:
        if variant == CONTROL:
            continue
        candidate = [row for row in rows if row["variant"] == variant]
        effect = paired_effect(
            baseline,
            candidate,
            value_key="target_nll",
            variant=variant,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
            expected_per_dataset=32,
        )
        effect["normalization"] = VARIANT_SPECS[variant][0]
        effect["prompt_objective_scale"] = VARIANT_SPECS[variant][1]
        effects.append(effect)
        if variant != P100:
            p100_effects[variant] = paired_effect(
                p100,
                candidate,
                value_key="target_nll",
                variant=f"{variant}_vs_p100",
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
                expected_per_dataset=32,
            )
    normalization_effects = {}
    for scale_tag in ("p025", "p050", "p100"):
        tw_variant = f"ava_ovpa_tw_{scale_tag}"
        mm_variant = f"ava_ovpa_mm_{scale_tag}"
        tw_baseline = {
            (str(row["dataset"]), str(row["id"])): float(row["target_nll"])
            for row in rows if row["variant"] == tw_variant
        }
        normalization_effects[scale_tag] = paired_effect(
            tw_baseline,
            [row for row in rows if row["variant"] == mm_variant],
            value_key="target_nll",
            variant=f"{mm_variant}_vs_{tw_variant}",
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
            expected_per_dataset=32,
        )
    _benjamini_hochberg(effects)
    effects.sort(key=lambda row: (
        -row["dataset_balanced_nll_improvement"],
        row["prompt_objective_scale"],
        row["variant"],
    ))
    for index, row in enumerate(effects, 1):
        row["rank"] = index
    positive = [
        row for row in effects
        if row["dataset_balanced_nll_improvement"] > 0
    ]
    promoted = [row["variant"] for row in positive[:2]]
    residual_variants = [CONTROL] + [row["variant"] for row in effects[:3]]
    selection = {
        "format_version": 1,
        "baseline": CONTROL,
        "selection_rule": (
            "dataset-balanced paired NLL improvement; ties prefer smaller "
            "prompt scale then variant id"
        ),
        "ranked": effects,
        "selected_for_promotion": promoted,
        "promotion_gate": bool(promoted),
        "promotion_labels": {
            row["variant"]: (
                "confirmatory" if row["ci_low"] > 0 else "exploratory"
            )
            for row in positive[:2]
        },
        "selected_for_residuals": residual_variants,
        "paired_vs_p100": p100_effects,
        "paired_modality_mean_vs_token_weighted": normalization_effects,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
    }
    _atomic_csv(args.output_dir / "analysis/objective_effects.csv", effects)
    _atomic_csv(args.output_dir / "analysis/search_summary.csv", _search_summary(args))
    atomic_json(args.output_dir / "analysis/selection.json", selection)
    best = effects[0]
    report = [
        "# 固定 A_VA 的 MBQ Objective 中期分析",
        "",
        f"NLL最优配置：`{best['variant']}`。",
        f"相对control的dataset-balanced改善为 "
        f"{best['dataset_balanced_nll_improvement']:.6f}，95% CI "
        f"[{best['ci_low']:.6f}, {best['ci_high']:.6f}]。",
        "",
        f"生成晋级：{', '.join(promoted) if promoted else '无'}。",
        f"残差配置：{', '.join(residual_variants)}。",
    ]
    (args.output_dir / "analysis").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis/report_zh.md").write_text(
        "\n".join(report) + "\n"
    )


def _residual_samples(args: argparse.Namespace) -> list:
    plan = json.loads(
        (args.formal_root / "residuals/residual_plan.json").read_text()
    )
    ids = list(plan["sample_ids"])
    lookup = {
        sample.id: sample for sample in load_manifest(args.evaluation_manifest)
    }
    if len(ids) != 32 or len(set(ids)) != 32 or not set(ids) <= set(lookup):
        raise RuntimeError("existing residual plan is invalid")
    samples = [lookup[identifier] for identifier in ids]
    counts = Counter(sample.dataset for sample in samples)
    if len(counts) != 4 or set(counts.values()) != {8}:
        raise RuntimeError(f"residual set is not fixed 8x4: {dict(counts)}")
    return samples


def _load_residual_cache(
    root: Path, variants: list[str],
) -> tuple[list[dict], list[dict]]:
    residuals = []
    predictions = []
    for variant in variants:
        directory = root / variant
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            if path.name.endswith(".error.json"):
                continue
            payload = json.loads(path.read_text())
            if payload.get("status") != "complete":
                raise RuntimeError(f"incomplete residual cache: {path}")
            residuals.extend(payload["residual_rows"])
            predictions.append(payload["prediction"])
    return residuals, predictions


def _run_residual_stage(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    configs = objective_configs(seed=args.seed)
    selection = json.loads(
        (args.output_dir / "analysis/selection.json").read_text()
    )
    variants = selection["selected_for_residuals"]
    calibration = load_manifest(args.calibration_manifest)[:128]
    samples = _residual_samples(args)
    bf16_model, processor, _ = load_reference(args.model_path)
    bf16_selected = select_linears(
        bf16_model,
        QuantScope.DECODER_CANONICAL,
        strict_qwen_3b=not args.allow_nonstandard_model,
    )
    anchor_names = [
        item.name
        for item in sensitivity_anchors(bf16_selected)
        if any(f".layers.{layer}." in item.name for layer in LAYERS)
    ]
    if len(anchor_names) != len(LAYERS) * 2:
        raise RuntimeError("residual anchor inventory does not match seven layers")
    cache_root = args.output_dir / "residuals/_sample_cache"
    statuses = []
    try:
        for variant in variants:
            config = configs[variant]
            candidate_model, candidate_processor, _ = load_reference(args.model_path)
            del candidate_processor
            selected = select_linears(
                candidate_model,
                QuantScope.DECODER_CANONICAL,
                strict_qwen_3b=not args.allow_nonstandard_model,
            )
            search, cache_hit, search_key = _load_search(
                variant,
                candidate_model,
                selected,
                processor,
                calibration,
                config,
                args,
            )
            applied = apply_mbq(candidate_model, selected, config, search)
            variant_cache = cache_root / variant
            variant_cache.mkdir(parents=True, exist_ok=True)
            try:
                for sample in samples:
                    path = variant_cache / _sample_cache_name(sample.id)
                    if args.resume and path.is_file():
                        continue
                    try:
                        batch = next(mbq_calibration_batches(
                            processor,
                            [sample],
                            min_pixels=config.min_pixels,
                            max_pixels=config.max_pixels,
                        ))
                        reference = capture_reconstruction_outputs(
                            bf16_model,
                            batch,
                            layers_to_test=LAYERS,
                            anchor_names=anchor_names,
                            capture_logits=True,
                        )
                        candidate = capture_reconstruction_outputs(
                            candidate_model,
                            batch,
                            layers_to_test=LAYERS,
                            anchor_names=anchor_names,
                            capture_logits=True,
                        )
                        reference_logits = reference.pop("logits")
                        candidate_logits = candidate.pop("logits")
                        sample_rows = reconstruction_rows(
                            batch, reference, candidate, method=variant,
                        )
                        if variant == CONTROL:
                            sample_rows.extend(reconstruction_rows(
                                batch, reference, reference, method="bf16",
                            ))
                        for row in sample_rows:
                            row.update({
                                "objective_variant": (
                                    None if row["method"] == "bf16" else variant
                                ),
                                "activation_mode": (
                                    None if row["method"] == "bf16" else VA
                                ),
                                "objective_normalization": (
                                    None if row["method"] == "bf16"
                                    else VARIANT_SPECS[variant][0]
                                ),
                                "prompt_objective_scale": (
                                    None if row["method"] == "bf16"
                                    else VARIANT_SPECS[variant][1]
                                ),
                            })
                        prediction = {
                            "id": sample.id,
                            "dataset": sample.dataset,
                            "objective_variant": variant,
                            **prediction_change_summary(
                                reference_logits,
                                candidate_logits,
                                batch["language_mask"],
                            ),
                        }
                        atomic_json(path, {
                            "status": "complete",
                            "id": sample.id,
                            "objective_variant": variant,
                            "config_digest": config.digest,
                            "residual_rows": sample_rows,
                            "prediction": prediction,
                        })
                    except Exception as error:
                        atomic_json(path.with_suffix(".error.json"), {
                            "status": "failed",
                            "id": sample.id,
                            "objective_variant": variant,
                            "error_type": type(error).__name__,
                            "error": str(error),
                        })
                        raise
            finally:
                applied.restore()
                del candidate_model
            statuses.append({
                "variant": variant,
                "status": "complete",
                "samples": len(samples),
                "search_cache_hit": cache_hit,
                "search_key": search_key,
            })
    finally:
        del bf16_model, processor
    residual_rows, predictions = _load_residual_cache(cache_root, variants)
    expected_rows = (len(variants) + 1) * 32 * len(LAYERS) * 3 * 4
    if (
        len(residual_rows) != expected_rows
        or len(predictions) != len(variants) * 32
    ):
        raise RuntimeError(
            f"residual matrix incomplete: {len(residual_rows)} rows, "
            f"{len(predictions)} predictions"
        )
    _atomic_jsonl(
        args.output_dir / "residuals/objective_residuals.jsonl",
        residual_rows,
    )
    _atomic_jsonl(
        args.output_dir / "residuals/prediction_changes.jsonl",
        predictions,
    )
    _analyze_residuals(args, residual_rows, variants)
    atomic_json(args.output_dir / "residuals/run_audit.json", {
        "status": "complete",
        "variants": variants,
        "samples": len(samples),
        "layers": list(LAYERS),
        "anchors": len(anchor_names),
        "rows": len(residual_rows),
        "prediction_rows": len(predictions),
        "variant_statuses": statuses,
    })


def _analyze_residuals(
    args: argparse.Namespace,
    rows: list[dict],
    variants: list[str],
) -> None:
    index = {
        (
            row["method"], str(row["dataset"]), str(row["id"]),
            int(row["layer"]), row["projection"], row["modality"],
        ): row
        for row in rows if row["method"] != "bf16"
    }
    deltas = []
    for variant in variants:
        if variant == CONTROL:
            continue
        for key, candidate in index.items():
            if key[0] != variant:
                continue
            control_key = (CONTROL, *key[1:])
            control = index.get(control_key)
            if control is None:
                raise RuntimeError(f"missing control residual for {key[1:]}")
            for metric in ("mae", "rmse"):
                deltas.append({
                    "variant": variant,
                    "dataset": key[1],
                    "id": key[2],
                    "layer": key[3],
                    "projection": key[4],
                    "modality": key[5],
                    "metric": metric,
                    "value": float(candidate[metric]) - float(control[metric]),
                })
    grouped = defaultdict(list)
    for row in deltas:
        grouped[(
            row["variant"], row["metric"], row["layer"],
            row["projection"], row["modality"],
        )].append(row)
    summaries = []
    for key, values in grouped.items():
        buckets = {
            dataset: [float(row["value"]) for row in values if row["dataset"] == dataset]
            for dataset in sorted({row["dataset"] for row in values})
        }
        mean = statistics.fmean(
            statistics.fmean(bucket) for bucket in buckets.values()
        )
        rng = random.Random(f"ava-residual:{args.seed}:{key}")
        boot = [
            statistics.fmean(
                statistics.fmean(rng.choices(bucket, k=len(bucket)))
                for bucket in buckets.values()
            )
            for _ in range(args.bootstrap_samples)
        ]
        summaries.append({
            "variant": key[0],
            "metric": key[1],
            "layer": key[2],
            "projection": key[3],
            "modality": key[4],
            "dataset_balanced_mean_delta_vs_control": mean,
            "ci_low": float(np.quantile(boot, 0.025)),
            "ci_high": float(np.quantile(boot, 0.975)),
            "samples": len(values),
        })
    _atomic_csv(args.output_dir / "analysis/objective_residual_effects.csv", summaries)
    _atomic_jsonl(args.output_dir / "analysis/sample_residual_deltas.jsonl", deltas)

    sensitivity = [
        row for row in _read_jsonl(
            args.formal_root / "collection/sample_sensitivity.jsonl"
        )
        if row.get("sample_kind") == "target"
    ]
    sensitivity_index = {}
    for row in sensitivity:
        projection = (
            "o_proj" if str(row["module"]).endswith(".o_proj") else "down_proj"
        )
        for modality, gradient_name in (
            ("vision", "vision"),
            ("prompt_language", "prompt_language"),
            ("answer_language", "answer_predictor_language"),
        ):
            sensitivity_index[(
                str(row["id"]), int(row["layer"]), projection, modality,
            )] = float(row["gradients"][gradient_name]["mean"])
    nll_rows = _read_jsonl(args.output_dir / "w4_nll/objective_nll_results.jsonl")
    nll_index = {
        (row["variant"], str(row["id"])): float(row["target_nll"])
        for row in nll_rows
    }
    products = []
    for row in rows:
        variant = row.get("objective_variant")
        if (
            variant is None
            or row["projection"] == "block_output"
            or row["modality"] == "all_tokens"
        ):
            continue
        gradient = sensitivity_index.get((
            str(row["id"]), int(row["layer"]),
            row["projection"], row["modality"],
        ))
        if gradient is None:
            continue
        control_nll = nll_index.get((CONTROL, str(row["id"])))
        candidate_nll = nll_index.get((variant, str(row["id"])))
        products.append({
            "variant": variant,
            "dataset": row["dataset"],
            "id": row["id"],
            "layer": row["layer"],
            "projection": row["projection"],
            "modality": row["modality"],
            "gradient_mean_abs": gradient,
            "reconstruction_rmse": row["rmse"],
            "sensitivity_x_reconstruction_error": gradient * float(row["rmse"]),
            "nll_improvement_vs_control": (
                None if control_nll is None or candidate_nll is None
                else control_nll - candidate_nll
            ),
        })
    _atomic_csv(
        args.output_dir / "analysis/sensitivity_residual_product.csv",
        products,
    )


def _prior_ids(args: argparse.Namespace) -> set[str]:
    plan = json.loads(
        (args.formal_root / "w4_nll/ablation_plan.json").read_text()
    )
    groups = [
        set(plan[name])
        for name in ("adaptation_ids", "nll_ids", "promotion_ids")
    ]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise RuntimeError("existing formal sample partitions overlap")
    return set().union(*groups)


def promotion_samples(args: argparse.Namespace) -> list:
    used = _prior_ids(args)
    evaluation = load_manifest(args.evaluation_manifest)
    samples = stable_hash_subset(
        evaluation,
        64,
        salt=f"ava_objective_promotion_v1:{args.seed}",
        exclude_ids=used,
    )
    counts = Counter(sample.dataset for sample in samples)
    if (
        len(samples) != 256
        or set(counts.values()) != {64}
        or {sample.id for sample in samples} & used
    ):
        raise RuntimeError(f"promotion subset is not disjoint 64x4: {dict(counts)}")
    return samples


def _promotion_cache_dir(
    variant: str, args: argparse.Namespace,
) -> Path:
    if variant in ENDPOINT_CELLS:
        return args.factorial_root / "_calibration_cache"
    return args.output_dir / "_calibration_cache"


def _run_promotion_stage(args: argparse.Namespace) -> None:
    selection = json.loads(
        (args.output_dir / "analysis/selection.json").read_text()
    )
    selected = selection["selected_for_promotion"]
    promotion_root = args.output_dir / "promotion"
    if not selected:
        atomic_json(promotion_root / "status.json", {
            "status": "skipped",
            "reason": "no objective variant has positive mean paired NLL improvement",
        })
        return
    samples = promotion_samples(args)
    manifest = promotion_root / "promotion_manifest.jsonl"
    write_manifest(manifest, samples)
    configs = objective_configs(seed=args.seed)
    variants = [CONTROL, *selected]
    atomic_json(promotion_root / "manifest_audit.json", {
        "status": "complete",
        "samples": len(samples),
        "ids": [sample.id for sample in samples],
        "manifest_sha256": file_digest(manifest),
        "excluded_prior_ids": len(_prior_ids(args)),
        "disjoint_from_all_prior_formal_sets": True,
        "salt": f"ava_objective_promotion_v1:{args.seed}",
    })
    common = [
        "--vlmevalkit-root", str(args.vlmevalkit_root),
        "--lmu-data-root", str(args.lmu_data_root),
        "--base-model-path", str(args.model_path),
        "--calibration-manifest", str(args.calibration_manifest),
        "--screen-manifest", str(manifest),
        "--expected-samples-per-dataset", "64",
        "--preset", "screen",
        "--threads", str(args.threads),
        "--seed", str(args.seed),
        "--resume",
    ]
    if args.allow_nonstandard_model:
        common.append("--allow-nonstandard-model")
    for variant in variants:
        config_path = promotion_root / f"{variant}_config.json"
        configs[variant].dump(config_path)
        _run([
            sys.executable,
            "-m", "experiments.vlm_quant_benchmark.run_vlmeval_official",
            *common,
            "--calibration-cache-dir", str(_promotion_cache_dir(variant, args)),
            "--experiment-config-json", str(config_path),
            "--output-dir", str(promotion_root / "runs" / variant),
        ], args.output_dir / "logs" / f"promotion_{variant}.log")
    atomic_json(promotion_root / "status.json", {
        "status": "complete",
        "variants": variants,
        "generated_predictions": 256 * len(variants),
        "candidate_predictions": 256 * len(selected),
    })


def _reference_scores(path: Path, ids: set[str], label: str) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("status") != "complete":
        raise RuntimeError(f"{label} reference is not complete: {path}")
    scores = {
        (str(row["dataset"]), str(row["id"])): float(row["official_score"])
        for row in payload["samples"]
        if str(row["id"]) in ids
    }
    if len(scores) != 256:
        raise RuntimeError(f"{label} reference does not cover promotion IDs")
    return scores


def _promotion_result(
    variant: str, args: argparse.Namespace,
) -> tuple[dict, dict]:
    path = args.output_dir / "promotion/runs" / variant / "official_result.json"
    payload = json.loads(path.read_text())
    if payload.get("status") != "complete":
        raise RuntimeError(f"incomplete promotion result: {variant}")
    scores = {
        (str(row["dataset"]), str(row["id"])): float(row["official_score"])
        for row in payload["samples"]
    }
    if len(scores) != 256:
        raise RuntimeError(f"{variant} promotion does not contain 256 samples")
    return payload, scores


def _run_final_stage(args: argparse.Namespace) -> None:
    selection = json.loads(
        (args.output_dir / "analysis/selection.json").read_text()
    )
    status = json.loads(
        (args.output_dir / "promotion/status.json").read_text()
    )
    comparisons = []
    score_rows = []
    if status["status"] == "complete":
        payloads = {}
        sample_scores = {}
        for variant in status["variants"]:
            payloads[variant], sample_scores[variant] = _promotion_result(
                variant, args,
            )
            for dataset, value in sorted(payloads[variant]["datasets"].items()):
                score_rows.append({
                    "method": variant,
                    "dataset": dataset,
                    "score": float(value["score"]),
                })
            score_rows.append({
                "method": variant,
                "dataset": "balanced_mean",
                "score": statistics.fmean(
                    float(value["score"])
                    for value in payloads[variant]["datasets"].values()
                ),
            })
        ids = {identifier for _dataset, identifier in sample_scores[CONTROL]}
        external = {
            "squeezellm_current": _reference_scores(
                args.squeezellm_result, ids, "SqueezeLLM",
            ),
            "mbq_current": _reference_scores(
                args.mbq_current_result, ids, "MBQ current",
            ),
        }
        for label, scores in external.items():
            by_dataset = defaultdict(list)
            for (dataset, _identifier), score in scores.items():
                by_dataset[dataset].append(score)
            for dataset, values in sorted(by_dataset.items()):
                score_rows.append({
                    "method": label,
                    "dataset": dataset,
                    "score": 100 * statistics.fmean(values),
                })
            score_rows.append({
                "method": label,
                "dataset": "balanced_mean",
                "score": 100 * statistics.fmean(
                    statistics.fmean(values) for values in by_dataset.values()
                ),
            })
        control = sample_scores[CONTROL]
        for variant in selection["selected_for_promotion"]:
            rows = [
                {"dataset": dataset, "id": identifier, "official_score": value}
                for (dataset, identifier), value in sample_scores[variant].items()
            ]
            effect = paired_effect(
                control,
                rows,
                value_key="official_score",
                variant=f"{variant}_generation",
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
                expected_per_dataset=64,
                lower_is_better=False,
            )
            comparisons.append(effect)
        _atomic_csv(args.output_dir / "analysis/promotion_scores.csv", score_rows)
        _atomic_csv(
            args.output_dir / "analysis/promotion_comparisons.csv",
            comparisons,
        )
    effects = json.loads(
        (args.output_dir / "analysis/selection.json").read_text()
    )["ranked"]
    p100_pairs = selection["paired_vs_p100"]
    intermediate = [
        row for row in effects
        if 0 < row["prompt_objective_scale"] < 1
    ]
    over_weight = any(
        row["ci_low"] > 0
        and p100_pairs[row["variant"]]["ci_low"] > 0
        for row in intermediate
    )
    normalization_pairs = selection["paired_modality_mean_vs_token_weighted"]
    token_imbalance = any(
        row["ci_low"] > 0 for row in normalization_pairs.values()
    )
    effect_map = {row["variant"]: row for row in effects}
    residual_effects = []
    residual_path = args.output_dir / "analysis/objective_residual_effects.csv"
    if residual_path.is_file():
        with residual_path.open() as handle:
            residual_effects = list(csv.DictReader(handle))
    prompt_better_variants = {
        row["variant"]
        for row in residual_effects
        if row["modality"] == "prompt_language"
        and row["metric"] == "rmse"
        and float(row["ci_high"]) < 0
    }
    best = effects[0]
    linear_mae_misaligned = any(
        effect_map[variant]["ci_low"] <= 0
        for variant in prompt_better_variants
        if variant in effect_map
    )
    no_improvement = best["dataset_balanced_nll_improvement"] <= 0
    promoted_map = {
        row["variant"].removesuffix("_generation"): row
        for row in comparisons
    }
    recommended = [
        variant for variant, row in promoted_map.items()
        if row["ci_low"] > 0
    ]
    decision = {
        "prompt_objective_overweighted": over_weight,
        "token_count_objective_imbalance": token_imbalance,
        "linear_mae_task_misalignment": linear_mae_misaligned,
        "fixed_ava_objective_scaling_has_no_gain": no_improvement,
        "recommended_replacement": recommended,
        "recommendation_rule": (
            "independent generation paired improvement 95% CI lower bound > 0"
        ),
        "promotion_status": status["status"],
    }
    atomic_json(args.output_dir / "analysis/attribution_decision.json", decision)
    nll_rows = _read_jsonl(
        args.output_dir / "w4_nll/objective_nll_results.jsonl"
    )
    residual_rows = _read_jsonl(
        args.output_dir / "residuals/objective_residuals.jsonl"
    )
    if len(nll_rows) != 1024:
        raise RuntimeError("final validation requires 1024 NLL rows")
    atomic_json(args.output_dir / "formal_run_audit.json", {
        "status": "complete",
        "nll_rows": len(nll_rows),
        "residual_rows": len(residual_rows),
        "promotion_status": status["status"],
        "promotion_candidate_rows": (
            int(status.get("candidate_predictions", 0))
        ),
        "selection": selection,
        "decision": decision,
        "input_digests": {
            "calibration_manifest": file_digest(args.calibration_manifest),
            "evaluation_manifest": file_digest(args.evaluation_manifest),
            "factorial_endpoint_results": file_digest(
                args.factorial_root / "w4_nll/factorial_nll_results.jsonl"
            ),
        },
    })
    summary = [
        "# 固定 A_VA 的 MBQ Objective 正式总结",
        "",
        f"NLL最优配置：`{best['variant']}`，相对control改善 "
        f"{best['dataset_balanced_nll_improvement']:.6f}，95% CI "
        f"[{best['ci_low']:.6f}, {best['ci_high']:.6f}]。",
        "",
        f"Promotion状态：`{status['status']}`。",
        f"推荐替换项：{', '.join(recommended) if recommended else '无'}。",
        "",
        "完整归因见 analysis/attribution_decision.json。",
    ]
    (args.output_dir / "final_summary_zh.md").write_text(
        "\n".join(summary) + "\n"
    )


def _real_smoke(args: argparse.Namespace) -> None:
    smoke_args = argparse.Namespace(**vars(args))
    smoke_args.output_dir = args.output_dir / "_preflight"
    smoke_args.raw_cache_dir = smoke_args.output_dir / "_calibration_cache"
    config = dataclasses.replace(
        objective_configs(calibration_samples=1, seed=args.seed)[
            "ava_ovpa_tw_p025"
        ],
        mbq_scale_grid=(0.0, 0.5),
        scheme_id="ava_objective_real_smoke",
    )
    model, processor, _ = load_reference(args.model_path)
    selected_all = select_linears(
        model,
        QuantScope.DECODER_CANONICAL,
        strict_qwen_3b=not args.allow_nonstandard_model,
    )
    prefix = selected_all[0].name.split(".layers.", 1)[0] + ".layers.0."
    selected = [item for item in selected_all if prefix in item.name]
    calibration = load_manifest(args.calibration_manifest)[:1]
    sample = _nll_samples(args)[:1]
    try:
        search, hit, key, raw_key = _new_search(
            model,
            selected,
            processor,
            calibration,
            config,
            smoke_args,
            require_shared_raw=False,
        )
        if any(
            group["activation_modalities"] != ["vision", "answer_language"]
            for group in search.audit["groups"].values()
        ):
            raise RuntimeError("real smoke did not preserve A_VA")
        applied = apply_mbq(model, selected, config, search)
        try:
            rows = _evaluate_samples_cached(
                model,
                processor,
                sample,
                variant="ava_objective_real_smoke",
                config_digest=config.digest,
                result_dir=smoke_args.output_dir / "sample_cache",
                analysis_ids=set(),
                promotion_ids=set(),
                min_pixels=config.min_pixels,
                max_pixels=config.max_pixels,
                resume=True,
            )
        finally:
            applied.restore()
    finally:
        del model, processor
    atomic_json(smoke_args.output_dir / "real_smoke.json", {
        "status": "complete",
        "groups": len(search.audit["groups"]),
        "scales": len(search.scales),
        "cache_hit": hit,
        "search_key": key,
        "raw_key": raw_key,
        "nll_rows": len(rows),
        "finite_nll": all(np.isfinite(row["target_nll"]) for row in rows),
        "activation_mode": VA,
        "prompt_objective_scale": 0.25,
    })


def execute_stage(args: argparse.Namespace, stage: str) -> None:
    root = args.output_dir
    if stage == "preflight":
        available = psutil.virtual_memory().available
        disk = shutil.disk_usage(
            root.parent if root.parent.exists() else Path(".")
        ).free
        active = _active_quant_processes()
        if available < 80 * 1024**3:
            raise RuntimeError("A_VA objective preflight requires 80 GiB memory")
        if disk < 100 * 1024**3:
            raise RuntimeError("A_VA objective preflight requires 100 GiB disk")
        if active:
            raise RuntimeError(f"other quantization processes are active: {active}")
        for path in (
            Path(args.model_path),
            args.calibration_manifest,
            args.evaluation_manifest,
            args.formal_root,
            args.trimodal_root,
            args.factorial_root,
            _raw_cache_dir(args),
            args.squeezellm_result,
            args.mbq_current_result,
        ):
            if not path.exists():
                raise FileNotFoundError(path)
        _nll_samples(args)
        _residual_samples(args)
        promotion_samples(args)
        test_environment = dict(os.environ)
        existing = test_environment.get("PYTHONPATH", "")
        test_environment["PYTHONPATH"] = os.pathsep.join(
            value
            for value in (str(args.vlmevalkit_root.resolve()), existing)
            if value
        )
        _run([
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "experiments/vlm_quant_benchmark/tests",
        ], root / "logs/preflight_tests.log", environment=test_environment)
        if not args.skip_real_smoke:
            _real_smoke(args)
        atomic_json(root / "preflight_audit.json", {
            "status": "complete",
            "environment": environment_record(),
            "available_memory_bytes": available,
            "disk_free_bytes": disk,
            "active_quant_processes": active,
            "threads": args.threads,
            "seed": args.seed,
            "variants": list(VARIANT_SPECS),
            "shared_raw_cache": str(_raw_cache_dir(args).resolve()),
            "calibration_manifest_sha256": file_digest(args.calibration_manifest),
            "evaluation_manifest_sha256": file_digest(args.evaluation_manifest),
            "real_smoke": not args.skip_real_smoke,
        })
    elif stage == "search":
        _run_search_stage(args)
    elif stage == "w4_nll":
        _run_nll_stage(args)
    elif stage == "screen_analysis":
        _run_screen_analysis(args)
    elif stage == "residuals":
        _run_residual_stage(args)
    elif stage == "promotion":
        _run_promotion_stage(args)
    elif stage == "final":
        _run_final_stage(args)
    else:
        raise ValueError(stage)


def main() -> None:
    args = parse_args()
    if args.threads != 16:
        raise ValueError("formal A_VA objective protocol requires exactly 16 threads")
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "stage_state.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.is_file()
        else {
            "format_version": 1,
            "stages": {
                stage: {"status": "pending"} for stage in STAGES
            },
        }
    )
    selected = STAGES if args.stage == "all" else (args.stage,)
    with pid_lock(args.output_dir / "formal_run.pid"):
        for stage in selected:
            if (
                args.resume
                and state["stages"].get(stage, {}).get("status") == "complete"
            ):
                continue
            started = time.time()
            state["stages"][stage] = {
                "status": "running",
                "started_unix": started,
            }
            atomic_json(state_path, state)
            try:
                execute_stage(args, stage)
            except Exception as error:
                state["stages"][stage] = {
                    "status": "failed",
                    "started_unix": started,
                    "finished_unix": time.time(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                atomic_json(state_path, state)
                raise
            state["stages"][stage] = {
                "status": "complete",
                "started_unix": started,
                "finished_unix": time.time(),
            }
            atomic_json(state_path, state)


if __name__ == "__main__":
    main()

