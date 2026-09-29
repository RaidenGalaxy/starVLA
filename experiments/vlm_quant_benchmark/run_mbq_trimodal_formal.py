"""Locked, resumable formal W4 validation for three-token-class MBQ."""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import psutil
import torch

from .config import ExperimentConfig, ImplementationTier, MethodName, QuantScope
from .data import file_digest, load_manifest, sample_manifest_digest, write_manifest
from .mbq import (
    apply_mbq,
    collect_mbq_search_cached,
    mbq_cache_key,
    mbq_calibration_batches,
    mbq_statistics_cache_key,
)
from .provenance import source_digest
from .reporting import atomic_json
from .run_sensitivity_ablation import _evaluate_samples_cached
from .run_sensitivity_formal import _active_quant_processes, _run, pid_lock
from .runner import environment_record, load_reference
from .selection import select_linears
from .sensitivity import stable_hash_subset


STAGES = ("preflight", "collect", "w4_nll", "analysis", "promotion", "final")
VARIANTS = ("mbq_trimodal_no_floor", "mbq_trimodal_floor")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run formal tri-modal MBQ W4 validation")
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/vlm_quant_benchmark/sensitivity_analysis_v1/trimodal_v1"),
    )
    parser.add_argument(
        "--formal-root", type=Path,
        default=Path("outputs/vlm_quant_benchmark/sensitivity_analysis_v1/formal_128"),
    )
    parser.add_argument(
        "--model-path", default="playground/Pretrained_models/Qwen2.5-VL-3B-Instruct",
    )
    parser.add_argument(
        "--calibration-manifest", type=Path,
        default=Path(
            "data/vlm_quant_benchmark/multimodal_calibration/"
            "multimodal_calibration_4x32.jsonl"
        ),
    )
    parser.add_argument(
        "--evaluation-manifest", type=Path,
        default=Path("data/vlm_quant_benchmark/top3_protocol_v4/vlm_confirm_4x256.jsonl"),
    )
    parser.add_argument("--vlmevalkit-root", type=Path, default=Path("third_party/VLMEvalKit"))
    parser.add_argument("--lmu-data-root", type=Path, default=Path("data/vlm_quant_benchmark/LMUData"))
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--skip-real-smoke", action="store_true")
    parser.add_argument("--allow-nonstandard-model", action="store_true")
    return parser.parse_args()


def trimodal_configs(*, calibration_samples: int = 128, seed: int = 42) -> dict[str, ExperimentConfig]:
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
        mbq_modality_mode="vision_prompt_answer",
        mbq_algorithm_version=5,
        mbq_gradient_aggregation="pooled_elements",
        mbq_objective_normalization="token_weighted",
        mbq_reconstruction_level="linear_group",
    )
    return {
        "mbq_trimodal_no_floor": ExperimentConfig(
            **common,
            mbq_median_floor=False,
            scheme_id="mbq_trimodal_no_floor_w4a16",
        ),
        "mbq_trimodal_floor": ExperimentConfig(
            **common,
            mbq_median_floor=True,
            scheme_id="mbq_trimodal_floor_w4a16",
        ),
    }


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = sorted({key for row in rows for key in row}) if rows else []
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(path)


def _prior_ids(args: argparse.Namespace) -> tuple[set[str], list]:
    evaluation = load_manifest(args.evaluation_manifest)
    plan_path = args.formal_root / "w4_nll/ablation_plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError(plan_path)
    plan = json.loads(plan_path.read_text())
    adaptation = set(plan["adaptation_ids"])
    nll = set(plan["nll_ids"])
    promotion = set(plan["promotion_ids"])
    if adaptation & nll or adaptation & promotion or nll & promotion:
        raise RuntimeError("existing formal sample partitions overlap")
    known = {sample.id for sample in evaluation}
    if not (adaptation | nll | promotion) <= known:
        raise RuntimeError("existing formal sample IDs do not match evaluation manifest")
    return adaptation | nll | promotion, evaluation


def _nll_samples(args: argparse.Namespace) -> list:
    plan = json.loads((args.formal_root / "w4_nll/ablation_plan.json").read_text())
    ids = set(plan["nll_ids"])
    samples = [sample for sample in load_manifest(args.evaluation_manifest) if sample.id in ids]
    if len(samples) != 128:
        raise RuntimeError(f"expected 128 fixed NLL samples, found {len(samples)}")
    return samples


def _new_promotion_samples(args: argparse.Namespace) -> list:
    used, evaluation = _prior_ids(args)
    samples = stable_hash_subset(
        evaluation,
        64,
        salt=f"trimodal_promotion_v1:{args.seed}",
        exclude_ids=used,
    )
    if len(samples) != 256 or {sample.id for sample in samples} & used:
        raise RuntimeError("tri-modal promotion partition is not a disjoint 64x4 set")
    counts = defaultdict(int)
    for sample in samples:
        counts[sample.dataset] += 1
    if set(counts.values()) != {64}:
        raise RuntimeError(f"tri-modal promotion dataset imbalance: {dict(counts)}")
    return samples


def _search(
    model,
    selected,
    processor,
    calibration,
    config: ExperimentConfig,
    args: argparse.Namespace,
    *,
    cache_dir: Path,
):
    calibration_digest = sample_manifest_digest(calibration)
    model_digest = source_digest(args.model_path)
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
        cache_dir=cache_dir,
        key=key,
        raw_key=raw_key,
        reservoir_rows=32,
        attention_implementation="sdpa",
    )
    return result, cache_hit, key, raw_key


def _paired_effect(
    baseline: dict[tuple[str, str], float],
    candidate_rows: list[dict],
    *,
    bootstrap_samples: int,
    seed: int,
    variant: str,
    expected_per_dataset: int = 32,
) -> dict:
    paired = []
    for row in candidate_rows:
        key = (str(row["dataset"]), str(row["id"]))
        if key in baseline:
            paired.append((key[0], baseline[key] - float(row["target_nll"])))
    expected_total = 4 * expected_per_dataset
    if len(paired) != expected_total:
        raise RuntimeError(
            f"{variant} must have {expected_total} paired rows, found {len(paired)}"
        )
    buckets = {
        dataset: [value for name, value in paired if name == dataset]
        for dataset in sorted({name for name, _value in paired})
    }
    if len(buckets) != 4 or set(map(len, buckets.values())) != {expected_per_dataset}:
        raise RuntimeError(
            f"{variant} does not contain {expected_per_dataset} samples per dataset"
        )
    mean = statistics.fmean(statistics.fmean(values) for values in buckets.values())
    rng = random.Random(f"trimodal:{seed}:{variant}")
    boot = [
        statistics.fmean(
            statistics.fmean(rng.choices(values, k=len(values)))
            for values in buckets.values()
        )
        for _ in range(bootstrap_samples)
    ]
    return {
        "variant": variant,
        "paired_samples": len(paired),
        "dataset_balanced_nll_improvement": mean,
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "exploratory": float(np.quantile(boot, 0.025)) <= 0,
        "per_dataset_improvement": {
            dataset: statistics.fmean(values) for dataset, values in buckets.items()
        },
    }


def _flatten_search_audits(root: Path) -> tuple[list[dict], list[dict]]:
    weights, masses = [], []
    for variant in VARIANTS:
        path = root / "search_audits" / f"{variant}.json"
        if not path.is_file():
            continue
        audit = json.loads(path.read_text())
        for group, value in audit["groups"].items():
            for modality, effective in value["effective_modality_weights"].items():
                weights.append({
                    "variant": variant,
                    "group": group,
                    "sensitivity_kind": value["sensitivity_kind"],
                    "modality": modality,
                    "raw_weight": value["raw_modality_gradient_weights"][modality],
                    "effective_weight": effective,
                    "median_floor": value["median_floor_by_modality"].get(modality, 1.0),
                    "rows": value["modality_rows"][modality],
                })
                masses.append({
                    "variant": variant,
                    "group": group,
                    "modality": modality,
                    "mass_share": value["modality_mass_share"][modality],
                })
    return weights, masses


def _control_config(args: argparse.Namespace) -> ExperimentConfig:
    path = args.formal_root / "promotion/mbq_no_median_floor_config.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    config = ExperimentConfig.from_dict(json.loads(path.read_text()))
    if (
        config.method != MethodName.MBQ
        or config.mbq_modality_mode != "vision_answer"
        or config.mbq_median_floor
    ):
        raise RuntimeError("formal control config is not two-modal MBQ no-floor")
    return config


def _real_smoke(args: argparse.Namespace) -> None:
    root = args.output_dir / "_preflight"
    config = dataclasses.replace(
        trimodal_configs(calibration_samples=1, seed=args.seed)["mbq_trimodal_no_floor"],
        mbq_scale_grid=(0.0, 0.5),
        scheme_id="mbq_trimodal_real_smoke",
    )
    model, processor, _conversion = load_reference(args.model_path)
    selected_all = select_linears(
        model,
        QuantScope.DECODER_CANONICAL,
        strict_qwen_3b=not args.allow_nonstandard_model,
    )
    first_block = selected_all[0].name.split(".layers.", 1)[0] + ".layers.0."
    selected = [item for item in selected_all if first_block in item.name]
    calibration = load_manifest(args.calibration_manifest)[:1]
    evaluation = _nll_samples(args)[:1]
    try:
        search, cache_hit, key, raw_key = _search(
            model, selected, processor, calibration, config, args,
            cache_dir=root / "cache",
        )
        applied = apply_mbq(model, selected, config, search)
        try:
            rows = _evaluate_samples_cached(
                model,
                processor,
                evaluation,
                variant="mbq_trimodal_real_smoke",
                config_digest=config.digest,
                result_dir=root / "sample_cache",
                analysis_ids=set(),
                promotion_ids=set(),
                min_pixels=config.min_pixels,
                max_pixels=config.max_pixels,
                resume=True,
            )
        finally:
            applied.restore()
        atomic_json(root / "real_smoke.json", {
            "status": "complete",
            "selected_modules": len(selected),
            "nll_rows": len(rows),
            "finite_nll": all(np.isfinite(row["target_nll"]) for row in rows),
            "cache_hit": cache_hit,
            "search_key": key,
            "raw_key": raw_key,
        })
    finally:
        del model, processor


def execute_stage(args: argparse.Namespace, stage: str) -> None:
    root = args.output_dir
    cache_dir = root / "_calibration_cache"
    configs = trimodal_configs(seed=args.seed)
    if stage == "preflight":
        available_memory = psutil.virtual_memory().available
        disk_free = shutil.disk_usage(root.parent if root.parent.exists() else Path(".")).free
        active = _active_quant_processes()
        if available_memory < 80 * 1024**3:
            raise RuntimeError("tri-modal preflight requires at least 80 GiB available memory")
        if disk_free < 100 * 1024**3:
            raise RuntimeError("tri-modal preflight requires at least 100 GiB free disk")
        if active:
            raise RuntimeError(f"other quantization processes are active: {active}")
        for path in (
            Path(args.model_path), args.calibration_manifest, args.evaluation_manifest,
            args.formal_root, args.vlmevalkit_root,
        ):
            if not path.exists():
                raise FileNotFoundError(path)
        _prior_ids(args)
        _new_promotion_samples(args)
        test_environment = dict(os.environ)
        existing_pythonpath = test_environment.get("PYTHONPATH", "")
        test_environment["PYTHONPATH"] = os.pathsep.join(
            value for value in (
                str(args.vlmevalkit_root.resolve()), existing_pythonpath,
            ) if value
        )
        _run(
            [sys.executable, "-m", "pytest", "-q", "experiments/vlm_quant_benchmark/tests"],
            root / "logs/preflight_tests.log",
            environment=test_environment,
        )
        if not args.skip_real_smoke:
            _real_smoke(args)
        atomic_json(root / "preflight_audit.json", {
            "status": "complete",
            "environment": environment_record(),
            "available_memory_bytes": available_memory,
            "disk_free_bytes": disk_free,
            "active_quant_processes": active,
            "threads": args.threads,
            "seed": args.seed,
            "model_path": str(Path(args.model_path).resolve()),
            "calibration_manifest_sha256": file_digest(args.calibration_manifest),
            "evaluation_manifest_sha256": file_digest(args.evaluation_manifest),
            "formal_root": str(args.formal_root.resolve()),
            "real_smoke": not args.skip_real_smoke,
        })
    elif stage == "collect":
        torch.set_num_threads(args.threads)
        torch.manual_seed(args.seed)
        model, processor, _conversion = load_reference(args.model_path)
        selected = select_linears(
            model, QuantScope.DECODER_CANONICAL,
            strict_qwen_3b=not args.allow_nonstandard_model,
        )
        calibration = load_manifest(args.calibration_manifest)[:128]
        if len(calibration) != 128:
            raise RuntimeError("tri-modal formal collection requires 128 calibration samples")
        try:
            config = configs["mbq_trimodal_no_floor"]
            search, cache_hit, key, raw_key = _search(
                model, selected, processor, calibration, config, args,
                cache_dir=cache_dir,
            )
            atomic_json(root / "search_audits/mbq_trimodal_no_floor.json", search.audit)
            atomic_json(root / "collection_audit.json", {
                "status": "complete",
                "samples": len(calibration),
                "selected_modules": len(selected),
                "anchors": len(search.audit["raw_gradient_ratios"]),
                "cache_hit": cache_hit,
                "search_key": key,
                "raw_key": raw_key,
                "token_counts": search.audit["token_counts"],
            })
        finally:
            del model, processor
    elif stage == "w4_nll":
        torch.set_num_threads(args.threads)
        torch.manual_seed(args.seed)
        model, processor, _conversion = load_reference(args.model_path)
        selected = select_linears(
            model, QuantScope.DECODER_CANONICAL,
            strict_qwen_3b=not args.allow_nonstandard_model,
        )
        calibration = load_manifest(args.calibration_manifest)[:128]
        samples = _nll_samples(args)
        rows = []
        statuses = []
        try:
            for variant in VARIANTS:
                config = configs[variant]
                search, cache_hit, key, raw_key = _search(
                    model, selected, processor, calibration, config, args,
                    cache_dir=cache_dir,
                )
                atomic_json(root / "search_audits" / f"{variant}.json", search.audit)
                applied = apply_mbq(model, selected, config, search)
                try:
                    variant_rows = _evaluate_samples_cached(
                        model,
                        processor,
                        samples,
                        variant=variant,
                        config_digest=config.digest,
                        result_dir=root / "w4_nll/_sample_cache",
                        analysis_ids=set(),
                        promotion_ids=set(),
                        min_pixels=config.min_pixels,
                        max_pixels=config.max_pixels,
                        resume=args.resume,
                    )
                finally:
                    applied.restore()
                rows.extend(variant_rows)
                statuses.append({
                    "variant": variant,
                    "status": "complete",
                    "rows": len(variant_rows),
                    "config": config.to_dict(),
                    "config_digest": config.digest,
                    "search_cache_hit": cache_hit,
                    "search_key": key,
                    "raw_key": raw_key,
                })
        finally:
            del model, processor
        if len(rows) != 256:
            raise RuntimeError(f"tri-modal W4 NLL requires 256 rows, found {len(rows)}")
        _atomic_jsonl(root / "w4_nll/trimodal_ablation_results.jsonl", rows)
        atomic_json(root / "w4_nll/run_audit.json", {
            "status": "complete", "rows": len(rows), "variants": statuses,
        })
    elif stage == "analysis":
        rows = _read_jsonl(root / "w4_nll/trimodal_ablation_results.jsonl")
        formal_rows = _read_jsonl(args.formal_root / "w4_nll/ablation_results.jsonl")
        baseline = {
            (str(row["dataset"]), str(row["id"])): float(row["target_nll"])
            for row in formal_rows
            if row.get("variant") == "mbq_no_median_floor"
            and row.get("evaluation_split") == "ablation"
        }
        if len(baseline) != 128:
            raise RuntimeError(f"expected 128 two-modal no-floor baselines, found {len(baseline)}")
        effects = [
            _paired_effect(
                baseline,
                [row for row in rows if row.get("variant") == variant],
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
                variant=variant,
            )
            for variant in VARIANTS
        ]
        effects.sort(key=lambda row: -row["dataset_balanced_nll_improvement"])
        selected = effects[0]
        promotion_gate = selected["dataset_balanced_nll_improvement"] > 0
        selection = {
            "format_version": 1,
            "baseline": "mbq_no_median_floor",
            "selection_rule": "lowest dataset-balanced paired target NLL",
            "effects": effects,
            "selected": selected["variant"],
            "promotion_gate": promotion_gate,
            "promotion_label": (
                "confirmatory" if promotion_gate and selected["ci_low"] > 0
                else "exploratory" if promotion_gate else "not_promoted"
            ),
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
        }
        weights, masses = _flatten_search_audits(root)
        _atomic_csv(root / "analysis/trimodal_effects.csv", effects)
        _atomic_csv(root / "analysis/trimodal_weights.csv", weights)
        _atomic_csv(root / "analysis/trimodal_mass_share.csv", masses)
        atomic_json(root / "analysis/selection.json", selection)
        report = [
            "# MBQ 三类 Token W4 中期分析",
            "",
            f"选择配置：`{selected['variant']}`。",
            f"相对两模态 no-floor 的 dataset-balanced ΔNLL 改善为 "
            f"{selected['dataset_balanced_nll_improvement']:.6f}，95% CI "
            f"[{selected['ci_low']:.6f}, {selected['ci_high']:.6f}]。",
            "",
            (
                f"晋级状态：{selection['promotion_label']}，将进入独立64×4生成评测。"
                if promotion_gate else
                "晋级状态：未通过；平均NLL没有改善，不启动生成评测。"
            ),
        ]
        (root / "analysis").mkdir(parents=True, exist_ok=True)
        (root / "analysis/report_zh.md").write_text("\n".join(report) + "\n")
    elif stage == "promotion":
        selection_path = root / "analysis/selection.json"
        if not selection_path.is_file():
            raise FileNotFoundError(selection_path)
        selection = json.loads(selection_path.read_text())
        promotion_root = root / "promotion"
        if not selection["promotion_gate"]:
            atomic_json(promotion_root / "status.json", {
                "status": "skipped", "reason": "mean paired NLL improvement <= 0",
            })
            return
        samples = _new_promotion_samples(args)
        manifest = promotion_root / "promotion_manifest.jsonl"
        write_manifest(manifest, samples)
        tri_config = configs[selection["selected"]]
        control_config = _control_config(args)
        tri_config.dump(promotion_root / "mbq_trimodal_config.json")
        control_config.dump(promotion_root / "mbq_two_modal_no_floor_config.json")
        atomic_json(promotion_root / "manifest_audit.json", {
            "status": "complete",
            "samples": len(samples),
            "ids": [sample.id for sample in samples],
            "manifest_sha256": file_digest(manifest),
            "disjoint_from_all_prior_formal_sets": True,
            "salt": f"trimodal_promotion_v1:{args.seed}",
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
        commands = (
            (
                "mbq_trimodal",
                promotion_root / "mbq_trimodal_config.json",
                root / "_calibration_cache",
            ),
            (
                "mbq_two_modal_no_floor",
                promotion_root / "mbq_two_modal_no_floor_config.json",
                args.formal_root / "_calibration_cache",
            ),
        )
        for name, config_path, method_cache in commands:
            _run([
                sys.executable,
                "-m", "experiments.vlm_quant_benchmark.run_vlmeval_official",
                *common,
                "--calibration-cache-dir", str(method_cache),
                "--experiment-config-json", str(config_path),
                "--output-dir", str(promotion_root / "runs" / name),
            ], root / "logs" / f"promotion_{name}.log")
        atomic_json(promotion_root / "status.json", {
            "status": "complete", "generated_predictions": 512,
        })
    elif stage == "final":
        selection = json.loads((root / "analysis/selection.json").read_text())
        status = json.loads((root / "promotion/status.json").read_text())
        scores = []
        paired_generation = None
        if status["status"] == "complete":
            sample_scores = {}
            for name in ("mbq_trimodal", "mbq_two_modal_no_floor"):
                result_path = root / "promotion/runs" / name / "official_result.json"
                result = json.loads(result_path.read_text())
                if result.get("status") != "complete":
                    raise RuntimeError(f"incomplete official result: {name}")
                if sum(int(value["samples"]) for value in result["datasets"].values()) != 256:
                    raise RuntimeError(f"official result does not contain 256 samples: {name}")
                for dataset, value in sorted(result["datasets"].items()):
                    scores.append({
                        "method": name, "dataset": dataset, "score": value["score"],
                    })
                scores.append({
                    "method": name,
                    "dataset": "balanced_mean",
                    "score": statistics.fmean(
                        value["score"] for value in result["datasets"].values()
                    ),
                })
                sample_scores[name] = {
                    (str(row["dataset"]), str(row["id"])): float(row["official_score"])
                    for row in result["samples"]
                }
            baseline = sample_scores["mbq_two_modal_no_floor"]
            candidate = sample_scores["mbq_trimodal"]
            paired_rows = [
                {"dataset": dataset, "id": identifier, "improvement": candidate[key] - value}
                for key, value in baseline.items()
                for dataset, identifier in [key]
                if key in candidate
            ]
            baseline_map = {
                (row["dataset"], row["id"]): 0.0 for row in paired_rows
            }
            candidate_nll_shape = [
                {
                    "dataset": row["dataset"], "id": row["id"],
                    "target_nll": -row["improvement"],
                }
                for row in paired_rows
            ]
            paired_generation = _paired_effect(
                baseline_map,
                candidate_nll_shape,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
                variant="mbq_trimodal_generation_score",
                expected_per_dataset=64,
            )
            _atomic_csv(root / "analysis/promotion_scores.csv", scores)
            atomic_json(root / "analysis/promotion_comparison.json", paired_generation)
        nll_rows = _read_jsonl(root / "w4_nll/trimodal_ablation_results.jsonl")
        if len(nll_rows) != 256:
            raise RuntimeError("final validation requires exactly 256 tri-modal NLL rows")
        atomic_json(root / "formal_run_audit.json", {
            "status": "complete",
            "nll_rows": len(nll_rows),
            "promotion_status": status["status"],
            "promotion_rows": 512 if status["status"] == "complete" else 0,
            "selection": selection,
            "paired_generation": paired_generation,
            "input_digests": {
                "calibration_manifest": file_digest(args.calibration_manifest),
                "evaluation_manifest": file_digest(args.evaluation_manifest),
            },
        })
        summary = [
            "# MBQ 三类 Token 正式验证总结",
            "",
            "正式实现将 token 分为 vision、prompt-language 和 answer-language，"
            "answer-language 为敏感度参考。",
            "",
            f"NLL最优配置：`{selection['selected']}`；晋级标签："
            f"`{selection['promotion_label']}`。",
            f"Promotion状态：`{status['status']}`。",
        ]
        if paired_generation is not None:
            summary.extend([
                "",
                "独立64×4配对生成分数改善："
                f"{paired_generation['dataset_balanced_nll_improvement'] * 100:.4f} 分，"
                f"95% CI [{paired_generation['ci_low'] * 100:.4f}, "
                f"{paired_generation['ci_high'] * 100:.4f}]。",
            ])
        (root / "final_summary_zh.md").write_text("\n".join(summary) + "\n")
    else:  # pragma: no cover
        raise ValueError(stage)


def main() -> None:
    args = parse_args()
    if args.threads != 16:
        raise ValueError("formal tri-modal protocol requires exactly 16 CPU threads")
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "stage_state.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {
        "format_version": 1,
        "stages": {stage: {"status": "pending"} for stage in STAGES},
    }
    selected_stages = STAGES if args.stage == "all" else (args.stage,)
    with pid_lock(args.output_dir / "formal_run.pid"):
        for stage in selected_stages:
            if args.resume and state["stages"].get(stage, {}).get("status") == "complete":
                continue
            started = time.time()
            state["stages"][stage] = {"status": "running", "started_unix": started}
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
                "elapsed_seconds": time.time() - started,
            }
            atomic_json(state_path, state)


if __name__ == "__main__":
    main()
