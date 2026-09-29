"""Locked 2x2 MBQ factorial: activation statistics x reconstruction objective."""
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
from collections import defaultdict
from pathlib import Path

import numpy as np
import psutil
import torch

from .config import ExperimentConfig, ImplementationTier, MethodName, QuantScope
from .data import file_digest, load_manifest, sample_manifest_digest
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
from .run_mbq_trimodal_formal import trimodal_configs
from .run_sensitivity_ablation import _evaluate_samples_cached
from .run_sensitivity_formal import _active_quant_processes, _run, pid_lock
from .runner import environment_record, load_reference
from .selection import select_linears
from .sensitivity import sensitivity_anchors


STAGES = ("preflight", "search", "w4_nll", "residuals", "analysis", "final")
VA = "vision_answer"
VPA = "vision_prompt_answer"
LAYERS = (0, 6, 7, 12, 15, 18, 19, 24, 30, 34, 35)
CELLS = {
    "A_VA__O_VA": (VA, VA),
    "A_VPA__O_VA": (VPA, VA),
    "A_VA__O_VPA": (VA, VPA),
    "A_VPA__O_VPA": (VPA, VPA),
}
ENDPOINTS = {
    "A_VA__O_VA": "two_modal_no_floor",
    "A_VPA__O_VPA": "trimodal_no_floor",
}
CONTRASTS = (
    "activation_main_effect",
    "objective_main_effect",
    "activation_objective_interaction",
    "trimodal_endpoint_effect",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the formal MBQ 2x2 factorial")
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/vlm_quant_benchmark/sensitivity_analysis_v1/"
            "trimodal_factorial_v1"
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
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--skip-real-smoke", action="store_true")
    parser.add_argument("--allow-nonstandard-model", action="store_true")
    return parser.parse_args()


def factorial_configs(
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
        mbq_objective_normalization="token_weighted",
        mbq_reconstruction_level="linear_group",
        mbq_median_floor=False,
        mbq_reweight=True,
    )
    return {
        cell: ExperimentConfig(
            **common,
            mbq_activation_mode=activation,
            mbq_objective_mode=objective,
            scheme_id=f"mbq_factorial_{cell.lower()}_w4a16",
        )
        for cell, (activation, objective) in CELLS.items()
    }


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


def _nll_samples(args: argparse.Namespace) -> list:
    plan_path = args.formal_root / "w4_nll/ablation_plan.json"
    plan = json.loads(plan_path.read_text())
    ids = set(plan["nll_ids"])
    samples = [
        sample
        for sample in load_manifest(args.evaluation_manifest)
        if sample.id in ids
    ]
    counts = defaultdict(int)
    for sample in samples:
        counts[sample.dataset] += 1
    if len(samples) != 128 or len(counts) != 4 or set(counts.values()) != {32}:
        raise RuntimeError(f"factorial NLL set is not fixed 32x4: {dict(counts)}")
    return samples


def _residual_samples(args: argparse.Namespace) -> list:
    plan = json.loads(
        (args.formal_root / "residuals/residual_plan.json").read_text()
    )
    ids = list(plan["sample_ids"])
    lookup = {
        sample.id: sample for sample in load_manifest(args.evaluation_manifest)
    }
    if len(ids) != 32 or len(set(ids)) != 32 or not set(ids) <= set(lookup):
        raise RuntimeError("existing formal residual sample plan is invalid")
    samples = [lookup[identifier] for identifier in ids]
    counts = defaultdict(int)
    for sample in samples:
        counts[sample.dataset] += 1
    if len(counts) != 4 or set(counts.values()) != {8}:
        raise RuntimeError(f"factorial residual set is not fixed 8x4: {dict(counts)}")
    return samples


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


def _search(
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


def _scale_digest(search: MBQSearchResult) -> str:
    digest = hashlib.sha256()
    for name in sorted(search.scales):
        value = search.scales[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _compare_searches(
    candidate: MBQSearchResult,
    reference: MBQSearchResult,
    *,
    candidate_name: str,
    reference_name: str,
) -> dict:
    candidate_names = set(candidate.scales)
    reference_names = set(reference.scales)
    common = sorted(candidate_names & reference_names)
    alpha_mismatches = []
    for group, value in candidate.audit["groups"].items():
        reference_group = reference.audit["groups"].get(group)
        if (
            reference_group is None
            or value["selected_alpha"] != reference_group["selected_alpha"]
        ):
            alpha_mismatches.append(group)
    exact = candidate_names == reference_names and all(
        torch.equal(candidate.scales[name], reference.scales[name])
        for name in common
    )
    max_abs = max(
        (
            float(
                (
                    candidate.scales[name].float()
                    - reference.scales[name].float()
                ).abs().max().item()
            )
            for name in common
        ),
        default=float("inf"),
    )
    return {
        "candidate": candidate_name,
        "reference": reference_name,
        "candidate_scale_digest": _scale_digest(candidate),
        "reference_scale_digest": _scale_digest(reference),
        "same_module_names": candidate_names == reference_names,
        "scale_exact": exact,
        "scale_max_abs_difference": max_abs,
        "alpha_exact": not alpha_mismatches,
        "alpha_mismatch_groups": alpha_mismatches,
        "numeric_endpoint_reproduction": (
            candidate_names == reference_names
            and not alpha_mismatches
            and max_abs <= 1e-7
        ),
        "nll_reuse_allowed": exact and not alpha_mismatches,
    }


def _reference_search(
    cell: str,
    model,
    selected,
    calibration,
    args: argparse.Namespace,
) -> tuple[MBQSearchResult, dict]:
    if cell == "A_VA__O_VA":
        config_path = (
            args.formal_root / "promotion/mbq_no_median_floor_config.json"
        )
        config = ExperimentConfig.from_dict(json.loads(config_path.read_text()))
        cache_dir = args.formal_root / "_calibration_cache"
    elif cell == "A_VPA__O_VPA":
        config = trimodal_configs(
            calibration_samples=128, seed=args.seed,
        )["mbq_trimodal_no_floor"]
        cache_dir = args.trimodal_root / "_calibration_cache"
    else:
        raise ValueError(f"no endpoint reference for {cell}")
    key, _raw_key = _search_identity(selected, calibration, config, args)
    path = cache_dir / f"mbq_{key}.pt"
    search = load_mbq_search(path, key=key, selected=selected)
    if search is None:
        raise FileNotFoundError(f"endpoint search cache is absent or invalid: {path}")
    return search, {
        "config": config.to_dict(),
        "config_digest": config.digest,
        "search_key": key,
        "search_path": str(path.resolve()),
        "search_sha256": file_digest(path),
    }


def _run_search_stage(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    configs = factorial_configs(seed=args.seed)
    calibration = load_manifest(args.calibration_manifest)[:128]
    model, processor, _conversion = load_reference(args.model_path)
    selected = select_linears(
        model,
        QuantScope.DECODER_CANONICAL,
        strict_qwen_3b=not args.allow_nonstandard_model,
    )
    searches = {}
    statuses = []
    try:
        for cell, config in configs.items():
            search, cache_hit, key, raw_key = _search(
                model, selected, processor, calibration, config, args,
            )
            searches[cell] = search
            audit = dict(search.audit)
            audit["factorial_cell"] = cell
            audit["scale_digest"] = _scale_digest(search)
            atomic_json(args.output_dir / "search_audits" / f"{cell}.json", audit)
            statuses.append({
                "cell": cell,
                "activation_mode": config.effective_mbq_activation_mode,
                "objective_mode": config.effective_mbq_objective_mode,
                "search_key": key,
                "raw_key": raw_key,
                "cache_hit": cache_hit,
                "groups": len(search.audit["groups"]),
                "scales": len(search.scales),
                "scale_digest": _scale_digest(search),
                "config": config.to_dict(),
                "config_digest": config.digest,
            })
        endpoint_verification = {}
        for cell, label in ENDPOINTS.items():
            reference, provenance = _reference_search(
                cell, model, selected, calibration, args,
            )
            result = _compare_searches(
                searches[cell],
                reference,
                candidate_name=cell,
                reference_name=label,
            )
            result["reference_provenance"] = provenance
            endpoint_verification[cell] = result
    finally:
        del model, processor
    if any(item["groups"] != 144 or item["scales"] != 252 for item in statuses):
        raise RuntimeError("factorial searches do not contain 144 groups / 252 scales")
    atomic_json(args.output_dir / "search_audits/endpoint_verification.json", {
        "status": "complete",
        "endpoints": endpoint_verification,
        "reuse_rule": "exact scales and exact selected alphas",
    })
    atomic_json(args.output_dir / "search_audits/run_audit.json", {
        "status": "complete",
        "cells": statuses,
        "shared_raw_cache": str(_raw_cache_dir(args).resolve()),
        "shared_raw_cache_read_only": True,
    })


def _endpoint_source_rows(
    cell: str, args: argparse.Namespace,
) -> tuple[list[dict], Path]:
    if cell == "A_VA__O_VA":
        path = args.formal_root / "w4_nll/ablation_results.jsonl"
        rows = [
            row
            for row in _read_jsonl(path)
            if row.get("variant") == "mbq_no_median_floor"
            and row.get("evaluation_split") == "ablation"
        ]
    elif cell == "A_VPA__O_VPA":
        path = (
            args.trimodal_root
            / "w4_nll/trimodal_ablation_results.jsonl"
        )
        rows = [
            row
            for row in _read_jsonl(path)
            if row.get("variant") == "mbq_trimodal_no_floor"
        ]
    else:
        raise ValueError(cell)
    return rows, path


def _run_nll_stage(args: argparse.Namespace) -> None:
    verification = json.loads(
        (args.output_dir / "search_audits/endpoint_verification.json").read_text()
    )["endpoints"]
    configs = factorial_configs(seed=args.seed)
    samples = _nll_samples(args)
    expected_ids = {sample.id for sample in samples}
    calibration = load_manifest(args.calibration_manifest)[:128]
    rows = []
    statuses = []
    model = processor = selected = None
    try:
        for cell, config in configs.items():
            can_import = (
                cell in ENDPOINTS
                and verification[cell]["nll_reuse_allowed"]
            )
            if can_import:
                source_rows, source_path = _endpoint_source_rows(cell, args)
                if (
                    len(source_rows) != 128
                    or {str(row["id"]) for row in source_rows} != expected_ids
                ):
                    raise RuntimeError(f"invalid endpoint NLL source for {cell}")
                cell_rows = []
                for source in source_rows:
                    row = dict(source)
                    row["source_variant"] = row.get("variant")
                    row["source_config_digest"] = row.get("config_digest")
                    row["variant"] = cell
                    row["config_digest"] = config.digest
                    row["result_origin"] = "verified_existing"
                    row["source_result_path"] = str(source_path.resolve())
                    row["source_result_sha256"] = file_digest(source_path)
                    cell_rows.append(row)
                search_cache_hit = True
                key = None
            else:
                if model is None:
                    torch.set_num_threads(args.threads)
                    torch.manual_seed(args.seed)
                    model, processor, _conversion = load_reference(args.model_path)
                    selected = select_linears(
                        model,
                        QuantScope.DECODER_CANONICAL,
                        strict_qwen_3b=not args.allow_nonstandard_model,
                    )
                search, search_cache_hit, key, _raw_key = _search(
                    model, selected, processor, calibration, config, args,
                )
                applied = apply_mbq(model, selected, config, search)
                try:
                    evaluated = _evaluate_samples_cached(
                        model,
                        processor,
                        samples,
                        variant=cell,
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
                cell_rows = [
                    dict(row, result_origin="new") for row in evaluated
                ]
            if len(cell_rows) != 128:
                raise RuntimeError(f"{cell} produced {len(cell_rows)} NLL rows")
            rows.extend(cell_rows)
            statuses.append({
                "cell": cell,
                "status": "complete",
                "rows": len(cell_rows),
                "result_origin": (
                    "verified_existing" if can_import else "new"
                ),
                "config": config.to_dict(),
                "config_digest": config.digest,
                "search_key": key,
                "search_cache_hit": search_cache_hit,
            })
    finally:
        if model is not None:
            del model, processor
    if len(rows) != 512:
        raise RuntimeError(f"factorial W4 NLL requires 512 rows, found {len(rows)}")
    _atomic_jsonl(args.output_dir / "w4_nll/factorial_nll_results.jsonl", rows)
    atomic_json(args.output_dir / "w4_nll/run_audit.json", {
        "status": "complete",
        "rows": len(rows),
        "cells": statuses,
    })


def _sample_cache_name(identifier: str) -> str:
    return hashlib.sha256(identifier.encode()).hexdigest() + ".json"


def _load_residual_cache(root: Path) -> tuple[list[dict], list[dict]]:
    residuals = []
    predictions = []
    for cell in CELLS:
        directory = root / cell
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            if path.name.endswith(".error.json"):
                continue
            payload = json.loads(path.read_text())
            if payload.get("status") != "complete":
                raise RuntimeError(f"incomplete factorial residual cache: {path}")
            residuals.extend(payload["residual_rows"])
            predictions.append(payload["prediction"])
    return residuals, predictions


def _run_residual_stage(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    configs = factorial_configs(seed=args.seed)
    calibration = load_manifest(args.calibration_manifest)[:128]
    samples = _residual_samples(args)
    bf16_model, processor, _conversion = load_reference(args.model_path)
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
        raise RuntimeError(
            f"expected {len(LAYERS) * 2} residual anchors, found {len(anchor_names)}"
        )
    cache_root = args.output_dir / "residuals/_sample_cache"
    statuses = []
    first_cell = next(iter(CELLS))
    try:
        for cell, config in configs.items():
            candidate_model, candidate_processor, _ = load_reference(args.model_path)
            del candidate_processor
            candidate_selected = select_linears(
                candidate_model,
                QuantScope.DECODER_CANONICAL,
                strict_qwen_3b=not args.allow_nonstandard_model,
            )
            search, search_hit, search_key, _raw_key = _search(
                candidate_model,
                candidate_selected,
                processor,
                calibration,
                config,
                args,
            )
            applied = apply_mbq(
                candidate_model, candidate_selected, config, search,
            )
            cell_cache = cache_root / cell
            cell_cache.mkdir(parents=True, exist_ok=True)
            try:
                for sample in samples:
                    path = cell_cache / _sample_cache_name(sample.id)
                    if args.resume and path.is_file():
                        continue
                    try:
                        batch = next(
                            mbq_calibration_batches(
                                processor,
                                [sample],
                                min_pixels=config.min_pixels,
                                max_pixels=config.max_pixels,
                            )
                        )
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
                        residual_rows = reconstruction_rows(
                            batch,
                            reference,
                            candidate,
                            method=cell,
                        )
                        if cell == first_cell:
                            residual_rows.extend(
                                reconstruction_rows(
                                    batch,
                                    reference,
                                    reference,
                                    method="bf16",
                                )
                            )
                        for row in residual_rows:
                            row["factorial_cell"] = (
                                None if row["method"] == "bf16" else cell
                            )
                            row["activation_mode"] = (
                                None
                                if row["method"] == "bf16"
                                else config.effective_mbq_activation_mode
                            )
                            row["objective_mode"] = (
                                None
                                if row["method"] == "bf16"
                                else config.effective_mbq_objective_mode
                            )
                        prediction = {
                            "id": sample.id,
                            "dataset": sample.dataset,
                            "factorial_cell": cell,
                            **prediction_change_summary(
                                reference_logits,
                                candidate_logits,
                                batch["language_mask"],
                            ),
                        }
                        atomic_json(path, {
                            "status": "complete",
                            "id": sample.id,
                            "factorial_cell": cell,
                            "config_digest": config.digest,
                            "residual_rows": residual_rows,
                            "prediction": prediction,
                        })
                    except Exception as error:
                        atomic_json(path.with_suffix(".error.json"), {
                            "status": "failed",
                            "id": sample.id,
                            "factorial_cell": cell,
                            "error_type": type(error).__name__,
                            "error": str(error),
                        })
                        raise
            finally:
                applied.restore()
                del candidate_model
            statuses.append({
                "cell": cell,
                "status": "complete",
                "samples": len(samples),
                "search_cache_hit": search_hit,
                "search_key": search_key,
            })
    finally:
        del bf16_model, processor
    residual_rows, predictions = _load_residual_cache(cache_root)
    expected_rows = 5 * 32 * len(LAYERS) * 3 * 4
    if len(residual_rows) != expected_rows or len(predictions) != 128:
        raise RuntimeError(
            "factorial residual matrix incomplete: "
            f"{len(residual_rows)} residuals, {len(predictions)} predictions"
        )
    _atomic_jsonl(
        args.output_dir / "residuals/factorial_residuals.jsonl",
        residual_rows,
    )
    _atomic_jsonl(
        args.output_dir / "residuals/prediction_changes.jsonl",
        predictions,
    )
    atomic_json(args.output_dir / "residuals/run_audit.json", {
        "status": "complete",
        "samples": len(samples),
        "layers": list(LAYERS),
        "anchors": len(anchor_names),
        "rows": len(residual_rows),
        "prediction_rows": len(predictions),
        "cells": statuses,
    })


def factorial_contrast_values(cell_values: dict[str, float]) -> dict[str, float]:
    missing = set(CELLS) - set(cell_values)
    if missing:
        raise ValueError(f"factorial cell values are missing: {sorted(missing)}")
    y00 = float(cell_values["A_VA__O_VA"])
    y10 = float(cell_values["A_VPA__O_VA"])
    y01 = float(cell_values["A_VA__O_VPA"])
    y11 = float(cell_values["A_VPA__O_VPA"])
    return {
        "activation_main_effect": 0.5 * ((y10 - y00) + (y11 - y01)),
        "objective_main_effect": 0.5 * ((y01 - y00) + (y11 - y10)),
        "activation_objective_interaction": (y11 - y10) - (y01 - y00),
        "trimodal_endpoint_effect": y11 - y00,
    }


def factorial_contrast_rows(
    rows: list[dict],
    *,
    value_key: str,
    key_fields: tuple[str, ...],
) -> list[dict]:
    grouped = defaultdict(dict)
    metadata = {}
    for row in rows:
        cell = row.get("factorial_cell") or row.get("variant")
        if cell not in CELLS:
            continue
        key = tuple(str(row[field]) for field in key_fields)
        if cell in grouped[key]:
            raise RuntimeError(f"duplicate factorial observation: {key}, {cell}")
        grouped[key][cell] = float(row[value_key])
        metadata[key] = {field: row[field] for field in key_fields}
    output = []
    for key, values in grouped.items():
        contrasts = factorial_contrast_values(values)
        for contrast, value in contrasts.items():
            output.append({
                **metadata[key],
                "contrast": contrast,
                "value": value,
                "source_metric": value_key,
            })
    return output


def _bootstrap_summary(
    rows: list[dict],
    *,
    bootstrap_samples: int,
    seed: int,
    label: str,
) -> dict:
    buckets = {
        dataset: [float(row["value"]) for row in rows if row["dataset"] == dataset]
        for dataset in sorted({str(row["dataset"]) for row in rows})
    }
    if len(buckets) != 4 or any(not values for values in buckets.values()):
        raise RuntimeError(f"{label} does not contain four non-empty datasets")
    mean = statistics.fmean(statistics.fmean(values) for values in buckets.values())
    rng = random.Random(f"factorial:{seed}:{label}")
    boot = [
        statistics.fmean(
            statistics.fmean(rng.choices(values, k=len(values)))
            for values in buckets.values()
        )
        for _ in range(bootstrap_samples)
    ]
    low = float(np.quantile(boot, 0.025))
    high = float(np.quantile(boot, 0.975))
    nonpositive = sum(value <= 0 for value in boot) / len(boot)
    nonnegative = sum(value >= 0 for value in boot) / len(boot)
    return {
        "dataset_balanced_mean": mean,
        "ci_low": low,
        "ci_high": high,
        "bootstrap_p_two_sided": min(1.0, 2 * min(nonpositive, nonnegative)),
        "samples": len(rows),
        "per_dataset_mean": json.dumps(
            {
                dataset: statistics.fmean(values)
                for dataset, values in buckets.items()
            },
            sort_keys=True,
        ),
    }


def _benjamini_hochberg(rows: list[dict]) -> None:
    ordered = sorted(
        enumerate(rows), key=lambda item: item[1]["bootstrap_p_two_sided"],
    )
    total = len(ordered)
    adjusted = [1.0] * total
    running = 1.0
    for rank_index in range(total - 1, -1, -1):
        original_index, row = ordered[rank_index]
        rank = rank_index + 1
        running = min(
            running,
            float(row["bootstrap_p_two_sided"]) * total / rank,
        )
        adjusted[original_index] = min(1.0, running)
    for row, value in zip(rows, adjusted):
        row["fdr_q"] = value


def _summarize_contrasts(
    rows: list[dict],
    *,
    group_fields: tuple[str, ...],
    bootstrap_samples: int,
    seed: int,
) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        key = (row["contrast"],) + tuple(row[field] for field in group_fields)
        grouped[key].append(row)
    summaries = []
    for key, values in sorted(grouped.items(), key=lambda item: str(item[0])):
        payload = {
            "contrast": key[0],
            **{
                field: key[index + 1]
                for index, field in enumerate(group_fields)
            },
            **_bootstrap_summary(
                values,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
                label=":".join(map(str, key)),
            ),
        }
        summaries.append(payload)
    _benjamini_hochberg(summaries)
    return summaries


def _decision(nll: list[dict], residual: list[dict]) -> dict:
    nll_map = {row["contrast"]: row for row in nll}
    activation = nll_map["activation_main_effect"]
    objective = nll_map["objective_main_effect"]
    interaction = nll_map["activation_objective_interaction"]
    block = [
        row
        for row in residual
        if row["contrast"] in {
            "activation_main_effect", "objective_main_effect",
        }
        and row["projection"] == "block_output"
        and row["modality"] == "all_tokens"
        and row["source_metric"] == "rmse"
    ]
    residual_means = {
        contrast: statistics.fmean(
            row["dataset_balanced_mean"]
            for row in block
            if row["contrast"] == contrast
        )
        for contrast in ("activation_main_effect", "objective_main_effect")
    }
    activation_harm = activation["ci_low"] > 0
    objective_harm = objective["ci_low"] > 0
    interaction_detected = (
        interaction["ci_low"] > 0 or interaction["ci_high"] < 0
    )
    if activation_harm and not objective_harm:
        conclusion = "主要退化来自prompt进入scale候选的激活统计"
    elif objective_harm and not activation_harm:
        conclusion = "主要退化来自prompt进入加权重建目标"
    elif activation_harm and objective_harm:
        conclusion = "激活统计与加权目标都造成退化"
    elif interaction_detected:
        conclusion = "退化主要由激活统计与加权目标的交互产生"
    else:
        conclusion = "当前2x2证据不足以将退化归因于单一因素"
    return {
        "activation_main_effect_harmful": activation_harm,
        "objective_main_effect_harmful": objective_harm,
        "interaction_detected": interaction_detected,
        "activation_residual_mean_effect": residual_means["activation_main_effect"],
        "objective_residual_mean_effect": residual_means["objective_main_effect"],
        "positive_effect_definition": "higher NLL or residual is worse",
        "conclusion": conclusion,
    }


def _run_analysis_stage(args: argparse.Namespace) -> None:
    nll_rows = _read_jsonl(
        args.output_dir / "w4_nll/factorial_nll_results.jsonl"
    )
    residual_rows = [
        row
        for row in _read_jsonl(
            args.output_dir / "residuals/factorial_residuals.jsonl"
        )
        if row.get("factorial_cell") in CELLS
    ]
    nll_contrasts = factorial_contrast_rows(
        nll_rows,
        value_key="target_nll",
        key_fields=("dataset", "id"),
    )
    nll_summary = _summarize_contrasts(
        nll_contrasts,
        group_fields=(),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    residual_contrasts = []
    for metric in ("mae", "rmse"):
        source = factorial_contrast_rows(
            residual_rows,
            value_key=metric,
            key_fields=(
                "dataset", "id", "layer", "projection", "modality",
            ),
        )
        residual_contrasts.extend(source)
    residual_summary = _summarize_contrasts(
        residual_contrasts,
        group_fields=("source_metric", "layer", "projection", "modality"),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    decision = _decision(nll_summary, residual_summary)
    search_diagnostics = {}
    for cell in CELLS:
        audit = json.loads(
            (args.output_dir / "search_audits" / f"{cell}.json").read_text()
        )
        search_diagnostics[cell] = {
            group: value
            for group, value in audit["groups"].items()
            if ".layers.30." in group
        }
    _atomic_csv(
        args.output_dir / "analysis/factorial_effects.csv", nll_summary,
    )
    _atomic_csv(
        args.output_dir / "analysis/layer_residual_effects.csv",
        residual_summary,
    )
    _atomic_jsonl(
        args.output_dir / "analysis/sample_factorial_contrasts.jsonl",
        nll_contrasts,
    )
    atomic_json(
        args.output_dir / "analysis/layer30_search_diagnostics.json",
        search_diagnostics,
    )
    atomic_json(
        args.output_dir / "analysis/attribution_decision.json", decision,
    )
    endpoint = next(
        row for row in nll_summary
        if row["contrast"] == "trimodal_endpoint_effect"
    )
    report = [
        "# MBQ 2×2 因果拆分报告",
        "",
        "正值表示加入prompt因素后NLL或残差变差。",
        "",
        f"- 激活统计主效应：{next(row for row in nll_summary if row['contrast'] == 'activation_main_effect')['dataset_balanced_mean']:.6f}",
        f"- 重建目标主效应：{next(row for row in nll_summary if row['contrast'] == 'objective_main_effect')['dataset_balanced_mean']:.6f}",
        f"- 交互效应：{next(row for row in nll_summary if row['contrast'] == 'activation_objective_interaction')['dataset_balanced_mean']:.6f}",
        f"- 三模态端点相对两模态：{endpoint['dataset_balanced_mean']:.6f}",
        "",
        f"归因结论：{decision['conclusion']}。",
        "",
        "完整置信区间、逐数据集结果和逐层FDR结果见同目录CSV与JSON。",
    ]
    report_path = args.output_dir / "analysis/report_zh.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report) + "\n")
    atomic_json(args.output_dir / "analysis/run_audit.json", {
        "status": "complete",
        "nll_rows": len(nll_rows),
        "nll_contrast_rows": len(nll_contrasts),
        "residual_rows": len(residual_rows),
        "residual_contrast_rows": len(residual_contrasts),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
    })


def _real_smoke(args: argparse.Namespace) -> None:
    smoke_args = argparse.Namespace(**vars(args))
    smoke_args.output_dir = args.output_dir / "_preflight"
    smoke_args.raw_cache_dir = smoke_args.output_dir / "_calibration_cache"
    configs = factorial_configs(calibration_samples=1, seed=args.seed)
    configs = {
        cell: dataclasses.replace(
            config,
            mbq_scale_grid=(0.0, 0.5),
            scheme_id=f"factorial_real_smoke_{cell}",
        )
        for cell, config in configs.items()
    }
    model, processor, _conversion = load_reference(args.model_path)
    selected_all = select_linears(
        model,
        QuantScope.DECODER_CANONICAL,
        strict_qwen_3b=not args.allow_nonstandard_model,
    )
    prefix = selected_all[0].name.split(".layers.", 1)[0] + ".layers.0."
    selected = [item for item in selected_all if prefix in item.name]
    calibration = load_manifest(args.calibration_manifest)[:1]
    sample = _nll_samples(args)[:1]
    statuses = []
    searches = {}
    try:
        for cell, config in configs.items():
            search, hit, key, raw_key = _search(
                model,
                selected,
                processor,
                calibration,
                config,
                smoke_args,
                require_shared_raw=False,
            )
            searches[cell] = search
            statuses.append({
                "cell": cell,
                "groups": len(search.audit["groups"]),
                "scales": len(search.scales),
                "cache_hit": hit,
                "search_key": key,
                "raw_key": raw_key,
            })
        cell = "A_VPA__O_VA"
        applied = apply_mbq(model, selected, configs[cell], searches[cell])
        try:
            nll = _evaluate_samples_cached(
                model,
                processor,
                sample,
                variant="factorial_real_smoke",
                config_digest=configs[cell].digest,
                result_dir=smoke_args.output_dir / "sample_cache",
                analysis_ids=set(),
                promotion_ids=set(),
                min_pixels=configs[cell].min_pixels,
                max_pixels=configs[cell].max_pixels,
                resume=True,
            )
        finally:
            applied.restore()
    finally:
        del model, processor
    atomic_json(smoke_args.output_dir / "real_smoke.json", {
        "status": "complete",
        "cells": statuses,
        "nll_rows": len(nll),
        "finite_nll": all(np.isfinite(row["target_nll"]) for row in nll),
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
            raise RuntimeError("factorial preflight requires 80 GiB available memory")
        if disk < 100 * 1024**3:
            raise RuntimeError("factorial preflight requires 100 GiB free disk")
        if active:
            raise RuntimeError(f"other quantization processes are active: {active}")
        for path in (
            Path(args.model_path),
            args.calibration_manifest,
            args.evaluation_manifest,
            args.formal_root,
            args.trimodal_root,
            _raw_cache_dir(args),
        ):
            if not path.exists():
                raise FileNotFoundError(path)
        _nll_samples(args)
        _residual_samples(args)
        tri_audit = json.loads(
            (args.trimodal_root / "w4_nll/run_audit.json").read_text()
        )
        raw_keys = {
            item["raw_key"] for item in tri_audit["variants"]
        }
        if len(raw_keys) != 1:
            raise RuntimeError("tri-modal endpoint does not have one shared raw key")
        raw_key = next(iter(raw_keys))
        raw_path = _raw_cache_dir(args) / f"mbq_stats_{raw_key}.pt"
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        test_environment = dict(os.environ)
        existing = test_environment.get("PYTHONPATH", "")
        test_environment["PYTHONPATH"] = os.pathsep.join(
            value
            for value in (str(args.vlmevalkit_root.resolve()), existing)
            if value
        )
        _run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "experiments/vlm_quant_benchmark/tests",
            ],
            root / "logs/preflight_tests.log",
            environment=test_environment,
        )
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
            "shared_raw_path": str(raw_path.resolve()),
            "shared_raw_sha256": file_digest(raw_path),
            "calibration_manifest_sha256": file_digest(args.calibration_manifest),
            "evaluation_manifest_sha256": file_digest(args.evaluation_manifest),
            "real_smoke": not args.skip_real_smoke,
        })
    elif stage == "search":
        _run_search_stage(args)
    elif stage == "w4_nll":
        _run_nll_stage(args)
    elif stage == "residuals":
        _run_residual_stage(args)
    elif stage == "analysis":
        _run_analysis_stage(args)
    elif stage == "final":
        nll = _read_jsonl(root / "w4_nll/factorial_nll_results.jsonl")
        residual = _read_jsonl(root / "residuals/factorial_residuals.jsonl")
        prediction = _read_jsonl(root / "residuals/prediction_changes.jsonl")
        decision = json.loads(
            (root / "analysis/attribution_decision.json").read_text()
        )
        if len(nll) != 512:
            raise RuntimeError("final factorial validation requires 512 NLL rows")
        expected_residual = 5 * 32 * len(LAYERS) * 3 * 4
        if len(residual) != expected_residual or len(prediction) != 128:
            raise RuntimeError("final factorial residual matrix is incomplete")
        atomic_json(root / "formal_run_audit.json", {
            "status": "complete",
            "nll_rows": len(nll),
            "residual_rows": len(residual),
            "prediction_rows": len(prediction),
            "factorial_cells": CELLS,
            "layers": list(LAYERS),
            "decision": decision,
            "input_digests": {
                "calibration_manifest": file_digest(args.calibration_manifest),
                "evaluation_manifest": file_digest(args.evaluation_manifest),
                "shared_raw_statistics": file_digest(
                    next(_raw_cache_dir(args).glob("mbq_stats_*.pt"))
                ),
            },
        })
        summary = [
            "# MBQ 2×2 因果拆分正式总结",
            "",
            "四格W4A16实验、128样本配对NLL和32样本定向残差均已完成。",
            "",
            f"归因结论：{decision['conclusion']}。",
            "",
            "本实验不包含生成评测，也未改变既有正式结果目录。",
        ]
        (root / "final_summary_zh.md").write_text(
            "\n".join(summary) + "\n"
        )
    else:
        raise ValueError(stage)


def main() -> None:
    args = parse_args()
    if args.threads != 16:
        raise ValueError("formal factorial protocol requires exactly 16 CPU threads")
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
