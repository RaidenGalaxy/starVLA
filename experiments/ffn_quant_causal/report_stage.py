from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from experiments.infinitevl_quant.progress import (
    StageProgress,
    atomic_json,
    file_sha256,
    json_digest,
)

from .config import STAGES, CausalConfig
from .reporting import write_report
from .statistics import (
    stratified_difference_in_differences,
    stratified_factorial_interaction,
    stratified_paired_bootstrap,
)


REPORT_ARTIFACTS = {
    "audit": {
        "weight_pair": "ffn_weight_pair.json",
    },
    "rtn": {
        "variant_metrics": "summary.json",
        "static_weight_quantization": "static_weight_quantization.json",
    },
    "trace": {
        "activation_trace": "summary.json",
    },
    "counterfactual": {
        "offline_replay": "summary.json",
        "model_replays": "model_summaries.json",
        "dynamics": "dynamics_summary.json",
    },
    "mbq_qig": {
        "format_selection": "format_selection.json",
        "search": "summary.json",
    },
    "confirm": {
        "within_protocol_holdout": "summary.json",
        "model_summaries": "model_summaries.json",
    },
}


def _metric_excerpt(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    keys = (
        "samples", "mean_score", "target_nll", "target_nll_sample_mean",
        "target_ppl", "mean_logits_kl", "target_argmax_consistency",
        "truncated_samples", "datasets",
    )
    return {key: value[key] for key in keys if key in value}


def _counterfactual_excerpt(payload: Mapping[str, Any]) -> dict[str, Any]:
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        rows = []
    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        try:
            key = (
                str(row["weight_source"]), str(row["trace_source"]), int(row["bits"]),
            )
            grouped[key].append(float(row["down_nmse"]))
        except (KeyError, TypeError, ValueError):
            continue
    means = {
        f"weights={weight}|trace={trace}|w{bits}": sum(values) / len(values)
        for (weight, trace, bits), values in sorted(grouped.items())
    }
    return {
        "analysis_scope": payload.get("analysis_scope"),
        "end_to_end_model_score": payload.get("end_to_end_model_score"),
        "aggregate_rows": len(rows),
        "mean_down_nmse_by_weight_trace_bits": means,
    }


def _search_excerpt(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for kind, row in sorted(payload.items()):
        if not isinstance(row, Mapping):
            continue
        audit = row.get("search_audit", {})
        if not isinstance(audit, Mapping):
            audit = {}
        vvs = audit.get("vvs", {})
        qig = audit.get("qig", {})
        acceptance = audit.get("acceptance", {})
        result[str(kind)] = {
            "weight_bits": row.get("weight_bits"),
            "cache_hit": row.get("cache_hit"),
            "scale_artifacts": row.get("scale_artifacts", {}),
            "vvs_selected_layers": (
                vvs.get("selected") if isinstance(vvs, Mapping) else None
            ),
            "qig_skipped": qig.get("skipped") if isinstance(qig, Mapping) else None,
            "qig_effective_layers": (
                qig.get("effective_layers") if isinstance(qig, Mapping) else None
            ),
            "accepted_layers": (
                acceptance.get("accepted_layers")
                if isinstance(acceptance, Mapping) else None
            ),
            "final_equals_mbq": (
                acceptance.get("final_equals_mbq")
                if isinstance(acceptance, Mapping) else None
            ),
        }
    return result


def _artifact_excerpt(stage: str, name: str, payload: Any) -> Any:
    if not isinstance(payload, Mapping):
        return {"json_type": type(payload).__name__}
    if stage == "audit":
        keys = (
            "analysis", "compute_dtype", "linears", "parameters_per_model",
            "global_cosine_similarity", "global_relative_l2_infinitevl_minus_qwen",
            "by_projection",
        )
        return {key: payload[key] for key in keys if key in payload}
    if stage == "rtn" and name == "variant_metrics":
        return {
            str(variant): _metric_excerpt(metrics)
            for variant, metrics in sorted(payload.items())
        }
    if stage == "rtn":
        return dict(payload)
    if stage == "trace":
        return dict(payload)
    if stage == "counterfactual" and name == "offline_replay":
        return _counterfactual_excerpt(payload)
    if stage == "counterfactual" and name == "model_replays":
        return {
            str(kind): _counterfactual_excerpt(value)
            for kind, value in sorted(payload.items())
            if isinstance(value, Mapping)
        }
    if stage == "counterfactual":
        return dict(payload)
    if stage == "mbq_qig" and name == "search":
        return _search_excerpt(payload)
    if stage == "mbq_qig":
        return dict(payload)
    if stage == "confirm" and name == "model_summaries":
        return {
            str(kind): {
                "split": value.get("split"),
                "scope": value.get("scope"),
                "metrics": {
                    str(method): _metric_excerpt(metrics)
                    for method, metrics in sorted(value.get("metrics", {}).items())
                },
                "comparisons_to_bf16": value.get("comparisons_to_bf16", {}),
            }
            for kind, value in sorted(payload.items())
            if isinstance(value, Mapping)
        }
    if stage == "confirm":
        return {
            "scheme": payload.get("scheme"),
            "models": {
                str(kind): {
                    "metrics": {
                        str(method): _metric_excerpt(metrics)
                        for method, metrics in sorted(value.get("metrics", {}).items())
                    }
                }
                for kind, value in sorted(payload.get("models", {}).items())
                if isinstance(value, Mapping)
            },
            "cross_model_difference_in_differences": payload.get(
                "cross_model_difference_in_differences", {}
            ),
            "report_label": "within_protocol_holdout192",
            "evaluation_history": "historically_reused_evaluation",
        }
    return dict(payload)


def _trace_rows_artifact(stage_root: Path, output_dir: Path) -> dict[str, Any]:
    """Hash and aggregate the numerical trace rows behind trace/summary.json."""

    file_index = []
    summaries = {}
    total_bytes = 0
    for kind in ("infinitevl", "qwen"):
        directory = stage_root / f"{kind}__trace_smoke4" / "samples"
        paths = sorted(directory.glob("*.json")) if directory.is_dir() else []
        if len(paths) != 4:
            raise ValueError(f"{kind} trace requires four sample rows, found {len(paths)}")
        grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        sample_ids = []
        for path in paths:
            row = json.loads(path.read_text())
            if row.get("model_kind") != kind:
                raise ValueError(f"trace model identity mismatch: {path}")
            layers = row.get("layer_summaries")
            if not isinstance(layers, list) or len(layers) != 36:
                raise ValueError(f"trace row must contain 36 layers: {path}")
            sample_ids.append(str(row.get("id")))
            for layer in layers:
                grouped[int(layer["layer"])].append(layer)
            relative = str(path.relative_to(output_dir))
            digest = file_sha256(path)
            size = path.stat().st_size
            total_bytes += size
            file_index.append({"path": relative, "bytes": size, "sha256": digest})
        if set(grouped) != set(range(36)) or any(len(rows) != 4 for rows in grouped.values()):
            raise ValueError(f"{kind} trace does not form an exact 4x36 matrix")
        layer_rows = []
        for layer, rows in sorted(grouped.items()):
            values: dict[str, Any] = {"layer": layer, "samples": len(rows)}
            for stage_name in ("gate", "up", "product", "down_output"):
                for metric in (
                    "rms", "abs_p999", "kurtosis", "top_0_1pct_energy_fraction",
                ):
                    numbers = [float(row["stages"][stage_name][metric]) for row in rows]
                    values[f"{stage_name}_{metric}"] = sum(numbers) / len(numbers)
            for metric in (
                "ffn_to_residual_rms", "ffn_residual_cosine",
                "merged_rms", "cancellation_ratio",
            ):
                numbers = [float(row["branch"][metric]) for row in rows]
                values[metric] = sum(numbers) / len(numbers)
            layer_rows.append(values)
        summaries[kind] = {
            "samples": len(paths),
            "sample_ids": sorted(sample_ids),
            "layers": layer_rows,
        }
    return {
        "path": str(stage_root / "*__trace_smoke4" / "samples" / "*.json"),
        "relative_path": "trace/*__trace_smoke4/samples/*.json",
        "bytes": total_bytes,
        "sha256": json_digest(file_index),
        "summary": {
            "analysis": "observed_ffn_activation_trace_4x36_per_model",
            "files": len(file_index),
            "file_index": file_index,
            "models": summaries,
        },
    }


def build_evidence_index(
    output_dir: Path,
    statuses: dict[str, dict[str, str]],
) -> dict[str, dict[str, Any]]:
    """Index only artifacts from stages whose identity and marker validated."""

    evidence: dict[str, dict[str, Any]] = {}
    for stage in STAGES:
        status = statuses[stage]["status"]
        record: dict[str, Any] = {
            "status": status,
            "artifacts": {},
        }
        if status == "completed" and stage in REPORT_ARTIFACTS:
            errors = []
            for name, relative in REPORT_ARTIFACTS[stage].items():
                path = output_dir / stage / relative
                try:
                    payload = json.loads(path.read_text())
                    artifact = {
                        "path": str(path),
                        "relative_path": str(path.relative_to(output_dir)),
                        "bytes": path.stat().st_size,
                        "sha256": file_sha256(path),
                        "summary": _artifact_excerpt(stage, name, payload),
                    }
                except (
                    OSError, json.JSONDecodeError, AttributeError,
                    KeyError, TypeError, ValueError,
                ) as error:
                    errors.append(f"{relative}: {type(error).__name__}: {error}")
                    continue
                record["artifacts"][name] = artifact
            if stage == "trace" and not errors:
                try:
                    record["artifacts"]["activation_rows"] = _trace_rows_artifact(
                        output_dir / stage, output_dir,
                    )
                except (
                    OSError, json.JSONDecodeError, AttributeError,
                    KeyError, TypeError, ValueError,
                ) as error:
                    errors.append(
                        f"trace sample rows: {type(error).__name__}: {error}"
                    )
            if errors:
                record.update({"status": "invalid_artifact", "errors": errors})
                statuses[stage] = {
                    "status": "invalid_artifact",
                    "detail": "报告所需 JSON 产物无法读取或汇总",
                }
        evidence[stage] = record
    return evidence


def _paired(
    progress: StageProgress,
    baseline: str,
    candidate: str,
    *,
    samples: int,
    seed: int,
) -> dict:
    return stratified_paired_bootstrap(
        progress.rows(baseline),
        progress.rows(candidate),
        samples=samples,
        seed=seed,
    )


def build_rtn_analyses(
    progress: StageProgress,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    paired = {}
    for kind in ("infinitevl", "qwen"):
        baseline = f"{kind}__screen32__bf16"
        for bits in (8, 4, 3):
            candidate = f"{kind}__screen32__rtn_w{bits}a16_ffn"
            paired[f"{kind} screen32 RTN W{bits}A16 FFN"] = _paired(
                progress, baseline, candidate,
                samples=bootstrap_samples, seed=seed,
            )
        paired[f"{kind} smoke4 RTN W2A16 FFN (exploratory)"] = _paired(
            progress,
            f"{kind}__smoke4__bf16",
            f"{kind}__smoke4__rtn_w2a16_ffn",
            samples=bootstrap_samples,
            seed=seed,
        )
        diagnostic_baseline = f"{kind}__diagnostic8__bf16"
        for bits in (4, 3):
            for label in (
                "all_gate_proj", "all_up_proj", "all_down_proj",
                "layer01_gate_proj", "layer01_ffn",
            ):
                candidate = f"{kind}__diagnostic8__rtn_w{bits}a16_{label}"
                paired[f"{kind} diagnostic8 exploratory RTN W{bits}A16 {label}"] = _paired(
                    progress, diagnostic_baseline, candidate,
                    samples=bootstrap_samples, seed=seed,
                )

    did = {}
    for bits in (8, 4, 3):
        did[f"screen32 RTN W{bits}A16 FFN"] = stratified_difference_in_differences(
            progress.rows("infinitevl__screen32__bf16"),
            progress.rows(f"infinitevl__screen32__rtn_w{bits}a16_ffn"),
            progress.rows("qwen__screen32__bf16"),
            progress.rows(f"qwen__screen32__rtn_w{bits}a16_ffn"),
            samples=bootstrap_samples,
            seed=seed,
        )
    factorial = {
        "InfiniteVL screen32 RTN W4A16": stratified_factorial_interaction(
            progress.rows("infinitevl__screen32__bf16"),
            progress.rows("infinitevl__screen32__rtn_w4a16_ffn"),
            progress.rows("infinitevl__screen32__rtn_w4a16_attn_main"),
            progress.rows("infinitevl__screen32__rtn_w4a16_decoder_safe"),
            samples=bootstrap_samples,
            seed=seed,
        ),
    }
    return {
        "paired": paired,
        "difference_in_differences": did,
        "factorial": factorial,
        "not_run": [],
    }


def add_confirm_analyses(
    analyses: dict,
    progress: StageProgress,
    *,
    weight_bits: int,
    bootstrap_samples: int,
    seed: int,
) -> None:
    for kind in ("infinitevl", "qwen"):
        baseline = f"{kind}__confirm192__bf16"
        for method in ("rtn", "mbq", "qig"):
            candidate = f"{kind}__confirm192__{method}_w{weight_bits}a16_ffn"
            analyses["paired"][
                f"{kind} within_protocol_holdout192 {method.upper()} "
                f"W{weight_bits}A16 FFN (historically reused evaluation)"
            ] = _paired(
                progress,
                baseline,
                candidate,
                samples=bootstrap_samples,
                seed=seed,
            )
    for method in ("rtn", "mbq", "qig"):
        analyses["difference_in_differences"][
            f"within_protocol_holdout192 {method.upper()} W{weight_bits}A16 FFN "
            "(historically reused evaluation)"
        ] = stratified_difference_in_differences(
            progress.rows("infinitevl__confirm192__bf16"),
            progress.rows(
                f"infinitevl__confirm192__{method}_w{weight_bits}a16_ffn"
            ),
            progress.rows("qwen__confirm192__bf16"),
            progress.rows(f"qwen__confirm192__{method}_w{weight_bits}a16_ffn"),
            samples=bootstrap_samples,
            seed=seed,
        )


def stage_status(
    output_dir: Path,
    *,
    config_digest: str,
    input_digests: dict,
) -> dict[str, dict[str, str]]:
    result = {}
    for stage in STAGES:
        stage_root = output_dir / stage
        marker = stage_root / ".complete"
        if marker.is_file():
            try:
                payload = json.loads(marker.read_text())
                if set(payload) != {
                    "format_version", "stage", "config_digest",
                    "input_digests", "required",
                }:
                    raise TypeError("completion marker keys do not match schema")
                required = payload["required"]
                if not required or not isinstance(required, list) or not all(
                    isinstance(item, str) for item in required
                ):
                    raise TypeError("required must be a non-empty list of paths")
                if len(required) != len(set(required)):
                    raise TypeError("required paths must be unique")
                required_paths = [Path(item) for item in required]
                if any(
                    item.is_absolute() or not item.parts or ".." in item.parts
                    for item in required_paths
                ):
                    raise TypeError("required paths must stay within the stage directory")
            except (OSError, json.JSONDecodeError, KeyError, TypeError):
                result[stage] = {
                    "status": "invalid_marker",
                    "detail": "存在 .complete，但标记结构无效",
                }
            else:
                expected = {
                    "format_version": 1,
                    "stage": stage,
                    "config_digest": config_digest,
                    "input_digests": input_digests,
                }
                identity_ok = all(payload.get(key) == value for key, value in expected.items())
                identity_path = stage_root / "identity.json"
                try:
                    identity_record = json.loads(identity_path.read_text())
                except (OSError, json.JSONDecodeError):
                    identity_record = None
                identity_file_ok = identity_record == expected
                files_ok = all((stage_root / item).is_file() for item in required_paths)
                if identity_ok and identity_file_ok and files_ok:
                    result[stage] = {
                        "status": "completed",
                        "detail": "身份标记与全部必需产物均已验证",
                    }
                else:
                    result[stage] = {
                        "status": "identity_or_artifact_mismatch",
                        "detail": "标记/identity.json 身份不匹配或必需文件缺失",
                    }
        elif (output_dir / stage).is_dir():
            result[stage] = {
                "status": "partial_or_running",
                "detail": "阶段目录存在，但没有 .complete",
            }
        else:
            result[stage] = {"status": "not_run", "detail": "尚无阶段目录"}
    return result


def run_report(
    config: CausalConfig,
    report_progress: StageProgress,
    rtn_progress: StageProgress,
) -> dict:
    statuses = stage_status(
        config.output_dir,
        config_digest=report_progress.config_digest,
        input_digests=report_progress.inputs,
    )
    evidence = build_evidence_index(config.output_dir, statuses)
    if statuses["rtn"]["status"] != "completed":
        raise RuntimeError("RTN stage identity/artifacts must validate before report")
    analyses = build_rtn_analyses(
        rtn_progress,
        bootstrap_samples=config.bootstrap_samples,
        seed=config.seed,
    )
    if statuses["confirm"]["status"] == "completed":
        add_confirm_analyses(
            analyses,
            StageProgress(
                config.output_dir,
                "confirm",
                config_digest=report_progress.config_digest,
                inputs=report_progress.inputs,
            ),
            weight_bits=config.search_weight_bits,
            bootstrap_samples=config.bootstrap_samples,
            seed=config.seed,
        )
    analyses["not_run"] = [
        f"{stage} 未形成可验证证据（{statuses[stage]['status']}）"
        for stage in STAGES[:-1]
        if statuses[stage]["status"] != "completed"
    ]
    write_report(
        report_progress.root,
        analyses,
        stage_status=statuses,
        evidence=evidence,
    )
    atomic_json(report_progress.root / "analyses.json", analyses)
    atomic_json(report_progress.root / "evidence_index.json", evidence)
    report_progress.mark_complete((
        "analyses.json", "evidence_index.json", "report_summary.json", "report_zh.md",
    ))
    # Refresh the rendered status after the completion marker exists.  If the
    # process dies before this point, the marker still truthfully distinguishes
    # a complete machine-readable stage from a partial render.
    final_statuses = stage_status(
        config.output_dir,
        config_digest=report_progress.config_digest,
        input_digests=report_progress.inputs,
    )
    final_evidence = build_evidence_index(config.output_dir, final_statuses)
    atomic_json(report_progress.root / "evidence_index.json", final_evidence)
    return write_report(
        report_progress.root,
        analyses,
        stage_status=final_statuses,
        evidence=final_evidence,
    )
