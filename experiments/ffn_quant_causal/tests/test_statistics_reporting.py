from __future__ import annotations

import json
import math

import pytest

from experiments.ffn_quant_causal.reporting import render_report_zh, write_report
from experiments.ffn_quant_causal.report_stage import (
    build_evidence_index,
    stage_status,
)
from experiments.ffn_quant_causal.statistics import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    factorial_interaction_from_progress,
    paired_bootstrap_from_progress,
    stratified_difference_in_differences,
    stratified_factorial_interaction,
    stratified_paired_bootstrap,
)
from experiments.infinitevl_quant.progress import StageProgress
from experiments.infinitevl_quant_ffn_bf16_4x64.scoring import SCORER_ID


def _rows(score: float, nll: float, *, scorer: str = SCORER_ID) -> list[dict]:
    rows = []
    for dataset in ("DocVQA_VAL", "MMMU_DEV_VAL"):
        for index, tokens in enumerate((1, 3, 2)):
            rows.append({
                "id": f"{dataset}:{index}",
                "dataset": dataset,
                "score": score,
                "scoring": {"scorer": scorer, "rule": "test", "extracted": "x"},
                "target_text": {
                    "tokens": tokens,
                    "total_nll": nll * tokens,
                    "nll": nll,
                },
            })
    return rows


def test_paired_bootstrap_is_stratified_paired_and_deterministic() -> None:
    baseline = _rows(0.50, 1.00)
    candidate = _rows(0.60, 1.20)
    first = stratified_paired_bootstrap(baseline, candidate, samples=250, seed=17)
    second = stratified_paired_bootstrap(baseline, candidate, samples=250, seed=17)
    assert first == second
    assert DEFAULT_BOOTSTRAP_SAMPLES == 10_000
    assert first["datasets"] == {"DocVQA_VAL": 3, "MMMU_DEV_VAL": 3}
    assert math.isclose(first["point"]["qa_macro_delta_points"], 10.0)
    assert math.isclose(first["point"]["target_nll_relative_change"], 0.2)
    assert first["qa_macro_delta_points_ci95"] == pytest.approx([10.0, 10.0])
    assert first["target_nll_relative_change_ci95"] == pytest.approx([0.2, 0.2])


def test_token_weighted_nll_is_not_an_unweighted_sample_mean() -> None:
    baseline = _rows(0.5, 1.0)
    candidate = _rows(0.5, 1.0)
    # One- and two-token rows get NLL 2; the three-token row gets NLL 4.
    # Repeated in both datasets, the pooled value is (2 + 12 + 4) / 6 = 3.
    for row in candidate:
        row["target_text"]["nll"] = 4.0 if row["target_text"]["tokens"] == 3 else 2.0
        row["target_text"]["total_nll"] = (
            row["target_text"]["nll"] * row["target_text"]["tokens"]
        )
    result = stratified_paired_bootstrap(baseline, candidate, samples=50, seed=1)
    assert math.isclose(result["point"]["candidate_target_nll"], 3.0)
    assert math.isclose(result["point"]["target_nll_relative_change"], 2.0)


def test_difference_in_differences_uses_the_same_sample_draw() -> None:
    qwen_baseline = _rows(0.50, 1.0)
    qwen_candidate = _rows(0.52, 1.05)
    # Cross-model target tokenization may differ; only baseline/candidate
    # token counts within each model must match.
    for baseline_row, candidate_row in zip(qwen_baseline, qwen_candidate, strict=True):
        tokens = 2 * baseline_row["target_text"]["tokens"]
        baseline_row["target_text"].update(tokens=tokens, total_nll=1.0 * tokens)
        candidate_row["target_text"].update(tokens=tokens, total_nll=1.05 * tokens)
    result = stratified_difference_in_differences(
        _rows(0.50, 1.0),
        _rows(0.60, 1.2),
        qwen_baseline,
        qwen_candidate,
        samples=200,
        seed=23,
    )
    point = result["point"]
    assert math.isclose(point["qa_macro_did_points"], 8.0)
    assert math.isclose(point["target_nll_relative_change_did"], 0.15)
    assert result["qa_macro_did_points_ci95"] == pytest.approx([8.0, 8.0])
    assert result["target_nll_relative_change_did_ci95"] == pytest.approx([0.15, 0.15])


def test_w4_factorial_reports_additive_and_multiplicative_interactions() -> None:
    result = stratified_factorial_interaction(
        _rows(0.50, 1.0),
        _rows(0.40, 1.5),
        _rows(0.48, 1.1),
        _rows(0.30, 2.0),
        samples=200,
        seed=29,
    )
    point = result["point"]
    assert math.isclose(point["qa_macro_additive_interaction_points"], -8.0)
    assert math.isclose(point["target_nll_additive_interaction"], 0.4)
    assert math.isclose(
        point["target_nll_multiplicative_interaction"], 2.0 / (1.5 * 1.1) - 1.0,
    )


def test_progress_wrappers_read_exact_stage_rows(tmp_path) -> None:
    progress = StageProgress(tmp_path, "rtn", config_digest="cfg", inputs={"data": "x"})
    variants = {
        "bf16": _rows(0.50, 1.0),
        "rtn_w4a16_ffn": _rows(0.40, 1.5),
        "rtn_w4a16_attn_main": _rows(0.48, 1.1),
        "rtn_w4a16_decoder_safe": _rows(0.30, 2.0),
    }
    for variant, rows in variants.items():
        for row in rows:
            progress.complete_sample(variant, row["id"], row)
    paired = paired_bootstrap_from_progress(
        progress, "bf16", "rtn_w4a16_ffn", samples=40, seed=3,
    )
    factorial = factorial_interaction_from_progress(progress, samples=40, seed=3)
    assert paired["paired_samples"] == 6
    assert factorial["paired_samples"] == 6


def test_legacy_or_untagged_qa_scores_are_rejected() -> None:
    with pytest.raises(ValueError, match="legacy"):
        stratified_paired_bootstrap(
            _rows(0.5, 1.0), _rows(0.4, 1.2, scorer="legacy_v1"),
            samples=5,
        )
    untagged = _rows(0.4, 1.2)
    untagged[0]["scoring"] = None
    with pytest.raises(ValueError, match=SCORER_ID):
        stratified_paired_bootstrap(_rows(0.5, 1.0), untagged, samples=5)


def test_report_separates_observation_uncertainty_and_not_run(tmp_path) -> None:
    paired = stratified_paired_bootstrap(
        _rows(0.5, 1.0), _rows(0.4, 1.2), samples=25, seed=7,
    )
    did = stratified_difference_in_differences(
        _rows(0.5, 1.0), _rows(0.4, 1.2),
        _rows(0.5, 1.0), _rows(0.48, 1.05),
        samples=25, seed=7,
    )
    factorial = stratified_factorial_interaction(
        _rows(0.5, 1.0), _rows(0.4, 1.5),
        _rows(0.48, 1.1), _rows(0.3, 2.0),
        samples=25, seed=7,
    )
    analyses = {
        "paired": {"InfiniteVL W4 FFN": paired},
        "difference_in_differences": {"W4 FFN": did},
        "factorial": {"InfiniteVL W4": factorial},
        "not_run": ["协议内留出复核尚未完成"],
    }
    statuses = {
        "audit": "completed",
        "rtn": "completed",
        "trace": {"status": "running", "detail": "已完成 2/4 数据集"},
    }
    report = render_report_zh(analyses, stage_status=statuses)
    for label in (
        "cpu_reference_backend", "fake_quantized", "ffn_only", "matched",
        "within_protocol_holdout192", "historically_reused_evaluation",
        "no_speed_or_memory_claim",
    ):
        assert f"`{label}`" in report
    assert "## 已观察结果" in report
    assert "## 统计不确定性" in report
    assert "## 尚未执行或尚未提供" in report
    assert "协议内留出复核尚未完成" in report
    assert "不是历史上未见的新测试集" in report
    assert "独立 confirm" not in report
    assert SCORER_ID in report

    summary = write_report(tmp_path / "report", analyses, stage_status=statuses)
    assert summary["scorer"] == SCORER_ID
    assert (tmp_path / "report" / "report_summary.json").is_file()
    assert (tmp_path / "report" / "report_zh.md").read_text() == report


def test_stage_status_requires_exact_identity_marker_and_required_files(tmp_path) -> None:
    inputs = {"implementation": "digest"}
    progress = StageProgress(tmp_path, "rtn", config_digest="cfg", inputs=inputs)
    (progress.root / "summary.json").write_text("{}")
    progress.mark_complete(("summary.json",))

    statuses = stage_status(tmp_path, config_digest="cfg", input_digests=inputs)
    assert statuses["rtn"]["status"] == "completed"

    (progress.root / "summary.json").unlink()
    statuses = stage_status(tmp_path, config_digest="cfg", input_digests=inputs)
    assert statuses["rtn"]["status"] == "identity_or_artifact_mismatch"

    (progress.root / "summary.json").write_text("{}")
    identity_path = progress.root / "identity.json"
    identity = json.loads(identity_path.read_text())
    identity["config_digest"] = "other"
    identity_path.write_text(json.dumps(identity))
    statuses = stage_status(tmp_path, config_digest="cfg", input_digests=inputs)
    assert statuses["rtn"]["status"] == "identity_or_artifact_mismatch"


def test_stage_status_rejects_unsafe_required_path(tmp_path) -> None:
    inputs = {"implementation": "digest"}
    progress = StageProgress(tmp_path, "trace", config_digest="cfg", inputs=inputs)
    marker = progress.identity | {"required": ["../foreign.json"]}
    (progress.root / ".complete").write_text(json.dumps(marker))
    statuses = stage_status(tmp_path, config_digest="cfg", input_digests=inputs)
    assert statuses["trace"]["status"] == "invalid_marker"


def test_evidence_index_hashes_and_summarizes_validated_artifacts(tmp_path) -> None:
    inputs = {"implementation": "digest"}
    audit = StageProgress(tmp_path, "audit", config_digest="cfg", inputs=inputs)
    audit_payload = {
        "analysis": "matched_ffn_checkpoint_weight_pair",
        "linears": 108,
        "parameters_per_model": 2_434_793_472,
        "global_cosine_similarity": 0.998,
        "global_relative_l2_infinitevl_minus_qwen": 0.061,
    }
    (audit.root / "ffn_weight_pair.json").write_text(json.dumps(audit_payload))
    audit.mark_complete(("ffn_weight_pair.json",))

    rtn = StageProgress(tmp_path, "rtn", config_digest="cfg", inputs=inputs)
    (rtn.root / "summary.json").write_text(json.dumps({
        "infinitevl__screen32__bf16": {"samples": 32, "target_nll": 0.7},
    }))
    (rtn.root / "static_weight_quantization.json").write_text(json.dumps({
        "analysis": "matched_rtn_static_weight_error", "models": {},
    }))
    rtn.mark_complete(("summary.json", "static_weight_quantization.json"))

    statuses = stage_status(tmp_path, config_digest="cfg", input_digests=inputs)
    evidence = build_evidence_index(tmp_path, statuses)
    weight = evidence["audit"]["artifacts"]["weight_pair"]
    assert len(weight["sha256"]) == 64
    assert weight["summary"]["linears"] == 108
    assert evidence["rtn"]["artifacts"]["variant_metrics"]["summary"][
        "infinitevl__screen32__bf16"
    ]["samples"] == 32

    analyses = {
        "paired": {}, "difference_in_differences": {}, "factorial": {},
        "not_run": ["其余阶段未运行"],
    }
    summary = write_report(
        tmp_path / "rendered", analyses, stage_status=statuses, evidence=evidence,
    )
    assert not summary["protocol_completion"]["all_experimental_stages_validated"]
    assert summary["evidence"]["audit"]["artifacts"]["weight_pair"]["sha256"]
    rendered = (tmp_path / "rendered" / "report_zh.md").read_text()
    assert "## 证据索引" in rendered
    assert "ffn_weight_pair.json" in rendered


def test_trace_evidence_requires_the_actual_sample_rows(tmp_path) -> None:
    inputs = {"implementation": "digest"}
    trace = StageProgress(tmp_path, "trace", config_digest="cfg", inputs=inputs)
    (trace.root / "summary.json").write_text(json.dumps({
        "infinitevl": {"samples": 4}, "qwen": {"samples": 4},
    }))
    trace.mark_complete(("summary.json",))
    statuses = stage_status(tmp_path, config_digest="cfg", input_digests=inputs)
    assert statuses["trace"]["status"] == "completed"
    evidence = build_evidence_index(tmp_path, statuses)
    assert statuses["trace"]["status"] == "invalid_artifact"
    assert evidence["trace"]["status"] == "invalid_artifact"
    assert "trace sample rows" in evidence["trace"]["errors"][0]
