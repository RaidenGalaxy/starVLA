import json

import pytest

from experiments.ffn_quant_rootcause.config import Config
from experiments.ffn_quant_rootcause.evidence import EvidenceProgress
from experiments.ffn_quant_rootcause.reporting import ratio_artifacts, report
from experiments.ffn_quant_rootcause.statistics import holm, paired_effect


def test_resumed_evidence_rejects_changed_input_and_tampering(tmp_path):
    progress = EvidenceProgress(tmp_path, "ratio", config_digest="config", inputs={"audit": "source"})
    unit = {"gamma": 0.5, "input": "image1"}
    progress.commit("variant", "a", unit, {"dataset": "dataset", "value": 3})
    assert progress.completed("variant", "a", unit)["value"] == 3
    progress.commit("states", "b", unit, {"layers": {2: {"energy": 1.0}, 10: {"energy": 2.0}}})
    assert progress.completed("states", "b", unit)["layers"]["10"]["energy"] == 2.0
    with pytest.raises(RuntimeError, match="different intervention"):
        progress.completed("variant", "a", {"gamma": 1})
    path = progress.sample_path("variant", "a")
    row = json.loads(path.read_text())
    row["value"] = 4
    path.write_text(json.dumps(row))
    with pytest.raises(RuntimeError, match="checksum"):
        progress.completed("variant", "a", unit)
    with pytest.raises(RuntimeError, match="different inputs"):
        EvidenceProgress(tmp_path, "ratio", config_digest="changed", inputs={"audit": "source"})


def rows(loss):
    return [
        {
            "id": str(i),
            "dataset": "a" if i < 4 else "b",
            "score": 0.5,
            "target_text": {"tokens": i + 1, "total_nll": (i + 1) * loss, "nll": loss},
        }
        for i in range(8)
    ]


def test_paired_bootstrap_weights_tokens_preserves_ids_and_unclipped_fraction():
    effect = paired_effect(rows(1), rows(2), rows(0.5), samples=500)
    assert effect["point"]["recovery_absolute"] == pytest.approx(1.5)
    assert effect["point"]["explained_fraction"] == pytest.approx(1.5)
    assert effect["ci95"]["recovery_absolute"] == pytest.approx([1.5, 1.5])
    zero = paired_effect(rows(1), rows(1), rows(0.5), samples=100)
    assert zero["point"]["explained_fraction"] is None
    assert zero["undefined_replicates"]["explained_fraction"] == 100
    bad = rows(2)
    bad[0]["id"] = "unmatched"
    with pytest.raises(ValueError, match="sets differ"):
        paired_effect(rows(1), rows(2), bad, samples=10)
    assert holm({"a": 0.01, "b": 0.04, "c": 0.9}) == pytest.approx({"a": 0.03, "b": 0.08, "c": 0.9})
    reversed_effect = paired_effect(rows(1), rows(2), rows(3), samples=1000)
    assert reversed_effect["p_degradation_one_sided"] < 0.02
    assert reversed_effect["p_recovery_one_sided"] == 1


def test_partial_report_never_promotes_unfinished_stage(tmp_path):
    config = Config(output_dir=tmp_path)
    stage = tmp_path / "ratio"
    stage.mkdir()
    (stage / "summary.json").write_text('{"false_result": 1}')
    result = report(config)
    assert result["experiment_complete"] is False
    assert not result["completed_stages"]
    assert "确认阶段尚未完成" in (tmp_path / "report/report_zh.md").read_text()


def test_summary_checksum_is_required(tmp_path):
    progress = EvidenceProgress(tmp_path, "audit", config_digest="config", inputs={"audit": "source"})
    progress.finish({"passed": True})
    assert progress.verified_complete()
    (progress.root / "summary.json").write_text('{"passed": false}')
    with pytest.raises(RuntimeError, match="checksum"):
        progress.verified_complete()


def test_config_caps_each_round_at_120_hours():
    with pytest.raises(ValueError):
        Config(wallclock_hours=121)


def test_historical_outputs_are_protected_before_any_cli_writes(tmp_path):
    parent = tmp_path / "history"
    with pytest.raises(ValueError, match="separate"):
        Config(output_dir=parent / "nested", parent_output=parent)
    with pytest.raises(ValueError, match="separate"):
        Config(output_dir=tmp_path, parent_output=parent)


def test_ratio_export_preserves_groups_trajectories_and_undefined_values(tmp_path):
    config = Config(output_dir=tmp_path)
    (tmp_path / "report").mkdir()
    rows = []
    for model in ("infinitevl", "qwen"):
        for trajectory in ("reference", "quantized"):
            for group in ("all_valid", "visual", "text_prompt", "answer_prediction"):
                for layer in (0, 1):
                    rows.append(
                        {
                            "model_kind": model,
                            "trajectory": trajectory,
                            "layer": layer,
                            "group": group,
                            "sample_medians": {
                                "rho": None if layer == 0 else 1.0,
                                "eta_f": 0.1,
                                "eta_r": 0.2,
                                "eta_y": 0.3,
                            },
                            "pooled_rms": {"residual": 1.0, "ffn": 2.0, "error": 0.2, "merged": 3.0},
                        }
                    )
    ratio_artifacts(config, {"aggregated": rows})
    assert len((tmp_path / "report/ratio_by_layer.csv").read_text().splitlines()) == len(rows) + 1
    for name in ("ratio_by_layer", "ratio_visual", "ratio_text_prompt", "ratio_answer_prediction"):
        assert "<svg" in (tmp_path / "report" / f"{name}.svg").read_text()
