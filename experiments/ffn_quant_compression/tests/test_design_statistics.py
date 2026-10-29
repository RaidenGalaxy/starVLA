import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.ffn_quant_compression import analysis, verification
from experiments.ffn_quant_compression.config import PARAMETERS_PER_LAYER, Config
from experiments.ffn_quant_compression.design import build_design, condition, semantic_layers
from experiments.ffn_quant_compression.statistics import Bootstrap, adjust_holm, average_ranks, spearman_curves


def design():
    return build_design(
        {
            kind: {"ranking_diagnostic8": [{"layer": i, "recovery_absolute": 36 - i} for i in range(36)]}
            for kind in ("infinitevl", "qwen")
        }
    )


def test_locked_masks_counts_and_equal_storage():
    data = design()
    assert json.loads(json.dumps(data)) == data
    for model in data["models"].values():
        assert [len(model[k]) for k in ("qa_order", "main_screen_order", "random_screen_order")] == [10, 72, 12]
        assert len(model["conditions"]) == 84
        for q in range(37):
            left = model["conditions"][model["aliases"]["sensitive_first"][str(q)]]
            right = model["conditions"][model["aliases"]["sensitive_protected"][str(q)]]
            assert left["mixed_weight_bits"] == right["mixed_weight_bits"]
            assert left["q"] == right["q"] == q
            assert semantic_layers(left["spec"]) == tuple(range(q))
            assert semantic_layers(right["spec"]) == tuple(range(36 - q, 36))
            assert left["compression_saving"] == pytest.approx(q / 48)
        assert model["conditions"]["W"]["mixed_weight_bits"] == 4 * 36 * PARAMETERS_PER_LAYER
        assert model["conditions"]["B"]["compression_saving"] == 0


def test_semantic_cache_excludes_oracles_and_mixed_projection_precision():
    from experiments.ffn_quant_rootcause.interventions import Intervention

    assert semantic_layers(Intervention(gamma=0.5).payload()) is None
    assert semantic_layers(Intervention(bits=3).payload()) is None
    assert semantic_layers(Intervention(direction="gaussian").payload()) is None
    assert semantic_layers(Intervention(projection_bits={"1.gate": None}).payload()) is None
    assert semantic_layers(Intervention(layer_bits={1: None}).payload()) == tuple(i for i in range(36) if i != 1)


def test_compression_rejects_duplicate_layers_and_historical_output_overlap(tmp_path):
    with pytest.raises(ValueError):
        condition([1, 1])
    with pytest.raises(ValueError):
        condition([36])
    with pytest.raises(ValueError):
        Config(output_dir=tmp_path, rootcause_output=tmp_path / "history")


def test_tie_ranks_and_constant_correlations():
    actual = average_ranks([[1, 1, 4, 2], [3, 2, 1, 0]])
    np.testing.assert_array_equal(actual, [[1.5, 1.5, 4, 3], [4, 3, 2, 1]])
    rho = spearman_curves([[1, 2, 3], [3, 2, 1], [2, 2, 2]])
    np.testing.assert_allclose(rho[:2], [1, -1])
    assert np.isnan(rho[2])
    assert spearman_curves([[1, 1 + 1e-14, 1]])[0] != spearman_curves([[1, 1 + 1e-14, 1]])[0]


def make_rows(n, level, qa):
    rows = []
    for i in range(n):
        token = 1 + i % 3
        value = 0.4 + (i % 11) * 0.03 + level * (1 + (i % 5) * 0.02)
        score = float((i % 10) / 10 > level * 0.2)
        rows.append(
            {
                "id": f"d{i // (n // 4)}:{i:03d}",
                "dataset": f"d{i // (n // 4)}",
                "target_text": {"tokens": token, "total_nll": value * token, "nll": value},
                "score": score if qa else float("nan"),
                "prediction": str(score),
            }
        )
    return rows


def test_nll_uses_token_weights_and_sign_flip_not_curve_points():
    a = make_rows(8, 0, True)
    a[0]["target_text"] = {"tokens": 100, "total_nll": 1000.0, "nll": 10.0}
    b = Bootstrap(a, count=100)
    point, _ = b.summarize(a)
    expected = sum(r["target_text"]["total_nll"] for r in a) / sum(r["target_text"]["tokens"] for r in a)
    assert point["nll"] == pytest.approx(expected)
    assert point["nll"] != pytest.approx(np.mean([r["target_text"]["nll"] for r in a]))
    wrong = make_rows(8, 1, True)
    with pytest.raises(ValueError, match="token"):
        b.summarize(wrong)
    assert adjust_holm({"a": 0.01, "b": 0.04, "c": 0.03}) == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})


def test_full_analysis_independent_verification_and_plot(tmp_path, monkeypatch):
    data = design()
    raw = {}
    for kind in ("infinitevl", "qwen"):
        model = data["models"][kind]
        for split, n, labels in (("screen32", 32, list(model["conditions"])), ("confirm192", 192, model["qa_order"])):
            cells = {}
            for label in labels:
                record = model["conditions"][label]
                level = sum((36 - i) / 1000 for i in record["quantized_layers"])
                cells[label] = make_rows(n, level, split == "confirm192")
            raw[kind, split] = cells
    monkeypatch.setattr(analysis, "collect", lambda *args: raw)
    config = Config(output_dir=tmp_path, bootstrap_samples=128)
    summary, _ = analysis.analyze(config, {}, {}, data, None)
    assert len(summary["tests"]) == 16 and summary["holm_family_complete"]
    splits = {
        "screen": [SimpleNamespace(id=r["id"]) for r in raw["qwen", "screen32"]["B"]],
        "confirm": [SimpleNamespace(id=r["id"]) for r in raw["qwen", "confirm192"]["B"]],
    }
    monkeypatch.setattr(
        verification, "score_prediction", lambda sample, prediction: SimpleNamespace(score=float(prediction))
    )
    checked = verification.verify(summary, raw, splits, data, count=128)
    assert checked["status"] == "passed"
    summary["verification"] = checked
    from experiments.ffn_quant_compression.reporting import report

    result = report(config, summary)
    assert result["experiment_complete"]
    assert all(
        (tmp_path / "report" / f"{name}.{suffix}").is_file()
        for name in ("qa_vs_compression", "nll_vs_compression")
        for suffix in ("svg", "png")
    )
    assert len((tmp_path / "report/curve_points.csv").read_text().splitlines()) == 209


def test_incomplete_holm_family_is_not_adjusted(tmp_path, monkeypatch):
    data = design()
    raw = {}
    for kind in ("infinitevl", "qwen"):
        raw[kind, "confirm192"] = {"B": make_rows(192, 0, True)}
        raw[kind, "screen32"] = {}
    monkeypatch.setattr(analysis, "collect", lambda *args: raw)
    result, _ = analysis.analyze(Config(output_dir=tmp_path, bootstrap_samples=32), {}, {}, data, None, partial=True)
    assert not result["holm_family_complete"] and not result["tests"]
