from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

from experiments.ffn_quant_generalization.analysis import analyze
from experiments.ffn_quant_generalization.config import DATASETS, MODELS, Config
from experiments.ffn_quant_generalization.design import build_design
from experiments.ffn_quant_generalization.reporting import report
from experiments.ffn_quant_generalization.runtime import progress
from experiments.ffn_quant_rootcause.interventions import Intervention
from experiments.infinitevl_quant.progress import atomic_json, json_digest


@pytest.fixture
def synthetic_experiment(tmp_path):
    config = Config(output_dir=tmp_path / "run")
    # Small resampling count exercises every mask/statistic without a long test.
    config = SimpleNamespace(output_dir=config.output_dir, bootstrap_samples=100, digest=config.digest)
    design = json.loads(json.dumps(build_design({k: list(range(v[0])) for k, v in MODELS.items()})))
    audit = {"test": True, "models": {kind: {"content_sha256": kind} for kind in MODELS}}
    samples = [SimpleNamespace(id=f"{ds}:{i}", dataset=ds, target="yes") for ds in DATASETS for i in range(3)]
    splits = {name: samples for name in ("screen32", "confirm128", "qa32")}
    for kind in MODELS:
        model = design["models"][kind]
        for stage, split, keys in (("sweep", "screen32", ("main", "random")), ("confirm", "confirm128", ("confirm",))):
            store = progress(config, stage, audit, design)
            for label in dict.fromkeys(label for key in keys for label in model[key]):
                record = model["conditions"][label]
                # Different layer identities cause a genuine allocation effect at equal q.
                mask_effect = sum(i + 1 for i in record["quantized_layers"]) * 0.0001
                for i, sample in enumerate(samples):
                    loss = [1 + 0.2 * i + mask_effect, 0.5 + 0.1 * i + mask_effect]
                    row = {
                        "dataset": sample.dataset,
                        "condition": record,
                        "input_digest": sample.id,
                        "model_content_sha256": kind,
                        "score": math.nan,
                        "target_text": {
                            "tokens": 2,
                            "total_nll": math.fsum(loss),
                            "token_losses": loss,
                            "token_ids": [1, 2],
                        },
                    }
                    store.commit(
                        f"{kind}__{split}__{label}",
                        sample.id,
                        {
                            "spec": Intervention.from_payload(record["spec"]).payload(),
                            "input_digest": sample.id,
                            "operation": "teacher",
                        },
                        row,
                    )
        qa_store = progress(config, "qa", audit, design)
        for label in model["qa"]:
            for sample in samples:
                qa_store.commit(
                    f"{kind}__qa32__{label}",
                    sample.id,
                    {},
                    {
                        "dataset": sample.dataset,
                        "prediction": "yes",
                        "score": 1.0,
                        "target_text": {"tokens": 1, "total_nll": 1.0},
                        "condition": model["conditions"][label],
                    },
                )
    return config, audit, splits, design


def test_complete_statistics_independent_checks_and_export(synthetic_experiment):
    config, audit, splits, design = synthetic_experiment
    summary = analyze(config, audit, splits, design)
    assert summary["core_complete"] and summary["verification"]["passed"]
    assert len(summary["tests"]) == 36 and summary["holm_family_complete"]
    assert summary["verification"]["qa_rescored"] == 3 * 5 * 6
    for kind, model in summary["models"].items():
        for ds in DATASETS:
            curves = model["cohorts"]["screen32"]["scopes"][ds]["curves"]
            for curve in curves.values():
                assert len(curve["points"]) == MODELS[kind][0] + 1
                assert len(curve["marginal"]) == MODELS[kind][0]
                assert curve["spearman"]["rho"] == pytest.approx(1)
        for value in model["qa"].values():
            assert value["metrics"]["qa_macro"] == 100
    result = report(config, summary, design)
    assert len([name for name in result["artifacts"] if name.endswith(".svg")]) == 4
    assert (config.output_dir / "report/curve_points.csv").exists()


def test_analysis_rejects_missing_condition_instead_of_reporting_complete(synthetic_experiment):
    config, audit, splits, design = synthetic_experiment
    store = progress(config, "sweep", audit, design)
    label = design["models"]["infinitevl"]["main"][2]
    store.sample_path(f"infinitevl__screen32__{label}", splits["screen32"][0].id).unlink()
    with pytest.raises(RuntimeError, match="cannot finalize incomplete"):
        analyze(config, audit, splits, design)
    partial = analyze(config, audit, splits, design, partial=True)
    assert partial["partial"] and not partial["core_complete"]


def test_runtime_unit_uses_integer_layers_after_json_roundtrip(synthetic_experiment):
    config, audit, _splits, design = synthetic_experiment
    model = design["models"]["infinitevl"]
    label = model["aliases"]["sensitive_first"]["18"]
    store = progress(config, "sweep", audit, design)
    row = store.rows(f"infinitevl__screen32__{label}")[0]
    raw_unit = {"spec": model["conditions"][label]["spec"], "input_digest": row["input_digest"], "operation": "teacher"}
    runtime_unit = {**raw_unit, "spec": Intervention.from_payload(raw_unit["spec"]).payload()}
    assert row["provenance"]["unit"] == json_digest(runtime_unit)
    assert json_digest(runtime_unit) != json_digest(raw_unit)


def test_analysis_still_rejects_binding_to_another_mask(synthetic_experiment):
    config, audit, splits, design = synthetic_experiment
    model = design["models"]["infinitevl"]
    label = model["aliases"]["sensitive_first"]["18"]
    store = progress(config, "sweep", audit, design)
    path = store.sample_path(f"infinitevl__screen32__{label}", splits["screen32"][0].id)
    row = json.loads(path.read_text())
    row["provenance"]["unit"] = json_digest(
        {"spec": Intervention(bits=4).payload(), "input_digest": row["input_digest"], "operation": "teacher"}
    )
    row["row_sha256"] = json_digest({k: v for k, v in row.items() if k != "row_sha256"})
    atomic_json(path, row)
    with pytest.raises(RuntimeError, match="different operation or mask"):
        analyze(config, audit, splits, design)
