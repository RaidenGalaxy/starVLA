from experiments.infinitevl_quant_ffn_bf16_4x64.statistics import (
    exact_mcnemar,
    stratified_gate_bootstrap,
)


def _row(identifier, dataset, score, nll):
    return {
        "id": identifier,
        "dataset": dataset,
        "score": score,
        "target_text": {"total_nll": nll, "tokens": 1},
    }


def test_gate_bootstrap_is_deterministic_and_paired():
    datasets = ("DocVQA_VAL", "MMMU_DEV_VAL", "ScienceQA_TEST", "TextVQA_VAL")
    baseline = [_row(f"{dataset}:{i}", dataset, 1.0, 1.0) for dataset in datasets for i in range(3)]
    candidate = [_row(row["id"], row["dataset"], row["score"], 1.01) for row in baseline]
    first = stratified_gate_bootstrap(baseline, candidate, samples=100, seed=7)
    second = stratified_gate_bootstrap(baseline, candidate, samples=100, seed=7)
    assert first == second
    assert first["gate_pass_probability"] == 1.0
    assert first["point"]["macro_delta"] == 0.0


def test_exact_mcnemar_counts_direction():
    baseline = [
        _row("MMMU_DEV_VAL:1", "MMMU_DEV_VAL", 0.0, 1.0),
        _row("MMMU_DEV_VAL:2", "MMMU_DEV_VAL", 1.0, 1.0),
        _row("ScienceQA_TEST:1", "ScienceQA_TEST", 1.0, 1.0),
    ]
    candidate = [
        _row("MMMU_DEV_VAL:1", "MMMU_DEV_VAL", 1.0, 1.0),
        _row("MMMU_DEV_VAL:2", "MMMU_DEV_VAL", 1.0, 1.0),
        _row("ScienceQA_TEST:1", "ScienceQA_TEST", 0.0, 1.0),
    ]
    result = exact_mcnemar(baseline, candidate)
    assert result["MMMU_DEV_VAL"]["improved"] == 1
    assert result["ScienceQA_TEST"]["regressed"] == 1
