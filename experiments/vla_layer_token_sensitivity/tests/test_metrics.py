import numpy as np
import pytest

from experiments.vla_layer_token_sensitivity.metrics import (
    action_error_by_phase,
    cosine_drift_by_group,
    cvar,
    linear_cka,
    non_additive_interaction,
    rank_stability,
    topk_overlap,
)


def test_grouped_cosine_exposes_sparse_action_token_change() -> None:
    reference = np.ones((1, 4, 2), dtype=np.float64)
    perturbed = reference.copy()
    perturbed[0, 3] = -reference[0, 3]
    groups = np.array([["vision", "vision", "vision", "action"]])

    drift = cosine_drift_by_group(reference, perturbed, groups)

    assert drift["vision"] == pytest.approx(0.0)
    assert drift["action"] == pytest.approx(2.0)
    assert np.mean(list(drift.values())) == pytest.approx(1.0)
    # A token-count-weighted global mean is much smaller and illustrates the
    # dilution failure that the group-balanced metric is designed to expose.
    assert cosine_drift_by_group(reference, perturbed, groups, group_names=["vision"])[
        "vision"
    ] == pytest.approx(0.0, abs=1e-12)


def test_linear_cka_is_invariant_to_isotropic_scaling() -> None:
    reference = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    assert linear_cka(reference, 3.0 * reference) == pytest.approx(1.0)


def test_linear_cka_gram_path_matches_feature_path() -> None:
    rng = np.random.default_rng(3)
    reference = rng.normal(size=(4, 12))
    perturbed = reference + 0.1 * rng.normal(size=(4, 12))
    centered_reference = reference - reference.mean(axis=0, keepdims=True)
    centered_perturbed = perturbed - perturbed.mean(axis=0, keepdims=True)
    expected = np.linalg.norm(centered_reference.T @ centered_perturbed, ord="fro") ** 2
    expected /= np.linalg.norm(centered_reference.T @ centered_reference, ord="fro")
    expected /= np.linalg.norm(centered_perturbed.T @ centered_perturbed, ord="fro")

    assert linear_cka(reference, perturbed) == pytest.approx(expected)


def test_action_phase_summary_reports_tail_risk() -> None:
    reference = np.zeros((4, 2))
    perturbed = np.array([[0.1, 0.1], [0.2, 0.2], [1.0, 1.0], [3.0, 3.0]])
    phases = np.array(["approach", "approach", "contact", "contact"])

    summary = action_error_by_phase(reference, perturbed, phases, quantile=0.5)

    assert summary["approach"]["mean"] == pytest.approx(0.025)
    assert summary["contact"]["cvar"] == pytest.approx(9.0)
    assert cvar([1.0, 2.0, 10.0], 0.8) == pytest.approx(10.0)


def test_ranking_and_interaction_metrics() -> None:
    assert rank_stability([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert topk_overlap([1, 2, 9, 10], [1, 2, 8, 9], 2) == pytest.approx(1.0)
    interaction = non_additive_interaction([0.9, 0.4], [0.2, 0.1], [0.3, 0.2])
    np.testing.assert_allclose(interaction, [0.4, 0.1])
