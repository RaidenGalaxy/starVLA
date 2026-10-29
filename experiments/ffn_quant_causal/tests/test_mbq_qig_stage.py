from __future__ import annotations

import contextlib
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.ffn_quant_causal import mbq_qig_stage as stage
from experiments.infinitevl_quant.inventory import LinearTarget
from experiments.vlm_quant_benchmark.mbq import MBQSearchResult


def _targets() -> list[LinearTarget]:
    result = []
    for layer in range(36):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            module = nn.Linear(2, 2, bias=False)
            result.append(LinearTarget(
                name=f"model.layers.{layer}.mlp.{projection}",
                module=module,
                scope="decoder_ffn",
                layer=layer,
            ))
    return result


def _splits():
    calibration = [SimpleNamespace(id=f"cal-{index}") for index in range(128)]
    return {
        "mbq_calibration": calibration,
        "vvs": calibration[:16],
        "qig": calibration[:32],
        "qig_acceptance": [
            SimpleNamespace(id=f"accept-{index}") for index in range(32)
        ],
    }


def _settings() -> stage.MatchedSearchSettings:
    return stage.MatchedSearchSettings(min_pixels=1, max_pixels=2)


def test_contract_and_format_resolution_are_explicit():
    audit = stage.validate_search_contract("infinitevl", _targets(), _splits())
    assert audit["target_count"] == 108
    assert audit["split_counts"] == stage.EXPECTED_SPLIT_SIZES

    bits, decision = stage.resolve_weight_bits()
    assert bits == 4
    assert decision["source"] == "preregistered_default_w4"
    bits, decision = stage.resolve_weight_bits({"selected_weight_bits": 3})
    assert bits == 3
    assert decision["source"] == "rtn_selection_record"

    bad = _splits()
    bad["qig_acceptance"][0] = bad["mbq_calibration"][0]
    with pytest.raises(ValueError, match="not independent"):
        stage.validate_search_contract("qwen", _targets(), bad)


def test_private_mask_marker_is_infinitevl_swa_only():
    traces = {
        0: [SimpleNamespace(kwargs={})],
        1: [SimpleNamespace(kwargs={})],
    }
    layers = [
        SimpleNamespace(layer_type="sliding_attention"),
        SimpleNamespace(layer_type="linear_attention"),
    ]
    audit = stage._prepare_trace_backend("infinitevl", traces, layers)
    assert traces[0][0].kwargs[stage.PRE_ROUTED_SLIDING_MASK] is True
    assert stage.PRE_ROUTED_SLIDING_MASK not in traces[1][0].kwargs
    assert audit["private_mask_markers_injected"] == 1

    clean_qwen = {0: [SimpleNamespace(kwargs={"attention_mask": None})]}
    stage._prepare_trace_backend("qwen", clean_qwen, [SimpleNamespace()])
    assert stage.PRE_ROUTED_SLIDING_MASK not in clean_qwen[0][0].kwargs
    contaminated = {
        0: [SimpleNamespace(kwargs={stage.PRE_ROUTED_SLIDING_MASK: True})],
    }
    with pytest.raises(RuntimeError, match="reached a Qwen trace"):
        stage._prepare_trace_backend("qwen", contaminated, [SimpleNamespace()])


def test_qig_backend_context_is_model_aware(monkeypatch):
    state = {"recurrent": False, "calls": []}

    @contextlib.contextmanager
    def recurrent():
        assert not state["recurrent"]
        state["recurrent"] = True
        try:
            yield
        finally:
            state["recurrent"] = False

    def importance(_fp, _quant, _trace, steps, *, iqr_multiplier):
        state["calls"].append((state["recurrent"], steps, iqr_multiplier))
        return {
            "vision": torch.tensor([1.0]),
            "answer_language": torch.tensor([1.0]),
        }

    monkeypatch.setattr(stage, "qig_recurrent_reference", recurrent)
    monkeypatch.setattr(stage, "qig_token_importance", importance)
    stage._backend_qig_importance("infinitevl", object(), object(), object(), 8)
    stage._backend_qig_importance("qwen", object(), object(), object(), 16)
    assert state["calls"] == [(True, 8, 1.5), (False, 16, 1.5)]
    assert state["recurrent"] is False


def test_masked_qig_convergence_excludes_structural_zeros_and_averages_ties():
    assert torch.equal(
        stage._average_ranks(torch.tensor([0.0, 0.0, 1.0, 1.0])),
        torch.tensor([0.5, 0.5, 2.5, 2.5]),
    )
    trace = SimpleNamespace(
        sample_id="sample",
        source="source-a",
        vision_mask=torch.tensor([[True, True, False, False, False, False]]),
        answer_mask=torch.tensor([[False, False, True, True, False, False]]),
    )
    low = {
        "vision": torch.tensor([[0.9, 0.1, 0.0, 0.0, 0.0, 0.0]]),
        "answer_language": torch.tensor([[0.0, 0.0, 0.8, 0.2, 0.0, 0.0]]),
    }
    high = {
        "vision": torch.tensor([[0.1, 0.9, 0.0, 0.0, 0.0, 0.0]]),
        "answer_language": torch.tensor([[0.0, 0.0, 0.2, 0.8, 0.0, 0.0]]),
    }
    low_record = stage._convergence_record(trace, low)
    high_record = stage._convergence_record(trace, high)
    assert low_record["vector"].numel() == 4
    chosen, audit = stage._choose_masked_qig_steps(
        {8: [low_record], 16: [high_record]},
        rank_threshold=0.98,
        jaccard_threshold=0.90,
    )
    assert chosen == 16
    assert not audit["converged"]
    comparison = audit["comparisons"][0]
    assert comparison["structural_zeros_excluded"] is True
    assert comparison["tie_ranking"] == "average"
    assert comparison["aggregation"] == (
        "median_within_source_then_median_across_sources"
    )
    assert comparison["positive_mass_tokens"] == 4
    assert comparison["empty_positive_mass_traces"] == 0
    assert comparison["zero_mass_valid_tokens_excluded"] is True
    assert comparison["median_spearman"] < 0.98
    assert comparison["median_top10_jaccard"] < 0.90


def test_qig_convergence_all_zero_positive_mass_is_not_stability_evidence():
    rows = [{
        "sample_id": "sample",
        "source": "source-a",
        "vector": torch.zeros(6),
    }]
    chosen, audit = stage._choose_masked_qig_steps(
        {8: rows, 16: rows},
        rank_threshold=0.98,
        jaccard_threshold=0.90,
    )
    assert chosen == 16
    assert audit["converged"] is False
    comparison = audit["comparisons"][0]
    assert comparison["valid_tokens"] == 6
    assert comparison["positive_mass_tokens"] == 0
    assert comparison["empty_positive_mass_traces"] == 1
    assert comparison["median_spearman"] == 0.0
    assert comparison["median_top10_jaccard"] == 0.0
    assert comparison["empty_positive_mass_policy"] == (
        "score_zero_not_converged_evidence"
    )


def test_qig_eta_normalizes_by_integrated_steps_and_reserves_block_search():
    projection = stage._qig_budget_projection(
        consumed_seconds=100.0,
        measured_seconds=24.0,
        measured_integrated_steps=24,
        trace_count=2,
        layer_count=3,
        step_grid=(8, 16, 32),
        scale_candidates=20,
        coordinate_sweeps=2,
    )
    assert projection["seconds_per_integrated_step_trace"] == 1.0
    assert projection["ig_upper_seconds"] == 336.0
    assert projection["block_search_units_per_trace_layer"] == 126.0
    assert projection["block_search_reserve_seconds"] == 756.0
    assert projection["projected_total_seconds"] == 1168.0
    assert projection["includes_capture_elapsed"] is True
    assert projection["includes_conservative_block_search_reserve"] is True


def test_frozen_budget_decision_survives_resume_without_reduction():
    cached = {
        "effective_layers": [9, 4, 20],
        "effective_steps": [8, 16, 32],
        "decision_frozen": True,
        "consumed_seconds": 321.5,
        "decision": {"outcome": "full_workload_within_budget"},
    }
    layers, steps, frozen, consumed, decision = stage._restore_budget_decision(
        cached,
        selected_layers=[9, 4, 20],
        configured_steps=(8, 16, 32),
    )
    assert layers == [9, 4, 20]
    assert steps == (8, 16, 32)
    assert frozen is True
    assert consumed == 321.5
    assert decision["outcome"] == "full_workload_within_budget"


def test_vvs_audit_states_one_sided_limitation():
    audit = stage._annotate_vvs_audit({"measurement": {"per_layer": {}}})
    assert audit["score_definition"] == "clip(1 - d_quant / d_fp, 0, 1)"
    assert audit["directionality"] == "one_sided_loss_of_vision_dependence"
    assert audit["measurement_contains_d_fp_and_d_quant"] is True
    assert audit["unselected_layers_are_not_evidence_of_ffn_insensitivity"] is True


def test_qig_budget_is_persisted_and_checked_after_block_search(
    monkeypatch, tmp_path,
):
    clock = {"value": 0.0}

    def now():
        return clock["value"]

    trace = SimpleNamespace(
        sample_id="sample",
        source="source-a",
        hidden=torch.zeros(1, 2, 2),
        kwargs={},
        vision_mask=torch.tensor([[True, False]]),
        answer_mask=torch.tensor([[False, True]]),
    )

    def capture(*_args, **_kwargs):
        clock["value"] += 0.1
        return {0: [trace]}

    def importance(**_kwargs):
        clock["value"] += 0.01
        return {
            "vision": torch.tensor([[1.0, 0.0]]),
            "answer_language": torch.tensor([[0.0, 1.0]]),
        }, False, 0.001

    def search(_fp, modules, scales, *_args, **_kwargs):
        clock["value"] += 10.0
        return {
            item.name: scales[item.name].clone() for item in modules
        }, {"non_regression": True}

    monkeypatch.setattr(stage.time, "perf_counter", now)
    monkeypatch.setattr(stage, "decoder_layers", lambda _model: [nn.Identity() for _ in range(36)])
    monkeypatch.setattr(stage, "capture_block_traces", capture)
    monkeypatch.setattr(stage, "make_quantized_block", lambda *_args, **_kwargs: nn.Identity())
    monkeypatch.setattr(stage, "_importance_cached", importance)
    monkeypatch.setattr(stage, "search_vulnerable_block", search)

    targets = _targets()
    base = MBQSearchResult({
        target.name: torch.ones(target.module.in_features) for target in targets
    }, {})
    settings = stage.MatchedSearchSettings(
        min_pixels=1,
        max_pixels=2,
        budget_hours=0.001,
    )
    scales, audit = stage.run_qig_backend_aware(
        "qwen",
        nn.Module(),
        object(),
        targets,
        [],
        settings,
        base,
        [0],
        cache_root=tmp_path,
        identity="budget-test",
    )
    assert all(torch.equal(scales[name], value) for name, value in base.scales.items())
    assert audit["skipped"] is True
    assert audit["candidate_equals_mbq"] is True
    assert audit["reason"] == "qig_cumulative_budget_exhausted_during_block_search"
    saved = torch.load(
        tmp_path / "budget_decision.pt", map_location="cpu", weights_only=True,
    )
    assert saved["decision_frozen"] is True
    assert saved["skipped"] is True
    assert saved["consumed_seconds"] >= 10.0


def test_resumed_frozen_budget_does_not_reestimate_after_cached_layer(
    monkeypatch, tmp_path,
):
    targets = _targets()
    base_scales = {
        target.name: torch.ones(target.module.in_features) for target in targets
    }
    base = MBQSearchResult(base_scales, {})
    identity = "resume-budget-test"
    settings = _settings()
    stage._atomic_torch(tmp_path / "budget_decision.pt", {
        "format_version": stage.FORMAT_VERSION,
        "identity": identity,
        "kind": "qig_budget_decision",
        "effective_layers": [0, 1],
        "effective_steps": [8, 16, 32],
        "decision_frozen": True,
        "decision": {"outcome": "full_workload_within_budget"},
        "consumed_seconds": 1.0,
        "budget_seconds": 18 * 3600.0,
        "skipped": False,
        "completed": False,
        "reason": "budget_decision_full_and_frozen",
    })
    layer_zero_identity = stage.json_digest({
        "identity": identity,
        "layer": 0,
        "input_scales": stage._scale_digest(base_scales),
        "configured_steps": list(settings.step_grid),
    })
    layer_zero_targets = [target for target in targets if target.layer == 0]
    stage._atomic_torch(tmp_path / "layer_00" / "search.pt", {
        "format_version": stage.FORMAT_VERSION,
        "identity": layer_zero_identity,
        "kind": "qig_layer_search",
        "scales": {
            target.name: base_scales[target.name].clone()
            for target in layer_zero_targets
        },
        "audit": {"synthetic_cached_layer": True},
    })

    traces = {}
    for layer in (0, 1):
        traces[layer] = [SimpleNamespace(
            sample_id=f"sample-{layer}",
            source="source-a",
            hidden=torch.zeros(1, 2, 2),
            kwargs={},
            vision_mask=torch.tensor([[True, False]]),
            answer_mask=torch.tensor([[False, True]]),
        )]
    monkeypatch.setattr(
        stage, "decoder_layers", lambda _model: [nn.Identity() for _ in range(36)],
    )
    monkeypatch.setattr(stage, "capture_block_traces", lambda *_args, **_kwargs: traces)
    monkeypatch.setattr(stage, "make_quantized_block", lambda *_args, **_kwargs: nn.Identity())
    monkeypatch.setattr(stage, "_importance_cached", lambda **_kwargs: ({
        "vision": torch.tensor([[1.0, 0.0]]),
        "answer_language": torch.tensor([[0.0, 1.0]]),
    }, True, 10**9))
    searched = []

    def search(_fp, modules, scales, *_args, **_kwargs):
        searched.append({int(item.name.split(".layers.")[1].split(".")[0]) for item in modules}.pop())
        return {
            item.name: scales[item.name].clone() for item in modules
        }, {"non_regression": True}

    monkeypatch.setattr(stage, "search_vulnerable_block", search)
    _scales, audit = stage.run_qig_backend_aware(
        "qwen",
        nn.Module(),
        object(),
        targets,
        [],
        settings,
        base,
        [0, 1],
        cache_root=tmp_path,
        identity=identity,
    )
    assert searched == [1]
    assert audit["effective_layers"] == [0, 1]
    assert audit["budget_decision_frozen"] is True
    assert "initial_budget_projection" not in audit
    assert audit["layers"]["0"]["cache_hit"] is True


def test_run_mbq_uses_resumable_global_reservoir_and_final_cache(
    monkeypatch, tmp_path,
):
    observed = {"collect": 0, "search": 0}

    def fake_batches(_processor, samples, *, min_pixels, max_pixels):
        assert (min_pixels, max_pixels) == (1, 2)
        return ({"id": sample.id} for sample in samples)

    def fake_collect(_model, selected, batches, **kwargs):
        observed["collect"] += 1
        assert len(selected) == 108
        assert [row["id"] for row in batches] == [
            f"cal-{index}" for index in range(128)
        ]
        assert kwargs["expected_ids"] == [
            f"cal-{index}" for index in range(128)
        ]
        return object()

    def fake_search(raw, selected, scheme, *, scale_grid):
        observed["search"] += 1
        assert raw is not None and scheme.weight_bits == 4
        assert scale_grid == stage.DEFAULT_SCALE_GRID
        return MBQSearchResult({
            item.name: torch.ones(item.module.in_features) for item in selected
        }, {"synthetic_test_double": True})

    monkeypatch.setattr(stage._legacy, "mbq_calibration_batches", fake_batches)
    monkeypatch.setattr(stage, "_collect_mbq_raw_statistics_resumable", fake_collect)
    monkeypatch.setattr(stage, "_matched_ffn_scale_search", fake_search)
    targets = _targets()
    artifact = tmp_path / "mbq.pt"
    result, cache_hit = stage.run_mbq_ffn(
        object(),
        object(),
        targets,
        _splits()["mbq_calibration"],
        _settings(),
        artifact=artifact,
        search_identity="search-identity",
    )
    assert not cache_hit
    assert len(result.scales) == 108
    assert observed == {"collect": 1, "search": 1}
    assert result.audit["per_sample_resume_supported"] is True
    assert result.audit["mathematically_equivalent_to_uninterrupted_collector"] is True
    assert result.audit["non_equivalent_per_sample_concatenation_used"] is False
    status = json.loads((tmp_path / "mbq.pt.resume.json").read_text())
    assert status["status"] == "completed"
    assert status["resume_granularity"] == "one_completed_sample_global_reservoir"

    cached, cache_hit = stage.run_mbq_ffn(
        object(), object(), targets, _splits()["mbq_calibration"], _settings(),
        artifact=artifact, search_identity="search-identity",
    )
    assert cache_hit
    assert observed == {"collect": 1, "search": 1}
    assert set(cached.scales) == set(result.scales)


def test_mbq_reservoir_checkpoint_restores_rng_and_global_state_exactly(tmp_path):
    selected = stage._legacy.selected_linears(_targets()[:3])
    modalities = ("vision", "answer_language")
    original = stage._mbq._ModalReservoir(selected, 3, 17, modalities)
    for item in selected:
        for offset, modality in enumerate(modalities):
            original.add(
                item.name,
                modality,
                torch.arange(10, dtype=torch.float32).reshape(5, 2) + offset,
            )
    down_name = selected[2].name
    path = tmp_path / "resume.pt"
    identity = "resume-identity"
    records = {down_name: [{"source": "s", "vision_sum": 1.0}]}
    counts = {"vision": 5, "language": 2, "samples": 1}
    stage._save_mbq_resume_state(
        path,
        identity=identity,
        reservoir=original,
        gradient_records=records,
        token_counts=counts,
        completed_ids=["sample-0"],
        modality_mode="vision_answer",
        algorithm_version=4,
    )
    restored, restored_records, restored_counts, completed = (
        stage._load_mbq_resume_state(
            path,
            identity=identity,
            expected_ids=["sample-0", "sample-1"],
            selected=selected,
            rows=3,
            seed=17,
            modalities=modalities,
            modality_mode="vision_answer",
            algorithm_version=4,
        )
    )
    assert completed == ["sample-0"]
    assert dict(restored_records) == records
    assert restored_counts == counts
    for item in selected:
        for offset, modality in enumerate(modalities):
            following = torch.arange(8, dtype=torch.float32).reshape(4, 2) + 20 + offset
            original.add(item.name, modality, following)
            restored.add(item.name, modality, following)
    assert original.counts == restored.counts
    for key in original.generators:
        assert torch.equal(
            original.generators[key].get_state(), restored.generators[key].get_state(),
        )
        assert torch.equal(original.values[key], restored.values[key])
        assert torch.equal(original.priorities[key], restored.priorities[key])


def test_vvs_resumes_exact_per_sample_rows(monkeypatch, tmp_path):
    calls = []

    def fake_batches(_processor, samples, *, min_pixels, max_pixels):
        assert (min_pixels, max_pixels) == (1, 2)
        sample, = samples
        return [{"id": str(sample.id), "source": str(sample.dataset)}]

    def fake_measure(_model, _selected, _base, batches, _config):
        batch, = list(batches)
        calls.append(batch["id"])
        scores = {
            layer: [(batch["id"], batch["source"], (layer + 1) / 36.0)]
            for layer in range(36)
        }
        measurement = {
            "per_layer": {
                str(layer): [{
                    "id": batch["id"],
                    "source": batch["source"],
                    "d_fp": 1.0,
                    "d_quant": 1.0 - (layer + 1) / 36.0,
                    "vvs": (layer + 1) / 36.0,
                }]
                for layer in range(36)
            },
            "seconds": 2.0,
            "intervention": "zero_vision_tokens_at_first_decoder_input",
            "measurement": "final_norm_answer_predictor_hidden_mae",
        }
        return scores, measurement

    monkeypatch.setattr(stage._legacy, "mbq_calibration_batches", fake_batches)
    monkeypatch.setattr(stage._legacy, "measure_vulnerability", fake_measure)
    samples = [
        SimpleNamespace(id="a", dataset="source-a"),
        SimpleNamespace(id="b", dataset="source-b"),
    ]
    targets = _targets()
    base = MBQSearchResult({
        target.name: torch.ones(target.module.in_features) for target in targets
    }, {})
    kwargs = dict(
        cache_root=tmp_path,
        identity="vvs-resume-test",
        min_pixels=1,
        max_pixels=2,
        relative_threshold=0.05,
        max_layers=2,
        bootstrap_samples=10,
        seed=42,
    )

    first_selected, first_audit = stage.run_vvs_resumable(
        nn.Module(), object(), targets, samples[:1], _settings().scheme, base,
        **kwargs,
    )
    assert calls == ["a"]
    assert first_selected == [35, 34]
    assert first_audit["sample_cache_hits"] == 0

    selected, audit = stage.run_vvs_resumable(
        nn.Module(), object(), targets, samples, _settings().scheme, base,
        **kwargs,
    )
    assert calls == ["a", "b"]
    assert selected == [35, 34]
    assert audit["sample_cache_hits"] == 1
    assert audit["per_sample_resume_supported"] is True
    assert len(audit["measurement"]["per_layer"]["0"]) == 2
    assert audit["measurement"]["seconds"] == 4.0

    _selected, cached_audit = stage.run_vvs_resumable(
        nn.Module(), object(), targets, samples, _settings().scheme, base,
        **kwargs,
    )
    assert calls == ["a", "b"]
    assert cached_audit["sample_cache_hits"] == 2


def test_pipeline_caches_distinct_scale_states_without_claiming_results(
    monkeypatch, tmp_path,
):
    targets = _targets()
    splits = _splits()
    calls = {"mbq": 0, "vvs": 0}

    def fake_mbq(
        model, processor, incoming_targets, samples, settings, *, artifact,
        search_identity, cooperative_check,
    ):
        calls["mbq"] += 1
        assert len(samples) == 128
        assert search_identity
        assert cooperative_check is None
        scales = {
            target.name: torch.ones(target.module.in_features)
            for target in incoming_targets
        }
        return MBQSearchResult(scales, {"synthetic_test_double": True}), False

    def fake_vvs(
        model, processor, incoming_targets, samples, scheme, base, **kwargs,
    ):
        calls["vvs"] += 1
        assert len(samples) == 16
        return [], {"synthetic_test_double": True, "selected": []}

    monkeypatch.setattr(stage, "run_mbq_ffn", fake_mbq)
    monkeypatch.setattr(stage, "run_vvs_resumable", fake_vvs)
    first = stage.run_matched_mbq_qig(
        "qwen",
        nn.Module(),
        object(),
        targets,
        splits,
        output_dir=tmp_path,
        protocol_identity="test-protocol-identity",
        settings=_settings(),
    )
    assert calls == {"mbq": 1, "vvs": 1}
    assert not first.cache_hit
    assert first.audit["acceptance"]["final_equals_mbq"] is True
    assert set(first.audit["scale_states"]) == {"mbq", "qig_candidate", "accepted"}
    assert first.audit["labels"][-1] == "no_speed_or_memory_claim"

    second = stage.run_matched_mbq_qig(
        "qwen",
        nn.Module(),
        object(),
        targets,
        splits,
        output_dir=tmp_path,
        protocol_identity="test-protocol-identity",
        settings=_settings(),
    )
    assert second.cache_hit
    assert calls == {"mbq": 1, "vvs": 1}
    assert (tmp_path / "qwen" / "w4a16" / ".complete").is_file()
