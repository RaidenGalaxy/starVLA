from __future__ import annotations

import json
from pathlib import Path

import pytest
from torch import nn

from experiments.ffn_quant_causal.audit import (
    audit_model_identity,
    audit_model_pair,
    build_protocol_identity,
    ensure_protocol_identity,
    protocol_input_digests,
    runtime_ffn_inventory,
    scorer_resource_audit,
    stage_progress,
    static_ffn_inventory,
)
from experiments.ffn_quant_causal.config import (
    EXPECTED_FFN_LINEAR_COUNT,
    MODEL_SPECS,
    CausalConfig,
)
from experiments.ffn_quant_causal.protocol import build_protocol_splits


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Real shapes exercise the strict runtime contract without allocating
        # roughly ten GiB of toy weights in the unit test.
        self.gate_proj = nn.Linear(2048, 11008, bias=False, device="meta")
        self.up_proj = nn.Linear(2048, 11008, bias=False, device="meta")
        self.down_proj = nn.Linear(11008, 2048, bias=False, device="meta")


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _MLP()


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Layer() for _ in range(36)])


class _WrappedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Decoder()


def _fake_source(root: Path, model_type: str, marker: bytes) -> Path:
    root.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps({
        "model_type": model_type,
        "num_hidden_layers": 36,
        "hidden_size": 2048,
        "intermediate_size": 11008,
    }))
    names = {
        f"model.layers.{layer}.mlp.{projection}.weight": "model-00001-of-00001.safetensors"
        for layer in range(36)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": names}))
    (root / "model-00001-of-00001.safetensors").write_bytes(marker)
    return root


def test_real_protocol_splits_are_exact_and_disjoint():
    root = Path("data/vlm_quant_benchmark")
    splits, audit = build_protocol_splits(
        root / "multimodal_calibration/multimodal_calibration_4x32.jsonl",
        root / "top4_protocol_v3/vlm_screen_4x64.jsonl",
    )
    assert {name: len(rows) for name, rows in splits.items()} == {
        "mbq_calibration": 128,
        "vvs": 16,
        "qig": 32,
        "qig_acceptance": 32,
        "screen": 32,
        "diagnostic8": 8,
        "confirm": 192,
        "full256": 256,
        "smoke4": 4,
    }
    assert all(
        value == 0
        for row in audit["zero_overlap"].values()
        for value in row.values()
    )
    assert audit["evaluation_partition"]["covers_full256_by_id"]
    assert audit["evaluation_partition"][
        "qig_acceptance_reserved_for_qig_layer_acceptance"
    ]
    assert audit["evaluation_partition"][
        "qig_acceptance_not_used_by_rtn_or_within_protocol_holdout"
    ]
    assert audit["evaluation_partition"]["smoke4_within_screen"]
    assert audit["evaluation_partition"]["diagnostic8_within_screen"]
    assert audit["holdout_semantics"] == {
        "compatibility_split_key": "confirm",
        "report_label": "within_protocol_holdout192",
        "evaluation_history": "historically_reused_evaluation",
        "historically_unseen": False,
        "interpretation": (
            "disjoint from calibration, acceptance, and screen inside this protocol; "
            "historically reused and not newly unseen across prior experiments"
        ),
    }
    assert set(audit["per_source_or_dataset"]["confirm"].values()) == {48}


@pytest.mark.parametrize("model_key", ("infinitevl", "qwen"))
def test_static_model_identity_and_ffn_inventory(tmp_path, model_key):
    spec = MODEL_SPECS[model_key]
    source = _fake_source(tmp_path / model_key, spec.model_type, model_key.encode())
    inventory = static_ffn_inventory(source, spec)
    assert inventory["linear_count"] == EXPECTED_FFN_LINEAR_COUNT
    assert inventory["by_projection"] == {
        "gate_proj": 36, "up_proj": 36, "down_proj": 36,
    }
    identity = audit_model_identity(source, spec)
    assert identity["weights_fully_hashed"]
    assert len(identity["content_sha256"]) == 64
    assert identity["static_ffn_inventory"]["linear_count"] == 108


def test_runtime_ffn_inventory_is_exact():
    targets, audit = runtime_ffn_inventory(_WrappedModel(), MODEL_SPECS["qwen"])
    assert len(targets) == EXPECTED_FFN_LINEAR_COUNT
    assert audit["decoder_path"] == "model.layers"
    assert audit["by_projection"] == {
        "gate_proj": 36, "up_proj": 36, "down_proj": 36,
    }
    assert targets[0].name == "model.layers.0.mlp.gate_proj"


def test_dual_model_pair_has_content_digest(tmp_path):
    sources = {
        key: _fake_source(tmp_path / key, MODEL_SPECS[key].model_type, key.encode())
        for key in ("infinitevl", "qwen")
    }
    audit = audit_model_pair(sources, MODEL_SPECS)
    assert audit["passed"]
    assert len(audit["pair_digest"]) == 64
    assert (
        audit["models"]["infinitevl"]["content_sha256"]
        != audit["models"]["qwen"]["content_sha256"]
    )


def test_input_identity_binds_deterministic_scorer_resource(tmp_path):
    sources = {
        key: _fake_source(tmp_path / key, MODEL_SPECS[key].model_type, key.encode())
        for key in ("infinitevl", "qwen")
    }
    pair = audit_model_pair(sources, MODEL_SPECS)
    calibration = tmp_path / "calibration.jsonl"
    evaluation = tmp_path / "evaluation.jsonl"
    calibration.write_text("calibration\n")
    evaluation.write_text("evaluation\n")
    inputs = protocol_input_digests(
        pair,
        {"digests": {"screen": "screen-digest"}},
        calibration_manifest=calibration,
        evaluation_manifest=evaluation,
    )
    resource = scorer_resource_audit()["textvqa_annotations"]
    assert inputs["scorer_textvqa_annotations"] == resource["sha256"]
    assert len(inputs["runtime_stack"]) == 64
    assert len(inputs["implementation"]) == 64


def test_stage_progress_and_protocol_identity_are_content_locked(tmp_path):
    config = CausalConfig(output_dir=tmp_path / "output")
    inputs = {
        "model_infinitevl": "a" * 64,
        "model_qwen": "b" * 64,
        "data_splits": "c" * 64,
    }
    identity = build_protocol_identity(config, inputs)
    path = ensure_protocol_identity(config.output_dir, identity)
    assert json.loads(path.read_text()) == identity
    assert ensure_protocol_identity(config.output_dir, identity) == path

    progress = stage_progress(config, "audit", inputs)
    progress.complete_sample("inventory", "infinitevl", {"id": "infinitevl"})
    assert progress.completed_sample("inventory", "infinitevl") == {"id": "infinitevl"}
    with pytest.raises(RuntimeError, match="different inputs/config"):
        stage_progress(config, "audit", inputs | {"data_splits": "changed"})
    with pytest.raises(RuntimeError, match="different causal protocol identity"):
        ensure_protocol_identity(config.output_dir, identity | {"format_version": 2})


def test_cpu_config_rejects_non_cpu(tmp_path):
    with pytest.raises(ValueError, match="CPU-only"):
        CausalConfig(output_dir=tmp_path, device="cuda")


def test_config_records_holdout_history_and_diagnostic8(tmp_path):
    payload = CausalConfig(output_dir=tmp_path).to_dict()
    assert "within_protocol_holdout192" in payload["result_labels"]
    assert "historically_reused_evaluation" in payload["result_labels"]
    assert payload["split_contract"]["diagnostic8"] == 8
    assert payload["split_contract"]["confirm"] == 192
    assert payload["split_contract"]["confirm_report_label"] == (
        "within_protocol_holdout192"
    )
    assert payload["split_contract"]["confirm_evaluation_history"] == (
        "historically_reused_evaluation"
    )
