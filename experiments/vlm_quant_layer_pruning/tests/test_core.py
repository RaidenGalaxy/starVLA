from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from experiments.vlm_quant_benchmark.data import MultimodalSample
from experiments.vlm_quant_benchmark.selection import SelectedLinear
from experiments.vlm_quant_layer_pruning.core import (
    TeacherReference,
    compare_teacher,
    load_validated_mbq,
    load_validated_mbq_w4,
    temporary_layer_set_skip,
    temporary_generation_token_group_freeze,
    temporary_token_group_freeze,
)
from experiments.vlm_quant_layer_pruning.prepare_splits import select_content_disjoint


class ResidualBlock(nn.Module):
    def forward(self, value):
        return value + 2


class TupleResidualBlock(nn.Module):
    def forward(self, value):
        return value + 2, "cache"


def test_layer_skip_is_reversible_for_tensor_and_tuple_outputs():
    tensor_layer = ResidualBlock()
    tuple_layer = TupleResidualBlock()
    value = torch.tensor([1.0])
    assert tensor_layer(value).item() == 3
    with temporary_layer_set_skip([tensor_layer], [0]):
        assert tensor_layer(value).item() == 1
    assert tensor_layer(value).item() == 3
    with temporary_layer_set_skip([tuple_layer], [0]):
        output = tuple_layer(value)
        assert output[0].item() == 1
        assert output[1] == "cache"


def test_token_group_freeze_only_replaces_selected_rows_and_is_reversible():
    layer = ResidualBlock()
    value = torch.tensor([[[1.0], [2.0], [3.0]]])
    mask = torch.tensor([[True, False, True]])
    with temporary_token_group_freeze([layer], [0], mask):
        output = layer(value)
        assert output[:, 0].item() == 1.0
        assert output[:, 1].item() == 4.0
        assert output[:, 2].item() == 3.0
    assert torch.equal(layer(value), value + 2)


def test_all_token_freeze_matches_whole_layer_identity():
    value = torch.tensor([[[1.0], [2.0]]])
    frozen_layer = ResidualBlock()
    skipped_layer = ResidualBlock()
    with temporary_token_group_freeze(
        [frozen_layer], [0], torch.ones((1, 2), dtype=torch.bool),
    ):
        frozen = frozen_layer(value)
    with temporary_layer_set_skip([skipped_layer], [0]):
        skipped = skipped_layer(value)
    assert torch.equal(frozen, skipped)


def test_token_group_freeze_rejects_invalid_masks():
    layer = ResidualBlock()
    with pytest.raises(ValueError, match="rank-2 boolean"):
        with temporary_token_group_freeze([layer], [0], torch.ones(2)):
            pass


def test_generation_freeze_only_changes_prefill_and_preserves_decode():
    layer = ResidualBlock()
    prefill = torch.tensor([[[1.0], [2.0], [3.0]]])
    decode = torch.tensor([[[4.0]]])
    mask = torch.tensor([[True, False, True]])
    with temporary_generation_token_group_freeze([layer], [0], mask) as audit:
        prefill_output = layer(prefill)
        decode_output = layer(decode)
    assert prefill_output.flatten().tolist() == [1.0, 4.0, 3.0]
    assert decode_output.item() == 6.0
    assert audit == {"prefill_calls": 1, "decode_calls": 1}
    assert torch.equal(layer(prefill), prefill + 2)


def test_generation_freeze_rejects_unexpected_sequence_shape():
    layer = ResidualBlock()
    mask = torch.tensor([[True, False, True]])
    with temporary_generation_token_group_freeze([layer], [0], mask):
        with pytest.raises(ValueError, match="unexpected non-decode shape"):
            layer(torch.ones((1, 2, 1)))
    with pytest.raises(ValueError, match="cannot be empty"):
        with temporary_token_group_freeze(
            [layer], [0], torch.zeros((1, 2), dtype=torch.bool),
        ):
            pass


def test_compare_teacher_separates_token_groups_and_nll():
    masks = {
        "vision": torch.tensor([[True, False, False]]),
        "prompt_language": torch.tensor([[False, True, False]]),
        "answer_predictor": torch.tensor([[False, False, True]]),
    }
    reference = TeacherReference(
        target_nll=1.0,
        final_hidden=torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]),
        answer_logits=torch.tensor([[2.0, 0.0]]),
        masks=masks,
        token_counts={key: 1 for key in masks},
    )
    candidate = TeacherReference(
        target_nll=1.25,
        final_hidden=torch.tensor([[[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]]),
        answer_logits=torch.tensor([[0.0, 2.0]]),
        masks=masks,
        token_counts={key: 1 for key in masks},
    )
    metrics = compare_teacher(candidate, reference)
    assert metrics["delta_target_nll"] == pytest.approx(0.25)
    assert metrics["final_hidden_cosine_drift"]["vision"] == pytest.approx(1.0)
    assert metrics["final_hidden_cosine_drift"]["prompt_language"] == pytest.approx(0.0)
    assert metrics["answer_logit_kl"] > 0


def _sample(identifier: str, question: str, image: str) -> MultimodalSample:
    return MultimodalSample(
        id=identifier,
        dataset="d",
        split="s",
        image=Path(f"/{identifier}.png"),
        prompt=identifier,
        target="x",
        question_sha256=question,
        image_sha256=image,
    )


def test_content_disjoint_partition_keeps_duplicate_components_together():
    samples = [
        _sample("a", "q1", "i1"),
        _sample("b", "q1", "i2"),
        _sample("c", "q3", "i2"),
        _sample("d", "q4", "i4"),
        _sample("e", "q5", "i5"),
    ]
    selected, remaining = select_content_disjoint(samples, 3, salt="test")
    selected_ids = {item.id for item in selected}
    assert selected_ids == {"a", "b", "c"}
    assert {item.question_sha256 for item in selected}.isdisjoint(
        {item.question_sha256 for item in remaining}
    )
    assert {item.image_sha256 for item in selected}.isdisjoint(
        {item.image_sha256 for item in remaining}
    )


def test_mbq_artifact_requires_matching_w4_audit(tmp_path):
    linear = nn.Linear(2, 2, bias=False)
    name = "model.language_model.layers.0.self_attn.q_proj"
    selected = [SelectedLinear(name, linear, "decoder_attention", 4)]
    scale = torch.tensor([1.0, 2.0])
    artifact = tmp_path / "mbq.pt"
    torch.save({"key": "k", "scales": {name: scale}, "audit": {}}, artifact)
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({
        "method": "mbq",
        "weight_bits": 4,
        "modules": {name: {"scale_min": 1.0, "scale_max": 2.0}},
    }))
    search, provenance = load_validated_mbq_w4(artifact, audit, selected)
    assert set(search.scales) == {name}
    assert provenance["weight_bits"] == 4
    bad = json.loads(audit.read_text())
    bad["weight_bits"] = 3
    audit.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="not the validated MBQ W4"):
        load_validated_mbq_w4(artifact, audit, selected)


def test_mbq_artifact_accepts_matching_w3_and_rejects_wrong_bits(tmp_path):
    linear = nn.Linear(2, 2, bias=False)
    name = "model.language_model.layers.0.self_attn.q_proj"
    selected = [SelectedLinear(name, linear, "decoder_attention", 4)]
    scale = torch.tensor([1.0, 2.0])
    artifact = tmp_path / "mbq_w3.pt"
    torch.save({"key": "w3", "scales": {name: scale}, "audit": {}}, artifact)
    audit = tmp_path / "audit_w3.json"
    audit.write_text(json.dumps({
        "method": "mbq",
        "weight_bits": 3,
        "activation_bits": None,
        "scope": "decoder_canonical",
        "modules": {name: {"scale_min": 1.0, "scale_max": 2.0}},
    }))
    search, provenance = load_validated_mbq(
        artifact, audit, selected, weight_bits=3,
    )
    assert set(search.scales) == {name}
    assert provenance["weight_bits"] == 3
    with pytest.raises(ValueError, match="not the validated MBQ W4"):
        load_validated_mbq(artifact, audit, selected, weight_bits=4)
