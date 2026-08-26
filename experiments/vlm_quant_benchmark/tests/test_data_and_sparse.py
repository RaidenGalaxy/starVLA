import json

import pytest
import pandas as pd
import torch
from PIL import Image
from torch import nn

from experiments.vlm_quant_benchmark.calibration import ActivationStats
from experiments.vlm_quant_benchmark.config import ExperimentConfig, MethodName
from experiments.vlm_quant_benchmark.data import MultimodalSample, assert_disjoint, load_manifest, write_manifest
from experiments.vlm_quant_benchmark.prepare_vlmeval_screen import (
    _canonical_target,
    _stratified_circular_indices,
)
from experiments.vlm_quant_benchmark.second_wave_methods import (
    SparseCorrectionLinear,
    omniquant_linear,
    squeezellm_linear,
)


def write_sample(root, name, identifier, split, color):
    image = root / f"{name}.png"
    Image.new("RGB", (8, 8), color).save(image)
    manifest = root / f"{name}.jsonl"
    manifest.write_text(json.dumps({
        "id": identifier, "dataset": "toy", "split": split,
        "image": image.name, "prompt": f"What color is sample {name}?", "target": "red",
    }) + "\n")
    return manifest


def test_manifest_and_image_leakage(tmp_path):
    calibration_path = write_sample(tmp_path, "cal", "cal_1", "calibration", "red")
    evaluation_path = write_sample(tmp_path, "eval", "eval_1", "evaluation", "blue")
    calibration, evaluation = load_manifest(calibration_path), load_manifest(evaluation_path)
    assert_disjoint(calibration, evaluation)
    evaluation[0].image.write_bytes(calibration[0].image.read_bytes())
    with pytest.raises(ValueError, match="image leakage"):
        assert_disjoint(calibration, evaluation)


def test_write_manifest_roundtrips_images_relative_to_manifest(tmp_path):
    root = tmp_path / "prepared"
    image = root / "images" / "sample.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), "red").save(image)
    manifest = root / "samples.jsonl"
    write_manifest(manifest, [MultimodalSample(
        id="sample:0", dataset="toy", split="train", image=image,
        prompt="What color?", target="red",
    )])
    raw = json.loads(manifest.read_text())
    assert raw["image"] == "images/sample.png"
    assert raw["images"] == ["images/sample.png"]
    loaded = load_manifest(manifest)
    assert loaded[0].image == image


def test_sparse_correction_linear_matches_dense_weight():
    source = nn.Linear(4, 3, bias=True)
    dense = torch.zeros_like(source.weight)
    rows, columns = torch.meshgrid(torch.arange(3), torch.arange(4), indexing="ij")
    indices = torch.stack((rows.reshape(-1), columns.reshape(-1)))
    values = source.weight.detach().reshape(-1)
    wrapper = SparseCorrectionLinear(source, dense, indices, values)
    value = torch.randn(2, 5, 4)
    torch.testing.assert_close(source(value), wrapper(value), atol=1e-6, rtol=1e-6)


def test_omniquant_accepts_inference_mode_calibration_tensors():
    source = nn.Linear(4, 3, bias=False)
    with torch.inference_mode():
        reservoir = torch.randn(2, 4)
        stats = ActivationStats(
            abs_sum=reservoir.abs().sum(0),
            abs_max=reservoir.abs().amax(0),
            rows=2,
            reservoir=reservoir,
        )
    config = ExperimentConfig(MethodName.OMNIQUANT, omniquant_steps=1)
    wrapper, audit = omniquant_linear(source, stats, config)
    assert audit["steps"] == 1
    assert torch.isfinite(wrapper(torch.randn(2, 4))).all()


def test_omniquant_enables_grad_locally_and_restores_global_state():
    source = nn.Linear(4, 3, bias=False)
    reservoir = torch.randn(2, 4)
    stats = ActivationStats(
        abs_sum=reservoir.abs().sum(0),
        abs_max=reservoir.abs().amax(0),
        rows=2,
        reservoir=reservoir,
    )
    config = ExperimentConfig(MethodName.OMNIQUANT, omniquant_steps=2)
    previous = torch.is_grad_enabled()
    try:
        torch.set_grad_enabled(False)
        wrapper, audit = omniquant_linear(source, stats, config)
        assert not torch.is_grad_enabled()
        assert len(audit["loss_curve"]) == 2
        assert all(torch.isfinite(torch.tensor(audit["loss_curve"])))
        assert torch.isfinite(wrapper(torch.randn(2, 4))).all()
    finally:
        torch.set_grad_enabled(previous)


def test_screen_sampler_keeps_exact_complete_circular_groups():
    rows = []
    for group in range(16):
        for variant in range(4):
            rows.append({
                "index": group * 100 + variant,
                "g_index": group,
                "answer_type": "mcq",
                "category": f"category-{group % 2}",
            })
    data = pd.DataFrame(rows)
    selected = _stratified_circular_indices(data, 64, seed=42)
    assert len(selected) == 64
    selected_groups = set(data.iloc[selected]["g_index"])
    for group in selected_groups:
        assert sum(data.iloc[selected]["g_index"] == group) == 4


def test_screen_target_uses_plain_first_docvqa_answer():
    assert _canonical_target("DocVQA_VAL", "['$2,000', '2000']") == "$2,000"
    assert _canonical_target("DocVQA_VAL", ["primary", "alternative"]) == "primary"


def test_screen_target_uses_first_reference_for_active_vqa_datasets():
    assert _canonical_target("TextVQA_VAL", "['one', '1']") == "one"
    malformed = "[not valid Python"
    assert _canonical_target("DocVQA_VAL", malformed) == malformed


def test_squeezellm_w3_uses_exactly_eight_centroids():
    source = nn.Linear(9, 3, bias=False)
    reservoir = torch.randn(4, 9)
    stats = ActivationStats(
        abs_sum=reservoir.abs().sum(0), abs_max=reservoir.abs().amax(0),
        rows=4, reservoir=reservoir,
    )
    config = ExperimentConfig(
        MethodName.SQUEEZELLM, weight_bits=3, group_size=4,
        scheme_id="squeezellm_w3a16_g4_asym",
    )
    wrapper, audit = squeezellm_linear(source, stats, config)
    assert audit["centroids"] == 8
    assert torch.isfinite(wrapper(torch.randn(2, 9))).all()
