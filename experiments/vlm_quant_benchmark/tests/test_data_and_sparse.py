import json

import pytest
import torch
from PIL import Image
from torch import nn

from experiments.vlm_quant_benchmark.data import assert_disjoint, load_manifest
from experiments.vlm_quant_benchmark.second_wave_methods import SparseCorrectionLinear


def write_sample(root, name, identifier, split, color):
    image = root / f"{name}.png"
    Image.new("RGB", (8, 8), color).save(image)
    manifest = root / f"{name}.jsonl"
    manifest.write_text(json.dumps({
        "id": identifier, "dataset": "toy", "split": split,
        "image": image.name, "prompt": "What color?", "target": "red",
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


def test_sparse_correction_linear_matches_dense_weight():
    source = nn.Linear(4, 3, bias=True)
    dense = torch.zeros_like(source.weight)
    rows, columns = torch.meshgrid(torch.arange(3), torch.arange(4), indexing="ij")
    indices = torch.stack((rows.reshape(-1), columns.reshape(-1)))
    values = source.weight.detach().reshape(-1)
    wrapper = SparseCorrectionLinear(source, dense, indices, values)
    value = torch.randn(2, 5, 4)
    torch.testing.assert_close(source(value), wrapper(value), atol=1e-6, rtol=1e-6)
