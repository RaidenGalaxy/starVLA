from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from experiments.ffn_quant_generalization.runtime import Engine
from experiments.infinitevl_quant.evaluation import _target_metrics


def fixture():
    logits = torch.randn((1, 11, 1024), generator=torch.Generator().manual_seed(1)) * 10
    positions = [2, 3, 5, 7, 9]
    labels = torch.full((1, 11), -100, dtype=torch.long)
    labels[0, positions] = torch.tensor([5, 10, 30, 100, 900])
    return logits, labels, positions


def engine(logits, labels, positions):
    return SimpleNamespace(
        inputs=lambda sample: (None, {}, labels, positions, None),
        bundle=SimpleNamespace(model=lambda **kwargs: SimpleNamespace(logits=logits)),
    )


@pytest.mark.parametrize("threads", [1, 16])
def test_forward_matches_historical_masked_loss_with_fp32_reduction(threads):
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        logits, labels, positions = fixture()
        expected = _target_metrics(logits, labels)
        fused_total = float(
            F.cross_entropy(logits[0, torch.tensor(positions) - 1], labels[0, positions], reduction="sum")
        )
        # Reproduce the guard failure that stopped the real 7B bridge run.
        assert abs(fused_total - expected["total_nll"]) > 1e-6
        _, actual = Engine.forward(engine(logits, labels, positions), None, None, native=True)
        assert actual["total_nll"] == expected["total_nll"]
        assert actual["tokens"] == expected["tokens"] == 5
        assert actual["positions"] == positions
        assert actual["token_ids"] == labels[0, positions].tolist()
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("positions", [[2, 3, 5, 7], [3, 2, 5, 7, 9], [2, 3, 5, 7, 7, 9]])
def test_forward_rejects_missing_reordered_or_duplicate_target_positions(positions):
    logits, labels, _ = fixture()
    with pytest.raises(RuntimeError, match="positions differ from label mask"):
        Engine.forward(engine(logits, labels, positions), None, None, native=True)


def test_forward_still_rejects_nonfinite_loss():
    logits, labels, positions = fixture()
    logits[0, positions[0] - 1, 0] = torch.nan
    with pytest.raises(FloatingPointError, match="nonfinite target loss"):
        Engine.forward(engine(logits, labels, positions), None, None, native=True)
