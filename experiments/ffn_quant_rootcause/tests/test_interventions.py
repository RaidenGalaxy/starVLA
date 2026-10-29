import pytest
import torch

from experiments.ffn_quant_mechanism.tests.test_shadow import ToyModel, _targets
from experiments.ffn_quant_rootcause.interventions import Intervention, QuantBank, capture_path, intervene
from experiments.infinitevl_quant.config import SCHEMES
from experiments.infinitevl_quant.quantization import quantized_linears


def test_endpoints_mixed_projection_and_exception_restore():
    torch.manual_seed(3)
    model = ToyModel()
    source = model.layers[0].mlp
    bank = QuantBank(model.layers)
    value = torch.randn(1, 4, 128)
    baseline = model(value)
    with intervene(model.layers, bank, Intervention(gamma=0)):
        assert torch.equal(model(value), baseline)
    with quantized_linears(model, _targets(model), SCHEMES["w4a16"]):
        existing = model(value)
    with intervene(model.layers, bank, Intervention()):
        assert torch.equal(model(value), existing)
    spec = Intervention(bits=None, projection_bits={"0.gate": 3})
    with intervene(model.layers, bank, spec):
        actual = model(value)
    gate = bank.linear(0, "gate", 3, value)
    expected = value + source.down_proj(source.act_fn(gate) * source.up_proj(value))
    assert torch.equal(actual, expected)
    hooks = len(model.layers[0].post_attention_layernorm._forward_pre_hooks)
    with pytest.raises(RuntimeError, match="forced"):
        with intervene(model.layers, bank, Intervention()):
            raise RuntimeError("forced")
    assert model.layers[0].mlp is source
    assert len(model.layers[0].post_attention_layernorm._forward_pre_hooks) == hooks
    assert torch.equal(model(value), baseline)


def test_live_observation_direction_energy_path_and_serialization():
    torch.manual_seed(3)
    model = ToyModel()
    bank = QuantBank(model.layers)
    value = torch.randn(1, 3, 128)
    spec = Intervention(direction="gaussian", seed=44, layer_gammas={0: 0.5})
    assert Intervention.from_payload(spec.payload()) == spec
    observations = []

    def observer(layer, residual, x, ref, q, out):
        observations.append((residual, ref, q, out))

    with intervene(model.layers, bank, spec, observer=observer, sample_id="sample"), capture_path(model.layers) as path:
        model(value)
    r, ref, q, out = observations[0]
    expected_energy = (q["down_output"] - ref["down_output"]).double().square().sum(-1) * 0.25
    actual_energy = (out - ref["down_output"]).double().square().sum(-1)
    assert torch.allclose(expected_energy, actual_energy, rtol=1e-5, atol=1e-7)
    assert torch.equal(path[0]["residual"], r)
    assert set(path[0]) == {"residual", "normalized", "ffn", "merged"}


def test_channel_repair_changes_only_chosen_product_channels():
    model = ToyModel()
    bank = QuantBank(model.layers)
    value = torch.randn(1, 2, 128)
    observations = []
    with intervene(
        model.layers, bank, Intervention(channel_repairs={0: (2, 5)}), observer=lambda *args: observations.append(args)
    ):
        model(value)
    _, _, _, ref, candidate, _ = observations[0]
    assert torch.equal(candidate["product"][..., [2, 5]], ref["product"][..., [2, 5]])
