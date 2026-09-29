from __future__ import annotations

from pathlib import Path

from torch import nn

from experiments.infinitevl_quant.data_protocol import build_splits, sample_hashes
from experiments.infinitevl_quant.inventory import build_inventory, select_scope


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 8)
        self.up_proj = nn.Linear(4, 8)
        self.down_proj = nn.Linear(8, 4)


class _SWA(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.k_proj = nn.Linear(4, 2)
        self.v_proj = nn.Linear(4, 2)
        self.o_proj = nn.Linear(4, 4)


class _GDN(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("q_proj", "k_proj", "v_proj", "g_proj", "o_proj", "a_proj", "b_proj"):
            setattr(self, name, nn.Linear(4, 4))


class _Layer(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type
        self.self_attn = _SWA() if layer_type == "sliding_attention" else _GDN()
        self.mlp = _MLP()


class _Language(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            _Layer("sliding_attention" if index % 4 == 0 else "linear_attention")
            for index in range(36)
        ])


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = _Language()


def test_inventory_exact_counts():
    targets, audit = build_inventory(_Model())
    assert audit["counts"] == {
        "decoder_ffn": 108, "swa_attention": 36, "gdn_main": 135,
        "gdn_dynamics": 54, "decoder_total": 333, "decoder_safe": 279,
    }
    assert len(select_scope(targets, "decoder_safe")) == 279
    assert len(select_scope(targets, "decoder_all")) == 333
    assert len(select_scope(targets, "attn_main")) == 171
    assert len(select_scope(targets, "attn_all")) == 225


def test_real_protocol_split_counts_and_hash_disjointness():
    root = Path("data/vlm_quant_benchmark")
    splits = build_splits(
        root / "multimodal_calibration/multimodal_calibration_4x32.jsonl",
        root / "top4_protocol_v3/vlm_screen_4x64.jsonl",
    )
    assert {name: len(value) for name, value in splits.items()} == {
        "mbq": 16, "vvs": 4, "qig": 8, "acceptance": 8,
        "smoke": 4, "screen": 32, "confirm": 32,
    }
    primary = ("mbq", "acceptance", "screen", "confirm")
    hashes = {name: {sample_hashes(sample) for sample in splits[name]} for name in primary}
    for index, left in enumerate(primary):
        for right in primary[index + 1:]:
            assert not hashes[left] & hashes[right]
    assert {sample.id for sample in splits["vvs"]} <= {sample.id for sample in splits["mbq"]}
    assert {sample.id for sample in splits["qig"]} <= {sample.id for sample in splits["mbq"]}
