from pathlib import Path

from experiments.infinitevl_quant_ffn_bf16_4x64.config import EXPECTED_SCOPE_COUNTS
from experiments.infinitevl_quant_ffn_bf16_4x64.data_protocol import build_full_splits
from experiments.infinitevl_quant_ffn_bf16_4x64.run_cpu_protocol import _select_candidate


def test_full_split_counts_and_holdout_disjointness():
    splits, audit = build_full_splits(Path(
        "data/vlm_quant_benchmark/top4_protocol_v3/vlm_screen_4x64.jsonl"
    ))
    assert len(splits["full256"]) == 256
    assert len(splits["legacy128"]) == 128
    assert len(splits["unseen_holdout128"]) == 128
    assert all(value == 0 for value in audit["zero_overlap"]["legacy_vs_unseen"].values())
    assert set(audit["per_dataset"]["full256"].values()) == {64}
    assert set(audit["per_dataset"]["unseen_holdout128"].values()) == {32}


def test_scope_counts_and_candidate_priority():
    assert EXPECTED_SCOPE_COUNTS["attn_main"] == 171
    assert EXPECTED_SCOPE_COUNTS["attn_all"] == 225
    variants = {
        f"{bits}__{scope}": "FAIL"
        for bits in ("w8a16", "w4a16", "w3a16")
        for scope in ("attn_main", "attn_all")
    }
    variants["w4a16__attn_all"] = "ROBUST_USABLE"
    variants["w3a16__attn_main"] = "ROBUST_USABLE"
    assert _select_candidate(variants) == {
        "candidate": "w4a16__attn_all", "decision": "GO_ATTN_ALL",
    }

