from __future__ import annotations

from experiments.infinitevl_quant_ffn_bf16.config import (
    CONFIRM_SCOPES,
    EXPECTED_SCOPE_COUNTS,
    FORMATS,
    SCREEN_SCOPES,
)
from experiments.infinitevl_quant_ffn_bf16.run_cpu_protocol import select_candidate


def _comparisons(passing=()):
    return {
        f"{fmt}__{scope}": {"gate": {"passed": f"{fmt}__{scope}" in passing}}
        for fmt in FORMATS for scope in CONFIRM_SCOPES
    }


def test_protocol_matrix_and_scope_counts_are_fixed():
    assert len(FORMATS) == 3
    assert len(SCREEN_SCOPES) == 5
    assert EXPECTED_SCOPE_COUNTS["attn_main"] == 171
    assert EXPECTED_SCOPE_COUNTS["attn_all"] == 225
    assert EXPECTED_SCOPE_COUNTS["ffn"] == 108


def test_candidate_selection_prefers_lowest_bit_full_gdn_then_main():
    result = select_candidate(_comparisons({"w4a16__attn_all", "w3a16__attn_main"}))
    assert result == {"candidate": "w4a16__attn_all", "decision": "GO_ATTN_ALL"}
    result = select_candidate(_comparisons({"w3a16__attn_main", "w8a16__attn_all"}))
    assert result == {"candidate": "w8a16__attn_all", "decision": "GO_ATTN_ALL"}
    result = select_candidate(_comparisons({"w3a16__attn_main"}))
    assert result["candidate"] == "w3a16__attn_main"
    assert "KEEP_GDN_DYNAMICS_REFERENCE" in result["decision"]

