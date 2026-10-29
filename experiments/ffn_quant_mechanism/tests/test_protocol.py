from __future__ import annotations

from pathlib import Path

import pytest

from experiments.infinitevl_quant.progress import atomic_json

from ..config import MechanismConfig
from ..protocol import (
    PARENT_REQUIRED,
    audited_splits,
    create_audit,
    load_audit,
    parent_evidence_audit,
    stage_progress,
)


def _fake_parent(root: Path, split_digests: dict[str, str]) -> Path:
    for relative in PARENT_REQUIRED:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("payload\n")
    atomic_json(root / "protocol_identity.json", {"protocol": "ffn_quant_causal"})
    atomic_json(
        root / "audit/models.json",
        {
            "models": {
                kind: {
                    "passed": True,
                    "content_sha256": character * 64,
                    "source": f"/{kind}",
                    "model": {"key": kind},
                }
                for kind, character in (("infinitevl", "a"), ("qwen", "b"))
            },
        },
    )
    atomic_json(root / "audit/splits.json", {"digests": split_digests})
    for kind in ("infinitevl", "qwen"):
        reference = root / "confirm/reference_artifacts" / kind
        reference.mkdir(parents=True, exist_ok=True)
        for index in range(192):
            (reference / f"{index:03d}.pt").write_bytes(b"reference")
        for suffix in ("bf16", "rtn_w4a16_ffn"):
            samples = root / "confirm" / f"{kind}__confirm192__{suffix}" / "samples"
            samples.mkdir(parents=True, exist_ok=True)
            for index in range(192):
                (samples / f"{index:03d}.json").write_text("{}\n")
    return root


def test_split_contract_and_audited_parent_cache_are_content_locked(tmp_path) -> None:
    base = Path("data/vlm_quant_benchmark")
    config = MechanismConfig(
        output_dir=tmp_path / "output",
        parent_output=tmp_path / "parent",
        calibration_manifest=base / "multimodal_calibration/multimodal_calibration_4x32.jsonl",
        evaluation_manifest=base / "top4_protocol_v3/vlm_screen_4x64.jsonl",
        bootstrap_samples=10,
    )
    splits, split_audit = audited_splits(config)
    assert {
        name: len(splits[name])
        for name in (
            "qig",
            "diagnostic8",
            "screen",
            "confirm",
            "qig_acceptance",
        )
    } == {
        "qig": 32,
        "diagnostic8": 8,
        "screen": 32,
        "confirm": 192,
        "qig_acceptance": 32,
    }
    parent = _fake_parent(tmp_path / "parent", split_audit["digests"])
    parent_audit = parent_evidence_audit(parent)
    assert parent_audit["read_only_contract"]
    assert parent_audit["reused_artifact_files"] == 1152
    audit = create_audit(config)
    assert audit["splits"]["zero_overlap"]["screen_vs_confirm"] == {
        "id": 0,
        "image": 0,
        "question": 0,
    }
    progress = stage_progress(config, "audit")
    progress.mark_complete(("audit.json", "manifests"))
    assert load_audit(config)["config_digest"] == config.digest()

    reused = parent / "confirm/infinitevl__confirm192__bf16/samples/000.json"
    reused.write_text('{"tampered": true}\n')
    with pytest.raises(RuntimeError, match="parent causal evidence changed"):
        load_audit(config)


def test_cli_exposes_all_preregistered_stages() -> None:
    from ..run_cpu_protocol import _parser

    parser = _parser()
    for stage in (
        "audit",
        "calibrate",
        "localize",
        "transport",
        "gdn_mediation",
        "confirm",
        "report",
        "all",
    ):
        assert parser.parse_args(["--stage", stage]).stage == stage
