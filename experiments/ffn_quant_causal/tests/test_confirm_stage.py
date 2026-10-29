from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.ffn_quant_causal import confirm_stage
from experiments.ffn_quant_causal.adapters import ModelBundle
from experiments.ffn_quant_causal.confirm_stage import (
    CONFIRM_SAMPLE_COUNT,
    SCALE_IDENTITY_SPLITS,
    build_scale_artifact_identity,
    load_scale_artifact,
    run_model_confirm,
    save_scale_artifact,
)
from experiments.ffn_quant_causal.evaluation import aggregate_rows
from experiments.infinitevl_quant.config import SCHEMES
from experiments.infinitevl_quant.inventory import LinearTarget
from experiments.infinitevl_quant.progress import StageProgress
from experiments.infinitevl_quant.quantization import FakeQuantLinear
from experiments.infinitevl_quant_ffn_bf16_4x64.scoring import SCORER_ID
from experiments.vlm_quant_benchmark.data import (
    MultimodalSample,
    sample_manifest_digest,
)


def _samples(tmp_path: Path, rows_per_dataset: int = 48) -> list[MultimodalSample]:
    image = tmp_path / "image.bin"
    image.write_bytes(b"toy image")
    result = []
    for dataset_index in range(4):
        dataset = f"dataset_{dataset_index}"
        for row in range(rows_per_dataset):
            result.append(MultimodalSample(
                id=f"{dataset}-{row}",
                dataset=dataset,
                split="test",
                image=image,
                prompt=f"question {dataset_index} {row}",
                target="answer",
                official_index=str(row),
            ))
    return result


def _split_digests(confirm_samples: list[MultimodalSample]) -> dict[str, str]:
    result = {name: f"digest-{name}" for name in SCALE_IDENTITY_SPLITS}
    result["confirm"] = sample_manifest_digest(confirm_samples)
    # Descriptive full256 may exist in the protocol audit but is intentionally
    # excluded from the scale identity.
    result["full256"] = "descriptive-only"
    return result


def _identity(
    confirm_samples: list[MultimodalSample], *, method: str = "mbq",
):
    return build_scale_artifact_identity(
        model_kind="infinitevl",
        model_digest="model-digest",
        method=method,
        target_names=("0",),
        scheme=SCHEMES["w4a16"],
        config_digest="config-digest",
        split_digests=_split_digests(confirm_samples),
        protocol_digest="protocol-digest",
        implementation_digest="implementation-digest",
        search_identity="search-identity",
        search_audit_digest="search-audit-digest",
    )


def _provenance() -> dict[str, str]:
    return {
        "protocol_digest": "protocol-digest",
        "implementation_digest": "implementation-digest",
        "search_identity": "search-identity",
        "search_audit_digest": "search-audit-digest",
    }


@pytest.mark.parametrize(
    "field",
    [
        "model", "targets", "scheme", "config", "split", "protocol",
        "implementation", "search_identity", "search_audit",
    ],
)
def test_scale_artifact_identity_is_checked_exactly(tmp_path: Path, field: str) -> None:
    samples = _samples(tmp_path)
    identity = _identity(samples)
    path = tmp_path / "mbq.pt"
    save_scale_artifact(path, {"0": torch.ones(3)}, identity=identity)
    if field == "model":
        expected = dataclasses.replace(identity, model_kind="qwen")
    elif field == "targets":
        expected = dataclasses.replace(identity, target_names=("other",))
    elif field == "scheme":
        expected = dataclasses.replace(identity, scheme={
            **identity.scheme, "name": "rtn_w3a16", "weight_bits": 3,
        })
    elif field == "config":
        expected = dataclasses.replace(identity, config_digest="other-config")
    elif field == "split":
        expected = dataclasses.replace(identity, split_digests={
            **identity.split_digests, "confirm": "other-confirm",
        })
    elif field == "protocol":
        expected = dataclasses.replace(identity, protocol_digest="other-protocol")
    elif field == "implementation":
        expected = dataclasses.replace(
            identity, implementation_digest="other-implementation",
        )
    elif field == "search_identity":
        expected = dataclasses.replace(identity, search_identity="other-search")
    else:
        expected = dataclasses.replace(
            identity, search_audit_digest="other-search-audit",
        )
    with pytest.raises(RuntimeError, match="identity mismatch"):
        load_scale_artifact(path, expected_identity=expected)


def test_scale_artifact_roundtrip_resume_and_checksum(tmp_path: Path) -> None:
    samples = _samples(tmp_path)
    identity = _identity(samples)
    path = tmp_path / "scales.pt"
    scales = {"0": torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)}
    first = save_scale_artifact(path, scales, identity=identity)
    assert first.scales["0"].dtype == torch.float32
    assert torch.equal(first.scales["0"], scales["0"].float())
    second = save_scale_artifact(path, scales, identity=identity)
    assert second.artifact_sha256 == first.artifact_sha256

    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(raw)
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        load_scale_artifact(path, expected_identity=identity)


def test_run_model_confirm_uses_confirm192_and_restores_modules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = _samples(tmp_path)
    assert len(samples) == CONFIRM_SAMPLE_COUNT
    split_digests = _split_digests(samples)
    model = nn.Sequential(nn.Linear(3, 2, bias=False))
    original = model[0]
    target = LinearTarget("0", original, "decoder_ffn", 0)
    bundle = ModelBundle(
        kind="infinitevl",
        model=model,
        processor=SimpleNamespace(),
        ffn_targets=[target],
        load_audit={},
        decoder_prefix="",
    )
    mbq_path, qig_path = tmp_path / "mbq.pt", tmp_path / "qig.pt"
    for method, path, scale in (
        ("mbq", mbq_path, 1.0),
        ("qig", qig_path, 1.25),
    ):
        save_scale_artifact(
            path,
            {"0": torch.full((3,), scale)},
            identity=build_scale_artifact_identity(
                model_kind="infinitevl",
                model_digest="model-digest",
                method=method,
                target_names=("0",),
                scheme=SCHEMES["w4a16"],
                config_digest="config-digest",
                split_digests=split_digests,
                **_provenance(),
            ),
        )

    observed: list[tuple[str, bool, type[nn.Module]]] = []
    newly_written = 0

    def fake_evaluate_variant(
        active_model,
        processor,
        active_samples,
        *,
        progress,
        variant,
        reference_dir,
        write_reference,
        min_pixels,
        max_pixels,
        max_new_tokens,
        generate,
        **_kwargs,
    ):
        nonlocal newly_written
        del processor, reference_dir, min_pixels, max_pixels, max_new_tokens
        observed.append((variant, write_reference, type(active_model[0])))
        assert write_reference is variant.endswith("__bf16")
        assert isinstance(active_model[0], nn.Linear) if write_reference else isinstance(
            active_model[0], FakeQuantLinear,
        )
        for sample in active_samples:
            if progress.completed_sample(variant, sample.id) is not None:
                continue
            newly_written += 1
            is_baseline = write_reference
            total_nll = 1.0 if is_baseline else 1.1
            progress.complete_sample(variant, sample.id, {
                "id": sample.id,
                "dataset": sample.dataset,
                "target": sample.target,
                "prediction": "answer" if generate else "",
                "score": 1.0 if generate else math.nan,
                "scoring": {
                    "scorer": SCORER_ID,
                    "rule": "toy",
                    "extracted": "answer",
                } if generate else None,
                "target_text": {
                    "tokens": 1,
                    "total_nll": total_nll,
                    "nll": total_nll,
                },
                "comparison_to_bf16": {},
                "teacher_seconds": 0.0,
                "generation_seconds": 0.0,
                "generation_tokens": 1 if generate else 0,
                "generation_truncated": False,
                "sequence_tokens": 1,
            })
        return aggregate_rows(progress.rows(variant))

    monkeypatch.setattr(confirm_stage, "evaluate_variant", fake_evaluate_variant)
    progress = StageProgress(
        tmp_path / "output", "confirm",
        config_digest="config-digest", inputs={"model": "model-digest"},
    )
    result = run_model_confirm(
        bundle,
        samples,
        progress=progress,
        scheme=SCHEMES["w4a16"],
        mbq_artifact_path=mbq_path,
        qig_artifact_path=qig_path,
        model_digest="model-digest",
        config_digest="config-digest",
        split_digests=split_digests,
        artifact_provenance=_provenance(),
        min_pixels=1,
        max_pixels=2,
        max_new_tokens=3,
        bootstrap_samples=20,
        seed=7,
    )
    assert model[0] is original
    assert result["split"]["samples"] == 192
    assert result["split"]["full256_is_descriptive_only"]
    assert set(result["variants"]) == {"bf16", "rtn", "mbq", "qig"}
    assert all("__confirm192__" in value for value in result["variants"].values())
    assert result["scorer"] == SCORER_ID
    assert newly_written == 4 * CONFIRM_SAMPLE_COUNT
    assert observed[0][2] is nn.Linear
    assert all(item[2] is FakeQuantLinear for item in observed[1:])

    # A rerun consults the same per-sample rows and does not rewrite them.
    newly_written = 0
    rerun = run_model_confirm(
        bundle,
        samples,
        progress=progress,
        scheme=SCHEMES["w4a16"],
        mbq_artifact_path=mbq_path,
        qig_artifact_path=qig_path,
        model_digest="model-digest",
        config_digest="config-digest",
        split_digests=split_digests,
        artifact_provenance=_provenance(),
        min_pixels=1,
        max_pixels=2,
        max_new_tokens=3,
        bootstrap_samples=20,
        seed=7,
    )
    assert newly_written == 0
    assert rerun["variants"] == result["variants"]
    assert model[0] is original


def test_run_model_confirm_rejects_descriptive_full256(tmp_path: Path) -> None:
    samples = _samples(tmp_path, rows_per_dataset=64)
    split_digests = _split_digests(samples)
    model = nn.Sequential(nn.Linear(3, 2, bias=False))
    bundle = ModelBundle(
        kind="infinitevl",
        model=model,
        processor=SimpleNamespace(),
        ffn_targets=[LinearTarget("0", model[0], "decoder_ffn", 0)],
        load_audit={},
        decoder_prefix="",
    )
    progress = StageProgress(
        tmp_path / "output", "confirm",
        config_digest="config-digest", inputs={"model": "model-digest"},
    )
    with pytest.raises(ValueError, match="full256 is descriptive only"):
        run_model_confirm(
            bundle,
            samples,
            progress=progress,
            scheme=SCHEMES["w4a16"],
            mbq_artifact_path=tmp_path / "unused-mbq.pt",
            qig_artifact_path=tmp_path / "unused-qig.pt",
            model_digest="model-digest",
            config_digest="config-digest",
            split_digests=split_digests,
            artifact_provenance=_provenance(),
            min_pixels=1,
            max_pixels=2,
            max_new_tokens=3,
            bootstrap_samples=2,
            seed=7,
        )
