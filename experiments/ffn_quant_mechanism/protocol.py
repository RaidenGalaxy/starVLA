from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.ffn_quant_causal.protocol import (
    audit_split_disjointness,
    build_protocol_splits,
    write_split_manifests,
)
from experiments.infinitevl_quant.progress import (
    StageProgress,
    atomic_json,
    file_sha256,
    json_digest,
)
from experiments.vlm_quant_benchmark.data import MultimodalSample

from .config import MechanismConfig

PARENT_REQUIRED = (
    "protocol_identity.json",
    "audit/implementation.json",
    "audit/models.json",
    "audit/splits.json",
    "counterfactual/dynamics_summary.json",
    "counterfactual/summary.json",
    "confirm/summary.json",
    "mbq_qig/scale_artifacts/infinitevl/mbq_w4a16_ffn.pt",
    "mbq_qig/scale_artifacts/infinitevl/mbq_w4a16_ffn.pt.identity.json",
    "mbq_qig/scale_artifacts/qwen/mbq_w4a16_ffn.pt",
    "mbq_qig/scale_artifacts/qwen/mbq_w4a16_ffn.pt.identity.json",
)


def _audited_reuse_files(parent: Path) -> list[Path]:
    files: list[Path] = []
    for kind in ("infinitevl", "qwen"):
        reference_dir = parent / "confirm/reference_artifacts" / kind
        references = sorted(reference_dir.glob("*.pt"))
        if len(references) != 192:
            raise RuntimeError(f"parent {kind} confirm reference cache has {len(references)} files, expected 192")
        files.extend(references)
        for suffix in ("bf16", "rtn_w4a16_ffn"):
            sample_dir = parent / "confirm" / f"{kind}__confirm192__{suffix}" / "samples"
            rows = sorted(sample_dir.glob("*.json"))
            if len(rows) != 192:
                raise RuntimeError(f"parent {kind}/{suffix} cache has {len(rows)} rows, expected 192")
            files.extend(rows)
    return files


def parent_evidence_audit(parent: Path) -> dict[str, Any]:
    parent = parent.resolve()
    missing = [relative for relative in PARENT_REQUIRED if not (parent / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"parent causal evidence is incomplete: {missing}")
    reused = _audited_reuse_files(parent)
    paths = [parent / relative for relative in PARENT_REQUIRED] + reused
    files = {
        str(path.relative_to(parent)): {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in paths
    }
    identity = json.loads((parent / "protocol_identity.json").read_text())
    if identity.get("protocol") != "ffn_quant_causal":
        raise RuntimeError("parent output is not an ffn_quant_causal protocol")
    models = json.loads((parent / "audit/models.json").read_text()).get("models", {})
    if set(models) != {"infinitevl", "qwen"} or any(
        not models[kind].get("passed") or not models[kind].get("content_sha256") for kind in ("infinitevl", "qwen")
    ):
        raise RuntimeError("parent model identity audit is incomplete")
    return {
        "parent": str(parent),
        "identity": identity,
        "models": {
            kind: {
                "content_sha256": models[kind]["content_sha256"],
                "source": models[kind]["source"],
                "model": models[kind]["model"],
            }
            for kind in ("infinitevl", "qwen")
        },
        "files": files,
        "reused_artifact_files": len(reused),
        "digest": json_digest(files),
        "read_only_contract": True,
    }


def implementation_audit() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]
    roots = (
        repository / "experiments/ffn_quant_mechanism",
        repository / "experiments/ffn_quant_causal",
        repository / "experiments/infinitevl_quant",
    )
    files = sorted({path for root in roots for path in root.glob("*.py") if path.is_file()})
    records: dict[str, str] = {}
    combined = hashlib.sha256()
    for path in files:
        relative = str(path.relative_to(repository))
        digest = file_sha256(path)
        records[relative] = digest
        combined.update(relative.encode())
        combined.update(digest.encode())
    return {
        "digest": combined.hexdigest(),
        "file_count": len(records),
        "files": records,
    }


def audited_splits(config: MechanismConfig) -> tuple[dict[str, list[MultimodalSample]], dict]:
    splits, audit = build_protocol_splits(
        config.calibration_manifest,
        config.evaluation_manifest,
    )
    required = {
        "qig": 32,
        "diagnostic8": 8,
        "screen": 32,
        "confirm": 192,
        "qig_acceptance": 32,
    }
    actual = {name: len(splits[name]) for name in required}
    if actual != required:
        raise RuntimeError(f"mechanism split counts changed: {actual} != {required}")
    audit_split_disjointness(splits)
    return splits, audit


def select_long16(
    samples: Sequence[MultimodalSample],
    token_counts: Mapping[str, int],
) -> list[MultimodalSample]:
    by_dataset: dict[str, list[MultimodalSample]] = {}
    for sample in samples:
        if int(token_counts.get(sample.id, 0)) >= 512:
            by_dataset.setdefault(sample.dataset, []).append(sample)
    selected: list[MultimodalSample] = []
    for dataset in sorted({sample.dataset for sample in samples}):
        candidates = by_dataset.get(dataset, [])
        ordered = sorted(
            candidates,
            key=lambda item: hashlib.sha256(f"ffn-quant-mechanism-long16:{item.id}".encode()).hexdigest(),
        )
        if len(ordered) < 4:
            raise RuntimeError(f"{dataset} has fewer than four confirm samples with >=512 tokens")
        selected.extend(ordered[:4])
    counts = Counter(sample.dataset for sample in selected)
    if len(selected) != 16 or set(counts.values()) != {4}:
        raise RuntimeError(f"invalid long16 selection: {dict(counts)}")
    return selected


def create_audit(config: MechanismConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    parent = parent_evidence_audit(config.parent_output)
    implementation = implementation_audit()
    splits, split_audit = audited_splits(config)
    parent_splits = json.loads((config.parent_output.resolve() / "audit/splits.json").read_text())
    split_names = ("qig", "diagnostic8", "screen", "confirm", "qig_acceptance")
    expected_digests = {name: parent_splits.get("digests", {}).get(name) for name in split_names}
    observed_digests = {name: split_audit["digests"][name] for name in split_names}
    if observed_digests != expected_digests:
        raise RuntimeError(
            f"current manifests do not reproduce parent split identities: {observed_digests} != {expected_digests}"
        )
    manifests = write_split_manifests(config.output_dir / "audit/manifests", splits)
    payload = {
        "format_version": 1,
        "protocol": "ffn_quant_mechanism",
        "config": config.to_dict(),
        "config_digest": config.digest(),
        "parent": parent,
        "implementation": implementation,
        "splits": split_audit,
        "manifests": manifests,
    }
    atomic_json(
        config.output_dir / "protocol_identity.json",
        {
            "format_version": 1,
            "protocol": "ffn_quant_mechanism",
            "config_digest": config.digest(),
            "parent_digest": parent["digest"],
            "implementation_digest": implementation["digest"],
            "split_digests": split_audit["digests"],
        },
    )
    atomic_json(config.output_dir / "audit/audit.json", payload)
    return payload


def load_audit(config: MechanismConfig) -> dict[str, Any]:
    path = config.output_dir / "audit/audit.json"
    if not path.is_file():
        raise FileNotFoundError("run --stage audit before other mechanism stages")
    payload = json.loads(path.read_text())
    if payload.get("config_digest") != config.digest():
        raise RuntimeError("mechanism audit belongs to another configuration")
    current_parent = parent_evidence_audit(config.parent_output)
    if payload.get("parent", {}).get("digest") != current_parent["digest"]:
        raise RuntimeError("parent causal evidence changed after mechanism audit")
    current_implementation = implementation_audit()
    if payload.get("implementation", {}).get("digest") != current_implementation["digest"]:
        raise RuntimeError("mechanism implementation changed after audit; choose a new --output-dir")
    return payload


def stage_progress(config: MechanismConfig, stage: str) -> StageProgress:
    audit = load_audit(config)
    inputs = {
        "parent": str(audit["parent"]["digest"]),
        "implementation": str(audit["implementation"]["digest"]),
        **{
            f"split:{name}": str(digest)
            for name, digest in audit["splits"]["digests"].items()
            if name in {"qig", "diagnostic8", "screen", "confirm", "qig_acceptance"}
        },
    }
    return StageProgress(
        config.output_dir,
        stage,
        config_digest=config.digest(),
        inputs=dict(sorted(inputs.items())),
    )
