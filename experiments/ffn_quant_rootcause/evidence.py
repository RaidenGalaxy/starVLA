from __future__ import annotations

import datetime
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

import torch

from experiments.ffn_quant_causal.audit import audit_model_identity, scorer_resource_audit
from experiments.ffn_quant_causal.config import MODEL_SPECS
from experiments.ffn_quant_causal.protocol import build_protocol_splits
from experiments.ffn_quant_mechanism.stages import resolve_infinitevl_path
from experiments.infinitevl_quant.progress import StageProgress, atomic_json, file_sha256, json_digest

from .config import MODELS


def implementation_identity():
    root = Path(__file__).resolve().parents[2]
    packages = (
        "ffn_quant_rootcause",
        "ffn_quant_causal",
        "ffn_quant_mechanism",
        "infinitevl_quant",
        "vlm_quant_benchmark",
        "infinitevl_quant_ffn_bf16_4x64",
    )
    files = {
        str(p.relative_to(root)): file_sha256(p)
        for package in packages
        for p in sorted((root / "experiments" / package).glob("*.py"))
    }
    return {"files": files, "digest": json_digest(files)}


def validate_historical_stage(root):
    identity = json.loads((root / "identity.json").read_text())
    marker = json.loads((root / ".complete").read_text())
    required = marker.get("required")
    if any(marker.get(k) != v for k, v in identity.items()) or not isinstance(required, list) or not required:
        raise RuntimeError(f"invalid historical completion identity: {root}")
    if any(not (root / name).exists() for name in required):
        raise RuntimeError(f"historical stage is missing required evidence: {root}")


def truth(config):
    paths = {
        "infinitevl": resolve_infinitevl_path(config.infinitevl_model_path),
        "qwen": config.qwen_model_path.resolve(),
    }
    parent = config.parent_output.resolve()
    if config.output_dir.resolve() in (parent, config.mechanism_output.resolve()):
        raise ValueError("new output must not overwrite historical evidence")
    for stage in ("audit", "rtn", "trace", "counterfactual", "confirm"):
        validate_historical_stage(parent / stage)
    for stage in ("audit", "calibrate", "localize"):
        validate_historical_stage(config.mechanism_output.resolve() / stage)
    splits, split_audit = build_protocol_splits(config.calibration_manifest, config.evaluation_manifest)
    previous_models = json.loads((parent / "audit/models.json").read_text())["models"]
    models = {kind: audit_model_identity(paths[kind], MODEL_SPECS[kind]) for kind in MODELS}
    for kind in MODELS:
        if models[kind]["content_sha256"] != previous_models[kind]["content_sha256"]:
            raise RuntimeError(f"{kind} checkpoint differs from historical comparison")
    previous_splits = json.loads((parent / "audit/splits.json").read_text())["digests"]
    if previous_splits != split_audit["digests"]:
        raise RuntimeError("input manifests do not reproduce historical split identities")
    images = {
        str(p.resolve()): file_sha256(p)
        for name in ("qig", "screen", "confirm")
        for sample in splits[name]
        for p in sample.images
    }
    history = {}
    for source in (parent, config.mechanism_output.resolve()):
        if not source.is_dir():
            raise FileNotFoundError(source)
        for path in sorted(source.rglob("*")):
            # Snapshot immutable evidence; no intermediate transport/mediator cache import.
            relative = str(path.relative_to(source))
            if path.is_file() and (source == parent or relative.split("/")[0] in ("audit", "calibrate", "localize")):
                if path.suffix in (".json", ".pt") or path.name == ".complete":
                    history[str(path)] = file_sha256(path)
    packages = {}
    for name in (
        "torch",
        "transformers",
        "numpy",
        "Pillow",
        "tokenizers",
        "safetensors",
        "qwen-vl-utils",
        "huggingface-hub",
        "accelerate",
        "einops",
        "torchvision",
        "matplotlib",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    payload = {
        "config": config.payload(),
        "model_paths": {k: str(v) for k, v in paths.items()},
        "models": models,
        "split_audit": split_audit,
        "images": images,
        "historical_files": history,
        "implementation": implementation_identity(),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": packages,
            "torch_build": torch.__config__.show(),
            "torch_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "cpu_capability": torch.backends.cpu.get_cpu_capability(),
            "mkldnn_enabled": torch.backends.mkldnn.enabled,
            "environment": {
                k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES")
            },
        },
        "scorer": scorer_resource_audit(),
        "reference_cache_policy": "rerun_under_current_identity",
        "confirmation_history": "historically_reused_evaluation",
    }
    return payload, splits


def audit(config):
    payload, splits = truth(config)
    path = config.output_dir / "audit/audit.json"
    identity = json_digest(payload)
    if path.is_file() and json.loads(path.read_text()) != payload:
        raise RuntimeError("audit identity changed; use a new output directory")
    atomic_json(path, payload)
    return payload, splits, identity


class EvidenceProgress(StageProgress):
    """Bind every row to its intervention and reject corrupt resumed evidence."""

    def __init__(self, root, stage, *, config_digest, inputs):
        super().__init__(root, stage, config_digest=config_digest, inputs=inputs)
        self.identity_digest = json_digest(self.identity)

    def completed(self, variant, identifier, unit):
        row = self.completed_sample(variant, identifier)
        if row is None:
            return None
        expected = {"stage_identity": self.identity_digest, "unit": json_digest(unit), "variant": variant}
        if row.get("provenance") != expected:
            raise RuntimeError("sample belongs to a different intervention/input")
        checksum = row.get("row_sha256")
        if checksum != json_digest({k: v for k, v in row.items() if k != "row_sha256"}):
            raise RuntimeError("completed sample checksum mismatch")
        return row

    def commit(self, variant, identifier, unit, row):
        row = dict(
            row,
            id=identifier,
            provenance={"stage_identity": self.identity_digest, "unit": json_digest(unit), "variant": variant},
        )
        # JSON changes integer layer keys to strings. Canonicalize before hashing
        # so recurrent-state and multi-layer rows survive a save/load round trip.
        row = json.loads(json.dumps(row, ensure_ascii=False))
        row["row_sha256"] = json_digest(row)
        self.complete_sample(variant, identifier, row)
        status_path = self.root.parent / "status.json"
        current = json.loads(status_path.read_text()) if status_path.is_file() else {}
        atomic_json(
            status_path,
            {
                **current,
                "status": "running",
                "stage": self.root.name,
                "last_variant": variant,
                "last_sample_id": identifier,
                "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
        )

    def finish(self, summary):
        atomic_json(self.root / "summary.json", summary)
        manifest = {str(p.relative_to(self.root)): file_sha256(p) for p in sorted(self.root.glob("*/samples/*.json"))}
        atomic_json(
            self.root / "summary.sha256.json",
            {"summary.json": file_sha256(self.root / "summary.json"), "sample_rows": manifest},
        )
        self.mark_complete(("summary.json", "summary.sha256.json"))

    def verified_complete(self):
        if not self.is_complete(("summary.json", "summary.sha256.json")):
            return False
        expected = json.loads((self.root / "summary.sha256.json").read_text())
        manifest = {str(p.relative_to(self.root)): file_sha256(p) for p in sorted(self.root.glob("*/samples/*.json"))}
        if expected != {"summary.json": file_sha256(self.root / "summary.json"), "sample_rows": manifest}:
            raise RuntimeError("stage summary checksum mismatch")
        for variant in self.root.iterdir():
            if variant.is_dir() and (variant / "samples").exists():
                for row in self.rows(variant.name):
                    if (
                        row.get("provenance", {}).get("stage_identity") != self.identity_digest
                        or row.get("provenance", {}).get("variant") != variant.name
                    ):
                        raise RuntimeError("completed stage contains foreign sample evidence")
                    if row.get("row_sha256") != json_digest({k: v for k, v in row.items() if k != "row_sha256"}):
                        raise RuntimeError("completed stage contains corrupt sample evidence")
        return True


def atomic_torch(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def input_digest(inputs):
    import hashlib

    digest = hashlib.sha256()
    for name, value in sorted(inputs.items()):
        digest.update(name.encode())
        if isinstance(value, torch.Tensor):
            digest.update(str((value.dtype, tuple(value.shape))).encode())
            digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        else:
            digest.update(json_digest(value).encode())
    return digest.hexdigest()
