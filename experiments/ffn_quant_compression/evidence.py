from __future__ import annotations

import dataclasses
import datetime
import json
from pathlib import Path

from experiments.ffn_quant_rootcause.config import Config as RootConfig
from experiments.ffn_quant_rootcause.evidence import EvidenceProgress, truth
from experiments.ffn_quant_rootcause.stages import load_summary
from experiments.infinitevl_quant.progress import atomic_json, file_sha256, json_digest

from .design import build_design


def source_config(config, previous):
    names = {f.name for f in dataclasses.fields(RootConfig)}
    paths = {f.name for f in dataclasses.fields(RootConfig) if isinstance(f.default, Path)}
    paths.add("infinitevl_model_path")
    values = {k: Path(v) if k in paths and v is not None else v for k, v in previous["config"].items() if k in names}
    values["output_dir"] = config.rootcause_output
    return RootConfig(**values)


def prepare(config):
    payload, splits = truth(config)
    previous = json.loads((config.rootcause_output / "audit/audit.json").read_text())
    old_config = source_config(config, previous)
    history = {stage: load_summary(old_config, stage) for stage in ("local", "confirm")}
    design = build_design(history["local"]["models"])
    expected = {"gate_proj": [11008, 2048], "up_proj": [11008, 2048], "down_proj": [2048, 11008]}
    for kind, model in payload["models"].items():
        inv = model["static_ffn_inventory"]
        if inv["decoder_layers"] != 36 or inv["expected_shapes"] != expected:
            raise RuntimeError(f"{kind} FFN inventory differs from compression formula")
    source_files = {
        str(p.resolve()): file_sha256(p)
        for stage in ("local", "confirm")
        for name in ("summary.json", "summary.sha256.json", "identity.json", ".complete")
        for p in (config.rootcause_output / stage / name,)
    }
    root = Path(__file__).resolve().parent
    payload.update(
        driver_files={str(p.relative_to(root)): file_sha256(p) for p in sorted(root.glob("*.py"))},
        rootcause_sources=source_files,
        rootcause_audit_sha256=file_sha256(config.rootcause_output / "audit/audit.json"),
        design_sha256=json_digest(design),
        reference_cache_policy="explicit_verified_import",
    )
    reasons = []
    for name in ("implementation", "runtime", "scorer", "images"):
        if payload[name] != previous[name]:
            reasons.append(name)
    if payload["split_audit"]["digests"] != previous["split_audit"]["digests"]:
        reasons.append("data_splits")
    for kind in payload["models"]:
        if payload["models"][kind]["content_sha256"] != previous["models"][kind]["content_sha256"]:
            reasons.append("checkpoint_" + kind)
    for name in ("min_pixels", "max_pixels", "max_new_tokens", "group_size", "threads"):
        if payload["config"][name] != previous["config"][name]:
            reasons.append("config_" + name)
    payload["historical_reuse_eligibility"] = {"eligible": not reasons, "mismatches": reasons}
    payload["historical_rootcause_identity"] = json_digest(previous)
    for path, value in (
        (config.output_dir / "audit/audit.json", payload),
        (config.output_dir / "audit/locked_design.json", design),
    ):
        if path.exists() and json.loads(path.read_text()) != value:
            raise RuntimeError(f"frozen evidence changed: {path}; use a new output directory")
        atomic_json(path, value)
    return payload, splits, design


class Progress(EvidenceProgress):
    def __init__(self, config, stage, audit, design, upstream=None):
        self.audit, self.design = audit, design
        super().__init__(
            config.output_dir,
            stage,
            config_digest=config.digest(),
            inputs={"audit": json_digest(audit), "design": json_digest(design), "upstream": upstream or {}},
        )

    def commit(self, variant, identifier, unit, row):
        pieces = variant.split("__", 2)
        if len(pieces) == 3 and pieces[0] in self.design["models"]:
            model, _, label = pieces
            record = self.design["models"][model]["conditions"].get(label)
            if record is not None:
                row = dict(
                    row,
                    compression={k: v for k, v in record.items() if k != "spec"},
                    model_content_sha256=self.audit["models"][model]["content_sha256"],
                    design_sha256=json_digest(self.design),
                )
        super().commit(variant, identifier, unit, row)


def write_status(config, state, stage, **extra):
    atomic_json(
        config.output_dir / "status.json",
        {
            "status": state,
            "stage": stage,
            "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            **extra,
        },
    )
