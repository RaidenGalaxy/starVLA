from __future__ import annotations

import json
import math

import torch

from experiments.ffn_quant_rootcause.evidence import atomic_torch
from experiments.infinitevl_quant.progress import file_sha256, json_digest, safe_id
from experiments.infinitevl_quant_ffn_bf16_4x64.scoring import score_prediction

from .design import condition, semantic_layers


class History:
    """Explicit import with semantic masks, content checks and live input binding."""

    def __init__(self, config, audit):
        self.config, self.audit = config, audit
        self.cache = {}

    def index(self, kind, split):
        key = kind, split
        if key in self.cache:
            return self.cache[key]
        stage = "confirm" if split == "confirm192" else "local"
        root = self.config.rootcause_output / stage
        identity = json.loads((root / "identity.json").read_text())
        manifest = json.loads((root / "summary.sha256.json").read_text())["sample_rows"]
        rows = {}
        for path in sorted(root.glob(f"{kind}__{split}__*/samples/*.json")):
            row = json.loads(path.read_text())
            layers = semantic_layers(row["intervention"])
            if layers is None:
                continue
            if file_sha256(path) != manifest[str(path.relative_to(root))]:
                raise RuntimeError("historical row changed after audit")
            if row["row_sha256"] != json_digest({k: v for k, v in row.items() if k != "row_sha256"}) or row[
                "provenance"
            ]["stage_identity"] != json_digest(identity):
                raise RuntimeError("historical row checksum or identity is invalid")
            mask = condition(layers)["id"]
            entry = {"row": row, "path": path, "stage": stage, "sha256": file_sha256(path)}
            rows.setdefault((mask, row["id"]), entry)
        self.cache[key] = rows
        return rows

    def find(self, kind, split, label, identifier):
        return self.index(kind, split).get((label, identifier))

    def try_import(self, engine, sample, inputs, split, label, spec, *, generate, permitted):
        if not permitted:
            return None
        entry = self.find(engine.bundle.kind, split, label, sample.id)
        if entry is None:
            return None
        if file_sha256(entry["path"]) != entry["sha256"]:
            raise RuntimeError("historical source changed after the reuse index was built")
        old = entry["row"]
        if old["input_digest"] != inputs[4] or math.isfinite(old["score"]) != generate:
            return None
        if generate:
            scored = score_prediction(sample, old["prediction"])
            if scored.score != old["score"]:
                raise RuntimeError("historical prediction does not reproduce its QA score")
        row = {k: v for k, v in old.items() if k not in ("provenance", "row_sha256")}
        row.update(
            intervention=spec.payload(),
            split=split,
            elapsed_seconds=0.0,
            teacher_seconds=0.0,
            generation_seconds=0.0,
            historical_reuse={
                "verified": True,
                "source_file": str(entry["path"].resolve()),
                "source_file_sha256": entry["sha256"],
                "source_row_sha256": old["row_sha256"],
                "historical_elapsed_seconds": old["elapsed_seconds"],
                "source_audit_sha256": self.audit["rootcause_audit_sha256"],
                "validation": "identity+live_inputs+model_sentinels",
            },
        )
        if label == "B":
            original = (
                self.config.rootcause_output
                / "references"
                / entry["stage"]
                / engine.bundle.kind
                / split
                / f"{safe_id(sample.id)}.pt"
            )
            if not original.is_file() or file_sha256(original) != old["reference_sha256"]:
                raise RuntimeError("historical reference logits are missing or corrupt")
            reference = torch.load(original, map_location="cpu", weights_only=True)
            if reference["input_digest"] != inputs[4] or reference["id"] != sample.id:
                raise RuntimeError("historical reference logits belong to a different input")
            reference["audit_digest"] = engine.progress.inputs["audit"]
            destination = engine.reference_path(split, sample)
            atomic_torch(destination, reference)
            row["reference_sha256"] = file_sha256(destination)
            row["historical_reuse"]["reference_source_sha256"] = old["reference_sha256"]
        unit = {"spec": spec.payload(), "generate": generate, "input_digest": inputs[4]}
        variant = f"{engine.bundle.kind}__{split}__{label}"
        engine.progress.commit(variant, sample.id, unit, row)
        return engine.progress.completed(variant, sample.id, unit)
