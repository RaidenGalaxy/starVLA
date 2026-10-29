from __future__ import annotations

import json
import statistics
import time

from experiments.ffn_quant_causal.adapters import load_model_bundle
from experiments.ffn_quant_causal.budget import WallclockBudgetExhausted
from experiments.ffn_quant_rootcause.evaluation import Engine as ReferenceEngine
from experiments.ffn_quant_rootcause.interventions import Intervention
from experiments.infinitevl_quant.progress import atomic_json, file_sha256

from .config import MODELS
from .reuse import History


class Engine(ReferenceEngine):
    def __init__(self, bundle, config, progress, budget, audit):
        super().__init__(bundle, config, progress, budget)
        self.audit = audit
        if len(self.layers) != 36:
            raise RuntimeError("loaded model does not contain 36 FFN layers")
        shapes = audit["models"][bundle.kind]["static_ffn_inventory"]["expected_shapes"]
        for layer in self.layers:
            if any(list(getattr(layer.mlp, name).weight.shape) != shape for name, shape in shapes.items()):
                raise RuntimeError("loaded FFN shapes differ from audited compression accounting")

    def inputs(self, sample):
        if sample.id not in self.input_cache:
            for path in sample.images:
                if file_sha256(path) != self.audit["images"].get(str(path.resolve())):
                    raise RuntimeError("image contents changed after the identity audit")
        return super().inputs(sample)


def engines(config, audit, progress, budget):
    for kind in MODELS:
        budget.check(stage=progress.root.name, boundary="before_model_load", variant=kind)
        bundle = load_model_bundle(kind, config_path(audit, kind))
        try:
            yield kind, Engine(bundle, config, progress, budget, audit)
        finally:
            bundle.close()


def config_path(audit, kind):
    from pathlib import Path

    return Path(audit["model_paths"][kind])


class InferenceBudget:
    def __init__(self, budget, reserve_seconds):
        self.actual, self.reserve_seconds = budget, reserve_seconds

    def remaining_seconds(self):
        return self.actual.remaining_seconds() - self.reserve_seconds

    def check(self, **kwargs):
        self.actual.check(**kwargs)
        if self.remaining_seconds() <= 0:
            raise WallclockBudgetExhausted("inference stopped to preserve analysis/report reserve")


def complete_rows(progress, variant, samples, *, required=True):
    rows = progress.rows(variant)
    expected = {s.id for s in samples}
    if len({r["id"] for r in rows}) != len(rows) or any(r["id"] not in expected for r in rows):
        raise RuntimeError("condition has duplicate or foreign samples")
    if len(rows) != len(expected):
        if required:
            raise RuntimeError(f"incomplete condition: {variant}")
        return None
    return rows


def audit_sentinels(config, payload, splits, design, progress, budget):
    history = History(config, payload)
    result = {}
    for kind, engine in engines(config, payload, progress, budget):
        model = design["models"][kind]
        eligible = payload["historical_reuse_eligibility"]["eligible"]
        comparisons = []
        selected = list(
            dict.fromkeys(
                [
                    "B",
                    "W",
                    model["aliases"]["sensitive_first"]["1"],
                    model["aliases"]["sensitive_first"]["3"],
                    model["aliases"]["sensitive_protected"]["33"],
                    model["aliases"]["sensitive_protected"]["35"],
                ]
            )
        )
        # Four dataset representatives; all checks use exact processed-input hashes.
        for label in selected:
            spec = Intervention.from_payload(model["conditions"][label]["spec"])
            for sample in splits["smoke4"]:
                inputs = engine.inputs(sample)
                row = engine.evaluate(sample, inputs, "smoke4", label, spec)
                prior = history.find(kind, "screen32", label, sample.id) if eligible else None
                passed = bool(
                    prior
                    and prior["row"]["input_digest"] == inputs[4]
                    and abs(row["target_text"]["nll"] - prior["row"]["target_text"]["nll"]) <= 1e-6
                )
                comparisons.append({"kind": "teacher", "label": label, "id": sample.id, "passed": passed})
        sentinels = [splits["confirm"][i] for i in (0, 48, 96, 144)]
        for label in ("B", "W"):
            spec = Intervention.from_payload(model["conditions"][label]["spec"])
            for sample in sentinels:
                inputs = engine.inputs(sample)
                row = engine.evaluate(sample, inputs, "sentinel192", label, spec, generate=True)
                prior = history.find(kind, "confirm192", label, sample.id) if eligible else None
                passed = bool(
                    prior
                    and prior["row"]["input_digest"] == inputs[4]
                    and abs(row["target_text"]["nll"] - prior["row"]["target_text"]["nll"]) <= 1e-6
                    and prior["row"]["prediction"] == row["prediction"]
                    and prior["row"]["score"] == row["score"]
                )
                comparisons.append({"kind": "generation", "label": label, "id": sample.id, "passed": passed})
        result[kind] = {
            "historical_reuse_verified": eligible and all(x["passed"] for x in comparisons),
            "comparisons": comparisons,
            "tested_endpoint_and_single_top3_masks": selected,
        }
        atomic_json(progress.root / "sentinels_in_progress.json", result)
    return {
        "models": result,
        "design_locked_before_inference": True,
        "history_eligibility": payload["historical_reuse_eligibility"],
    }


def evaluate_condition(engine, samples, split, label, record, history, permitted, generate):
    spec = Intervention.from_payload(record["spec"])
    for sample in samples:
        inputs = engine.inputs(sample)
        variant = f"{engine.bundle.kind}__{split}__{label}"
        unit = {"spec": spec.payload(), "generate": generate, "input_digest": inputs[4]}
        previous = engine.progress.completed(variant, sample.id, unit)
        if previous is not None:
            if label == "B":
                ref = engine.reference_path(split, sample)
                if not ref.is_file() or file_sha256(ref) != previous["reference_sha256"]:
                    raise RuntimeError("resumed reference logits are missing or corrupt")
            continue
        engine.check("before_import_or_inference", variant, sample.id)
        if (
            history.try_import(engine, sample, inputs, split, label, spec, generate=generate, permitted=permitted)
            is None
        ):
            engine.evaluate(sample, inputs, split, label, spec, generate=generate)
    return complete_rows(engine.progress, f"{engine.bundle.kind}__{split}__{label}", samples)


def sweep(config, payload, splits, design, audit_summary, progress, budget, *, on_checkpoint=None):
    inference = InferenceBudget(budget, config.reserve_hours * 3600)
    history = History(config, payload)
    omissions = {}
    # All QA points on BOTH models precede dense teacher-forced scans.
    phases = (
        ("qa", "confirm192", "confirm", "qa_order", True),
        ("main_nll", "screen32", "screen", "main_screen_order", False),
        ("random_nll", "screen32", "screen", "random_screen_order", False),
    )
    for phase, split, sample_key, order_key, generate in phases:
        for kind, engine in engines(config, payload, progress, inference):
            model = design["models"][kind]
            permitted = audit_summary["models"][kind]["historical_reuse_verified"]
            order = model[order_key]
            observed = []
            if phase == "random_nll":
                for folder in progress.root.glob("*__screen32__*/samples"):
                    for path in folder.glob("*.json"):
                        row = json.loads(path.read_text())
                        if not row.get("historical_reuse"):
                            observed.append(row["elapsed_seconds"])
                seconds = statistics.mean(observed) if observed else 40.0
                remaining = sum(
                    not progress.sample_path(f"{kind}__{split}__{label}", s.id).exists()
                    for label in order
                    for s in splits[sample_key]
                )
                # Preserve both models' optional family as a declared unit.
                if kind == MODELS[0]:
                    other = design["models"][MODELS[1]][order_key]
                    remaining += sum(
                        not progress.sample_path(f"{MODELS[1]}__{split}__{label}", s.id).exists()
                        for label in other
                        for s in splits[sample_key]
                    )
                forecast = remaining * seconds * 1.2
                if forecast > inference.remaining_seconds():
                    omissions["random_nll"] = {
                        "reason": "preserve_five_hour_analysis_reserve",
                        "estimated_seconds": forecast,
                        "models": list(MODELS),
                        "all_seeds": [42, 43, 44],
                    }
                    break
            start = time.perf_counter()
            done = []
            for label in order:
                rows = evaluate_condition(
                    engine, splits[sample_key], split, label, model["conditions"][label], history, permitted, generate
                )
                done.append(label)
                times = [r["elapsed_seconds"] for r in rows if not r.get("historical_reuse")]
                atomic_json(
                    progress.root / "schedule.json",
                    {
                        "phase": phase,
                        "model": kind,
                        "completed_conditions_in_model_phase": done,
                        "remaining_conditions_in_model_phase": [v for v in order if v not in done],
                        "last_condition_seconds_mean": statistics.mean(times) if times else None,
                        "remaining_inference_hours": inference.remaining_seconds() / 3600,
                        "actual_wallclock_remaining_hours": budget.remaining_seconds() / 3600,
                        "phase_elapsed_seconds": time.perf_counter() - start,
                    },
                )
                print(f"[sweep/{phase}/{kind}] complete {label} q={model['conditions'][label]['q']}", flush=True)
            if on_checkpoint is not None:
                on_checkpoint(omissions)
    completed = {}
    for kind in MODELS:
        model = design["models"][kind]
        for split, samples, labels in (
            ("confirm192", splits["confirm"], model["qa_order"]),
            ("screen32", splits["screen"], model["main_screen_order"]),
        ):
            for label in labels:
                complete_rows(progress, f"{kind}__{split}__{label}", samples)
        completed[kind] = {
            "qa_conditions": len(model["qa_order"]),
            "main_nll_conditions": len(model["main_screen_order"]),
        }
    return {
        "mandatory_conditions_complete": True,
        "models": completed,
        "budgeted_omissions": omissions,
        "history_reuse_is_explicit": True,
    }
