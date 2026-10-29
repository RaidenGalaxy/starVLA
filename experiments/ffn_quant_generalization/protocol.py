from __future__ import annotations

import contextlib
import json
import math
import statistics
from pathlib import Path

from experiments.ffn_quant_rootcause.interventions import Intervention
from experiments.ffn_quant_rootcause.numerics import scalar_quantize
from experiments.infinitevl_quant.progress import atomic_json, file_sha256, json_digest
from experiments.infinitevl_quant.quantization import quantize_weight

from .config import MODELS
from .design import condition
from .runtime import engine_for, progress, rows


def mean_nll(values):
    return math.fsum(r["target_text"]["total_nll"] for r in values) / sum(r["target_text"]["tokens"] for r in values)


@contextlib.contextmanager
def native_rtn(engine):
    originals = []
    try:
        for i, layer in enumerate(engine.layers):
            for projection in ("gate", "up", "down"):
                parameter = getattr(layer.mlp, projection + "_proj").weight
                quantized = engine.bank.weight(i, projection, 4)
                originals.append((parameter, parameter.data))
                parameter.data = quantized
        yield
    finally:
        for parameter, value in originals:
            parameter.data = value


def audit_live(config, audit, splits, budget):
    evidence = progress(config, "audit", audit)
    if evidence.verified_complete():
        return json.loads((evidence.root / "summary.json").read_text())
    result = {"models": {}, "historical_reuse": {}}
    history_audit = json.loads((config.history_dir / "audit/audit.json").read_text())
    old_map = {"infinitevl": "infinitevl", "qwen25vl_3b": "qwen"}
    history_design = json.loads((config.history_dir / "audit/locked_design.json").read_text())
    for kind in MODELS:
        with engine_for(config, kind, audit, evidence, budget) as engine:
            scalar_records = []
            for index in (0, 1, len(engine.layers) - 1):
                for projection in ("gate", "up", "down"):
                    value = getattr(engine.layers[index].mlp, projection + "_proj").weight.detach()[:3, :256]
                    expected, _ = scalar_quantize(value, bits=4, group_size=128)
                    actual = quantize_weight(value, bits=4, group_size=128, zero_point=True).dequantized
                    delta = float((actual - expected).abs().max())
                    if delta > 1e-7:
                        raise ArithmeticError("independent scalar quantizer mismatch")
                    scalar_records.append({"layer": index, "projection": projection, "max_abs": delta})
            parity = []
            for sample in (
                splits["pilot8"][:1] + [s for s in splits["pilot8"] if s.dataset != splits["pilot8"][0].dataset][:1]
            ):
                budget.check(stage="audit", boundary="before_native_parity", sample_id=sample.id)
                native, _ = engine.forward(sample, Intervention(bits=None), native=True)
                reference, _ = engine.forward(sample, Intervention(bits=None))
                quantized, _ = engine.forward(sample, Intervention(bits=4))
                with native_rtn(engine):
                    rtn, _ = engine.forward(sample, Intervention(bits=None), native=True)
                restored, _ = engine.forward(sample, Intervention(bits=None), native=True)
                errors = {
                    "reference_vs_native": float((native - reference).abs().max()),
                    "W4_vs_native_RTN": float((quantized - rtn).abs().max()),
                    "restored_vs_native": float((restored - native).abs().max()),
                }
                if max(errors.values()) > 1e-5:
                    raise ArithmeticError(f"native endpoint/restoration mismatch: {errors}")
                parity.append({"id": sample.id, **errors})
            pilot = {}
            for label, spec in (("B", Intervention(bits=None)), ("W", Intervention(bits=4))):
                pilot[label] = [
                    engine.evaluate(
                        s, "pilot8", label, spec, condition(kind, [] if label == "B" else range(MODELS[kind][0]))
                    )
                    for s in splits["pilot8"]
                ]
            # One fixed sample of each dataset measures real greedy generation cost.
            generation_pilot = []
            for ds in sorted({s.dataset for s in splits["pilot8"]}):
                sample = next(s for s in splits["pilot8"] if s.dataset == ds)
                teacher = next(r for r in pilot["B"] if r["id"] == sample.id)
                generation_pilot.append(
                    engine.generate(sample, "B", condition(kind, []), teacher, split="pilot_generation")
                )
            result["models"][kind] = {
                "scalar_quantizer": scalar_records,
                "native_parity": parity,
                "teacher_mean_seconds": statistics.mean(r["teacher_seconds"] for r in pilot["B"] + pilot["W"]),
                "generation_mean_seconds": statistics.mean(r["generation_seconds"] for r in generation_pilot),
                "teacher_by_dataset": {
                    ds: statistics.mean(r["teacher_seconds"] for r in pilot["B"] + pilot["W"] if r["dataset"] == ds)
                    for ds in sorted({s.dataset for s in splits["pilot8"]})
                },
            }
            if kind not in old_map:
                continue
            old = old_map[kind]
            # Every historical model file is checked, not just the checkpoint path.
            source_files = history_audit["models"][old]["files"]
            path = Path(audit["models"][kind]["path"])
            file_identity = all(
                (path / name).is_file() and file_sha256(path / name) == rec["sha256"]
                for name, rec in source_files.items()
            )
            implementation_same = audit["implementation"] == history_audit["implementation"]
            matching_runtime = all(
                audit["runtime"]["packages"].get(k) == v
                for k, v in history_audit["runtime"]["packages"].items()
                if k in audit["runtime"]["packages"]
            )
            matches = []
            saved_model = history_design["models"][old]
            labels = list(
                dict.fromkeys(
                    [
                        "B",
                        "W",
                        saved_model["aliases"]["sensitive_first"]["1"],
                        saved_model["aliases"]["sensitive_protected"]["35"],
                    ]
                )
            )
            for label in labels:
                for sample in splits["smoke4"]:
                    row = engine.evaluate(
                        sample,
                        "history_sentinel",
                        label,
                        Intervention.from_payload(saved_model["conditions"][label]["spec"]),
                    )
                    # Locate by recorded ID: filename hashes use a different canonicalization.
                    prior = next(
                        json.loads(p.read_text())
                        for p in (config.history_dir / "sweep" / f"{old}__screen32__{label}" / "samples").glob("*.json")
                        if json.loads(p.read_text())["id"] == sample.id
                    )
                    ok = (
                        row["input_digest"] == prior["input_digest"]
                        and abs(row["target_text"]["nll"] - prior["target_text"]["nll"]) <= 1e-6
                    )
                    matches.append({"id": sample.id, "label": label, "passed": ok})
            eligible = file_identity and implementation_same and matching_runtime and all(v["passed"] for v in matches)
            result["historical_reuse"][kind] = {
                "eligible": eligible,
                "model_files_match": file_identity,
                "implementation_match": implementation_same,
                "runtime_match": matching_runtime,
                "sentinels": matches,
            }
            if not eligible:
                raise RuntimeError(f"historical bridge identity failed for {kind}; no silent reuse permitted")
    evidence.finish(result)
    return result


def rank(config, audit, splits, budget):
    evidence = progress(config, "rank", audit)
    if evidence.verified_complete():
        return json.loads((evidence.root / "summary.json").read_text())["orders"]
    old = json.loads((config.history_dir / "audit/locked_design.json").read_text())["models"]
    orders = {"infinitevl": old["infinitevl"]["sensitivity_order"], "qwen25vl_3b": old["qwen"]["sensitivity_order"]}
    kind = "qwen25vl_7b"
    with engine_for(config, kind, audit, evidence, budget) as engine:
        variants = {
            "B": Intervention(bits=None),
            "W": Intervention(bits=4),
            **{f"restore_{i:02d}": Intervention(bits=4, layer_bits={i: None}) for i in range(28)},
        }
        collected = {}
        for label, spec in variants.items():
            for sample in splits["diagnostic8"]:
                engine.evaluate(sample, "diagnostic8", label, spec)
            collected[label] = rows(evidence, kind, "diagnostic8", label, splits["diagnostic8"])
        orders[kind] = sorted(range(28), key=lambda i: (mean_nll(collected[f"restore_{i:02d}"]), i))
        gains = [
            {"layer": i, "absolute_nll_recovery": mean_nll(collected["W"]) - mean_nll(collected[f"restore_{i:02d}"])}
            for i in orders[kind]
        ]
    evidence.finish(
        {
            "orders": orders,
            "qwen7b_recovery": gains,
            "selection_data": "old_diagnostic8_within_oldscreen32",
            "new_evaluation_used": False,
        }
    )
    return orders


def execute_matrix(config, audit, splits, design, budget, stage, split, keys):
    evidence = progress(config, stage, audit, design)
    for kind in MODELS:
        model = design["models"][kind]
        labels = list(dict.fromkeys(["B", "W", *[v for key in keys for v in model[key]]]))
        pending = [
            label for label in labels if rows(evidence, kind, split, label, splits[split], required=False) is None
        ]
        if not pending:
            continue
        with engine_for(config, kind, audit, evidence, budget) as engine:
            for label in pending:
                record = model["conditions"][label]
                for sample in splits[split]:
                    engine.evaluate(sample, split, label, Intervention.from_payload(record["spec"]), record)
                update_schedule(config, audit, splits, design, budget)
    return evidence


def bridge(config, audit, splits, design, budget, audit_summary):
    evidence = progress(config, "bridge", audit, design)
    if evidence.verified_complete():
        return evidence
    oldroot = config.history_dir / "sweep"
    oldidentity = json.loads((oldroot / "identity.json").read_text())
    oldmarker = json.loads((oldroot / ".complete").read_text())
    oldhashes = json.loads((oldroot / "summary.sha256.json").read_text())
    if any(oldmarker.get(k) != v for k, v in oldidentity.items()) or oldhashes["summary.json"] != file_sha256(
        oldroot / "summary.json"
    ):
        raise RuntimeError("historical sweep completion identity or summary is invalid")
    history_manifest = oldhashes["sample_rows"]
    oldmap = {"infinitevl": "infinitevl", "qwen25vl_3b": "qwen"}
    for kind, old in oldmap.items():
        if not audit_summary["historical_reuse"][kind]["eligible"]:
            raise RuntimeError("historical import has not passed its live audit")
        for label in design["models"][kind]["main"]:
            root = config.history_dir / "sweep" / f"{old}__screen32__{label}" / "samples"
            by_id = {json.loads(p.read_text())["id"]: p for p in root.glob("*.json")}
            for sample in splits["oldscreen32"]:
                source = by_id[sample.id]
                if file_sha256(source) != history_manifest[str(source.relative_to(config.history_dir / "sweep"))]:
                    raise RuntimeError("historical row differs from verified completion manifest")
                value = json.loads(source.read_text())
                if value["row_sha256"] != json_digest({k: v for k, v in value.items() if k != "row_sha256"}):
                    raise RuntimeError("historical row payload corrupt")
                record = design["models"][kind]["conditions"][label]
                unit = {"source_sha256": file_sha256(source), "condition": record}
                variant = f"{kind}__oldscreen32__{label}"
                if evidence.completed(variant, sample.id, unit) is None:
                    evidence.commit(
                        variant,
                        sample.id,
                        unit,
                        {
                            "dataset": sample.dataset,
                            "target_text": value["target_text"],
                            "input_digest": value["input_digest"],
                            "condition": record,
                            "model_kind": kind,
                            "score": math.nan,
                            "teacher_seconds": 0,
                            "historical_import": {"path": str(source), "sha256": file_sha256(source), "verified": True},
                        },
                    )
    kind = "qwen25vl_7b"
    with engine_for(config, kind, audit, evidence, budget) as engine:
        for label in design["models"][kind]["bridge"]:
            record = design["models"][kind]["conditions"][label]
            for sample in splits["oldscreen32"]:
                engine.evaluate(sample, "oldscreen32", label, Intervention.from_payload(record["spec"]), record)
    evidence.finish(
        {
            "historical_reuse_verified": True,
            "qwen7b_anchor_conditions": 4,
            "cohort": "oldscreen32_contains_ranking_samples_not_independent_confirmation",
        }
    )
    return evidence


def update_schedule(config, audit, splits, design, budget):
    tasks = (("confirm", "confirm128", ("confirm",)), ("sweep", "screen32", ("main", "random")))
    by_model = {}
    for kind in MODELS:
        complete = pending_rows = 0
        durations = []
        for stage, split, keys in tasks:
            evidence = progress(config, stage, audit, design)
            labels = set(label for key in keys for label in design["models"][kind][key])
            for label in labels:
                current = evidence.rows(f"{kind}__{split}__{label}")
                complete += len(current)
                pending_rows += len(splits[split]) - len(current)
                durations.extend(r["teacher_seconds"] for r in current)
        estimated = statistics.mean(durations) if durations else None
        by_model[kind] = {
            "complete_rows": complete,
            "pending_rows": pending_rows,
            "mean_teacher_seconds": estimated,
            "estimated_remaining_seconds": pending_rows * estimated if estimated else None,
        }
    estimates = [m["estimated_remaining_seconds"] for m in by_model.values()]
    atomic_json(
        config.output_dir / "schedule.json",
        {
            "models": by_model,
            "remaining_inference_hours": budget.remaining_seconds() / 3600,
            "estimated_core_remaining_hours": sum(estimates) / 3600 if all(v is not None for v in estimates) else None,
            "deadline_unix": budget.deadline_unix,
            "estimate_is_not_completion_guarantee": True,
        },
    )


def qa(config, audit, splits, design, budget):
    evidence = progress(config, "qa", audit, design)
    teacher_progress = progress(config, "confirm", audit, design)
    if evidence.verified_complete():
        return evidence
    for kind in MODELS:
        with engine_for(config, kind, audit, evidence, budget) as engine:
            for label in design["models"][kind]["qa"]:
                condition_rows = {
                    r["id"]: r for r in rows(teacher_progress, kind, "confirm128", label, splits["confirm128"])
                }
                for sample in splits["qa32"]:
                    engine.generate(
                        sample, label, design["models"][kind]["conditions"][label], condition_rows[sample.id]
                    )
    evidence.finish(
        {
            "conditions": 15,
            "generation_calls": 480,
            "teacher_metrics_reused_from_confirm128": True,
            "role": "directional_sanity_check_not_equivalence_or_no_degradation_claim",
        }
    )
    return evidence


def optional_old_sweep(config, audit, splits, design, budget, audit_summary):
    evidence = progress(config, "old_supplement", audit, design)
    bridge_evidence = progress(config, "bridge", audit, design)
    if evidence.verified_complete():
        return True
    model = design["models"]["qwen25vl_7b"]
    seconds = audit_summary["models"]["qwen25vl_7b"]["teacher_mean_seconds"]
    observed = [
        r["teacher_seconds"]
        for label in model["bridge"]
        for r in bridge_evidence.rows(f"qwen25vl_7b__oldscreen32__{label}")
    ]
    if observed:
        seconds = statistics.mean(observed)
    pending = sum(
        len(splits["oldscreen32"]) - len(evidence.rows(f"qwen25vl_7b__oldscreen32__{label}"))
        for label in model["main"]
        if label not in model["bridge"]
    )
    if budget.remaining_seconds() < pending * seconds * 1.3 + 3600:
        atomic_json(
            config.output_dir / "optional_omissions.json",
            {
                "old_qwen7b_dense": {
                    "reason": "insufficient_complete_unit_budget",
                    "predicted_seconds_with_margin": pending * seconds * 1.3,
                    "remaining_seconds": budget.remaining_seconds(),
                }
            },
        )
        return False
    with engine_for(config, "qwen25vl_7b", audit, evidence, budget) as engine:
        for label in model["main"]:
            if label in model["bridge"]:
                continue
            record = model["conditions"][label]
            for sample in splits["oldscreen32"]:
                engine.evaluate(sample, "oldscreen32", label, Intervention.from_payload(record["spec"]), record)
    evidence.finish({"qwen7b_old_full_curve": True, "anchors_from_bridge_stage": True})
    return True
