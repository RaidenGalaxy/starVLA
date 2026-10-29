from __future__ import annotations

import math

import numpy as np

from experiments.ffn_quant_rootcause.interventions import Intervention
from experiments.infinitevl_quant.progress import file_sha256, json_digest

from .config import MODELS, POLICIES, QA_COUNTS, SEEDS
from .statistics import Bootstrap, adjust_holm, interval, spearman_curves
from .sweep import complete_rows


def collect(config, payload, splits, design, progress):
    result = {}
    for kind in MODELS:
        model = design["models"][kind]
        for split, key, count in (("screen32", "screen", 8), ("confirm192", "confirm", 48)):
            cells = {}
            for label, record in model["conditions"].items():
                rows = complete_rows(progress, f"{kind}__{split}__{label}", splits[key], required=False)
                if rows is None:
                    continue
                for row in rows:
                    if row["row_sha256"] != json_digest({k: v for k, v in row.items() if k != "row_sha256"}):
                        raise RuntimeError("analysis found a corrupt row")
                    if row["provenance"]["stage_identity"] != progress.identity_digest:
                        raise RuntimeError("analysis found a foreign stage identity")
                    expected = {k: v for k, v in record.items() if k != "spec"}
                    if (
                        row["compression"] != expected
                        or row["model_content_sha256"] != payload["models"][kind]["content_sha256"]
                    ):
                        raise RuntimeError("compression/model metadata differ from frozen design")
                    spec = Intervention.from_payload(record["spec"])
                    unit = {
                        "spec": spec.payload(),
                        "generate": split == "confirm192",
                        "input_digest": row["input_digest"],
                    }
                    if row["provenance"]["unit"] != json_digest(unit):
                        raise RuntimeError("row belongs to a different inference configuration")
                datasets = {r["dataset"] for r in rows}
                if len(datasets) != 4 or any(sum(r["dataset"] == ds for r in rows) != count for ds in datasets):
                    raise RuntimeError("dataset quota differs from approved sample counts")
                cells[label] = rows
            if "B" in cells:
                baseline = {r["id"]: r for r in cells["B"]}
                for rows in cells.values():
                    if any(r["input_digest"] != baseline[r["id"]]["input_digest"] for r in rows):
                        raise RuntimeError("processed inputs changed between masks")
                for row in baseline.values():
                    from experiments.infinitevl_quant.progress import safe_id

                    path = config.output_dir / "references/sweep" / kind / split / f"{safe_id(row['id'])}.pt"
                    if not path.is_file() or file_sha256(path) != row["reference_sha256"]:
                        raise RuntimeError("baseline reference logits are missing or corrupt")
            result[kind, split] = cells
    return result


def analyze(config, payload, splits, design, progress, *, partial=False, omissions=None):
    raw = collect(config, payload, splits, design, progress)
    summary = {
        "models": {},
        "tests": {},
        "holm_family_complete": False,
        "history": "historically_reused_evaluation",
        "partial": partial,
        "budgeted_omissions": omissions or {},
        "bootstrap_samples": config.bootstrap_samples,
        "correlation_interpretation": "descriptive paired-bootstrap intervals, not independent x points",
        "spearman_rank_rounding_decimals": 12,
        "confidence_intervals": "pointwise95_not_simultaneous",
    }
    pvalues = {}
    for kind in MODELS:
        model, out = design["models"][kind], {"cohorts": {}}
        summary["models"][kind] = out
        for split in ("confirm192", "screen32"):
            cells = raw[kind, split]
            if "B" not in cells:
                continue
            bootstrap = Bootstrap(cells["B"], config.bootstrap_samples)
            points, distributions = {}, {}
            for label, rows in cells.items():
                points[label], distributions[label] = bootstrap.summarize(rows)
                points[label]["condition"] = model["conditions"][label]
            cohort = {"points": points, "correlations": {}, "policy_points": {}}
            out["cohorts"][split] = cohort
            qs = list(QA_COUNTS) if split == "confirm192" else list(range(37))
            for policy in POLICIES:
                labels = [model["aliases"][policy][str(q)] for q in qs]
                cohort["policy_points"][policy] = [
                    {"q": q, "compression_percent": 100 * q / 48, "id": label, "metrics": points.get(label)}
                    for q, label in zip(qs, labels, strict=True)
                ]
                if not all(label in points for label in labels):
                    continue
                for metric in ("nll", "qa_macro") if split == "confirm192" else ("nll",):
                    observed = spearman_curves([points[label][metric] for label in labels])[0]
                    boot = spearman_curves(np.stack([distributions[label][metric] for label in labels], axis=1))
                    cohort["correlations"][policy + ":" + metric] = {
                        "rho": float(observed) if math.isfinite(observed) else None,
                        "ci95": interval(boot),
                        "undefined_bootstrap_replicates": int((~np.isfinite(boot)).sum()),
                        "compression_points": len(qs),
                        "sample_unit": "dataset_stratified_sample_id",
                    }
            if split == "confirm192":
                for q in QA_COUNTS[1:-1]:
                    damage, protect = (model["aliases"][p][str(q)] for p in POLICIES)
                    if damage not in points or protect not in points:
                        continue
                    contrast = bootstrap.contrast(
                        cells[damage], cells[protect], distributions[damage], distributions[protect]
                    )
                    for metric, value in contrast.items():
                        name = f"{kind}:q{q}:{metric}"
                        summary["tests"][name] = dict(
                            value, model=kind, q=q, metric=metric, compression_percent=100 * q / 48
                        )
                        pvalues[name] = value["p_two_sided"]
            else:
                cohort["random_points"] = []
                cohort["random_seed_points"] = {}
                if not (omissions or {}).get("random_nll"):
                    for seed in SEEDS:
                        cohort["random_seed_points"][f"random_{seed}"] = [
                            {
                                "q": q,
                                "compression_percent": 100 * q / 48,
                                "id": model["aliases"][f"random_{seed}"][str(q)],
                                "metrics": points.get(model["aliases"][f"random_{seed}"][str(q)]),
                            }
                            for q in QA_COUNTS
                        ]
                    for q in QA_COUNTS:
                        labels = [model["aliases"][f"random_{z}"][str(q)] for z in SEEDS]
                        if all(label in points for label in labels):
                            values = [points[label]["nll"] for label in labels]
                            cohort["random_points"].append(
                                {
                                    "q": q,
                                    "compression_percent": 100 * q / 48,
                                    "nll_mean": float(np.mean(values)),
                                    "nll_min": min(values),
                                    "nll_max": max(values),
                                    "seeds": list(SEEDS),
                                    "band": "three_seed_range_not_sample_confidence_interval",
                                }
                            )
    if len(pvalues) == 16:
        for name, value in adjust_holm(pvalues).items():
            summary["tests"][name]["p_holm"] = value
        summary["holm_family_complete"] = True
    if not partial:
        for kind in MODELS:
            for split, labels in (
                ("confirm192", design["models"][kind]["qa_order"]),
                ("screen32", design["models"][kind]["main_screen_order"]),
            ):
                if not all(v in raw[kind, split] for v in labels):
                    raise RuntimeError("mandatory curves are incomplete")
        if not summary["holm_family_complete"]:
            raise RuntimeError("sixteen-comparison Holm family is incomplete")
    return summary, raw
