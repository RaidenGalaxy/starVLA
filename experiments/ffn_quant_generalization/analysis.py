from __future__ import annotations

import math

import numpy as np
from scipy.stats import rankdata

from experiments.ffn_quant_compression.statistics import Bootstrap, adjust_holm, interval, spearman_curves
from experiments.infinitevl_quant.progress import json_digest

from .config import DATASETS, MODELS, POLICIES, SEEDS
from .runtime import progress, rows
from .scoring import score_prediction


def independent_rho(values):
    ranks = rankdata(values)
    return float(np.corrcoef(np.arange(len(values)), ranks)[0, 1]) if np.ptp(ranks) else None


def analyze(config, audit, splits, design, partial=False):
    output = {
        "format_version": 1,
        "models": {},
        "tests": {},
        "bootstrap_samples": config.bootstrap_samples,
        "partial": partial,
        "core_complete": True,
        "holm_family_size": 36,
        "uncertainty": "pointwise paired image-sample bootstrap; random-seed range is separate",
        "compute": "CPU_FP32_fake_quant_only",
        "coverage_axis": "FFN quantization coverage percent 0..100",
    }
    verification = {
        "passed": True,
        "checked_rows": 0,
        "nll_points": 0,
        "nll_intervals": 0,
        "correlations": 0,
        "contrasts": 0,
        "qa_rescored": 0,
        "max_token_sum_difference": 0.0,
    }
    pending_p = {}
    for kind, (size, _, _) in MODELS.items():
        model = design["models"][kind]
        out = output["models"][kind] = {"layers": size, "cohorts": {}, "qa": {}, "background_interaction": {}}
        for stage, split, order in (
            ("sweep", "screen32", model["main"] + model["random"]),
            ("confirm", "confirm128", model["confirm"]),
        ):
            evidence = progress(config, stage, audit, design)
            cells = {}
            for label in dict.fromkeys(order):
                values = rows(evidence, kind, split, label, splits[split], required=False)
                if values is None:
                    output["core_complete"] = False
                    continue
                for row in values:
                    if row["condition"] != model["conditions"][label]:
                        raise RuntimeError("row mask metadata differs from locked design")
                    expected_unit = json_digest(
                        {
                            "spec": model["conditions"][label]["spec"],
                            "input_digest": row["input_digest"],
                            "operation": "teacher",
                        }
                    )
                    if row["provenance"]["unit"] != expected_unit:
                        raise RuntimeError("row is bound to a different operation or mask")
                    if row["model_content_sha256"] != audit["models"][kind]["content_sha256"]:
                        raise RuntimeError("row checkpoint identity differs")
                    metrics = row["target_text"]
                    if (
                        len(metrics["token_losses"]) != metrics["tokens"]
                        or len(metrics["token_ids"]) != metrics["tokens"]
                    ):
                        raise RuntimeError("target token arrays are not aligned")
                    difference = abs(math.fsum(metrics["token_losses"]) - metrics["total_nll"])
                    verification["max_token_sum_difference"] = max(verification["max_token_sum_difference"], difference)
                    if difference > 1e-6 * max(1, abs(metrics["total_nll"])):
                        raise ArithmeticError("independent token-loss sum differs beyond FP32 reduction tolerance")
                    verification["checked_rows"] += 1
                cells[label] = values
            if "B" not in cells:
                continue
            baseline = {r["id"]: r for r in cells["B"]}
            for values in cells.values():
                if any(r["input_digest"] != baseline[r["id"]]["input_digest"] for r in values):
                    raise RuntimeError("paired input digest differs across layer masks")
            cohort = out["cohorts"][split] = {"scopes": {}, "complete_conditions": len(cells)}
            qs = list(range(size + 1)) if split == "screen32" else model["confirm_q"]
            for scope in (*DATASETS, "overall"):
                chosen = {
                    label: [r for r in values if scope == "overall" or r["dataset"] == scope]
                    for label, values in cells.items()
                }
                boot = Bootstrap(chosen["B"], config.bootstrap_samples)
                # Independently regroup bootstrap draws as sample multiplicities.
                multiplicity = np.stack([np.bincount(draw, minlength=len(boot.ids)) for draw in boot.draw])
                point, dist = {}, {}
                for label, values in chosen.items():
                    point[label], dist[label] = boot.summarize(values)
                    by_id = {r["id"]: r for r in values}
                    losses = np.array([by_id[i]["target_text"]["total_nll"] for i in boot.ids])
                    token = np.array([by_id[i]["target_text"]["tokens"] for i in boot.ids])
                    expected = math.fsum(losses) / sum(token)
                    oracle = (multiplicity @ losses) / (multiplicity @ token)
                    if abs(expected - point[label]["nll"]) > 1e-12 or not np.allclose(
                        np.quantile(oracle, [0.025, 0.975]), point[label]["nll_ci95"], atol=1e-12, rtol=0
                    ):
                        raise ArithmeticError("independent NLL point/bootstrap interval mismatch")
                    verification["nll_points"] += 1
                    verification["nll_intervals"] += 1
                    point[label]["delta_nll"] = point[label]["nll"] - point["B"]["nll"] if "B" in point else 0
                    point[label]["relative_nll_percent"] = (
                        100 * (point[label]["nll"] / point["B"]["nll"] - 1) if point.get("B", {}).get("nll") else None
                    )
                result = cohort["scopes"][scope] = {"points": point, "curves": {}, "random": {}}
                for policy in POLICIES:
                    labels = [model["aliases"][policy][str(q)] for q in qs]
                    curve = result["curves"][policy] = {
                        "points": [],
                        "complete": all(label in point for label in labels),
                    }
                    for q, label in zip(qs, labels, strict=True):
                        if label in point:
                            curve["points"].append(
                                {
                                    "q": q,
                                    "coverage_percent": 100 * q / size,
                                    "bf16_saving_percent": 75 * q / size,
                                    "id": label,
                                    "metrics": point[label],
                                }
                            )
                    if not curve["complete"]:
                        continue
                    values = [point[label]["nll"] for label in labels]
                    observed = float(spearman_curves(values)[0])
                    boot_values = np.stack([dist[label]["nll"] for label in labels], axis=1)
                    rho_draw = spearman_curves(boot_values)
                    oracle_rho = independent_rho(np.round(values, 12))
                    if oracle_rho is not None and abs(oracle_rho - observed) > 1e-12:
                        raise ArithmeticError("independent Spearman point mismatch")
                    # Check the complete CI through an independent rank implementation.
                    ranks = rankdata(np.round(boot_values, 12), axis=1)
                    ranks -= ranks.mean(axis=1, keepdims=True)
                    x = np.arange(len(qs), dtype=float) - (len(qs) - 1) / 2
                    denom = np.sqrt((ranks * ranks).sum(1) * (x * x).sum())
                    oracle_draw = np.divide((ranks * x).sum(1), denom, out=np.full(len(ranks), np.nan), where=denom != 0)
                    if interval(oracle_draw) is not None and not np.allclose(
                        interval(oracle_draw), interval(rho_draw), atol=1e-12, rtol=0
                    ):
                        raise ArithmeticError("independent Spearman interval mismatch")
                    curve["spearman"] = {
                        "rho": observed if math.isfinite(observed) else None,
                        "ci95": interval(rho_draw),
                        "undefined_replicates": int((~np.isfinite(rho_draw)).sum()),
                    }
                    verification["correlations"] += 1
                    if split == "screen32":
                        curve["marginal"] = []
                        for q in range(1, size + 1):
                            left, right = labels[q - 1], labels[q]
                            curve["marginal"].append(
                                {
                                    "q": q,
                                    "newly_quantized_layer": model["sensitivity_order"][q - 1]
                                    if policy == POLICIES[0]
                                    else model["sensitivity_order"][size - q],
                                    "delta": point[right]["nll"] - point[left]["nll"],
                                    "ci95": interval(dist[right]["nll"] - dist[left]["nll"]),
                                }
                            )
                if split == "screen32":
                    for q in (size // 4, size // 2, 3 * size // 4):
                        labels = [model["aliases"][f"random_{seed}"][str(q)] for seed in SEEDS]
                        if all(label in point for label in labels):
                            values = [point[label]["nll"] for label in labels]
                            random_draw = np.stack([dist[label]["nll"] for label in labels]).mean(0)
                            sensitive = model["aliases"][POLICIES[0]][str(q)]
                            protected = model["aliases"][POLICIES[1]][str(q)]
                            result["random"][str(q)] = {
                                "coverage_percent": 100 * q / size,
                                "seeds": list(SEEDS),
                                "values": values,
                                "mean": float(np.mean(values)),
                                "seed_range": [min(values), max(values)],
                                "sample_mean_ci95": interval(np.stack([dist[label]["nll"] for label in labels]).mean(0)),
                            }
                            if protected in point and sensitive in point:
                                result["random"][str(q)].update(
                                    protection_advantage=float(np.mean(values)) - point[protected]["nll"],
                                    protection_advantage_ci95=interval(random_draw - dist[protected]["nll"]),
                                    sensitive_damage_vs_random=point[sensitive]["nll"] - float(np.mean(values)),
                                    sensitive_damage_vs_random_ci95=interval(dist[sensitive]["nll"] - random_draw),
                                )
                else:
                    for q in model["confirm_q"][1:-1]:
                        left, right = [model["aliases"][p][str(q)] for p in POLICIES]
                        if left not in chosen or right not in chosen:
                            continue
                        dl, _ = boot.arrays(chosen[left])
                        pl, _ = boot.arrays(chosen[right])
                        differences = (dl - pl) / boot.token.sum()
                        value = float(differences.sum())
                        randomized = (boot.signs * differences).sum(1)
                        p = float((1 + (np.abs(randomized) >= abs(value)).sum()) / (config.bootstrap_samples + 1))
                        test = {
                            "model": kind,
                            "scope": scope,
                            "q": q,
                            "coverage_percent": 100 * q / size,
                            "protection_advantage_nll": value,
                            "ci95": interval(dist[left]["nll"] - dist[right]["nll"]),
                            "p_two_sided": p,
                            "p_holm": None,
                            "test": "paired_image_sample_sign_flip_10000",
                        }
                        # Independent scalar summation verifies every randomized tail count.
                        manual = np.einsum("ij,j->i", boot.signs, differences, optimize=False)
                        oracle_p = (1 + (np.abs(manual) >= abs(value)).sum()) / (config.bootstrap_samples + 1)
                        if abs(oracle_p - p) > 1e-12:
                            # Rounding exactly at the observed tail is handled consistently.
                            tolerance = 1e-14 * max(1.0, abs(value))
                            p = float(
                                (1 + (np.abs(randomized) >= abs(value) - tolerance).sum())
                                / (config.bootstrap_samples + 1)
                            )
                            oracle_p = float(
                                (1 + (np.abs(manual) >= abs(value) - tolerance).sum()) / (config.bootstrap_samples + 1)
                            )
                            if abs(oracle_p - p) > 1e-12:
                                raise ArithmeticError("independent sign randomization mismatch")
                            test["p_two_sided"] = p
                        key = f"{kind}:{scope}:q{q}"
                        output["tests"][key] = test
                        pending_p[key] = p
                        verification["contrasts"] += 1
                if split == "screen32":
                    only = model["aliases"][POLICIES[0]]["1"]
                    except_one = model["aliases"][POLICIES[1]][str(size - 1)]
                    if all(k in point for k in ("B", "W", only, except_one)):
                        contrast = dist["W"]["nll"] - dist[except_one]["nll"] - dist[only]["nll"] + dist["B"]["nll"]
                        out["background_interaction"][scope] = {
                            "layer": model["sensitivity_order"][0],
                            "single_layer_damage": point[only]["nll"] - point["B"]["nll"],
                            "damage_on_other_layers_W4": point["W"]["nll"] - point[except_one]["nll"],
                            "interaction": point["W"]["nll"]
                            - point[except_one]["nll"]
                            - point[only]["nll"]
                            + point["B"]["nll"],
                            "ci95": interval(contrast),
                            "exploratory": True,
                        }
        qa_progress = progress(config, "qa", audit, design)
        qa_sample = {s.id: s for s in splits["qa32"]}
        for label in model["qa"]:
            values = rows(qa_progress, kind, "qa32", label, splits["qa32"], required=False)
            if values is None:
                continue
            for row in values:
                if score_prediction(qa_sample[row["id"]], row["prediction"]) != row["score"]:
                    raise ArithmeticError("QA independent rescoring mismatch")
                verification["qa_rescored"] += 1
            boot = Bootstrap(values, config.bootstrap_samples)
            metrics = boot.summarize(values)[0]
            by_id = {r["id"]: r for r in values}
            scores = np.array([by_id[i]["score"] for i in boot.ids])
            independent_macro = 100 * np.mean([scores[g].mean() for g in boot.groups.values()])
            independent_draw = []
            offset = 0
            for group in boot.groups.values():
                independent_draw.append(
                    np.array(
                        [math.fsum(scores[draw]) / len(group) for draw in boot.draw[:, offset : offset + len(group)]]
                    )
                )
                offset += len(group)
            oracle_ci = interval(100 * np.stack(independent_draw).mean(0))
            if abs(independent_macro - metrics["qa_macro"]) > 1e-12 or not np.allclose(
                oracle_ci, metrics["qa_ci95"], atol=1e-12, rtol=0
            ):
                raise ArithmeticError("independent QA aggregation/interval mismatch")
            out["qa"][label] = {"condition": model["conditions"][label], "metrics": metrics}
    if len(pending_p) == 36:
        adjusted = adjust_holm(pending_p)
        sorted_items = sorted(pending_p.items(), key=lambda item: item[1])
        last = 0.0
        for i, (key, p) in enumerate(sorted_items):
            last = max(last, min(1.0, (36 - i) * p))
            if abs(last - adjusted[key]) > 1e-15:
                raise ArithmeticError("independent Holm adjustment mismatch")
            output["tests"][key]["p_holm"] = adjusted[key]
        output["holm_family_complete"] = True
    else:
        output["holm_family_complete"] = False
    if not partial and not output["core_complete"]:
        raise RuntimeError("cannot finalize incomplete primary curves, random controls or confirmation")
    output["partial"] = not output["core_complete"]
    output["old_bridge"] = []
    bridge = progress(config, "bridge", audit, design)
    supplement = progress(config, "old_supplement", audit, design)
    if "oldscreen32" in splits:
        for kind in MODELS:
            for label in design["models"][kind]["main"]:
                values = rows(bridge, kind, "oldscreen32", label, splits["oldscreen32"], required=False)
                if values is None:
                    values = rows(supplement, kind, "oldscreen32", label, splits["oldscreen32"], required=False)
                if values is None:
                    continue
                for scope in [*sorted({r["dataset"] for r in values}), "overall"]:
                    selected = [r for r in values if scope == "overall" or r["dataset"] == scope]
                    count = sum(r["target_text"]["tokens"] for r in selected)
                    total = math.fsum(r["target_text"]["total_nll"] for r in selected)
                    output["old_bridge"].append(
                        {
                            "model": kind,
                            "scope": scope,
                            "mask_id": label,
                            "q": design["models"][kind]["conditions"][label]["q"],
                            "samples": len(selected),
                            "target_tokens": count,
                            "total_nll": total,
                            "nll": total / count,
                            "historical_import": kind != "qwen25vl_7b",
                            "independent_confirmation": False,
                        }
                    )
    output["verification"] = verification
    output["design_sha256"] = json_digest(design)
    return output
