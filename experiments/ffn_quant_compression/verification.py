"""Independent reductions, scorer replay and resampling audit; no model calls."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from experiments.infinitevl_quant_ffn_bf16_4x64.scoring import score_prediction

from .config import POLICIES, QA_COUNTS


def reference_rho(values):
    values = np.round(np.atleast_2d(values), 12)
    output = []
    x = np.arange(values.shape[1], dtype=float)
    x -= np.mean(x)
    for start in range(0, len(values), 256):
        chunk = values[start : start + 256]
        less = (chunk[:, None, :] < chunk[:, :, None]).sum(axis=-1)
        ties = (chunk[:, None, :] == chunk[:, :, None]).sum(axis=-1)
        ranks = 1 + less + (ties - 1) / 2
        ranks -= np.mean(ranks, axis=1, keepdims=True)
        den = np.sqrt((ranks * ranks).sum(axis=1) * (x * x).sum())
        output.extend(np.divide((ranks * x).sum(axis=1), den, out=np.full(len(chunk), np.nan), where=den != 0).tolist())
    return np.array(output)


def verify(summary, raw, splits, design, count=10_000):
    checked, maximum = Counter(), Counter()

    def close(a, b, label, tol=1e-9):
        if a is None or b is None:
            if a != b:
                raise ArithmeticError(f"independent audit null mismatch: {label}")
            return
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        error = float(np.max(np.abs(a - b)))
        maximum[label] = max(maximum[label], error)
        if not np.allclose(a, b, rtol=tol, atol=tol):
            raise ArithmeticError(f"independent audit mismatch {label}: {a} versus {b}")
        checked[label] += 1

    references = {}
    for (kind, split), cells in raw.items():
        if not cells:
            continue
        ids = sorted(r["id"] for r in cells["B"])
        samples = {s.id: s for s in splits["confirm" if split == "confirm192" else "screen"]}
        base = {r["id"]: r for r in cells["B"]}
        datasets = sorted({r["dataset"] for r in cells["B"]})
        groups = [np.array([i for i, name in enumerate(ids) if base[name]["dataset"] == ds]) for ds in datasets]
        token = np.array([base[name]["target_text"]["tokens"] for name in ids])
        rng = np.random.default_rng(42)
        draw = np.concatenate([g[rng.integers(len(g), size=(count, len(g)))] for g in groups], axis=1)
        signals = 2 * np.random.default_rng(42).integers(2, size=(count, len(ids))) - 1
        cohort = summary["models"][kind]["cohorts"][split]
        for label, rows in cells.items():
            by_id = {r["id"]: r for r in rows}
            losses = np.array([by_id[name]["target_text"]["total_nll"] for name in ids])
            scores = np.array([by_id[name]["score"] for name in ids])
            point = cohort["points"][label]
            close(math.fsum(losses) / int(token.sum()), point["nll"], "nll_point")
            distribution = losses[draw].sum(axis=1) / token[draw].sum(axis=1)
            close(np.quantile(distribution, [0.025, 0.975]), point["nll_ci95"], "nll_ci")
            values = {"nll": distribution, "loss": losses, "scores": scores}
            if split == "confirm192":
                for row in rows:
                    rescored = score_prediction(samples[row["id"]], row["prediction"])
                    close(rescored.score, row["score"], "qa_rescore", tol=1e-12)
                    if abs(
                        row["target_text"]["total_nll"] / row["target_text"]["tokens"] - row["target_text"]["nll"]
                    ) > 1e-6 * max(1, row["target_text"]["nll"]):
                        raise ArithmeticError("stored FP32 sample sum/mean mismatch")
                qa = math.fsum(math.fsum(scores[g]) / len(g) for g in groups) * 100 / len(groups)
                close(qa, point["qa_macro"], "qa_macro_point")
                boot_groups, offset = [], 0
                for group in groups:
                    boot_groups.append(scores[draw[:, offset : offset + len(group)]].mean(axis=1))
                    offset += len(group)
                values["qa_macro"] = np.mean(boot_groups, axis=0) * 100
                close(np.quantile(values["qa_macro"], [0.025, 0.975]), point["qa_ci95"], "qa_ci")
            references[kind, split, label] = values
        qs = QA_COUNTS if split == "confirm192" else range(37)
        for policy in POLICIES:
            labels = [design["models"][kind]["aliases"][policy][str(q)] for q in qs]
            for metric in ("nll", "qa_macro") if split == "confirm192" else ("nll",):
                saved = cohort["correlations"][policy + ":" + metric]
                observed = reference_rho([cohort["points"][label][metric] for label in labels])[0]
                close(float(observed) if math.isfinite(observed) else None, saved["rho"], "spearman_point")
                boot = reference_rho(np.stack([references[kind, split, label][metric] for label in labels], axis=1))
                finite = boot[np.isfinite(boot)]
                close(
                    np.quantile(finite, [0.025, 0.975]).tolist() if len(finite) else None, saved["ci95"], "spearman_ci"
                )
                if int((~np.isfinite(boot)).sum()) != saved["undefined_bootstrap_replicates"]:
                    raise ArithmeticError("undefined correlation count differs")
        if split == "confirm192":
            weights = np.zeros(len(ids))
            for group in groups:
                weights[group] = 100 / (len(groups) * len(group))
            for q in QA_COUNTS[1:-1]:
                left, right = (design["models"][kind]["aliases"][p][str(q)] for p in POLICIES)
                dl, pl = references[kind, split, left], references[kind, split, right]
                for metric in ("nll", "qa_macro"):
                    saved = summary["tests"][f"{kind}:q{q}:{metric}"]
                    delta = (
                        (dl["loss"] - pl["loss"]) / token.sum()
                        if metric == "nll"
                        else (pl["scores"] - dl["scores"]) * weights
                    )
                    obs = float(np.sum(delta))
                    null = (signals * delta).sum(axis=1)
                    p = (np.count_nonzero(np.abs(null) >= abs(obs)) + 1) / (count + 1)
                    close(obs, saved["advantage_of_protection"], "contrast_point")
                    close(p, saved["p_two_sided"], "contrast_p", tol=1e-12)
                    boot = dl[metric] - pl[metric] if metric == "nll" else pl[metric] - dl[metric]
                    close(np.quantile(boot, [0.025, 0.975]), saved["ci95"], "contrast_ci")
    values = sorted((v["p_two_sided"], k) for k, v in summary["tests"].items())
    for rank, (_, name) in enumerate(values):
        adjusted = min(1.0, max((len(values) - j) * values[j][0] for j in range(rank + 1)))
        close(adjusted, summary["tests"][name]["p_holm"], "holm_p", tol=1e-12)
    return {
        "status": "passed",
        "method": "independent scalar aggregation, scorer replay, integer bootstrap draws and pairwise tie ranks",
        "checks": dict(checked),
        "maximum_absolute_differences": dict(maximum),
        "new_model_inference": False,
    }
