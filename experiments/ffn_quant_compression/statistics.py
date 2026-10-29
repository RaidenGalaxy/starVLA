from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


def average_ranks(value):
    """Vectorized average tie ranks; rows are bootstrap curves, not samples."""
    # Equal categorical QA aggregates can differ by a few FP64 ulps.
    values = np.round(np.atleast_2d(np.asarray(value, dtype=float)), 12)
    order = np.argsort(values, axis=1, kind="stable")
    sorted_values = np.take_along_axis(values, order, axis=1)
    n = values.shape[1]
    start = np.concatenate((np.ones((len(values), 1), dtype=bool), np.diff(sorted_values, axis=1) != 0), axis=1)
    end = np.concatenate((np.diff(sorted_values, axis=1) != 0, np.ones((len(values), 1), dtype=bool)), axis=1)
    first = np.maximum.accumulate(np.where(start, np.arange(n), 0), axis=1)
    last = np.minimum.accumulate(np.where(end, np.arange(n) + 1, n)[:, ::-1], axis=1)[:, ::-1]
    ranks = np.empty_like(values)
    np.put_along_axis(ranks, order, (first + last + 1) / 2, axis=1)
    return ranks


def spearman_curves(value):
    ranks = average_ranks(value)
    x = np.arange(ranks.shape[1], dtype=float)
    x -= x.mean()
    ranks -= ranks.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.square(ranks).sum(axis=1) * np.square(x).sum())
    return np.divide((ranks * x).sum(axis=1), denominator, out=np.full(len(ranks), np.nan), where=denominator != 0)


def interval(values):
    finite = np.asarray(values)[np.isfinite(values)]
    return np.quantile(finite, [0.025, 0.975]).tolist() if len(finite) else None


def adjust_holm(values):
    result, previous = {}, 0.0
    for rank, key in enumerate(sorted(values, key=values.get)):
        previous = max(previous, min(1.0, (len(values) - rank) * values[key]))
        result[key] = previous
    return result


class Bootstrap:
    def __init__(self, baseline, count=10_000):
        self.ids = sorted(r["id"] for r in baseline)
        self.baseline = {r["id"]: r for r in baseline}
        if len(self.baseline) != len(self.ids) or not self.ids:
            raise ValueError("empty or duplicate sample IDs")
        self.groups = defaultdict(list)
        for i, name in enumerate(self.ids):
            self.groups[self.baseline[name]["dataset"]].append(i)
        self.token = np.array([self.baseline[i]["target_text"]["tokens"] for i in self.ids], dtype=float)
        if (self.token <= 0).any():
            raise ValueError("nonpositive target-token count")
        rng = np.random.default_rng(42)
        self.draw = np.concatenate([rng.choice(g, (count, len(g)), replace=True) for g in self.groups.values()], axis=1)
        self.signs = np.random.default_rng(42).choice((-1, 1), (count, len(self.ids)))
        self.count = count

    def arrays(self, rows):
        by_id = {r["id"]: r for r in rows}
        if len(by_id) != len(rows) or set(by_id) != set(self.ids):
            raise ValueError("sample sets differ between masks")
        for i, identifier in enumerate(self.ids):
            row = by_id[identifier]
            if row["dataset"] != self.baseline[identifier]["dataset"] or row["target_text"]["tokens"] != self.token[i]:
                raise ValueError("paired dataset/token mismatch")
        losses = np.array([by_id[i]["target_text"]["total_nll"] for i in self.ids])
        scores = np.array([by_id[i]["score"] for i in self.ids])
        if not np.isfinite(losses).all():
            raise FloatingPointError("nonfinite NLL")
        return losses, scores

    def summarize(self, rows):
        losses, scores = self.arrays(rows)
        nll = math.fsum(losses) / math.fsum(self.token)
        nll_boot = losses[self.draw].sum(axis=1) / self.token[self.draw].sum(axis=1)
        point = {
            "samples": len(rows),
            "target_tokens": int(self.token.sum()),
            "total_nll": math.fsum(losses),
            "nll": nll,
            "nll_ci95": interval(nll_boot),
        }
        distributions = {"nll": nll_boot}
        if np.isfinite(scores).all():
            point["qa_macro"] = float(100 * np.mean([scores[g].mean() for g in self.groups.values()]))
            offset, parts = 0, []
            for group in self.groups.values():
                parts.append(scores[self.draw[:, offset : offset + len(group)]].mean(axis=1))
                offset += len(group)
            distributions["qa_macro"] = 100 * np.mean(parts, axis=0)
            point["qa_ci95"] = interval(distributions["qa_macro"])
        else:
            point.update(qa_macro=None, qa_ci95=None)
        point["datasets"] = {
            ds: {
                "samples": len(g),
                "target_tokens": int(self.token[g].sum()),
                "nll": float(losses[g].sum() / self.token[g].sum()),
                "qa": float(100 * scores[g].mean()) if np.isfinite(scores[g]).all() else None,
            }
            for ds, g in self.groups.items()
        }
        point["reused_rows"] = sum(bool(r.get("historical_reuse")) for r in rows)
        return point, distributions

    def contrast(self, damage, protect, d_boot, p_boot):
        dl, ds = self.arrays(damage)
        pl, ps = self.arrays(protect)
        differences = {"nll": (dl - pl) / self.token.sum()}
        qa_weights = np.zeros(len(self.ids))
        for group in self.groups.values():
            qa_weights[group] = 100 / (len(self.groups) * len(group))
        differences["qa_macro"] = (ps - ds) * qa_weights
        result = {}
        for metric, difference in differences.items():
            observed = float(difference.sum())
            randomized = (self.signs * difference).sum(axis=1)
            p = float((1 + (np.abs(randomized) >= abs(observed)).sum()) / (self.count + 1))
            distribution = d_boot[metric] - p_boot[metric] if metric == "nll" else p_boot[metric] - d_boot[metric]
            result[metric] = {
                "advantage_of_protection": observed,
                "ci95": interval(distribution),
                "p_two_sided": p,
                "p_holm": None,
                "test": "paired_sample_two_sided_sign_flip",
                "randomizations": self.count,
            }
        return result
