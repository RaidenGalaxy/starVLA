from __future__ import annotations

import json
import math

import numpy as np

from experiments.ffn_quant_rootcause.interventions import Intervention
from experiments.infinitevl_quant.progress import json_digest

from .config import MODELS, PARAMETERS_PER_LAYER, POLICIES, QA_COUNTS, SEEDS


def condition(layers):
    chosen = tuple(sorted(layers))
    if len(chosen) != len(set(chosen)) or any(not 0 <= i < 36 for i in chosen):
        raise ValueError("invalid FFN layer mask")
    q = len(chosen)
    mask = sum(1 << i for i in chosen)
    identifier = "B" if q == 0 else "W" if q == 36 else f"m{mask:09x}"
    total, low = 36 * PARAMETERS_PER_LAYER, q * PARAMETERS_PER_LAYER
    bits = 16 * (total - low) + 4 * low
    saving = 1 - bits / (16 * total)
    if not math.isclose(saving, q / 48, abs_tol=1e-15):
        raise ArithmeticError("parameter and layer compression accounting differ")
    spec = Intervention(bits=None, layer_bits={i: 4 for i in chosen})
    if q == 36:
        spec = Intervention(bits=4)
    return {
        "id": identifier,
        "mask_hex": f"{mask:09x}",
        "mask_bits": "".join("1" if i in chosen else "0" for i in range(36)),
        "quantized_layers": list(chosen),
        "q": q,
        "reference_layers": 36 - q,
        "ffn_parameters": total,
        "quantized_parameters": low,
        "reference_weight_bits": 16 * total,
        "mixed_weight_bits": bits,
        "theoretical_weight_bytes": bits // 8,
        "compression_saving": saving,
        "metadata_bytes_included": False,
        "native_lowbit_memory_claim": False,
        "spec": spec.payload(),
    }


def semantic_layers(payload):
    spec = Intervention.from_payload(payload)
    if (
        spec.gamma != 1
        or spec.layer_gammas
        or spec.direction != "real"
        or spec.channel_repairs
        or spec.gdn_groups
        or spec.carry
        or spec.recurrent
    ):
        return None
    selected = []
    for i in range(36):
        values = [
            spec.projection_bits.get(f"{i}.{p}", spec.layer_bits.get(i, spec.bits)) for p in ("gate", "up", "down")
        ]
        if all(v is None for v in values):
            continue
        if values != [4, 4, 4]:
            return None
        selected.append(i)
    return tuple(selected)


def build_design(models):
    result = {
        "format_version": 1,
        "qa_counts": list(QA_COUNTS),
        "models": {},
        "ranking_source": "frozen_previous_diagnostic8",
        "qa_used_for_design": False,
    }
    for kind in MODELS:
        ranking = models[kind]["ranking_diagnostic8"]
        order = [r["layer"] for r in ranking]
        if sorted(order) != list(range(36)):
            raise ValueError("ranking must be a permutation of all 36 layers")
        conditions, aliases, random_orders = {}, {}, {}

        def add(policy, q, layers, *, conditions=conditions, aliases=aliases):
            row = condition(layers)
            existing = conditions.setdefault(row["id"], row)
            if existing != row:
                raise RuntimeError("mask identifier collision")
            aliases.setdefault(policy, {})[str(q)] = row["id"]
            return row["id"]

        for q in range(37):
            add(POLICIES[0], q, order[:q])
            add(POLICIES[1], q, order[36 - q :])
        for seed in SEEDS:
            permutation = np.random.default_rng(seed).permutation(36).tolist()
            random_orders[str(seed)] = permutation
            for q in QA_COUNTS:
                add(f"random_{seed}", q, permutation[:q])
        qa = ["B", "W"]
        for policy in POLICIES:
            for q in (35, 33, 18, 3):
                qa.append(aliases[policy][str(q)])
        main = [aliases[p][str(q)] for p in POLICIES for q in range(37)]
        main = list(dict.fromkeys(["B", "W", *main]))
        random = [aliases[f"random_{z}"][str(q)] for z in SEEDS for q in QA_COUNTS]
        random = list(dict.fromkeys(x for x in random if x not in main))
        qa = list(dict.fromkeys(qa))
        if (len(qa), len(main), len(random)) != (10, 72, 12):
            raise ValueError("frozen matrix count differs from approved design")
        for label, record in conditions.items():
            record["allocations"] = [
                {
                    "strategy": policy,
                    "q": int(q),
                    "seed": int(policy.removeprefix("random_")) if policy.startswith("random_") else None,
                }
                for policy, values in aliases.items()
                for q, identifier in values.items()
                if identifier == label
            ]
        result["models"][kind] = {
            "sensitivity_order": order,
            "ranking_sha256": json_digest(ranking),
            "random_orders": random_orders,
            "conditions": conditions,
            "aliases": aliases,
            "qa_order": qa,
            "main_screen_order": main,
            "random_screen_order": random,
        }
    return json.loads(json.dumps(result))
