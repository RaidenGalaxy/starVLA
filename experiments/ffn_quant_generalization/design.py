from __future__ import annotations

import numpy as np

from experiments.ffn_quant_rootcause.interventions import Intervention
from experiments.infinitevl_quant.progress import json_digest

from .config import MODELS, POLICIES, SEEDS


def counts(layers):
    if layers % 4:
        raise ValueError("registered comparison requires quarter-layer counts")
    return [0, layers // 4, layers // 2, 3 * layers // 4, layers - 1, layers]


def condition(model, layers):
    size, hidden, intermediate = MODELS[model]
    chosen = tuple(sorted(layers))
    if len(chosen) != len(set(chosen)) or any(not 0 <= i < size for i in chosen):
        raise ValueError("invalid layer mask")
    q = len(chosen)
    mask = sum(1 << i for i in chosen)
    key = "B" if q == 0 else "W" if q == size else f"m{mask:0{(size + 3) // 4}x}"
    n_layer = 3 * hidden * intermediate
    total, low = size * n_layer, q * n_layer
    mixed = 16 * (total - low) + 4 * low
    spec = Intervention(bits=4) if q == size else Intervention(bits=None, layer_bits={i: 4 for i in chosen})
    return {
        "id": key,
        "model": model,
        "q": q,
        "layers": size,
        "mask_bits": "".join("1" if i in chosen else "0" for i in range(size)),
        "quantized_layers": list(chosen),
        "ffn_parameters": total,
        "quantized_parameters": low,
        "coverage_percent": 100 * low / total,
        "bf16_saving_percent": 100 * (1 - mixed / (16 * total)),
        "mixed_weight_bits": mixed,
        "metadata_bytes_included": False,
        "spec": spec.payload(),
    }


def build_design(orders):
    design = {
        "format_version": 1,
        "models": {},
        "test_family_size": 36,
        "ranking_data": "historical_diagnostic8",
        "new_data_used_for_ranking": False,
    }
    for kind, (size, _, _) in MODELS.items():
        order = orders[kind]
        if sorted(order) != list(range(size)):
            raise ValueError(f"{kind} ranking is not a full permutation")
        conditions, aliases = {}, {}

        def add(policy, q, chosen, *, kind=kind, conditions=conditions, aliases=aliases):
            item = condition(kind, chosen)
            old = conditions.setdefault(item["id"], item)
            if old != item:
                raise RuntimeError("mask collision")
            aliases.setdefault(policy, {})[str(q)] = item["id"]
            return item["id"]

        for q in range(size + 1):
            add(POLICIES[0], q, order[:q])
            add(POLICIES[1], q, order[size - q :])
        permutations = {}
        for seed in SEEDS:
            permutation = np.random.default_rng(seed).permutation(size).tolist()
            permutations[str(seed)] = permutation
            for q in counts(size)[1:4]:
                add(f"random_{seed}", q, permutation[:q])
        main = list(dict.fromkeys(["B", "W", *[aliases[p][str(q)] for p in POLICIES for q in range(size + 1)]]))
        confirm = list(dict.fromkeys(["B", "W", *[aliases[p][str(q)] for q in counts(size)[1:-1] for p in POLICIES]]))
        random = list(dict.fromkeys(aliases[f"random_{s}"][str(q)] for s in SEEDS for q in counts(size)[1:4]))
        qa = list(
            dict.fromkeys(
                [
                    "B",
                    "W",
                    aliases[POLICIES[0]][str(size // 2)],
                    aliases[POLICIES[1]][str(size // 2)],
                    aliases[POLICIES[1]][str(size - 1)],
                ]
            )
        )
        bridge = qa[:4]
        if (len(main), len(confirm), len(qa), len(bridge)) != (2 * size, 10, 5, 4):
            raise RuntimeError("registered matrix size mismatch")
        design["models"][kind] = {
            "sensitivity_order": order,
            "ranking_sha256": json_digest(order),
            "conditions": conditions,
            "aliases": aliases,
            "main": main,
            "confirm": confirm,
            "random": random,
            "qa": qa,
            "bridge": bridge,
            "random_orders": permutations,
            "confirm_q": counts(size),
        }
    return design
