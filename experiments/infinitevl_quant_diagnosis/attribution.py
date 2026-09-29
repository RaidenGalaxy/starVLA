from __future__ import annotations

import itertools
import math
from collections.abc import Mapping


MODULES = ("decoder_ffn", "swa_attention", "gdn_main")


def mask_key(members: frozenset[str] | set[str]) -> str:
    selected = set(members)
    return "+".join(name for name in MODULES if name in selected) or "none"


def all_scope_masks() -> list[frozenset[str]]:
    return [
        frozenset(module for index, module in enumerate(MODULES) if bits & (1 << index))
        for bits in range(1 << len(MODULES))
    ]


def shapley_three(values: Mapping[str, float]) -> dict[str, float]:
    expected = {mask_key(mask) for mask in all_scope_masks()}
    if set(values) != expected:
        raise ValueError(f"factorial values mismatch: missing={expected - set(values)}")
    result = {module: 0.0 for module in MODULES}
    permutations = list(itertools.permutations(MODULES))
    for ordering in permutations:
        present: set[str] = set()
        for module in ordering:
            before = float(values[mask_key(present)])
            present.add(module)
            after = float(values[mask_key(present)])
            result[module] += (after - before) / len(permutations)
    total = float(values[mask_key(set(MODULES))]) - float(values["none"])
    fractions = {
        module: result[module] / total if not math.isclose(total, 0.0) else math.nan
        for module in MODULES
    }
    return {"contribution": result, "fraction": fractions, "total_effect": total}


def pair_interactions(values: Mapping[str, float]) -> dict[str, float]:
    result = {}
    for left, right in itertools.combinations(MODULES, 2):
        third = next(module for module in MODULES if module not in {left, right})
        effects = []
        for background in (set(), {third}):
            effects.append(
                float(values[mask_key(background | {left, right})])
                - float(values[mask_key(background | {left})])
                - float(values[mask_key(background | {right})])
                + float(values[mask_key(background)])
            )
        result[f"{left}*{right}"] = sum(effects) / len(effects)
    return result

