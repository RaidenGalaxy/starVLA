"""Run the ablation with strict-equivalence AWQ fallback enabled."""

from __future__ import annotations

from . import run_ablation
from .awq_safe import search_awq_layer


def main() -> int:
    run_ablation.search_awq_layer = search_awq_layer
    return run_ablation.main()


if __name__ == "__main__":
    raise SystemExit(main())

