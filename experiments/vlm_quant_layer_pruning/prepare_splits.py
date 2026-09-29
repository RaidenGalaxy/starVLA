from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

from experiments.vlm_quant_benchmark.data import (
    MultimodalSample,
    assert_disjoint,
    load_manifest,
    write_manifest,
)

from .core import atomic_json, file_sha256


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _content_components(samples: list[MultimodalSample]) -> list[list[MultimodalSample]]:
    union = _UnionFind(len(samples))
    seen: dict[tuple[str, str], int] = {}
    for index, sample in enumerate(samples):
        for key in (("question", sample.question_sha256), ("image", sample.image_sha256)):
            if key in seen:
                union.union(index, seen[key])
            else:
                seen[key] = index
    groups: dict[int, list[MultimodalSample]] = defaultdict(list)
    for index, sample in enumerate(samples):
        groups[union.find(index)].append(sample)
    return list(groups.values())


def _component_hash(component: list[MultimodalSample], salt: str) -> bytes:
    ids = ",".join(sorted(item.id for item in component))
    return hashlib.sha256(f"{salt}:{ids}".encode()).digest()


def select_content_disjoint(
    samples: list[MultimodalSample], count: int, *, salt: str,
) -> tuple[list[MultimodalSample], list[MultimodalSample]]:
    components = sorted(
        _content_components(samples), key=lambda group: _component_hash(group, salt),
    )
    reachable: dict[int, tuple[int, ...]] = {0: ()}
    for component_index, component in enumerate(components):
        size = len(component)
        for total, chosen in sorted(list(reachable.items()), reverse=True):
            candidate = total + size
            if candidate <= count and candidate not in reachable:
                reachable[candidate] = (*chosen, component_index)
    if count not in reachable:
        raise ValueError(f"cannot form an exact content-disjoint subset of {count}")
    selected_components = set(reachable[count])
    selected = [
        sample for index, component in enumerate(components)
        if index in selected_components for sample in component
    ]
    remaining = [
        sample for index, component in enumerate(components)
        if index not in selected_components for sample in component
    ]
    return selected, remaining


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare leak-free QCLP-VLM splits")
    parser.add_argument(
        "--calibration-manifest", type=Path,
        default=Path("data/vlm_quant_benchmark/multimodal_calibration/multimodal_calibration_4x32.jsonl"),
    )
    parser.add_argument(
        "--confirm-manifest", type=Path,
        default=Path("data/vlm_quant_benchmark/top3_protocol_v4/vlm_confirm_4x256.jsonl"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/vlm_quant_layer_pruning/splits"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration = load_manifest(args.calibration_manifest)
    calibration_by_dataset: dict[str, list[MultimodalSample]] = defaultdict(list)
    for sample in calibration:
        calibration_by_dataset[sample.dataset].append(sample)
    discovery: list[MultimodalSample] = []
    confirmatory: list[MultimodalSample] = []
    for dataset, samples in sorted(calibration_by_dataset.items()):
        selected, remaining = select_content_disjoint(
            samples, 8, salt=f"qclp-vlm-discovery-v1:{dataset}",
        )
        discovery.extend(selected)
        confirmatory.extend(remaining)
    if len(discovery) != 32 or len(confirmatory) != 96:
        raise ValueError("expected 32 discovery and 96 confirmatory calibration samples")
    assert_disjoint(discovery, confirmatory)

    evaluation = load_manifest(args.confirm_manifest)
    by_dataset: dict[str, list[MultimodalSample]] = defaultdict(list)
    for sample in evaluation:
        by_dataset[sample.dataset].append(sample)
    search: list[MultimodalSample] = []
    final: list[MultimodalSample] = []
    for dataset, samples in sorted(by_dataset.items()):
        selected, remaining = select_content_disjoint(
            samples, 64, salt=f"qclp-vlm-search-v1:{dataset}",
        )
        if len(remaining) != 192:
            raise ValueError(f"{dataset} final split has {len(remaining)} rows")
        search.extend(selected)
        final.extend(remaining)
    assert_disjoint(calibration, search)
    assert_disjoint(calibration, final)
    assert_disjoint(search, final)

    paths = {
        "discovery": args.output_dir / "calibration_discovery_4x8.jsonl",
        "confirmatory": args.output_dir / "calibration_confirmatory_4x24.jsonl",
        "search": args.output_dir / "evaluation_search_4x64.jsonl",
        "final": args.output_dir / "evaluation_final_4x192.jsonl",
    }
    for name, samples in (
        ("discovery", discovery),
        ("confirmatory", confirmatory),
        ("search", search),
        ("final", final),
    ):
        write_manifest(paths[name], samples)
    atomic_json(args.output_dir / "split_audit.json", {
        "schema_version": 1,
        "salt": {
            "discovery": "qclp-vlm-discovery-v1",
            "search": "qclp-vlm-search-v1:<dataset>",
        },
        "source": {
            "calibration": str(args.calibration_manifest.resolve()),
            "calibration_sha256": file_sha256(args.calibration_manifest),
            "evaluation": str(args.confirm_manifest.resolve()),
            "evaluation_sha256": file_sha256(args.confirm_manifest),
        },
        "outputs": {
            name: {
                "path": str(path.resolve()),
                "rows": len({
                    "discovery": discovery,
                    "confirmatory": confirmatory,
                    "search": search,
                    "final": final,
                }[name]),
                "sha256": file_sha256(path),
            }
            for name, path in paths.items()
        },
        "leakage_checks": {
            "id": 0,
            "question_sha256": 0,
            "image_sha256": 0,
        },
    })


if __name__ == "__main__":
    main()
