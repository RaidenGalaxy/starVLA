from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .core import atomic_json, file_sha256


DATASETS = ("DocVQA_VAL", "MMMU_DEV_VAL", "ScienceQA_TEST", "TextVQA_VAL")


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def content_keys(row: dict) -> tuple[str, str, set[str]]:
    images = set(str(value) for value in row.get("image_sha256s", []) if value)
    if row.get("image_sha256"):
        images.add(str(row["image_sha256"]))
    return str(row.get("id", "")), str(row.get("question_sha256", "")), images


def select_first_per_dataset(rows: list[dict], count: int) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("dataset"))].append(row)
    if set(grouped) != set(DATASETS):
        raise ValueError(f"unexpected source datasets: {sorted(grouped)}")
    if any(len(grouped[name]) < count for name in DATASETS):
        raise ValueError("source manifest does not contain enough rows per dataset")
    selected = [row for name in DATASETS for row in grouped[name][:count]]
    ids = [str(row["id"]) for row in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("selected generation sample IDs are not unique")
    return selected


def assert_content_disjoint(selected: list[dict], discovery: list[dict]) -> dict:
    selected_ids, selected_questions, selected_images = set(), set(), set()
    for row in selected:
        identifier, question, images = content_keys(row)
        selected_ids.add(identifier)
        if question:
            selected_questions.add(question)
        selected_images.update(images)
    discovery_ids, discovery_questions, discovery_images = set(), set(), set()
    for row in discovery:
        identifier, question, images = content_keys(row)
        discovery_ids.add(identifier)
        if question:
            discovery_questions.add(question)
        discovery_images.update(images)
    overlap = {
        "id": sorted(selected_ids & discovery_ids),
        "question_sha256": sorted(selected_questions & discovery_questions),
        "image_sha256": sorted(selected_images & discovery_images),
    }
    if any(overlap.values()):
        raise ValueError(
            "generation subset overlaps discovery data: "
            f"{ {key: len(value) for key, value in overlap.items()} }"
        )
    return {key: len(value) for key, value in overlap.items()}


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the fixed Q-ShortV generation subset")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--discovery", type=Path, action="append", default=[])
    parser.add_argument("--per-dataset", type=int, default=24)
    args = parser.parse_args()
    if args.per_dataset <= 0:
        raise ValueError("--per-dataset must be positive")
    source = read_jsonl(args.source)
    selected = select_first_per_dataset(source, args.per_dataset)
    discovery = [row for path in args.discovery for row in read_jsonl(path)]
    overlap = assert_content_disjoint(selected, discovery)
    atomic_jsonl(args.output, selected)
    atomic_json(args.audit, {
        "selection": "first rows in source-manifest order within each dataset",
        "source": str(args.source.resolve()),
        "source_sha256": file_sha256(args.source),
        "output": str(args.output.resolve()),
        "output_sha256": file_sha256(args.output),
        "rows": len(selected),
        "per_dataset": args.per_dataset,
        "datasets": {name: args.per_dataset for name in DATASETS},
        "discovery_manifests": [
            {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for path in args.discovery
        ],
        "overlap": overlap,
        "sample_id_sha256": hashlib.sha256(
            "\n".join(str(row["id"]) for row in selected).encode()
        ).hexdigest(),
    })


if __name__ == "__main__":
    main()
