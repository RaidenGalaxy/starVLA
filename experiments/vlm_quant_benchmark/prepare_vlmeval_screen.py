from __future__ import annotations

import argparse
import ast
import hashlib
import os
import random
from pathlib import Path

from .data import (
    MultimodalSample,
    assert_disjoint,
    file_digest,
    load_manifest,
    normalized_question_digest,
    write_manifest,
)
from .dataset_protocol import (
    ACTIVE_DATASET_NAMES,
    DATASET_SPECS,
    PROTOCOL_VERSION,
    canonical_answers,
)
from .reporting import atomic_json


DATASETS = ACTIVE_DATASET_NAMES
VLMEVALKIT_REVISION = "4c4540cb19610b166499d004873674d00c88680f"
CONFIRM_PROTOCOL_VERSION = 4


def _value(line, names, default=""):
    for name in names:
        if name in line and line[name] is not None:
            value = line[name]
            try:
                if value != value:
                    continue
            except (TypeError, ValueError):
                pass
            return value
    return default


def _canonical_target(dataset_name: str, value) -> str:
    """Return one plain target string for local teacher-forcing diagnostics."""
    target = canonical_answers(value)[0]
    if not target:
        raise ValueError(f"empty canonical target for {dataset_name}")
    return target


def _stratified_indices(data, count: int, seed: int) -> list[int]:
    buckets = {}
    for position, (_, line) in enumerate(data.iterrows()):
        key = (
            str(_value(line, ("answer_type", "question_type"), "open")),
            str(_value(line, ("category", "subject", "task", "l2-category"), "")),
        )
        buckets.setdefault(key, []).append(position)
    rng = random.Random(seed)
    for values in buckets.values():
        rng.shuffle(values)
    selected, keys = [], sorted(buckets)
    while len(selected) < min(count, len(data)):
        progressed = False
        for key in keys:
            if buckets[key] and len(selected) < count:
                selected.append(buckets[key].pop())
                progressed = True
        if not progressed:
            break
    return sorted(selected)


def _stratified_circular_indices(data, count: int, seed: int) -> list[int]:
    if "g_index" not in data.columns:
        return _stratified_indices(data, count, seed)
    groups = {}
    for position, (_, row) in enumerate(data.iterrows()):
        group = str(row["g_index"])
        groups.setdefault(group, []).append(position)
    buckets = {}
    for group, positions in groups.items():
        row = data.iloc[positions[0]]
        key = (
            str(_value(row, ("answer_type", "question_type"), "open")),
            str(_value(row, ("category", "subject", "task", "l2-category"), "")),
        )
        buckets.setdefault(key, []).append(group)
    rng = random.Random(seed)
    for values in buckets.values():
        rng.shuffle(values)
    selected, keys, total = [], sorted(buckets), 0
    while total < count:
        progressed = False
        for key in keys:
            while buckets[key]:
                group = buckets[key].pop()
                size = len(groups[group])
                if total + size <= count:
                    selected.extend(groups[group])
                    total += size
                    progressed = True
                    break
            if total == count:
                break
        if not progressed:
            break
    if total != count:
        raise ValueError(
            f"cannot form exactly {count} rows from complete circular groups; got {total}"
        )
    return sorted(selected)


def extend_positions(
    data,
    *,
    count: int,
    seed: int,
    preserved_indices: list[str],
) -> tuple[list[int], list[int]]:
    """Preserve a pinned prefix and draw a disjoint stratified extension."""
    if count <= 0 or count > len(data):
        raise ValueError(f"sample count must be in [1, {len(data)}], got {count}")
    if len(preserved_indices) > count:
        raise ValueError(
            f"preserved manifest has {len(preserved_indices)} rows, exceeding target {count}"
        )
    if len(set(preserved_indices)) != len(preserved_indices):
        raise ValueError("preserved manifest contains duplicate official indices")
    positions_by_index = {
        str(_value(row, ("index", "id"), position)): position
        for position, (_, row) in enumerate(data.iterrows())
    }
    try:
        preserved = [positions_by_index[str(index)] for index in preserved_indices]
    except KeyError as error:
        raise ValueError(f"preserved index absent upstream: {error}") from error
    preserved_set = set(preserved)
    if "g_index" in data.columns and preserved_set:
        groups: dict[str, set[int]] = {}
        for position, (_, row) in enumerate(data.iterrows()):
            groups.setdefault(str(row["g_index"]), set()).add(position)
        partial = [
            name for name, positions in groups.items()
            if positions & preserved_set and not positions <= preserved_set
        ]
        if partial:
            raise ValueError(
                f"preserved manifest splits {len(partial)} circular groups: {partial[:4]}"
            )
    remaining_positions = [
        position for position in range(len(data)) if position not in preserved_set
    ]
    additional_count = count - len(preserved)
    if additional_count == 0:
        return preserved, []
    remaining = data.iloc[remaining_positions].reset_index(drop=True)
    relative = _stratified_circular_indices(remaining, additional_count, seed)
    additional = [remaining_positions[position] for position in relative]
    if preserved_set & set(additional):
        raise AssertionError("stratified extension overlaps preserved positions")
    return preserved + additional, additional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare pinned VLMEvalKit screen/confirm manifest")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lmu-data-root", type=Path)
    parser.add_argument("--calibration-manifest", type=Path)
    parser.add_argument("--previous-manifest", type=Path)
    parser.add_argument("--samples-per-dataset", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lmu_root = args.lmu_data_root or args.output_dir.parent / "LMUData"
    lmu_root.mkdir(parents=True, exist_ok=True)
    os.environ["LMUData"] = str(lmu_root.resolve())
    try:
        from vlmeval.dataset import build_dataset
        from vlmeval.vlm.qwen2_vl.prompt import Qwen2VLPromptMixin
    except ImportError as error:
        raise ImportError(
            f"VLMEvalKit {VLMEVALKIT_REVISION} must be available on PYTHONPATH"
        ) from error

    class PromptBuilder(Qwen2VLPromptMixin):
        def __init__(self, dataset_object):
            self.dataset_object = dataset_object
            self._use_custom_prompt = True

        def dump_image(self, line, dataset):
            return self.dataset_object.dump_image(line)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    preserved: dict[str, list[str]] = {}
    if args.previous_manifest is not None:
        for sample in load_manifest(args.previous_manifest):
            if sample.dataset in DATASETS:
                preserved.setdefault(sample.dataset, []).append(sample.official_index)
        missing = set(DATASETS) - set(preserved)
        if missing:
            raise ValueError(f"previous manifest is missing datasets: {sorted(missing)}")
    samples = []
    selection_metadata = {}
    for dataset_name in DATASETS:
        dataset = build_dataset(dataset_name)
        positions, additional = extend_positions(
            dataset.data,
            count=args.samples_per_dataset,
            seed=args.seed,
            preserved_indices=preserved.get(dataset_name, []),
        )
        selection_metadata[dataset_name] = {
            "preserved_count": len(preserved.get(dataset_name, [])),
            "additional_count": len(additional),
            "preserved_indices": list(preserved.get(dataset_name, [])),
            "additional_indices": [
                str(_value(dataset.data.iloc[position], ("index", "id"), position))
                for position in additional
            ],
        }
        builder = PromptBuilder(dataset)
        for position in positions:
            line = dataset.data.iloc[position]
            messages = builder.build_prompt(line, dataset_name)
            images = [Path(item["value"]) for item in messages if item["type"] == "image"]
            prompt = "\n".join(
                str(item["value"]) for item in messages if item["type"] == "text"
            ).strip()
            if not images or not prompt:
                raise ValueError(f"invalid prompt at {dataset_name}[{position}]")
            target = _canonical_target(
                dataset_name, _value(line, ("answer", "target")),
            )
            official_index = str(_value(line, ("index", "id"), position))
            samples.append(MultimodalSample(
                id=f"{dataset_name}:{official_index}",
                dataset=dataset_name,
                split="screen" if args.samples_per_dataset == 64 else "confirm",
                image=images[0],
                extra_images=tuple(images[1:]),
                prompt=prompt,
                target=target,
                answer_type=str(_value(line, ("answer_type", "question_type"), "open")),
                category=str(_value(line, ("category", "subject", "task", "l2-category"), "")),
                official_index=official_index,
                upstream_revision=VLMEVALKIT_REVISION,
                image_sha256=file_digest(images[0]),
                question_sha256=normalized_question_digest(dataset_name, prompt),
            ))
    if args.calibration_manifest is not None:
        assert_disjoint(load_manifest(args.calibration_manifest), samples)
    manifest_name = (
        "vlm_screen_4x64.jsonl"
        if args.samples_per_dataset == 64
        else f"vlm_confirm_4x{args.samples_per_dataset}.jsonl"
    )
    manifest_path = args.output_dir / manifest_name
    write_manifest(manifest_path, samples)

    def md5(path: Path) -> str:
        digest = hashlib.md5()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    dataset_files = {}
    for dataset_name in DATASETS:
        source = lmu_root / f"{dataset_name}.tsv"
        if not source.is_file():
            raise FileNotFoundError(source)
        dataset_files[dataset_name] = {
            "path": str(source.resolve()),
            "bytes": source.stat().st_size,
            "md5": md5(source),
            "sha256": file_digest(source),
            "source_split": DATASET_SPECS[dataset_name].source_split,
        }
    atomic_json(args.output_dir / "dataset_protocol.json", {
        "protocol_version": (
            PROTOCOL_VERSION if args.samples_per_dataset == 64
            else CONFIRM_PROTOCOL_VERSION
        ),
        "vlmevalkit_revision": VLMEVALKIT_REVISION,
        "seed": args.seed,
        "samples_per_dataset": args.samples_per_dataset,
        "selection": selection_metadata,
        "parent_manifest": (
            None if args.previous_manifest is None else {
                "path": str(args.previous_manifest.resolve()),
                "sha256": file_digest(args.previous_manifest),
                "rows": sum(len(value) for value in preserved.values()),
            }
        ),
        "datasets": dataset_files,
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": file_digest(manifest_path),
            "rows": len(samples),
        },
        "generation": {
            "temperature": 0,
            "max_new_tokens": 128,
            "prompt_builder": "pinned_Qwen2VLPromptMixin",
            "answer_postprocess_version": PROTOCOL_VERSION,
        },
    })


if __name__ == "__main__":
    main()
