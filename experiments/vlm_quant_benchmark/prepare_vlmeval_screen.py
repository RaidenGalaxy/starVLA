from __future__ import annotations

import argparse
import ast
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


DATASETS = ("DocVQA_VAL", "MathVista_MINI", "MMMU_DEV_VAL", "MMBench_DEV_EN")
VLMEVALKIT_REVISION = "4c4540cb19610b166499d004873674d00c88680f"


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
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"empty answer list for {dataset_name}")
        target = str(value[0]).strip()
    else:
        target = str(value).strip()
        if dataset_name == "DocVQA_VAL":
            try:
                parsed = ast.literal_eval(target)
            except (SyntaxError, ValueError):
                parsed = None
            if isinstance(parsed, (list, tuple)):
                if not parsed:
                    raise ValueError("empty DocVQA answer list")
                target = str(parsed[0]).strip()
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare pinned VLMEvalKit 4x64 screen manifest")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lmu-data-root", type=Path)
    parser.add_argument("--calibration-manifest", type=Path)
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
    samples = []
    for dataset_offset, dataset_name in enumerate(DATASETS):
        dataset = build_dataset(dataset_name)
        positions = _stratified_circular_indices(
            dataset.data,
            args.samples_per_dataset,
            args.seed + dataset_offset,
        )
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
                split="screen",
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
    write_manifest(args.output_dir / "vlm_screen_4x64.jsonl", samples)


if __name__ == "__main__":
    main()
