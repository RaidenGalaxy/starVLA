from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MultimodalSample:
    id: str
    dataset: str
    split: str
    image: Path
    prompt: str
    target: str
    answer_type: str = "open"


def load_manifest(path: Path) -> list[MultimodalSample]:
    root, records, seen = path.parent, [], set()
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            identifier = str(raw["id"])
            if identifier in seen:
                raise ValueError(f"duplicate id {identifier!r} at line {line_number}")
            seen.add(identifier)
            image = Path(raw["image"])
            image = image if image.is_absolute() else root / image
            if not image.is_file():
                raise FileNotFoundError(image)
            target = str(raw["target"]).strip()
            if not target:
                raise ValueError(f"empty target at line {line_number}")
            records.append(MultimodalSample(
                id=identifier,
                dataset=str(raw.get("dataset", "custom")),
                split=str(raw["split"]),
                image=image,
                prompt=str(raw["prompt"]),
                target=target,
                answer_type=str(raw.get("answer_type", "open")),
            ))
    return records


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def assert_disjoint(calibration: list[MultimodalSample], evaluation: list[MultimodalSample]) -> None:
    overlap = {item.id for item in calibration} & {item.id for item in evaluation}
    if overlap:
        raise ValueError(f"calibration/evaluation ID leakage: {sorted(overlap)[:8]}")
    image_overlap = ({_file_digest(item.image) for item in calibration}
                     & {_file_digest(item.image) for item in evaluation})
    if image_overlap:
        raise ValueError(f"calibration/evaluation image leakage: {len(image_overlap)} duplicate files")


def write_protocol(path: Path, seed: int = 42) -> None:
    protocol = {
        "seed": seed,
        "calibration": {
            "text": {"dataset": "allenai/c4", "split": "train", "samples": 128, "tokens": 2048},
            "multimodal": {"samples": 128, "sources": {
                "docvqa_train": 32, "mathvista_train": 32,
                "mmmu_dev": 32, "chartqa_train": 32}},
        },
        "evaluation": {
            "vlm": ["MMMU_VAL", "DocVQA_VAL", "MMBench_DEV_EN", "MathVista_MINI"],
            "language": ["wikitext2_test", "ptb_test", "c4_validation_256x2048"],
        },
        "reference": "BF16 checkpoint values widened to FP32 for CPU compute",
        "warning": "conditional answer PPL is diagnostic and is not corpus language-model perplexity",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def fixed_subset(samples: list[MultimodalSample], count: int, seed: int) -> list[MultimodalSample]:
    if count >= len(samples):
        return samples
    indices = list(range(len(samples)))
    random.Random(seed).shuffle(indices)
    return [samples[index] for index in sorted(indices[:count])]
