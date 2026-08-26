from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .dataset_protocol import ACTIVE_DATASET_NAMES, PROTOCOL_VERSION


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_question_digest(dataset: str, prompt: str) -> str:
    """Dataset-independent normalized question hash for cross-source leakage checks."""
    del dataset  # Retained in the signature for manifest compatibility.
    normalized = re.sub(r"\s+", " ", prompt).strip().lower()
    return hashlib.sha256(normalized.encode()).hexdigest()


@dataclass(frozen=True)
class MultimodalSample:
    id: str
    dataset: str
    split: str
    image: Path
    prompt: str
    target: str
    answer_type: str = "open"
    extra_images: tuple[Path, ...] = ()
    category: str = ""
    official_index: str = ""
    upstream_revision: str = ""
    image_sha256: str = ""
    question_sha256: str = ""

    @property
    def images(self) -> tuple[Path, ...]:
        return (self.image, *self.extra_images)


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
            raw_images = raw.get("images")
            if raw_images is None:
                raw_images = [raw["image"]]
            images = []
            for raw_image in raw_images:
                image = Path(raw_image)
                image = image if image.is_absolute() else root / image
                if not image.is_file():
                    raise FileNotFoundError(image)
                images.append(image)
            if not images:
                raise ValueError(f"sample has no images at line {line_number}")
            target = str(raw["target"]).strip()
            if not target:
                raise ValueError(f"empty target at line {line_number}")
            dataset, prompt = str(raw.get("dataset", "custom")), str(raw["prompt"])
            image_sha256s = [file_digest(item) for item in images]
            image_sha256 = image_sha256s[0]
            question_sha256 = normalized_question_digest(dataset, prompt)
            if raw.get("image_sha256s") not in (None, [], image_sha256s):
                raise ValueError(f"image hashes mismatch at line {line_number}")
            if raw.get("image_sha256") not in (None, "", image_sha256):
                raise ValueError(f"image hash mismatch at line {line_number}")
            if raw.get("question_sha256") not in (None, "", question_sha256):
                raise ValueError(f"question hash mismatch at line {line_number}")
            records.append(MultimodalSample(
                id=identifier,
                dataset=dataset,
                split=str(raw["split"]),
                image=images[0],
                extra_images=tuple(images[1:]),
                prompt=prompt,
                target=target,
                answer_type=str(raw.get("answer_type", "open")),
                category=str(raw.get("category", "")),
                official_index=str(raw.get("official_index", "")),
                upstream_revision=str(raw.get("upstream_revision", "")),
                image_sha256=image_sha256,
                question_sha256=question_sha256,
            ))
    return records


def manifest_digest(path: Path) -> str:
    return file_digest(path)


def sample_manifest_digest(samples: list[MultimodalSample]) -> str:
    rows = [{
        "id": item.id,
        "dataset": item.dataset,
        "split": item.split,
        "official_index": item.official_index,
        "upstream_revision": item.upstream_revision,
        "image_sha256s": [file_digest(image) for image in item.images],
        "question_sha256": item.question_sha256 or normalized_question_digest(
            item.dataset, item.prompt
        ),
        "target": item.target,
    } for item in samples]
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def write_manifest(path: Path, samples: list[MultimodalSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    manifest_root = path.parent.resolve()

    def serialized_image(image: Path) -> str:
        resolved = image.resolve()
        try:
            return str(resolved.relative_to(manifest_root))
        except ValueError:
            return str(resolved)

    with temporary.open("w") as handle:
        for item in samples:
            row = asdict(item)
            row["image"] = serialized_image(item.image)
            row.pop("extra_images", None)
            row["images"] = [serialized_image(image) for image in item.images]
            row["image_sha256s"] = [file_digest(image) for image in item.images]
            row["image_sha256"] = item.image_sha256 or file_digest(item.image)
            row["question_sha256"] = item.question_sha256 or normalized_question_digest(
                item.dataset, item.prompt
            )
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def assert_disjoint(calibration: list[MultimodalSample], evaluation: list[MultimodalSample]) -> None:
    overlap = {item.id for item in calibration} & {item.id for item in evaluation}
    if overlap:
        raise ValueError(f"calibration/evaluation ID leakage: {sorted(overlap)[:8]}")
    image_overlap = {
        file_digest(image) for item in calibration for image in item.images
    } & {
        file_digest(image) for item in evaluation for image in item.images}
    if image_overlap:
        raise ValueError(f"calibration/evaluation image leakage: {len(image_overlap)} duplicate files")
    question_overlap = {
        item.question_sha256 or normalized_question_digest(item.dataset, item.prompt)
        for item in calibration
    } & {
        item.question_sha256 or normalized_question_digest(item.dataset, item.prompt)
        for item in evaluation
    }
    if question_overlap:
        raise ValueError(
            f"calibration/evaluation question leakage: {len(question_overlap)} duplicate questions"
        )


def write_protocol(path: Path, seed: int = 42) -> None:
    protocol = {
        "seed": seed,
        "pixel_protocol": {
            "min_pixels": 256 * 28 * 28,
            "max_pixels": 1280 * 28 * 28,
        },
        "calibration": {
            "text": {"dataset": "allenai/c4", "split": "train", "samples": 128, "tokens": 2048},
            "multimodal": {
                "samples": 128,
                "per_source": 32,
                "sources": {
                    "HuggingFaceM4/ChartQA": {
                        "split": "train",
                        "revision": "b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5",
                    },
                    "lmms-lab/textvqa": {
                        "split": "train",
                        "revision": "9c0699cd19768ac5ab97568f6b3cbac4c0062884",
                    },
                    "lmms-lab/ScienceQA-IMG": {
                        "split": "train",
                        "revision": "b9984af01798586705cee1fbdfa80a6dcbef04f1",
                    },
                    "lmms-lab/DocVQA:InfographicVQA": {
                        "split": "train",
                        "revision": "539088ef8a8ada01ac8e2e6d4e372586748a265e",
                    },
                },
            },
        },
        "evaluation": {
            "screen_samples_per_dataset": 64,
            "vlm": list(ACTIVE_DATASET_NAMES),
            "protocol_version": PROTOCOL_VERSION,
            "language": ["wikitext2_test", "ptb_test", "c4_validation_256x2048"],
            "vlmevalkit_revision": "4c4540cb19610b166499d004873674d00c88680f",
        },
        "reference": "BF16 checkpoint values widened to FP32 for CPU compute",
        "warning": "target_text_diagnostic is not used for method ranking",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def fixed_subset(samples: list[MultimodalSample], count: int, seed: int) -> list[MultimodalSample]:
    if count >= len(samples):
        return samples
    buckets: dict[tuple[str, str, str], list[int]] = {}
    for index, item in enumerate(samples):
        key = (item.dataset, item.answer_type, item.category)
        buckets.setdefault(key, []).append(index)
    rng = random.Random(seed)
    for indices in buckets.values():
        rng.shuffle(indices)
    selected = []
    keys = sorted(buckets)
    while len(selected) < count:
        progressed = False
        for key in keys:
            if buckets[key] and len(selected) < count:
                selected.append(buckets[key].pop())
                progressed = True
        if not progressed:
            break
    return [samples[index] for index in sorted(selected)]
