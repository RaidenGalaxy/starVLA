"""Deterministic tiny tabletop dataset for calibration and evaluation."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw


@dataclass(frozen=True)
class QuantSample:
    id: str
    split: str
    image: str
    prompt: str
    target: str


COLORS = {
    "red": (215, 55, 55),
    "blue": (55, 95, 215),
    "green": (55, 165, 90),
    "yellow": (225, 185, 45),
}
SHAPES = ("circle", "square", "triangle")


def _draw_shape(draw: ImageDraw.ImageDraw, shape: str, xy: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = xy
    box = (x - 12, y - 12, x + 12, y + 12)
    if shape == "circle":
        draw.ellipse(box, fill=color, outline=(20, 20, 20), width=1)
    elif shape == "square":
        draw.rectangle(box, fill=color, outline=(20, 20, 20), width=1)
    else:
        draw.polygon(((x, y - 14), (x - 14, y + 12), (x + 14, y + 12)), fill=color, outline=(20, 20, 20))


def _make_sample(index: int, split: str, root: Path, rng: random.Random) -> QuantSample:
    color_names = rng.sample(list(COLORS), 3)
    shapes = rng.sample(list(SHAPES), 3)
    positions = [(25, 37), (56, 74), (88, 37)]
    rng.shuffle(positions)

    image = Image.new("RGB", (112, 112), (235, 226, 208))
    draw = ImageDraw.Draw(image)
    draw.rectangle((3, 3, 108, 108), outline=(120, 105, 85), width=2)
    for color_name, shape, position in zip(color_names, shapes, positions):
        _draw_shape(draw, shape, position, COLORS[color_name])

    if index % 4 == 0:
        target_index = min(range(3), key=lambda item: positions[item][0])
        prompt = "Which object is furthest to the left on the tabletop?"
        target = f"The {color_names[target_index]} {shapes[target_index]}."
    elif index % 4 == 1:
        target_index = max(range(3), key=lambda item: positions[item][0])
        prompt = "Which object is furthest to the right on the tabletop?"
        target = f"The {color_names[target_index]} {shapes[target_index]}."
    elif index % 4 == 2:
        color_name = color_names[1]
        prompt = f"What shape is the {color_name} object?"
        target = f"It is a {shapes[1]}."
    else:
        shape = shapes[2]
        prompt = f"What color is the {shape}?"
        target = f"It is {color_names[2]}."

    sample_id = f"{'calib' if split == 'calibration' else 'eval'}_{index:04d}"
    image_path = root / "images" / f"{sample_id}.png"
    image.save(image_path)
    return QuantSample(sample_id, split, str(image_path.resolve()), prompt, target)


def generate_tiny_dataset(root: str | Path, *, seed: int = 42) -> Path:
    root = Path(root)
    (root / "images").mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    samples = [
        *[_make_sample(index, "calibration", root, rng) for index in range(8)],
        *[_make_sample(index, "evaluation", root, rng) for index in range(16)],
    ]
    manifest = root / "manifest.jsonl"
    temporary = manifest.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.__dict__, ensure_ascii=False) + "\n")
    temporary.replace(manifest)
    return manifest


def load_manifest(path: str | Path) -> list[QuantSample]:
    path = Path(path)
    samples: list[QuantSample] = []
    ids: set[str] = set()
    split_ids: dict[str, set[str]] = {"calibration": set(), "evaluation": set()}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                sample = QuantSample(**raw)
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(f"invalid manifest row {line_number}: {exc}") from exc
            if sample.split not in split_ids:
                raise ValueError(f"invalid split {sample.split!r} in row {line_number}")
            if sample.id in ids:
                raise ValueError(f"duplicate sample id: {sample.id}")
            if not sample.target.strip():
                raise ValueError(f"empty target for sample {sample.id}")
            image_path = Path(sample.image)
            if not image_path.is_absolute():
                image_path = (path.parent / image_path).resolve()
                sample = QuantSample(sample.id, sample.split, str(image_path), sample.prompt, sample.target)
            if not image_path.is_file():
                raise FileNotFoundError(f"image for sample {sample.id} does not exist: {image_path}")
            ids.add(sample.id)
            split_ids[sample.split].add(sample.id)
            samples.append(sample)
    if split_ids["calibration"] & split_ids["evaluation"]:
        raise ValueError("calibration and evaluation IDs overlap")
    if not split_ids["calibration"] or not split_ids["evaluation"]:
        raise ValueError("manifest must contain both calibration and evaluation samples")
    return samples


def select_samples(samples: list[QuantSample], *, split: str, limit: int) -> list[QuantSample]:
    selected = [sample for sample in samples if sample.split == split][:limit]
    if len(selected) != limit:
        raise ValueError(f"requested {limit} {split} samples, found {len(selected)}")
    return selected
