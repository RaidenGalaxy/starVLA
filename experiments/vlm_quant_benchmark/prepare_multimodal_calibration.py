from __future__ import annotations

import argparse
import collections
import hashlib
import json
import time
from pathlib import Path

from PIL import Image

from .data import (
    MultimodalSample,
    file_digest,
    load_manifest,
    normalized_question_digest,
    write_manifest,
)


SOURCES = (
    {
        "name": "chartqa_train",
        "repo": "HuggingFaceM4/ChartQA",
        "config": None,
        "split": "train",
        "revision": "b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5",
    },
    {
        "name": "textvqa_train",
        "repo": "lmms-lab/textvqa",
        "config": None,
        "split": "train",
        "revision": "9c0699cd19768ac5ab97568f6b3cbac4c0062884",
    },
    {
        "name": "scienceqa_img_train",
        "repo": "lmms-lab/ScienceQA-IMG",
        "config": None,
        "split": "train",
        "revision": "b9984af01798586705cee1fbdfa80a6dcbef04f1",
    },
    {
        "name": "infographicvqa_train",
        "repo": "lmms-lab/DocVQA",
        "config": "InfographicVQA",
        "split": "train",
        "data_file": "InfographicVQA/train-00008-of-00024.parquet",
        "shard_selection": "random.Random(42+3).randrange(24)=8",
        "revision": "539088ef8a8ada01ac8e2e6d4e372586748a265e",
    },
)


def _first(row: dict, names: tuple[str, ...], default=None):
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return default


def _answer(row: dict) -> str:
    value = _first(row, ("answer", "answers", "label", "target"))
    choices = _first(row, ("choices", "options"), [])
    if isinstance(value, int) and choices and 0 <= value < len(choices):
        value = choices[value]
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("empty answer list")
        value = collections.Counter(map(str, value)).most_common(1)[0][0]
    if isinstance(value, dict):
        value = _first(value, ("answer", "text", "label"))
    value = str(value).strip()
    if not value:
        raise ValueError("empty calibration answer")
    return value


def _prompt(row: dict) -> str:
    question = str(_first(row, ("question", "query", "prompt", "text"))).strip()
    choices = _first(row, ("choices", "options"), [])
    if choices:
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        question += "\nOptions:\n" + "\n".join(
            f"{letters[index]}. {choice}" for index, choice in enumerate(choices)
        )
    hint = _first(row, ("hint", "context"))
    if hint:
        question = f"Context: {hint}\nQuestion: {question}"
    return question


def _images(row: dict) -> list[Image.Image]:
    value = _first(row, ("images", "image"))
    if value is None:
        raise ValueError("calibration row has no image")
    values = value if isinstance(value, (list, tuple)) else [value]
    images = []
    for item in values:
        if isinstance(item, dict):
            item = item.get("path") or item.get("bytes")
        if isinstance(item, (str, Path)):
            with Image.open(item) as opened:
                images.append(opened.convert("RGB"))
        elif isinstance(item, Image.Image):
            images.append(item.convert("RGB"))
        else:
            raise TypeError(f"unsupported image value: {type(item).__name__}")
    return images


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def source_protocol(
    source: dict,
    source_index: int,
    *,
    samples_per_source: int,
    seed: int,
    shuffle_buffer: int,
) -> dict:
    return {
        "format_version": 1,
        "source": source,
        "samples_per_source": samples_per_source,
        "seed": seed + source_index,
        "shuffle_buffer": shuffle_buffer,
        "selection": (
            "pinned_train_shard_then_dataset_shuffle" if source.get("data_file")
            else "datasets_streaming_shuffle_then_first_usable"
        ),
    }


def protocol_digest(protocol: dict) -> str:
    payload = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def load_completed_source(
    state_dir: Path,
    protocol: dict,
    *,
    samples_per_source: int,
) -> list[MultimodalSample] | None:
    marker = state_dir / ".complete.json"
    manifest = state_dir / "samples.jsonl"
    if not marker.is_file():
        return None
    payload = json.loads(marker.read_text())
    expected_protocol = protocol_digest(protocol)
    if payload.get("protocol_sha256") != expected_protocol:
        raise ValueError(f"source resume protocol mismatch: {state_dir.name}")
    if not manifest.is_file():
        raise ValueError(f"source complete marker has no manifest: {state_dir.name}")
    if payload.get("manifest_sha256") != file_digest(manifest):
        raise ValueError(f"source resume manifest mismatch: {state_dir.name}")
    samples = load_manifest(manifest)
    if len(samples) != samples_per_source:
        raise ValueError(f"source resume row count mismatch: {state_dir.name}")
    source = protocol["source"]
    if any(
        item.dataset != source["name"]
        or item.upstream_revision != source["revision"]
        for item in samples
    ):
        raise ValueError(f"source resume provenance mismatch: {state_dir.name}")
    return samples


def collect_source(
    load_dataset,
    source: dict,
    source_index: int,
    args: argparse.Namespace,
    image_dir: Path,
    shuffle_buffer: int,
) -> list[MultimodalSample]:
    if source.get("data_file"):
        from huggingface_hub import hf_hub_download

        local_file = hf_hub_download(
            repo_id=source["repo"],
            repo_type="dataset",
            revision=source["revision"],
            filename=source["data_file"],
            local_dir=args.output_dir / "_downloads" / source["name"],
        )
        stream = load_dataset(
            "parquet",
            data_files={source["split"]: [local_file]},
            split=source["split"],
            streaming=False,
        ).shuffle(seed=args.seed + source_index)
    else:
        positional = [source["repo"]]
        if source["config"] is not None:
            positional.append(source["config"])
        stream = load_dataset(
            *positional,
            split=source["split"],
            revision=source["revision"],
            streaming=True,
        ).shuffle(
            seed=args.seed + source_index,
            buffer_size=shuffle_buffer,
        )
    samples: list[MultimodalSample] = []
    for upstream_index, row in enumerate(stream):
        try:
            prompt, target, images = _prompt(row), _answer(row), _images(row)
        except (KeyError, TypeError, ValueError, OSError):
            continue
        paths = []
        for image_index, image in enumerate(images):
            path = image_dir / (
                f"{source['name']}__{len(samples):04d}__{image_index}.png"
            )
            image.save(path, format="PNG")
            paths.append(path)
        identifier = f"{source['name']}:{len(samples):04d}"
        category = str(_first(row, ("category", "subject", "topic", "type"), ""))
        samples.append(MultimodalSample(
            id=identifier,
            dataset=source["name"],
            split="calibration_train",
            image=paths[0],
            extra_images=tuple(paths[1:]),
            prompt=prompt,
            target=target,
            answer_type=str(_first(row, ("answer_type", "question_type"), "open")),
            category=category,
            official_index=str(
                _first(row, ("id", "question_id", "questionId"), upstream_index)
            ),
            upstream_revision=source["revision"],
            image_sha256=file_digest(paths[0]),
            question_sha256=normalized_question_digest(source["name"], prompt),
        ))
        if len(samples) == args.samples_per_source:
            return samples
    raise RuntimeError(
        f"{source['name']} yielded only {len(samples)} usable rows"
    )


def prepare_source(
    load_dataset,
    source: dict,
    source_index: int,
    args: argparse.Namespace,
    image_dir: Path,
) -> tuple[list[MultimodalSample], bool]:
    shuffle_buffer = (
        args.shuffle_buffer if source_index == 0
        else args.large_source_shuffle_buffer
    )
    protocol = source_protocol(
        source,
        source_index,
        samples_per_source=args.samples_per_source,
        seed=args.seed,
        shuffle_buffer=shuffle_buffer,
    )
    state_dir = args.output_dir / "_source_state" / source["name"]
    completed = load_completed_source(
        state_dir,
        protocol,
        samples_per_source=args.samples_per_source,
    )
    if completed is not None:
        return completed, True

    state_dir.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, args.retries + 1):
        try:
            samples = collect_source(
                load_dataset, source, source_index, args, image_dir, shuffle_buffer,
            )
            manifest = state_dir / "samples.jsonl"
            write_manifest(manifest, samples)
            verified = load_manifest(manifest)
            if len(verified) != args.samples_per_source:
                raise ValueError("source manifest verification failed")
            atomic_json(state_dir / ".complete.json", {
                "status": "complete",
                "protocol": protocol,
                "protocol_sha256": protocol_digest(protocol),
                "manifest_sha256": file_digest(manifest),
                "samples": len(verified),
            })
            return verified, False
        except Exception as error:
            last_error = error
            atomic_json(state_dir / "error.json", {
                "status": "retrying" if attempt < args.retries else "failed",
                "attempt": attempt,
                "max_attempts": args.retries,
                "type": type(error).__name__,
                "message": str(error),
                "protocol_sha256": protocol_digest(protocol),
            })
            if attempt < args.retries:
                time.sleep(min(args.retry_base_seconds * (2 ** (attempt - 1)), 30))
    raise RuntimeError(
        f"{source['name']} failed after {args.retries} attempts"
    ) from last_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare pinned, source-resumable 4x32 multimodal calibration data"
    )
    parser.add_argument(
        "--large-source-shuffle-buffer", type=int, default=64
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-source", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=2048)
    parser.add_argument("--retries", type=int, default=10)
    parser.add_argument("--retry-base-seconds", type=float, default=5)
    args = parser.parse_args()
    if args.shuffle_buffer <= 0 or args.large_source_shuffle_buffer <= 0:
        parser.error(
            "shuffle-buffer and large-source-shuffle-buffer must be positive"
        )
    if args.retries <= 0 or args.retry_base_seconds < 0:
        parser.error("retries must be positive and retry-base-seconds non-negative")
    return args


def main() -> None:
    args = parse_args()
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ImportError(
            "datasets is required to prepare multimodal calibration"
        ) from error

    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    samples: list[MultimodalSample] = []
    metadata = {
        "seed": args.seed,
        "samples_per_source": args.samples_per_source,
        "sources": [],
        "source_resumable": True,
    }
    for source_index, source in enumerate(SOURCES):
        source_samples, resumed = prepare_source(
            load_dataset, source, source_index, args, image_dir,
        )
        samples.extend(source_samples)
        metadata["sources"].append(
            source | {"selected": len(source_samples), "resumed": resumed}
        )

    expected = len(SOURCES) * args.samples_per_source
    if len(samples) != expected:
        raise RuntimeError(f"combined calibration rows {len(samples)} != {expected}")
    manifest = args.output_dir / "multimodal_calibration_4x32.jsonl"
    write_manifest(manifest, samples)
    metadata["manifest_sha256"] = file_digest(manifest)
    atomic_json(args.output_dir / "calibration_sources.json", metadata)


if __name__ == "__main__":
    main()

