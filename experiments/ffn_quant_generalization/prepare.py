from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.infinitevl_quant.progress import atomic_json, file_sha256, json_digest
from experiments.vlm_quant_benchmark.data import MultimodalSample, load_manifest, write_manifest

from .config import DATASETS, SOURCES, Config


def start_clock(config):
    path = config.output_dir / "budget/started.json"
    if not path.exists():
        atomic_json(path, {"started_unix": time.time(), "total_hours": 240, "round_hours": 120})
    value = json.loads(path.read_text())
    if value["total_hours"] != 240 or value["round_hours"] != 120:
        raise RuntimeError("persistent preparation clock differs from protocol")
    if time.time() >= value["started_unix"] + 240 * 3600:
        raise RuntimeError("total preparation/execution deadline exceeded")
    return value


def download(config):
    from huggingface_hub import constants, snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    from requests.exceptions import RequestException

    constants.HF_HUB_DOWNLOAD_TIMEOUT = max(120, constants.HF_HUB_DOWNLOAD_TIMEOUT)

    start_clock(config)
    patterns = {
        "model7b": ["*.json", "*.safetensors", "*.txt", "*.model"],
        "chartqa": ["README.md", "data/test-*.parquet"],
        "gqa": ["README.md", "testdev_balanced_images/*.parquet", "testdev_balanced_instructions/*.parquet"],
    }

    def fetch(key):
        repo, revision = SOURCES[key]
        print(f"[prepare] downloading {repo}@{revision}", flush=True)
        attempt = 0
        while True:
            clock = start_clock(config)
            inference_deadline = clock["started_unix"] + (clock["total_hours"] - config.reserve_hours) * 3600
            if time.time() >= inference_deadline:
                raise RuntimeError("download stopped to preserve the existing analysis reserve")
            try:
                path = snapshot_download(
                    repo,
                    revision=revision,
                    repo_type="model" if key == "model7b" else "dataset",
                    allow_patterns=patterns[key],
                    max_workers=3,
                    etag_timeout=120,
                )
                break
            except Exception as error:
                code = getattr(getattr(error, "response", None), "status_code", None)
                # A large response may be truncated repeatedly by a proxy. Keep
                # its existing .incomplete file and resume until the original
                # deadline; never exhaust a fixed small number of attempts.
                if code in (401, 403, 404) or not isinstance(
                    error, (RequestException, LocalEntryNotFoundError, TimeoutError, ConnectionError)
                ):
                    raise
                attempt += 1
                delay = min(30, 2 ** min(attempt - 1, 5), max(0, inference_deadline - time.time()))
                atomic_json(
                    config.output_dir / "prepare" / f"{key}_retry.json",
                    {
                        "repo": repo,
                        "revision": revision,
                        "attempt": attempt,
                        "error_type": type(error).__name__,
                        "updated_unix": time.time(),
                        "next_retry_seconds": delay,
                        "original_deadline_unix": inference_deadline,
                        "existing_partial_downloads_preserved": True,
                    },
                )
                print(f"[prepare] retry {repo}, attempt={attempt}, error={type(error).__name__}", flush=True)
                time.sleep(delay)
        print(f"[prepare] downloaded {key}: {path}", flush=True)
        return key, str(Path(path).resolve())

    # snapshot_download already has an internal worker pool. Serializing repos
    # avoids a tqdm global-lock race between simultaneous snapshot downloads.
    paths = dict(fetch(key) for key in patterns)
    atomic_json(config.output_dir / "prepare/downloads.json", {"paths": paths, "sources": SOURCES})
    return paths


def pixels(image):
    if isinstance(image, dict):
        image = Image.open(io.BytesIO(image["bytes"])) if image.get("bytes") else Image.open(image["path"])
    image = image.convert("RGB")
    digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
    return image, digest


def historical_images(config):
    all_samples = load_manifest(config.calibration_manifest) + load_manifest(config.evaluation_manifest)
    hashes = set()
    for sample in all_samples:
        for path in sample.images:
            with Image.open(path) as image:
                hashes.add(pixels(image)[1])
    return hashes


def select_groups(candidates, blocked, seed):
    """Select one question per decoded RGB image; split before model execution."""
    selected, exclusions = {}, collections.Counter()
    seen = set(blocked)
    for group, rows in sorted(candidates.items()):
        order = np.random.default_rng(seed + (0 if group == "human" else 1)).permutation(len(rows))
        available = []
        local_seen = set(seen)
        for index in order:
            item = rows[int(index)]
            if item["pixel_sha256"] in local_seen:
                exclusions["duplicate_or_historical_image"] += 1
                continue
            if not str(item["target"]).strip():
                exclusions["empty_target"] += 1
                continue
            local_seen.add(item["pixel_sha256"])
            available.append(item)
        multiplier = 2 if group == "gqa" else 1
        required = 42 * multiplier
        if len(available) < required:
            raise RuntimeError(f"insufficient unique {group} images: {len(available)} < {required}")
        seen.update(item["pixel_sha256"] for item in available[:required])
        selected[group] = {
            "pilot8": available[: 2 * multiplier],
            "screen32": available[2 * multiplier : 10 * multiplier],
            "confirm128": available[10 * multiplier : 42 * multiplier],
        }
    return selected, dict(exclusions)


def make_data(config, paths):
    import pyarrow.parquet as pq

    base = config.output_dir / "prepare"
    marker = base / "data.json"
    if marker.exists():
        info = json.loads(marker.read_text())
        for name, digest in info["files"].items():
            if file_sha256(Path(name)) != digest:
                raise RuntimeError("prepared data checksum changed")
        return info
    candidates = {"human": [], "machine": [], "gqa": []}
    chart = pq.read_table(next(Path(paths["chartqa"]).glob("data/test-*.parquet"))).to_pylist()
    cache = {}
    for index, row in enumerate(chart):
        image, digest = pixels(row["image"])
        cache.setdefault(digest, image)
        group = "human" if row["human_or_machine"] == 0 else "machine"
        candidates[group].append(
            {
                "id": f"ChartQA_TEST:{index}",
                "question": row["query"],
                "target": str(row["label"][0]),
                "pixel_sha256": digest,
                "dataset": DATASETS[0],
                "group": group,
            }
        )
    gqa_root = Path(paths["gqa"])
    images = pq.read_table(next(gqa_root.glob("testdev_balanced_images/*.parquet"))).to_pylist()
    image_map = {}
    for row in images:
        image, digest = pixels(row["image"])
        cache.setdefault(digest, image)
        image_map[str(row["id"])] = digest
    questions = pq.read_table(next(gqa_root.glob("testdev_balanced_instructions/*.parquet"))).to_pylist()
    for row in questions:
        if not row["isBalanced"]:
            continue
        digest = image_map[str(row["imageId"])]
        candidates["gqa"].append(
            {
                "id": f"GQA_TESTDEV_BALANCED:{row['id']}",
                "question": row["question"],
                "target": row["answer"],
                "pixel_sha256": digest,
                "dataset": DATASETS[1],
                "group": "gqa",
            }
        )
    blocked = historical_images(config)
    chosen, exclusions = select_groups(candidates, blocked, config.seed)
    splits = {name: [] for name in ("pilot8", "screen32", "confirm128", "qa32")}
    identities = {}
    for group, values in chosen.items():
        for split, rows in values.items():
            for row in rows:
                digest = row["pixel_sha256"]
                path = base / "images" / f"{digest}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                cache[digest].save(path)
                revision = SOURCES["gqa" if group == "gqa" else "chartqa"][1]
                sample = MultimodalSample(
                    id=row["id"],
                    dataset=row["dataset"],
                    split=split,
                    image=path.resolve(),
                    prompt=row["question"].strip() + "\nAnswer the question using a single word or phrase.",
                    target=str(row["target"]).strip(),
                    category=group,
                    official_index=row["id"].split(":", 1)[1],
                    upstream_revision=revision,
                    image_sha256=file_sha256(path),
                )
                splits[split].append(sample)
                identities[sample.id] = {"pixel_sha256": digest, "source_group": group}
            if split == "confirm128":
                n = 16 if group == "gqa" else 8
                ids = {r["id"] for r in rows[:n]}
                splits["qa32"].extend(s for s in splits[split] if s.id in ids)
    for name, expected in (("pilot8", 4), ("screen32", 16), ("confirm128", 64), ("qa32", 16)):
        counts = collections.Counter(s.dataset for s in splits[name])
        if counts != dict.fromkeys(DATASETS, expected):
            raise RuntimeError(f"incorrect {name} quotas: {counts}")
        write_manifest(base / f"{name}.jsonl", sorted(splits[name], key=lambda s: s.id))
    assert_disjoint_splits(splits, identities, blocked)
    files = {str(p.resolve()): file_sha256(p) for p in sorted(base.glob("*.jsonl"))}
    files.update({str(p.resolve()): file_sha256(p) for p in sorted((base / "images").glob("*.png"))})
    info = {
        "files": files,
        "pixel_identities": identities,
        "exclusions": exclusions,
        "historical_blocked_images": len(blocked),
        "chartqa_train_previously_used_for_calibration": True,
        "seed": config.seed,
        "sources": SOURCES,
        "selection": "one_question_per_RGB_image_then_disjoint_splits",
    }
    atomic_json(marker, info)
    print(f"[prepare] frozen data identity={json_digest(info)}", flush=True)
    return info


def assert_disjoint_splits(splits, identities, blocked):
    seen = set(blocked)
    for name in ("pilot8", "screen32", "confirm128"):
        hashes = [identities[s.id]["pixel_sha256"] for s in splits[name]]
        if len(set(hashes)) != len(hashes) or seen.intersection(hashes):
            raise RuntimeError("image leakage between new/historical splits")
        seen.update(hashes)
    if not {s.id for s in splits["qa32"]} <= {s.id for s in splits["confirm128"]}:
        raise RuntimeError("QA must be a fixed confirmation subset")


def prepare(config):
    start_clock(config)
    marker = config.output_dir / "prepare/downloads.json"
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved["sources"] != json.loads(json.dumps(SOURCES)):
            raise RuntimeError("download revisions changed")
        paths = saved["paths"]
    else:
        paths = download(config)
    make_data(config, paths)
    return paths


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=Config().output_dir)
    args = p.parse_args()
    prepare(Config(output_dir=args.output_dir))
