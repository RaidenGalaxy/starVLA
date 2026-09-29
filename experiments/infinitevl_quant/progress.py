from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


def json_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    temporary.replace(path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(value)
    temporary.replace(path)


def safe_id(value: str) -> str:
    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")[:80]
    digest = hashlib.sha256(value.encode()).hexdigest()[:12]
    return f"{prefix}-{digest}"


class StageProgress:
    def __init__(self, root: Path, stage: str, *, config_digest: str, inputs: dict) -> None:
        self.root = root / stage
        self.root.mkdir(parents=True, exist_ok=True)
        self.config_digest = config_digest
        self.inputs = inputs
        self.identity = {
            "format_version": 1, "stage": stage,
            "config_digest": config_digest, "input_digests": inputs,
        }
        identity_path = self.root / "identity.json"
        if identity_path.exists():
            existing = json.loads(identity_path.read_text())
            if existing != self.identity:
                raise RuntimeError(
                    f"stage {stage} output belongs to different inputs/config; "
                    "choose a new --output-dir"
                )
        else:
            atomic_json(identity_path, self.identity)

    def sample_path(self, variant: str, sample_id: str) -> Path:
        return self.root / variant / "samples" / f"{safe_id(sample_id)}.json"

    def complete_sample(self, variant: str, sample_id: str, row: dict) -> None:
        atomic_json(self.sample_path(variant, sample_id), row)

    def completed_sample(self, variant: str, sample_id: str) -> dict | None:
        path = self.sample_path(variant, sample_id)
        if not path.is_file():
            return None
        row = json.loads(path.read_text())
        return row if row.get("id") == sample_id else None

    def rows(self, variant: str) -> list[dict]:
        directory = self.root / variant / "samples"
        if not directory.is_dir():
            return []
        return [json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))]

    def is_complete(self, required: tuple[str, ...]) -> bool:
        marker = self.root / ".complete"
        if not marker.is_file() or any(not (self.root / name).exists() for name in required):
            return False
        try:
            payload = json.loads(marker.read_text())
        except (OSError, json.JSONDecodeError):
            return False
        return (
            all(payload.get(key) == value for key, value in self.identity.items())
            and tuple(payload.get("required", ())) == required
        )

    def mark_complete(self, required: tuple[str, ...]) -> None:
        missing = [name for name in required if not (self.root / name).exists()]
        if missing:
            raise FileNotFoundError(f"cannot complete {self.root.name}; missing {missing}")
        atomic_json(self.root / ".complete", self.identity | {"required": list(required)})
