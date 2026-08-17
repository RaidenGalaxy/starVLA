from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_digest(source: str | Path) -> str:
    path = Path(source)
    if not path.exists():
        return hashlib.sha256(str(source).encode()).hexdigest()
    if path.is_file():
        return file_sha256(path)
    digest = hashlib.sha256()
    candidates = []
    for pattern in (
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors",
        "model-*.safetensors",
        "model.safetensors.index.json",
    ):
        candidates.extend(path.glob(pattern))
    for item in sorted(set(candidates)):
        digest.update(str(item.relative_to(path)).encode())
        digest.update(file_sha256(item).encode())
    if not candidates:
        raise ValueError(f"model source contains no recognized files: {path}")
    return digest.hexdigest()


def run_input_digests(
    *,
    model_source: str | Path,
    calibration_manifest: Path | None,
    evaluation_manifest: Path,
    processor_revision: str,
) -> dict[str, str]:
    return {
        "model": source_digest(model_source),
        "calibration_manifest": (
            file_sha256(calibration_manifest) if calibration_manifest is not None else ""
        ),
        "evaluation_manifest": file_sha256(evaluation_manifest),
        "processor_revision": hashlib.sha256(processor_revision.encode()).hexdigest(),
    }


def completed_run(
    output_dir: Path,
    *,
    config_digest: str,
    input_digests: dict[str, str],
    required_artifacts: tuple[str, ...] = ("metrics.json",),
) -> bool:
    marker = output_dir / ".complete"
    if not marker.is_file() or any(
        not (output_dir / name).is_file() for name in required_artifacts
    ):
        return False
    try:
        payload = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("format_version") == 1
        and payload.get("config_digest") == config_digest
        and payload.get("input_digests") == input_digests
        and tuple(payload.get("required_artifacts", ("metrics.json",))) == required_artifacts
    )


def mark_complete(
    output_dir: Path,
    *,
    config_digest: str,
    input_digests: dict[str, str],
    required_artifacts: tuple[str, ...] = ("metrics.json",),
) -> None:
    missing = [name for name in required_artifacts if not (output_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"cannot mark incomplete run complete; missing: {missing}")
    marker = output_dir / ".complete"
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps({
        "format_version": 1,
        "config_digest": config_digest,
        "input_digests": input_digests,
        "required_artifacts": list(required_artifacts),
    }, indent=2, sort_keys=True) + "\n")
    temporary.replace(marker)
