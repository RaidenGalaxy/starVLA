from __future__ import annotations

import json

import huggingface_hub
import pytest
from requests import HTTPError, Response
from requests.exceptions import ChunkedEncodingError

from experiments.ffn_quant_generalization import prepare
from experiments.ffn_quant_generalization.config import Config
from experiments.infinitevl_quant.progress import atomic_json


def test_truncated_downloads_resume_after_more_than_six_failures(tmp_path, monkeypatch):
    config = Config(output_dir=tmp_path / "run")
    initial = prepare.start_clock(config)
    partial = tmp_path / "weights.incomplete"
    partial.write_bytes(b"already downloaded bytes")
    attempts = []

    def snapshot(repo, **kwargs):
        attempts.append(repo)
        assert kwargs["revision"]
        if len(attempts) <= 8:
            raise ChunkedEncodingError("transfer truncated")
        return str(tmp_path / "cached_snapshot")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    monkeypatch.setattr(prepare.time, "sleep", lambda value: None)
    result = prepare.download(config)
    assert len(result) == 3 and len(attempts) == 11
    assert prepare.start_clock(config) == initial
    assert partial.read_bytes() == b"already downloaded bytes"
    retry = json.loads((config.output_dir / "prepare/model7b_retry.json").read_text())
    assert retry["attempt"] == 8
    assert retry["existing_partial_downloads_preserved"]
    assert retry["original_deadline_unix"] == initial["started_unix"] + 228 * 3600


@pytest.mark.parametrize("code", [401, 403, 404])
def test_auth_or_revision_errors_are_not_retried(tmp_path, monkeypatch, code):
    config = Config(output_dir=tmp_path / "run")
    calls = []
    response = Response()
    response.status_code = code

    def snapshot(*args, **kwargs):
        calls.append(1)
        raise HTTPError("permanent error", response=response)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    with pytest.raises(HTTPError):
        prepare.download(config)
    assert len(calls) == 1


def test_programming_errors_are_not_silently_retried(tmp_path, monkeypatch):
    config = Config(output_dir=tmp_path / "run")

    def snapshot(*args, **kwargs):
        raise ValueError("invalid arguments")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    with pytest.raises(ValueError, match="invalid arguments"):
        prepare.download(config)


def test_download_stops_at_original_inference_deadline(tmp_path, monkeypatch):
    config = Config(output_dir=tmp_path / "run")
    atomic_json(
        config.output_dir / "budget/started.json", {"started_unix": 1000, "total_hours": 240, "round_hours": 120}
    )
    monkeypatch.setattr(prepare.time, "time", lambda: 1000 + 229 * 3600)
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda *args, **kwargs: pytest.fail("deadline already passed")
    )
    with pytest.raises(RuntimeError, match="analysis reserve"):
        prepare.download(config)
