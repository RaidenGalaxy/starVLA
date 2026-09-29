from __future__ import annotations

from experiments.streaming_moment_benchmark.prepare_cpu_pilot import balanced_select
from experiments.streaming_moment_benchmark.download_media import _hf_root


def test_balanced_select_is_deterministic_and_identity_disjoint() -> None:
    rows = [
        {"id": "same", "stratum": "a", "value": 1},
        {"id": "same", "stratum": "b", "value": 2},
        {"id": "two", "stratum": "a", "value": 3},
        {"id": "three", "stratum": "b", "value": 4},
    ]
    kwargs = {
        "stratum": lambda row: row["stratum"],
        "identity": lambda row: row["id"],
        "seed": 7,
    }
    first = balanced_select(rows, 3, **kwargs)
    second = balanced_select(rows, 3, **kwargs)
    assert first == second
    assert len({row["id"] for row in first}) == 3


def test_huggingface_local_root_matches_repo_filename(tmp_path) -> None:
    output = tmp_path / "repo" / "data" / "videos" / "GUI" / "app" / "1.mp4"
    job = {"filename": "data/videos/GUI/app/1.mp4", "output_path": str(output)}
    assert _hf_root(job) == tmp_path / "repo"
