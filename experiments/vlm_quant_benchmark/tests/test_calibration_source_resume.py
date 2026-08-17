import pytest
from PIL import Image

from experiments.vlm_quant_benchmark.data import (
    MultimodalSample,
    file_digest,
    write_manifest,
)
from experiments.vlm_quant_benchmark.prepare_multimodal_calibration import (
    atomic_json,
    load_completed_source,
    protocol_digest,
    source_protocol,
)
from experiments.vlm_quant_benchmark.run_corpus_ppl import (
    load_partial_results,
    save_partial_results,
)


SOURCE = {
    "name": "toy_train",
    "repo": "toy/repo",
    "config": None,
    "split": "train",
    "revision": "revision-1",
}


def completed_state(tmp_path):
    state = tmp_path / "toy_train"
    image = state / "images" / "sample.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), "red").save(image)
    manifest = state / "samples.jsonl"
    write_manifest(manifest, [MultimodalSample(
        id="toy_train:0000",
        dataset="toy_train",
        split="calibration_train",
        image=image,
        prompt="What color?",
        target="red",
        upstream_revision="revision-1",
    )])
    protocol = source_protocol(
        SOURCE,
        0,
        samples_per_source=1,
        seed=42,
        shuffle_buffer=2048,
    )
    atomic_json(state / ".complete.json", {
        "status": "complete",
        "protocol": protocol,
        "protocol_sha256": protocol_digest(protocol),
        "manifest_sha256": file_digest(manifest),
        "samples": 1,
    })
    return state, protocol, manifest


def test_completed_source_cache_hit_validates_manifest_and_images(tmp_path):
    state, protocol, _ = completed_state(tmp_path)
    loaded = load_completed_source(state, protocol, samples_per_source=1)
    assert loaded is not None
    assert len(loaded) == 1
    assert loaded[0].dataset == "toy_train"


def test_completed_source_rejects_protocol_and_manifest_mismatch(tmp_path):
    state, protocol, manifest = completed_state(tmp_path)
    changed = protocol | {"shuffle_buffer": 1024}
    with pytest.raises(ValueError, match="protocol mismatch"):
        load_completed_source(state, changed, samples_per_source=1)

    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="manifest mismatch"):
        load_completed_source(state, protocol, samples_per_source=1)

def test_corpus_partial_resume_requires_matching_digests(tmp_path):
    path = tmp_path / "corpus_metrics.partial.json"
    results = {"wikitext2_test": {"ppl": 10.0, "windows": 16}}
    digests = {"model": "model-a", "wikitext2_test": "data-a"}
    save_partial_results(
        path, results,
        config_digest="config-a",
        input_digests=digests,
    )
    assert load_partial_results(
        path, config_digest="config-a", input_digests=digests,
    ) == results

    with pytest.raises(ValueError, match="config digest mismatch"):
        load_partial_results(
            path, config_digest="config-b", input_digests=digests,
        )
    with pytest.raises(ValueError, match="input digest mismatch"):
        load_partial_results(
            path, config_digest="config-a", input_digests=digests | {"model": "model-b"},
        )
