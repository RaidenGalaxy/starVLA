from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from .data import fixed_subset, load_manifest, manifest_digest
from .evaluation import evaluate_multimodal
from .official_awq import load_official_awq_dequantized
from .provenance import completed_run, mark_complete, sha256_json, source_digest
from .reporting import atomic_json
from .runner import environment_record, load_reference


OFFICIAL_AWQ_REVISION = "e7b623934290c5a4da0ee3c6e1e57bfb6b5abbf2"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Qwen's official AWQ checkpoint on CPU")
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--awq-model-path", default="Qwen/Qwen2.5-VL-3B-Instruct-AWQ")
    parser.add_argument("--awq-revision", default=OFFICIAL_AWQ_REVISION)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preset", choices=("smoke", "screen", "full"), default="smoke")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    awq_path = Path(args.awq_model_path)
    if not awq_path.exists():
        awq_path = Path(snapshot_download(
            repo_id=args.awq_model_path,
            revision=args.awq_revision,
        ))
    samples = load_manifest(args.evaluation_manifest)
    limit = {"smoke": 2, "screen": 256, "full": len(samples)}[args.preset]
    samples = fixed_subset(samples, limit, args.seed)
    config = {
        "kind": "official_anchor",
        "method": "official_qwen_awq",
        "scheme_id": "official_w4a16_g128_asym",
        "scope": "decoder_canonical",
        "implementation_tier": "official_anchor",
        "base_model_path": args.base_model_path,
        "awq_model_path": str(awq_path),
        "awq_revision": args.awq_revision,
        "min_pixels": 256 * 28 * 28,
        "max_pixels": 1280 * 28 * 28,
        "evaluation_manifest_sha256": manifest_digest(args.evaluation_manifest),
        "expected_quantization": {
            "bits": 4,
            "group_size": 128,
            "zero_point": True,
            "modules_to_not_convert": ["visual"],
        },
    }
    config_digest = sha256_json(config)
    digests = {
        "base_model": source_digest(args.base_model_path),
        "official_awq_model": source_digest(awq_path),
        "evaluation_manifest": manifest_digest(args.evaluation_manifest),
        "processor_revision": "transformers-4.57.0-use_fast",
    }
    if args.resume and completed_run(
        args.output_dir,
        config_digest=config_digest,
        input_digests=digests,
    ):
        return
    for stale in (".complete", "metrics.json", "error.json", "generations.jsonl"):
        (args.output_dir / stale).unlink(missing_ok=True)
    atomic_json(args.output_dir / "config.json", config | {
        "config_hash": config_digest,
        "input_digests": digests,
        "environment": environment_record(),
    })
    try:
        model, processor, conversion = load_reference(args.base_model_path)
        atomic_json(args.output_dir / "reference_conversion.json", conversion)
        audit = load_official_awq_dequantized(model, awq_path)
        if audit["packed_modules"] != 252:
            raise ValueError(
                f"official AWQ expected 252 decoder modules, got {audit['packed_modules']}"
            )
        atomic_json(args.output_dir / "quantization_audit.json", audit)
        metrics = evaluate_multimodal(
            model,
            processor,
            samples,
            max_new_tokens=args.max_new_tokens,
            output_jsonl=args.output_dir / "generations.jsonl",
            min_pixels=config["min_pixels"],
            max_pixels=config["max_pixels"],
        )
        atomic_json(args.output_dir / "metrics.json", metrics)
        mark_complete(
            args.output_dir,
            config_digest=config_digest,
            input_digests=digests,
        )
    except Exception as error:
        atomic_json(args.output_dir / "error.json", {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "config_hash": config_digest,
            "input_digests": digests,
            "hint": "CPU dequantization failed; this result must not fall back to local RTN.",
        })
        raise


if __name__ == "__main__":
    main()
