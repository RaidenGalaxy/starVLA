from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from .corpus_evaluation import evaluate_token_sequences_ppl, paired_window_delta
from .official_awq import load_official_awq_dequantized
from .provenance import completed_run, mark_complete, sha256_json, source_digest
from .reporting import atomic_json
from .run_corpus_ppl import load_sequences
from .run_official_awq_anchor import OFFICIAL_AWQ_REVISION
from .runner import environment_record, load_reference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Official Qwen AWQ corpus PPL anchor")
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--awq-model-path", default="Qwen/Qwen2.5-VL-3B-Instruct-AWQ")
    parser.add_argument("--awq-revision", default=OFFICIAL_AWQ_REVISION)
    parser.add_argument("--corpora-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preset", choices=("smoke", "screen", "full"), default="screen")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--reference-metrics", type=Path)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    awq_path = Path(args.awq_model_path)
    if not awq_path.exists():
        awq_path = Path(snapshot_download(
            repo_id=args.awq_model_path,
            revision=args.awq_revision,
        ))
    corpus_files = {
        "wikitext2_test": args.corpora_dir / "wikitext2_test.pt",
        "ptb_test": args.corpora_dir / "ptb_test.pt",
        "c4_validation": args.corpora_dir / "c4_validation_256x2048.pt",
    }
    config = {
        "method": "official_qwen_awq",
        "scheme_id": "official_w4a16_g128_asym",
        "scope": "decoder_canonical",
        "implementation_tier": "official_anchor",
        "awq_revision": args.awq_revision,
        "preset": args.preset,
        "batch_size": args.batch_size,
        "sequence_length": 2048,
    }
    config_digest = sha256_json(config)
    digests = {
        "base_model": source_digest(args.base_model_path),
        "official_awq_model": source_digest(awq_path),
        "preset": sha256_json(args.preset),
        **{name: source_digest(path) for name, path in corpus_files.items()},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    required = ("corpus_metrics.json",)
    if args.resume and completed_run(
        args.output_dir,
        config_digest=config_digest,
        input_digests=digests,
        required_artifacts=required,
    ):
        return
    for stale in (".complete", "corpus_metrics.json", "error.json"):
        (args.output_dir / stale).unlink(missing_ok=True)
    atomic_json(args.output_dir / "config.json", config | {
        "config_hash": config_digest,
        "input_digests": digests,
        "environment": environment_record(),
    })
    try:
        model, _processor, conversion = load_reference(args.base_model_path)
        atomic_json(args.output_dir / "reference_conversion.json", conversion)
        audit = load_official_awq_dequantized(model, awq_path, group_size=128)
        if audit["packed_modules"] != 252:
            raise ValueError(
                f"official AWQ expected 252 decoder modules, got {audit['packed_modules']}"
            )
        max_windows = {"smoke": 2, "screen": 16, "full": None}[args.preset]
        metrics = {}
        for name, path in corpus_files.items():
            metrics[name] = evaluate_token_sequences_ppl(
                model,
                load_sequences(path),
                sequence_length=2048,
                batch_size=args.batch_size,
                max_windows=max_windows,
            )
        if args.reference_metrics is not None:
            reference = json.loads(args.reference_metrics.read_text())
            for name in metrics:
                metrics[name]["paired_vs_bf16"] = paired_window_delta(
                    metrics[name], reference[name], seed=42,
                )
        atomic_json(args.output_dir / "corpus_metrics.json", metrics)
        atomic_json(args.output_dir / "quantization_audit.json", audit)
        mark_complete(
            args.output_dir,
            config_digest=config_digest,
            input_digests=digests,
            required_artifacts=required,
        )
    except Exception as error:
        atomic_json(args.output_dir / "error.json", {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "config_hash": config_digest,
            "input_digests": digests,
        })
        raise


if __name__ == "__main__":
    main()
