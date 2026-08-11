from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from .data import fixed_subset, load_manifest
from .evaluation import evaluate_multimodal
from .reporting import atomic_json
from .runner import environment_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Qwen's official packed AWQ checkpoint")
    parser.add_argument("--model-path", default="Qwen/Qwen2.5-VL-3B-Instruct-AWQ")
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preset", choices=("smoke", "screen", "full"), default="smoke")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_manifest(args.evaluation_manifest)
    limit = {"smoke": 2, "screen": 256, "full": len(samples)}[args.preset]
    samples = fixed_subset(samples, limit, args.seed)
    atomic_json(args.output_dir / "config.json", {
        "kind": "external_official_anchor",
        "model_path": args.model_path,
        "expected_quantization": {
            "bits": 4, "group_size": 128, "zero_point": True,
            "modules_to_not_convert": ["visual"],
        },
        "environment": environment_record(),
    })
    try:
        processor = AutoProcessor.from_pretrained(args.model_path)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path, device_map="cpu", attn_implementation="eager",
            low_cpu_mem_usage=True,
        ).eval()
        metrics = evaluate_multimodal(
            model, processor, samples,
            max_new_tokens=args.max_new_tokens,
            output_jsonl=args.output_dir / "generations.jsonl",
        )
        atomic_json(args.output_dir / "metrics.json", metrics)
    except Exception as error:
        atomic_json(args.output_dir / "error.json", {
            "type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc(),
            "hint": "Official packed AWQ may require a backend unavailable on CPU; do not replace it with local RTN.",
        })
        raise


if __name__ == "__main__":
    main()
