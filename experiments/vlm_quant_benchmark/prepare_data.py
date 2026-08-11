from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .data import write_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare deterministic text corpora and protocol metadata")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--skip-corpora", action="store_true")
    return parser.parse_args()


def encode(tokenizer, text: str) -> torch.Tensor:
    return tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0].cpu()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_protocol(args.output_dir / "protocol.json", args.seed)
    if args.skip_corpora:
        return
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ImportError("Install `datasets` to prepare WikiText-2, PTB, and C4") from error

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    wikitext = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    wiki_tokens = encode(tokenizer, "\n\n".join(row["text"] for row in wikitext))
    torch.save([wiki_tokens], args.output_dir / "wikitext2_test.pt")

    ptb = load_dataset("ptb_text_only", "penn_treebank", split="test", trust_remote_code=True)
    ptb_tokens = encode(tokenizer, "\n\n".join(row["sentence"] for row in ptb))
    torch.save([ptb_tokens], args.output_dir / "ptb_test.pt")

    def select_c4(split: str, count: int) -> list[torch.Tensor]:
        stream = load_dataset("allenai/c4", "en", split=split, streaming=True)
        stream = stream.shuffle(seed=args.seed, buffer_size=10_000)
        selected = []
        for row in stream:
            tokens = encode(tokenizer, row["text"])
            if tokens.numel() >= args.sequence_length:
                selected.append(tokens[:args.sequence_length])
            if len(selected) == count:
                break
        if len(selected) != count:
            raise RuntimeError(f"C4 {split} yielded only {len(selected)} usable documents")
        return selected

    torch.save(select_c4("validation", 256), args.output_dir / "c4_validation_256x2048.pt")
    torch.save(select_c4("train", 128), args.output_dir / "c4_calibration_128x2048.pt")
    metadata = {
        "seed": args.seed,
        "sequence_length": args.sequence_length,
        "wikitext2_tokens": wiki_tokens.numel(),
        "ptb_tokens": ptb_tokens.numel(),
        "c4_validation_documents": 256,
        "c4_calibration_documents": 128,
    }
    (args.output_dir / "corpora.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
