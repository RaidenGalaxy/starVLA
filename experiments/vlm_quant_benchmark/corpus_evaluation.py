from __future__ import annotations

from collections.abc import Iterable

import torch

from .metrics import TokenNLL


@torch.inference_mode()
def evaluate_token_sequences_ppl(model, sequences: Iterable[torch.Tensor], *,
                                 sequence_length: int = 2048) -> dict:
    """Paper-style PPL without introducing artificial cross-document transitions."""
    metric, windows, documents = TokenNLL(), 0, 0
    for sequence in sequences:
        sequence = sequence.reshape(-1)
        documents += 1
        for start in range(0, sequence.numel() - 1, sequence_length):
            window = sequence[start:start + sequence_length].unsqueeze(0)
            if window.shape[-1] < 2:
                continue
            output = model(input_ids=window, use_cache=False, return_dict=True)
            metric.update(output.logits, window)
            windows += 1
    return metric.to_dict() | {
        "documents": documents,
        "windows": windows,
        "sequence_length": sequence_length,
    }
