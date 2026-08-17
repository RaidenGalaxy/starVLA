from __future__ import annotations

from collections.abc import Iterable
import random
import time

import torch

from .metrics import TokenNLL


def _build_windows(
    sequences: Iterable[torch.Tensor],
    sequence_length: int,
    max_windows: int | None,
) -> tuple[int, list[dict]]:
    materialized = list(sequences)
    windows = []
    for document_index, sequence in enumerate(materialized):
        sequence = sequence.reshape(-1)
        for start in range(0, sequence.numel() - 1, sequence_length):
            token_ids = sequence[start:start + sequence_length]
            if token_ids.numel() < 2:
                continue
            windows.append({
                "document_index": document_index,
                "window_index": len(windows),
                "start": start,
                "input_ids": token_ids,
            })
            if max_windows is not None and len(windows) >= max_windows:
                return len(materialized), windows
    return len(materialized), windows


def _evaluate_windows(
    model,
    windows: list[dict],
    batch_size: int,
    *,
    collect_token_nlls: bool = False,
) -> tuple[TokenNLL, list[dict]]:
    aggregate, records = TokenNLL(), []
    for offset in range(0, len(windows), batch_size):
        batch = windows[offset:offset + batch_size]
        maximum = max(item["input_ids"].numel() for item in batch)
        input_ids = torch.zeros((len(batch), maximum), dtype=batch[0]["input_ids"].dtype)
        attention_mask = torch.zeros_like(input_ids)
        labels = torch.full_like(input_ids, -100)
        lengths = []
        for row, item in enumerate(batch):
            length = item["input_ids"].numel()
            lengths.append(length)
            input_ids[row, :length] = item["input_ids"]
            attention_mask[row, :length] = 1
            labels[row, :length] = item["input_ids"]
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        aggregate.update(output.logits, labels)
        for row, (item, length) in enumerate(zip(batch, lengths)):
            metric = TokenNLL()
            metric.update(output.logits[row:row + 1, :length], labels[row:row + 1, :length])
            record = {
                key: value for key, value in item.items() if key != "input_ids"
            } | metric.to_dict()
            if collect_token_nlls:
                token_losses = torch.nn.functional.cross_entropy(
                    output.logits[row, :length - 1].float(),
                    labels[row, 1:length],
                    reduction="none",
                )
                record["_token_nlls"] = token_losses.cpu()
            records.append(record)
    return aggregate, records


@torch.inference_mode()
def evaluate_token_sequences_ppl(
    model,
    sequences: Iterable[torch.Tensor],
    *,
    sequence_length: int = 2048,
    batch_size: int = 4,
    max_windows: int | None = None,
    validate_batch: bool = True,
    batch_tolerance: float = 1e-4,
) -> dict:
    """Token-weighted PPL with no artificial cross-document transitions."""
    documents, windows = _build_windows(sequences, sequence_length, max_windows)
    if not windows:
        return TokenNLL().to_dict() | {
            "documents": documents,
            "windows": 0,
            "sequence_length": sequence_length,
            "batch_size": batch_size,
            "window_records": [],
        }

    started = time.perf_counter()
    actual_batch_size = max(1, batch_size)
    while True:
        try:
            metric, records = _evaluate_windows(model, windows, actual_batch_size)
            break
        except (RuntimeError, MemoryError) as error:
            message = str(error).lower()
            if actual_batch_size == 1 or (
                "memory" not in message and "alloc" not in message
            ):
                raise
            actual_batch_size = max(1, actual_batch_size // 2)

    validation_error = 0.0
    if validate_batch and actual_batch_size > 1:
        count = min(actual_batch_size, len(windows))
        _batch_metric, batched = _evaluate_windows(
            model, windows[:count], count, collect_token_nlls=True,
        )
        _single_metric, singles = _evaluate_windows(
            model, windows[:count], 1, collect_token_nlls=True,
        )
        validation_error = max(
            float((batched[index]["_token_nlls"] - singles[index]["_token_nlls"]).abs().max())
            for index in range(count)
        )
        if validation_error > batch_tolerance:
            raise ValueError(
                f"batch per-token NLL validation failed: {validation_error} > {batch_tolerance}"
            )

    return metric.to_dict() | {
        "documents": documents,
        "windows": len(windows),
        "sequence_length": sequence_length,
        "batch_size": actual_batch_size,
        "batch_validation_max_abs_token_nll_diff": validation_error,
        "window_records": records,
        "elapsed_seconds": time.perf_counter() - started,
    }


def paired_window_delta(
    candidate: dict,
    reference: dict,
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 42,
) -> dict:
    """Paired window bootstrap of token-weighted candidate-minus-reference NLL."""
    reference_rows = {
        (row["document_index"], row["start"], row["tokens"]): row
        for row in reference["window_records"]
    }
    pairs = []
    for row in candidate["window_records"]:
        key = (row["document_index"], row["start"], row["tokens"])
        if key not in reference_rows:
            raise ValueError(f"candidate/reference corpus window mismatch: {key}")
        reference_row = reference_rows[key]
        pairs.append((
            float(row["total_nll"]),
            float(reference_row["total_nll"]),
            int(row["tokens"]),
        ))
    if len(pairs) != len(reference_rows):
        raise ValueError("candidate/reference corpus window counts differ")

    def statistic(indices):
        candidate_nll = sum(pairs[index][0] for index in indices)
        reference_nll = sum(pairs[index][1] for index in indices)
        tokens = sum(pairs[index][2] for index in indices)
        return candidate_nll / tokens - reference_nll / tokens

    observed = statistic(range(len(pairs)))
    rng = random.Random(seed)
    estimates = []
    for _ in range(bootstrap_samples):
        indices = [rng.randrange(len(pairs)) for _ in pairs]
        estimates.append(statistic(indices))
    estimates.sort()
    lower = estimates[int(0.025 * bootstrap_samples)]
    upper = estimates[min(int(0.975 * bootstrap_samples), bootstrap_samples - 1)]
    conclusion = "improved" if upper < 0 else "regressed" if lower > 0 else "indistinguishable"
    return {
        "paired_windows": len(pairs),
        "token_weighted_mean_nll_delta": observed,
        "bootstrap_95_ci": [lower, upper],
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "conclusion": conclusion,
    }
