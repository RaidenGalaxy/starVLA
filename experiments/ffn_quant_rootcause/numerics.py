from __future__ import annotations

import math
from itertools import product

import numpy as np
import torch
import torch.nn.functional as F


def finite(value: torch.Tensor) -> torch.Tensor:
    value = value.detach()
    if not torch.isfinite(value).all():
        raise FloatingPointError("diagnostic tensor contains NaN or Inf")
    return value


def quotient(numerator: float, denominator: float):
    return numerator / denominator if denominator != 0 else None


def quantiles(value: torch.Tensor):
    flat = finite(value).double().flatten()
    if flat.numel() == 0:
        return None
    return {str(p): float(torch.quantile(flat, p)) for p in (0.0, 0.5, 0.9, 0.99, 0.999, 1.0)}


def geometry(residual, reference, error, mask=None):
    """FP64 sums and token ratios; no epsilon can disguise a zero denominator."""
    if residual.shape != reference.shape or reference.shape != error.shape:
        raise ValueError("geometry tensors must have identical shapes")
    r, f, d = (finite(t).double().reshape(-1, t.shape[-1]) for t in (residual, reference, error))
    if mask is not None:
        if mask.numel() != r.shape[0]:
            raise ValueError("token mask does not align with the tensor")
        mask = mask.flatten().bool().to(r.device)
        r, f, d = r[mask], f[mask], d[mask]
    if r.numel() == 0:
        return {"tokens": 0, "numel": 0, "status": "empty_group"}
    powers = {
        "residual": r.square().sum(-1),
        "ffn": f.square().sum(-1),
        "error": d.square().sum(-1),
        "merged": (r + f).square().sum(-1),
    }
    rms = {k: math.sqrt(float(v.sum()) / r.numel()) for k, v in powers.items()}
    pairs = {
        "rho": ("ffn", "residual"),
        "eta_f": ("error", "ffn"),
        "eta_r": ("error", "residual"),
        "eta_y": ("error", "merged"),
    }
    ratios, tokens = {}, {}
    for name, (n, den) in pairs.items():
        ratios[name] = quotient(rms[n], rms[den])
        valid = powers[den] != 0
        tokens[name] = {
            "undefined_tokens": int((~valid).sum()),
            "quantiles": quantiles((powers[n][valid] / powers[den][valid]).sqrt()),
        }
    identity_error = None
    if all(ratios[k] is not None for k in ("eta_r", "eta_f", "rho")):
        identity_error = abs(ratios["eta_r"] - ratios["eta_f"] * ratios["rho"])
        if identity_error > 1e-10 * max(1.0, abs(ratios["eta_r"])):
            raise ArithmeticError("eta_r = eta_f * rho failed before aggregation")

    def cosine(a, b):
        return quotient(float((a * b).sum()), float(a.norm() * b.norm()))

    return {
        "tokens": r.shape[0],
        "numel": r.numel(),
        "rms": rms,
        "sum_squared": {k: float(v.sum()) for k, v in powers.items()},
        **ratios,
        "token_ratios": tokens,
        "identity_absolute_error": identity_error,
        "ffn_residual_cosine": cosine(f, r),
        "error_residual_cosine": cosine(d, r),
        "error_ffn_nmse": quotient(float(powers["error"].sum()), float(powers["ffn"].sum())),
    }


def token_groups(teacher, prompt_length, positions, *, visual_ids):
    ids = teacher["input_ids"]
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError("the protocol processes one unpadded sample at a time")
    valid = teacher.get("attention_mask", torch.ones_like(ids)).bool()
    if valid.shape != ids.shape or not 0 < prompt_length <= ids.shape[1]:
        raise ValueError("invalid attention mask or prompt length")
    visual = torch.zeros_like(valid)
    for token_id in visual_ids:
        visual |= ids.eq(token_id)
    visual &= valid
    prompt = torch.arange(ids.shape[1], device=ids.device)[None] < prompt_length
    answer = torch.zeros_like(valid)
    for position in positions:
        if not 1 <= position < ids.shape[1] or not valid[0, position] or not valid[0, position - 1]:
            raise ValueError("invalid answer prediction position")
        answer[0, position - 1] = True
    return {"all_valid": valid, "visual": visual, "text_prompt": prompt & valid & ~visual, "answer_prediction": answer}


def visual_token_ids(processor, model):
    ids = set()
    vocab = processor.tokenizer.get_vocab()
    for name in ("<|image_pad|>", "<|video_pad|>"):
        if name in vocab:
            ids.add(int(vocab[name]))
    for name in ("image_token_id", "video_token_id"):
        value = getattr(model.config, name, None)
        if value is not None:
            ids.add(int(value))
    if not ids:
        raise ValueError("processor/model has no verified visual token IDs")
    return ids


def scalar_quantize(weight, *, bits, group_size):
    """Independent scalar affine QDQ with explicit FP32 arithmetic/ties-to-even."""
    raw = finite(weight).float().cpu().numpy()
    if raw.ndim != 2 or bits not in (2, 3, 4, 8) or group_size <= 0:
        raise ValueError("invalid scalar QDQ input")
    restored = np.empty_like(raw)
    max_code = 2**bits - 1
    records = []
    for row in range(raw.shape[0]):
        for start in range(0, raw.shape[1], group_size):
            values = raw[row, start : start + group_size]
            lo, hi = min(map(float, values)), max(map(float, values))
            if lo == hi:
                restored[row, start : start + len(values)] = values
                records.append({"row": row, "start": start, "constant": True})
                continue
            scale = np.float32(np.float32(hi - lo) / np.float32(max_code))
            if scale <= 0:
                scale = np.float32(1)
            zero = min(max(round(float(np.float32(-np.float32(lo) / scale))), 0), max_code)
            clipped = 0
            for column, value in enumerate(values):
                code = round(float(np.float32(np.float32(value / scale) + np.float32(zero))))
                clipped += int(code < 0 or code > max_code)
                code = min(max(code, 0), max_code)
                restored[row, start + column] = np.float32(np.float32(code - zero) * scale)
            records.append(
                {
                    "row": row,
                    "start": start,
                    "constant": False,
                    "scale": float(scale),
                    "zero_point": zero,
                    "saturated": clipped,
                    "all_positive": lo > 0,
                    "all_negative": hi < 0,
                }
            )
    return torch.from_numpy(restored), records


def matching_noise(error, *, direction, seed):
    finite(error)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if direction == "real":
        return error
    if direction == "permuted":
        return error[..., torch.randperm(error.shape[-1], generator=generator).to(error.device)]
    if direction != "gaussian":
        raise ValueError(f"unknown error direction {direction}")
    noise = torch.randn(error.shape, generator=generator, dtype=torch.float32).to(error.device)
    norm = finite(error).double().square().sum(-1, keepdim=True).sqrt()
    noise = noise.double()
    return (noise * norm / noise.square().sum(-1, keepdim=True).sqrt()).to(error.dtype)


@torch.no_grad()
def decompose_ffn(source, value, bank, layer, bits, chunk_tokens=128):
    """All eight projection combinations, exact SwiGLU cross terms and channel energy."""
    x = finite(value).float().reshape(-1, value.shape[-1])
    combinations = {
        "".join(map(str, mask)): {"error_energy": 0.0, "reference_energy": 0.0} for mask in product((0, 1), repeat=3)
    }
    terms = {
        k: 0.0
        for k in (
            "gate",
            "up",
            "interaction",
            "gate_up_cross",
            "gate_interaction_cross",
            "up_interaction_cross",
            "product_error",
            "reconstruction_error",
            "down_input",
            "down_weight",
            "down_interaction",
            "down_reconstruction_error",
            "linear_gate_remainder",
            "negative_gate_energy",
            "negative_gate_leakage_energy",
        )
    }
    channel_energy = torch.zeros(source.gate_proj.out_features, dtype=torch.float64)
    stage_errors = {
        k: {"error_energy": 0.0, "reference_energy": 0.0, "numel": 0} for k in ("gate", "up", "silu", "product", "down")
    }
    signs = negative = count = 0
    negative_to_positive = positive_to_negative = 0
    gate_reference_sum = gate_quantized_sum = 0.0
    down_reference_energy = product_reference_energy = 0.0
    for chunk in x.split(chunk_tokens):
        ref = {p: getattr(source, p + "_proj")(chunk) for p in ("gate", "up")}
        q = {p: bank.linear(layer, p, bits, chunk) for p in ("gate", "up")}
        a, aq = source.act_fn(ref["gate"]), source.act_fn(q["gate"])
        m, mq = a * ref["up"], aq * q["up"]
        f = source.down_proj(m)
        fq = bank.linear(layer, "down", bits, mq)
        finite(fq)
        for name, left, right in (
            ("gate", ref["gate"], q["gate"]),
            ("up", ref["up"], q["up"]),
            ("silu", a, aq),
            ("product", m, mq),
            ("down", f, fq),
        ):
            stage_errors[name]["error_energy"] += float((right.double() - left.double()).square().sum())
            stage_errors[name]["reference_energy"] += float(left.double().square().sum())
            stage_errors[name]["numel"] += left.numel()
        for mask in product((0, 1), repeat=3):
            gate, up = q["gate"] if mask[0] else ref["gate"], q["up"] if mask[1] else ref["up"]
            intermediate = source.act_fn(gate) * up
            out = bank.linear(layer, "down", bits, intermediate) if mask[2] else source.down_proj(intermediate)
            row = combinations["".join(map(str, mask))]
            row["error_energy"] += float((finite(out).double() - f.double()).square().sum())
            row["reference_energy"] += float(f.double().square().sum())
        da, du = aq.double() - a.double(), q["up"].double() - ref["up"].double()
        t = (da * ref["up"].double(), a.double() * du, da * du)
        dm = mq.double() - m.double()
        for name, v in zip(("gate", "up", "interaction"), t, strict=True):
            terms[name] += float(v.square().sum())
        for name, i, j in (("gate_up_cross", 0, 1), ("gate_interaction_cross", 0, 2), ("up_interaction_cross", 1, 2)):
            terms[name] += float(2 * (t[i] * t[j]).sum())
        terms["product_error"] += float(dm.square().sum())
        terms["reconstruction_error"] += float((dm - sum(t)).square().sum())
        channel_energy += dm.square().sum(0).cpu()
        # Linear compute stays FP32 like the actual model; reductions remain FP64.
        wd = source.down_proj.weight.float()
        dw = bank.weight(layer, "down", bits).float() - wd
        dt = tuple(t.double() for t in (F.linear(dm.float(), wd), F.linear(m, dw), F.linear(dm.float(), dw)))
        for name, v in zip(("down_input", "down_weight", "down_interaction"), dt, strict=True):
            terms[name] += float(v.square().sum())
        terms["down_reconstruction_error"] += float((fq.double() - f.double() - sum(dt)).square().sum())
        g = ref["gate"].double()
        sigmoid = torch.sigmoid(g)
        derivative = sigmoid * (1 + g * (1 - sigmoid))
        terms["linear_gate_remainder"] += float((da - derivative * (q["gate"].double() - g)).square().sum())
        neg = g < 0
        terms["negative_gate_energy"] += float(a.double()[neg].square().sum())
        terms["negative_gate_leakage_energy"] += float(da[neg].square().sum())
        negative += int(neg.sum())
        signs += int((neg != (q["gate"] < 0)).sum())
        negative_to_positive += int((neg & (q["gate"] >= 0)).sum())
        positive_to_negative += int((~neg & (q["gate"] < 0)).sum())
        gate_reference_sum += float(g.sum())
        gate_quantized_sum += float(q["gate"].double().sum())
        count += g.numel()
        down_reference_energy += float(f.double().square().sum())
        product_reference_energy += float(m.double().square().sum())
    for row in combinations.values():
        row["nmse"] = quotient(row["error_energy"], row["reference_energy"])
    top_count = max(1, math.ceil(channel_energy.numel() * 0.01))
    top = torch.topk(channel_energy, top_count)
    for row in stage_errors.values():
        row["nmse"] = quotient(row["error_energy"], row["reference_energy"])
        row["error_rms"] = math.sqrt(row["error_energy"] / row["numel"])
    if terms["reconstruction_error"] > max(1e-18, terms["product_error"] * 1e-8):
        raise ArithmeticError("SwiGLU decomposition does not reconstruct the product error")
    if terms["down_reconstruction_error"] > max(1e-16, stage_errors["down"]["error_energy"] * 1e-5):
        raise ArithmeticError("down-projection decomposition exceeds FP32 roundoff tolerance")
    return {
        "bits": bits,
        "combinations_gate_up_down": combinations,
        "terms": terms,
        "product_reference_energy": product_reference_energy,
        "down_reference_energy": down_reference_energy,
        "gate_sign_flip_fraction": quotient(signs, count),
        "negative_gate_fraction": quotient(negative, count),
        "negative_to_positive_fraction": quotient(negative_to_positive, count),
        "positive_to_negative_fraction": quotient(positive_to_negative, count),
        "gate_reference_mean": quotient(gate_reference_sum, count),
        "gate_quantized_mean": quotient(gate_quantized_sum, count),
        "stage_errors": stage_errors,
        "channel_error_energy": channel_energy.tolist(),
        "top_1pct_channels": top.indices.tolist(),
        "top_1pct_energy_fraction": quotient(float(top.values.sum()), float(channel_energy.sum())),
    }
