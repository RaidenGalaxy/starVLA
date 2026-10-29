from __future__ import annotations

import dataclasses
import json
import math
import statistics
import time
from collections import defaultdict

import numpy as np
import torch
from safetensors import safe_open

from experiments.ffn_quant_causal.adapters import load_model_bundle
from experiments.ffn_quant_mechanism.gdn import MEDIATOR_GROUPS
from experiments.infinitevl_quant.progress import atomic_json, file_sha256, json_digest
from experiments.infinitevl_quant.quantization import quantize_weight

from .config import DOSES, MODELS, PROJECTIONS, SEEDS
from .evaluation import Engine, path_difference
from .interventions import Intervention
from .numerics import decompose_ffn, geometry, quotient, scalar_quantize, token_groups, visual_token_ids
from .statistics import holm, nll, paired_effect


def engines(config, audit, progress, budget):
    for kind in MODELS:
        budget.check(stage=progress.root.name, boundary="before_model_load", variant=kind)
        bundle = load_model_bundle(kind, config_path(audit, kind))
        try:
            yield kind, Engine(bundle, config, progress, budget)
        finally:
            bundle.close()


def config_path(audit, kind):
    from pathlib import Path

    return Path(audit["model_paths"][kind])


def scalar_audit(config, audit, progress, budget):
    records, gate_stats = [], {}
    for kind in MODELS:
        source = config_path(audit, kind)
        weights = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
        gate_name = "model.layers.1.mlp.gate_proj.weight"
        with safe_open(source / weights[gate_name], framework="pt", device="cpu") as handle:
            gate_weight = handle.get_tensor(gate_name).float()
        groups = gate_weight.reshape(gate_weight.shape[0], -1, config.group_size)
        lower, upper = groups.amin(-1), groups.amax(-1)
        gate_stats[kind] = {
            "one_sided_group_fraction": float(((lower > 0) | (upper < 0)).float().mean()),
            "abs_max": float(gate_weight.abs().max()),
            "std": float(gate_weight.std()),
            "group_range_quantiles": torch.quantile(
                (upper - lower).flatten(), torch.tensor([0.0, 0.5, 0.9, 0.99, 1.0])
            ).tolist(),
            "weight_error": {},
        }
        for bits in (3, 4, 8):
            q = quantize_weight(gate_weight, bits=bits, group_size=config.group_size, zero_point=True)
            error = q.dequantized - gate_weight
            gate_stats[kind]["weight_error"][str(bits)] = {
                "nmse": q.nmse,
                "mean_error": float(error.mean()),
                "row_mean_error_quantiles": torch.quantile(
                    error.mean(-1), torch.tensor([0.0, 0.5, 0.9, 0.99, 1.0])
                ).tolist(),
            }
        del gate_weight, groups
        for layer in range(36):
            for projection in PROJECTIONS:
                name = f"model.layers.{layer}.mlp.{projection}_proj.weight"
                unit = {"model": kind, "tensor": name, "sampled_rows": "first_middle_last", "bits": [3, 4, 8]}
                identifier = f"{kind}:{name}"
                row = progress.completed("scalar_quantizer", identifier, unit)
                if row is None:
                    budget.check(stage="audit", boundary="before_scalar_tensor", sample_id=identifier)
                    with safe_open(source / weights[name], framework="pt", device="cpu") as handle:
                        tensor = handle.get_slice(name)
                        rows = [0, tensor.get_shape()[0] // 2, tensor.get_shape()[0] - 1]
                        sampled = torch.cat([tensor[i : i + 1, :] for i in rows]).float()
                    values = {}
                    for bits in (3, 4, 8):
                        expected, groups = scalar_quantize(sampled, bits=bits, group_size=config.group_size)
                        actual = quantize_weight(
                            sampled, bits=bits, group_size=config.group_size, zero_point=True
                        ).dequantized
                        maximum = float((expected - actual).abs().max())
                        tolerance = 1e-6 * max(1.0, float(sampled.abs().max()))
                        if maximum > tolerance:
                            raise ArithmeticError(f"scalar quantizer mismatch: {identifier} W{bits}: {maximum}")
                        values[str(bits)] = {
                            "max_abs_difference": maximum,
                            "tolerance": tolerance,
                            "mean_weight_error": float((actual - sampled).mean()),
                            "constant_groups": sum(g["constant"] for g in groups),
                            "groups": groups if layer == 1 and projection == "gate" else None,
                            "sampled_groups": len(groups),
                        }
                    row = {
                        "model_kind": kind,
                        "layer": layer,
                        "projection": projection,
                        "row_indices": rows,
                        "results": values,
                    }
                    progress.commit("scalar_quantizer", identifier, unit, row)
                    row = progress.completed("scalar_quantizer", identifier, unit)
                records.append(row)
    # Also cover tail, sign-only, constant and rounding-boundary groups independent of checkpoint contents.
    cases = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [-5.0, -4.0, -3.0, -2.0, -1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, -0.5, 0.0, 0.5, 1.0],
        ]
    )
    for bits in (3, 4, 8):
        expected, _ = scalar_quantize(cases, bits=bits, group_size=3)
        actual = quantize_weight(cases, bits=bits, group_size=3, zero_point=True).dequantized
        if not torch.equal(expected, actual):
            raise ArithmeticError("synthetic quantizer edge-case mismatch")
    return {
        "passed": True,
        "sampled_tensors": len(records),
        "coverage": "three complete rows per FFN tensor; all g128 groups in those rows",
        "independence": "scalar FP32 arithmetic and Python ties-to-even; no production QDQ calls in reference",
        "historical_code_changed": False,
        "layer1_gate_full_tensor": gate_stats,
        "results": records,
    }


def ratio(config, audit, splits, progress, budget):
    all_rows, historical, endpoints = {}, [], []
    for kind, engine in engines(config, audit, progress, budget):
        # Recompute the actual retained historical tensors, never call them full sequences.
        for path in sorted((config.parent_output / "trace/artifacts" / kind).rglob("*.pt")):
            engine.check("before_historical_tensor", sample_id=str(path))
            payload = torch.load(path, map_location="cpu", weights_only=True)
            layer = int(payload["layer"])
            unit = {"sha256": file_sha256(path), "model": kind, "bits": 4}
            identifier = str(path.relative_to(config.parent_output))
            row = progress.completed("historical_fp64", identifier, unit)
            if row is None:
                source = engine.bank.sources[layer]
                x = payload["ffn_input"].float()
                with torch.inference_mode():
                    f = source(x)
                    g = engine.bank.linear(layer, "gate", 4, x)
                    u = engine.bank.linear(layer, "up", 4, x)
                    fq = engine.bank.linear(layer, "down", 4, source.act_fn(g) * u)
                saved_f = payload["down_output"].float()
                parity = float((f - saved_f).double().square().mean().sqrt())
                signal = float(saved_f.double().square().mean().sqrt())
                if parity > 2e-5 * max(signal, 1e-12):
                    raise ArithmeticError(f"historical FFN replay mismatch {path}")
                geo = geometry(payload["residual"], saved_f, fq - saved_f)
                fp32_r = float(payload["residual"].float().square().mean().sqrt())
                fp32_f = float(saved_f.square().mean().sqrt())
                row = {
                    "model_kind": kind,
                    "sample_id": payload["sample_id"],
                    "layer": layer,
                    "sampling": "historical_uniform_at_most_256_tokens",
                    "geometry_fp64": geo,
                    "fp32_rho": quotient(fp32_f, fp32_r),
                    "rho_fp32_minus_fp64": quotient(fp32_f, fp32_r) - geo["rho"] if fp32_r else None,
                    "replay_relative_rms": quotient(parity, signal),
                    "tokens": int(x.numel() // x.shape[-1]),
                }
                progress.commit("historical_fp64", identifier, unit, row)
            historical.append(row)
        for trajectory, gamma in (("reference", 0.0), ("quantized", 1.0)):
            variant = f"{kind}__calibration32__{trajectory}"
            for sample in splits["qig"]:
                inputs = engine.inputs(sample)
                prompt, teacher, _labels, positions, digest = inputs
                unit = {"input": digest, "gamma": gamma, "bits": 4, "full_sequence": True}
                row = progress.completed(variant, sample.id, unit)
                if row is None:
                    engine.check("before_ratio_sample", variant, sample.id)
                    groups = token_groups(
                        teacher,
                        prompt["input_ids"].shape[-1],
                        positions,
                        visual_ids=visual_token_ids(engine.bundle.processor, engine.bundle.model),
                    )
                    observations = []

                    def observe(
                        layer, residual, x, reference, candidate, output, *, observations=observations, groups=groups
                    ):
                        observations.append(
                            {
                                "layer": layer,
                                "groups": {
                                    group: geometry(
                                        residual,
                                        reference["down_output"],
                                        candidate["down_output"] - reference["down_output"],
                                        mask,
                                    )
                                    for group, mask in groups.items()
                                },
                            }
                        )

                    start = time.perf_counter()
                    engine.forward(sample, teacher, Intervention(gamma=gamma), observer=observe)
                    if [v["layer"] for v in observations] != list(range(36)):
                        raise RuntimeError("ratio stage did not observe one aligned call per layer")
                    row = {
                        "model_kind": kind,
                        "dataset": sample.dataset,
                        "trajectory": trajectory,
                        "sequence_tokens": teacher["input_ids"].shape[-1],
                        "prompt_tokens": prompt["input_ids"].shape[-1],
                        "group_counts": {k: int(v.sum()) for k, v in groups.items()},
                        "groups_may_overlap": "answer_prediction includes the last prompt position",
                        "layers": observations,
                        "elapsed_seconds": time.perf_counter() - start,
                    }
                    progress.commit(variant, sample.id, unit, row)
                    print(f"[ratio/{variant}] {sample.id} seconds={row['elapsed_seconds']:.1f}", flush=True)
                    engine.check("after_ratio_sample", variant, sample.id)
            all_rows[(kind, trajectory)] = complete_rows(progress, variant, splits["qig"])
        # Real-model endpoint parity against both unwrapped source and existing FakeQuantLinear.
        from experiments.infinitevl_quant.config import SCHEMES
        from experiments.infinitevl_quant.quantization import quantized_linears

        sample = splits["smoke4"][0]
        inputs = engine.inputs(sample)
        teacher = inputs[1]
        unit = {"model": kind, "input": inputs[4], "endpoint": [0, 1]}
        row = progress.completed("endpoint_parity", kind, unit)
        if row is None:
            with torch.inference_mode():
                engine.check("before_unwrapped_parity", sample_id=sample.id)
                baseline = engine.bundle.model(**teacher, use_cache=False, return_dict=True).logits
                zero, _, _ = engine.forward(sample, teacher, Intervention(gamma=0))
                one, _, _ = engine.forward(sample, teacher, Intervention(gamma=1))
                with quantized_linears(engine.bundle.model, engine.bundle.ffn_targets, SCHEMES["w4a16"]):
                    engine.check("before_existing_quantizer_parity", sample_id=sample.id)
                    existing = engine.bundle.model(**teacher, use_cache=False, return_dict=True).logits
                differences = {
                    "gamma0_max_error": float((zero.logits - baseline).abs().max()),
                    "gamma1_max_error": float((one.logits - existing).abs().max()),
                }
                if any(v > 1e-5 for v in differences.values()):
                    raise ArithmeticError(f"shadow endpoint mismatch: {differences}")
                row = {"model_kind": kind, "sample_id": sample.id, **differences}
                progress.commit("endpoint_parity", kind, unit, row)
        endpoints.append(row)
    buckets = defaultdict(list)
    for (kind, trajectory), rows in all_rows.items():
        for row in rows:
            for layer in row["layers"]:
                for group, geo in layer["groups"].items():
                    if geo["tokens"]:
                        buckets[(kind, trajectory, layer["layer"], group)].append(geo)
    aggregated = []
    for (kind, trajectory, layer, group), values in sorted(buckets.items()):
        total_numel = sum(v["numel"] for v in values)
        pooled = {k: math.sqrt(sum(v["sum_squared"][k] for v in values) / total_numel) for k in values[0]["sum_squared"]}
        aggregated.append(
            {
                "model_kind": kind,
                "trajectory": trajectory,
                "layer": layer,
                "group": group,
                "pooled_rms": pooled,
                "pooled_eta_r": quotient(pooled["error"], pooled["residual"]),
                "pooled_rho": quotient(pooled["ffn"], pooled["residual"]),
                "sample_medians": {
                    k: statistics.median(v[k] for v in values if v[k] is not None)
                    if any(v[k] is not None for v in values)
                    else None
                    for k in ("rho", "eta_f", "eta_r", "eta_y")
                },
            }
        )
    scales = transport_scales(all_rows)
    atomic_json(progress.root / "transport_scales.json", scales)
    return {
        "ratio_definitions": {
            "rho": "RMS(f)/RMS(r)",
            "eta_f": "RMS(delta)/RMS(f)",
            "eta_r": "RMS(delta)/RMS(r)",
            "eta_y": "RMS(delta)/RMS(r+f)",
        },
        "aggregation": "energy-pooled RMS and sample-median ratios are separately labeled",
        "historical_fp64": historical,
        "endpoint_parity": endpoints,
        "aggregated": aggregated,
        "transport_scales": scales,
        "transport_scales_sha256": file_sha256(progress.root / "transport_scales.json"),
    }


def transport_scales(rows):
    metrics = {}
    for kind in MODELS:
        for layer in range(36):
            by_dataset = defaultdict(list)
            for row in rows[(kind, "reference")]:
                by_dataset[row["dataset"]].append(row["layers"][layer]["groups"]["all_valid"])
            if len(by_dataset) != 4:
                raise ValueError("transport calibration requires four data sources")
            for mode in ("absolute", "relative"):
                values = {
                    ds: statistics.median(v["rms"]["error"] if mode == "absolute" else v["eta_r"] for v in cells)
                    for ds, cells in by_dataset.items()
                }
                metrics[(kind, layer, mode)] = statistics.mean(values.values())
    result = {kind: {mode: {} for mode in ("absolute", "relative")} for kind in MODELS}
    for kind in MODELS:
        target = "qwen" if kind == "infinitevl" else "infinitevl"
        for mode in ("absolute", "relative"):
            for layer in range(36):
                gamma = quotient(metrics[(target, layer, mode)], metrics[(kind, layer, mode)])
                if gamma is None or not math.isfinite(gamma) or gamma <= 0:
                    raise ArithmeticError("invalid transport calibration scale")
                result[kind][mode][str(layer)] = gamma
    return {
        "gammas": result,
        "calibration_split": "qig32",
        "confirm_used": False,
        "aggregation": "within_source_median_then_four_source_macro_mean",
        "clipping": None,
    }


def complete_rows(progress, variant, samples):
    rows = progress.rows(variant)
    expected = {s.id: s.dataset for s in samples}
    if len(rows) != len(expected) or {r["id"]: r["dataset"] for r in rows} != expected:
        raise RuntimeError(f"incomplete or foreign sample set: {variant}")
    return rows


def run_matrix(engine, samples, split, variants, *, generate=False, observe=False):
    collected = {}
    # Finish whole conditions before starting the next, so budget boundaries never
    # spread an incomplete sample subset over every comparison in the matrix.
    for label, spec in variants.items():
        for sample in samples:
            inputs = engine.inputs(sample)
            observed = []
            valid_mask = inputs[1]["attention_mask"].bool()

            def collector(
                layer,
                residual,
                x,
                reference,
                candidate,
                output,
                *,
                observed=observed,
                mask=valid_mask,
            ):
                observed.append(
                    {
                        "layer": layer,
                        "geometry": geometry(
                            residual, reference["down_output"], output - reference["down_output"], mask
                        ),
                    }
                )

            # Observations are committed as their own bound unit, since a resumed row must not lose them.
            variant = f"{engine.bundle.kind}__{split}__{label}"
            unit = {"spec": spec.payload(), "input": inputs[4]}
            measurement = engine.progress.completed(variant + "__achieved", sample.id, unit) if observe else True
            if observe and measurement is None and engine.progress.sample_path(variant, sample.id).exists():
                engine.check("before_missing_observation", variant, sample.id)
                engine.forward(sample, inputs[1], spec, observer=collector)
            engine.evaluate(
                sample,
                inputs,
                split,
                label,
                spec,
                generate=generate,
                observer=collector if observe and measurement is None and not observed else None,
            )
            if observe and measurement is None:
                engine.progress.commit(
                    variant + "__achieved",
                    sample.id,
                    unit,
                    {"dataset": sample.dataset, "layers": observed, "model_kind": engine.bundle.kind},
                )
        collected[label] = complete_rows(engine.progress, f"{engine.bundle.kind}__{split}__{label}", samples)
        times = [r["elapsed_seconds"] for values in collected.values() for r in values]
        observed_median = statistics.median(times)
        pending = (len(variants) - len(collected)) * len(samples)
        atomic_json(
            engine.progress.root / f"{engine.bundle.kind}__{split}__schedule.json",
            {
                "completed_conditions": list(collected),
                "remaining_conditions": [k for k in variants if k not in collected],
                "observed_cell_seconds_median": observed_median,
                "estimated_remaining_seconds": pending * observed_median,
                "budget_remaining_seconds": engine.budget.remaining_seconds(),
                "estimate_limit": "observed runtime including warmup; GDN and generation may differ",
            },
        )
    return collected


def local(config, audit, splits, progress, budget):
    result = {}
    for kind, engine in engines(config, audit, progress, budget):
        for sample in splits["diagnostic8"]:
            inputs = engine.inputs(sample)
            variant = f"{kind}__diagnostic8__decomposition"
            unit = {"input": inputs[4], "bits": [3, 4], "chunk_tokens": config.chunk_tokens}
            if progress.completed(variant, sample.id, unit) is not None:
                continue
            observations = []

            def observe(
                layer,
                residual,
                x,
                reference,
                candidate,
                output,
                *,
                engine=engine,
                variant=variant,
                sample_id=sample.id,
                observations=observations,
            ):
                engine.check("before_decomposition_layer", variant, f"{sample_id}:{layer}")
                observations.append(
                    {
                        "layer": layer,
                        "bits": [
                            decompose_ffn(engine.bank.sources[layer], x, engine.bank, layer, bits, config.chunk_tokens)
                            for bits in (4, 3)
                        ],
                    }
                )

            start = time.perf_counter()
            engine.forward(sample, inputs[1], Intervention(bits=None), observer=observe)
            progress.commit(
                variant,
                sample.id,
                unit,
                {"dataset": sample.dataset, "layers": observations, "elapsed_seconds": time.perf_counter() - start},
            )
            print(f"[local/{variant}] {sample.id} seconds={time.perf_counter() - start:.1f}", flush=True)
        decomp = complete_rows(progress, f"{kind}__diagnostic8__decomposition", splits["diagnostic8"])
        variants = {
            "B": Intervention(bits=None),
            "W": Intervention(),
            **{f"restore_layer_{i:02d}": Intervention(layer_bits={i: None}) for i in range(36)},
        }
        rows = run_matrix(engine, splits["diagnostic8"], "diagnostic8", variants)
        ranked = sorted(range(36), key=lambda i: (nll(rows[f"restore_layer_{i:02d}"]), i))
        best = ranked[:3]
        channels = {}
        for layer in best:
            energy = torch.zeros(engine.bank.sources[layer].gate_proj.out_features, dtype=torch.float64)
            for sample_row in decomp:
                bit_row = next(v for v in sample_row["layers"][layer]["bits"] if v["bits"] == 4)
                energy += torch.tensor(bit_row["channel_error_energy"], dtype=torch.float64)
            top = torch.topk(energy, max(1, math.ceil(energy.numel() * 0.01)))
            fraction = quotient(float(top.values.sum()), float(energy.sum()))
            channels[str(layer)] = {
                "indices": top.indices.tolist(),
                "energy_fraction": fraction,
                "triggered": fraction is not None and fraction >= 0.25,
            }
        screen = {"B": Intervention(bits=None), "W": Intervention()}
        for index, layer in enumerate(best, 1):
            screen[f"restore_top{index}"] = Intervention(layer_bits={layer: None})
            screen[f"only_top{index}"] = Intervention(bits=None, layer_bits={layer: 4})
        screen["restore_top3_combined"] = Intervention(layer_bits={i: None for i in best})
        screen["only_top3_combined"] = Intervention(bits=None, layer_bits={i: 4 for i in best})
        screen_rows = run_matrix(engine, splits["screen"], "screen32", screen)
        w3 = {
            "B": Intervention(bits=None),
            "W3": Intervention(bits=3),
            "only_layer1_gate_w3": Intervention(bits=None, projection_bits={"1.gate": 3}),
            "w3_gate1_reference": Intervention(bits=3, projection_bits={"1.gate": None}),
            "w3_gate1_w4": Intervention(bits=3, projection_bits={"1.gate": 4}),
            "w3_gate1_w8": Intervention(bits=3, projection_bits={"1.gate": 8}),
        }
        w3_rows = run_matrix(engine, splits["screen"], "w3_screen32", w3)
        result[kind] = {
            "top3_layers": best,
            "ranking_diagnostic8": [
                {"layer": i, "recovery_absolute": nll(rows["W"]) - nll(rows[f"restore_layer_{i:02d}"])} for i in ranked
            ],
            "concentrated_channels": channels,
            "decomposition_aggregates": summarize_decompositions(decomp),
            "screen_specs": {k: v.payload() for k, v in screen.items()},
            "screen_nll": {k: nll(v) for k, v in screen_rows.items()},
            "w3_nll": {k: nll(v) for k, v in w3_rows.items()},
            "w3_effects": {
                label: paired_effect(w3_rows["B"], w3_rows["W3"], values, samples=config.bootstrap_samples)
                for label, values in w3_rows.items()
                if label not in ("B", "W3")
            },
        }
        atomic_json(progress.root / "selection_in_progress.json", result)
    return {"models": result, "selection_split": "diagnostic8_within_screen32", "confirm_used": False}


def summarize_decompositions(samples):
    buckets = defaultdict(list)
    for sample in samples:
        for layer in sample["layers"]:
            for row in layer["bits"]:
                buckets[(layer["layer"], row["bits"])].append(row)
    result = []
    for (layer, bits), rows in sorted(buckets.items()):
        combinations = {}
        for mask in rows[0]["combinations_gate_up_down"]:
            values = [r["combinations_gate_up_down"][mask] for r in rows]
            combinations[mask] = quotient(
                sum(v["error_energy"] for v in values), sum(v["reference_energy"] for v in values)
            )
        stage_nmse = {
            name: quotient(
                sum(r["stage_errors"][name]["error_energy"] for r in rows),
                sum(r["stage_errors"][name]["reference_energy"] for r in rows),
            )
            for name in rows[0]["stage_errors"]
        }
        result.append(
            {
                "layer": layer,
                "bits": bits,
                "samples": len(rows),
                "combination_output_nmse": combinations,
                "stage_nmse": stage_nmse,
                "summed_terms": {key: sum(r["terms"][key] for r in rows) for key in rows[0]["terms"]},
                "gate_sign_flip_fraction_sample_mean": statistics.mean(r["gate_sign_flip_fraction"] for r in rows),
            }
        )
    return result


def load_summary(config, stage):
    root = config.output_dir / stage
    if not (root / ".complete").is_file():
        raise RuntimeError(f"{stage} must be complete before proceeding")
    identity = json.loads((root / "identity.json").read_text())
    marker = json.loads((root / ".complete").read_text())
    if identity.get("config_digest") != config.digest() or any(marker.get(k) != v for k, v in identity.items()):
        raise RuntimeError(f"{stage} prerequisite identity mismatch")
    if marker.get("required") != ["summary.json", "summary.sha256.json"]:
        raise RuntimeError(f"{stage} prerequisite has invalid artifact declarations")
    expected = json.loads((root / "summary.sha256.json").read_text())
    manifest = {str(p.relative_to(root)): file_sha256(p) for p in sorted(root.glob("*/samples/*.json"))}
    if expected != {"summary.json": file_sha256(root / "summary.json"), "sample_rows": manifest}:
        raise RuntimeError(f"{stage} summary was modified")
    return json.loads((root / "summary.json").read_text())


def dose(config, audit, splits, progress, budget):
    locality = load_summary(config, "local")["models"]
    scales = load_summary(config, "ratio")["transport_scales"]
    result = {}
    for kind, engine in engines(config, audit, progress, budget):
        top = locality[kind]["top3_layers"][0]
        variants = {"B": Intervention(bits=None), "W": Intervention()}
        for mode in ("absolute", "relative"):
            variants["transport_" + mode] = Intervention(
                layer_gammas={int(k): v for k, v in scales["gammas"][kind][mode].items()}
            )
        variants.update({f"dose_{gamma:g}": Intervention(gamma=gamma) for gamma in DOSES if gamma not in (0, 1)})
        direction_variants = {"direction_real": Intervention(bits=None, layer_bits={top: 4})}
        for direction in ("permuted", "gaussian"):
            for seed in SEEDS:
                direction_variants[f"direction_{direction}_{seed}"] = Intervention(
                    bits=None, layer_bits={top: 4}, direction=direction, direction_layers=(top,), seed=seed
                )
        channels = locality[kind]["concentrated_channels"].get(str(top))
        infinite_top = locality["infinitevl"]["top3_layers"][0]
        infinite_trigger = locality["infinitevl"]["concentrated_channels"][str(infinite_top)]["triggered"]
        channel_variants = {}
        if channels and (channels["triggered"] or infinite_trigger):
            indices = tuple(channels["indices"])
            channel_variants["repair_channels"] = Intervention(channel_repairs={top: indices})
            for seed in SEEDS:
                generator = torch.Generator().manual_seed(seed)
                pool = [i for i in range(engine.bank.sources[top].gate_proj.out_features) if i not in indices]
                random = tuple(pool[i] for i in torch.randperm(len(pool), generator=generator)[: len(indices)].tolist())
                channel_variants[f"repair_random_channels_{seed}"] = Intervention(
                    channel_repairs={top: random}, seed=seed
                )
        rows = run_matrix(engine, splits["screen"], "screen32", variants, observe=True)
        omissions = {}
        # Only entire optional families can be omitted; all seeds of a direction
        # family stay together, and required amplitude/transport cells are preserved.
        for name, family in (("directions", direction_variants), ("concentrated_channels", channel_variants)):
            if not family:
                continue
            seconds = statistics.median(r["elapsed_seconds"] for values in rows.values() for r in values)
            estimate = seconds * len(family) * len(splits["screen"]) * 1.3
            reserve = seconds * len(variants) * len(splits["screen"]) * 1.3 if kind == "infinitevl" else 0
            if estimate + reserve >= budget.remaining_seconds():
                omissions[name] = {
                    "reason": "whole_family_deferred_to_preserve_required_cross_model_matrix",
                    "conditions": list(family),
                    "estimated_seconds": estimate,
                    "reserved_seconds": reserve,
                }
                continue
            rows.update(run_matrix(engine, splits["screen"], "screen32", family, observe=True))
            variants.update(family)
        result[kind] = {
            "specs": {k: v.payload() for k, v in variants.items()},
            "nll": {k: nll(v) for k, v in rows.items()},
            "effects": {
                k: paired_effect(rows["B"], rows["W"], v, samples=config.bootstrap_samples)
                for k, v in rows.items()
                if k not in ("B", "W")
            },
            "direction_layer": top,
            "budgeted_omissions": omissions,
            "concentrated_channel_trigger": "top1pct_product_error_energy_at_least_25pct_on_diagnostic8",
        }
    return {
        "models": result,
        "confirmation_used_for_scales_or_selection": False,
        "achieved_distribution_rows": "per-variant __achieved sample files; measured on live candidate trajectories",
        "dose_zero": "B",
        "dose_one": "W",
        "interpretation": "error scaling tests amplitude sufficiency, not residual-scale causation",
    }


def propagation(config, audit, splits, progress, budget):
    locality = load_summary(config, "local")["models"]
    results = {}
    for kind, engine in engines(config, audit, progress, budget):
        best = locality[kind]["top3_layers"]
        structures = {
            "single": Intervention(bits=None, layer_bits={best[0]: 4}),
            "combined": Intervention(bits=None, layer_bits={i: 4 for i in best}),
            "all": Intervention(),
        }
        for sample in splits["diagnostic8"]:
            inputs = engine.inputs(sample)
            pending = []
            for label, spec in structures.items():
                variant = f"{kind}__diagnostic8__path_{label}"
                unit = {"spec": spec.payload(), "epsilon": config.linear_epsilon, "input": inputs[4]}
                if progress.completed(variant, sample.id, unit) is None:
                    pending.append((label, spec, variant, unit))
            if not pending:
                continue
            baseline, ref_path, _ = engine.forward(sample, inputs[1], Intervention(bits=None), trace=True)
            prediction_rows = torch.tensor([p - 1 for p in inputs[3]])
            base_logits = baseline.logits[0, prediction_rows].detach().float()
            for _label, spec, variant, unit in pending:
                engine.check("before_propagation_cell", variant, sample.id)
                start = time.perf_counter()
                small, small_path, _ = engine.forward(
                    sample, inputs[1], dataclasses.replace(spec, gamma=config.linear_epsilon), trace=True
                )
                candidate, candidate_path, _ = engine.forward(sample, inputs[1], spec, trace=True)
                tangent = {
                    i: {k: (value - ref_path[i][k]) / config.linear_epsilon for k, value in row.items()}
                    for i, row in small_path.items()
                }
                candidate_logits = candidate.logits[0, prediction_rows].detach().float()
                linear_logits = (small.logits[0, prediction_rows].detach().float() - base_logits) / config.linear_epsilon
                actual = candidate_logits - base_logits
                from experiments.infinitevl_quant.evaluation import _compare_logits, _target_metrics

                target = _target_metrics(candidate.logits, inputs[2])
                row = {
                    "dataset": sample.dataset,
                    "path": path_difference(ref_path, candidate_path, linear=tangent),
                    "logit_actual_error_energy": float(actual.double().square().sum()),
                    "logit_nonlinear_remainder_energy": float((actual.double() - linear_logits.double()).square().sum()),
                    "comparison_to_bf16": _compare_logits(candidate_logits, base_logits),
                    "target_text": target,
                    "epsilon": config.linear_epsilon,
                    "elapsed_seconds": time.perf_counter() - start,
                }
                progress.commit(variant, sample.id, unit, row)
                engine.check("after_propagation_cell", variant, sample.id)
        result = {"path_variants": list(structures), "path_samples": len(splits["diagnostic8"])}
        if kind == "infinitevl":
            variants = {
                "B": Intervention(bits=None),
                "W": Intervention(),
                "recurrent_B": Intervention(bits=None, recurrent=True),
                "recurrent_W": Intervention(recurrent=True),
            }
            repairs = {
                "carry": ((), True),
                **{g: ((g,), False) for g in MEDIATOR_GROUPS},
                "joint": (tuple(MEDIATOR_GROUPS), True),
            }
            for name, (groups, carry) in repairs.items():
                variants[f"gdn_{name}"] = Intervention(gdn_groups=groups, carry=carry, recurrent=True)
                variants[f"identity_{name}"] = Intervention(bits=None, gdn_groups=groups, carry=carry, recurrent=True)
            # Parity first on all exploratory IDs; reject the intervention backend before using its effects.
            parity_variants = {k: v for k, v in variants.items() if k in ("B", "W", "recurrent_B", "recurrent_W")}
            parity = run_matrix(engine, splits["screen"], "screen32", parity_variants)
            gates = {}
            for name, original in (("recurrent_B", "B"), ("recurrent_W", "W")):
                left, right = {r["id"]: r for r in parity[name]}, {r["id"]: r for r in parity[original]}
                maximum = max(abs(left[i]["target_text"]["nll"] - right[i]["target_text"]["nll"]) for i in left)
                gates[name] = maximum
                if maximum > 1e-4:
                    raise ArithmeticError(f"recurrent/chunk NLL parity failed: {name} {maximum}")
            rows = run_matrix(engine, splits["screen"], "screen32", variants)
            for name in repairs:
                left = {r["id"]: r for r in rows[f"identity_{name}"]}
                right = {r["id"]: r for r in rows["recurrent_B"]}
                if max(abs(left[i]["target_text"]["nll"] - right[i]["target_text"]["nll"]) for i in left) > 1e-4:
                    raise ArithmeticError(f"reference self-repair is not an identity: {name}")
            result.update(
                gdn_backend_parity=gates,
                specs={k: v.payload() for k, v in variants.items()},
                effects={
                    k: paired_effect(rows["recurrent_B"], rows["recurrent_W"], values, samples=config.bootstrap_samples)
                    for k, values in rows.items()
                    if k.startswith("gdn_")
                },
                state_audits="raw per-sample states at real prefixes 64/128/256/512/final; no synthetic extension",
                oracle_generation=False,
            )
        results[kind] = result
    return {
        "models": results,
        "gdn_inclusion": "unconditional; not gated by local GDN/SWA gain",
        "confirmation_used": False,
    }


def screen_candidates(config, splits, local_summary, dose_summary, propagation_summary):
    result = {}
    for kind in MODELS:
        candidates = {}
        for stage, summary in (("local", local_summary), ("dose", dose_summary), ("propagation", propagation_summary)):
            model = summary["models"][kind]
            specs = model.get("screen_specs", model.get("specs", {}))
            for label, payload in specs.items():
                eligible = (
                    label.startswith("restore_")
                    or label in ("transport_absolute", "transport_relative", "repair_channels")
                    or label in ("dose_0.25", "dose_0.5", "dose_0.75")
                    or label.startswith("gdn_")
                )
                if not eligible:
                    continue
                p = config.output_dir / stage
                paths = sorted((p / f"{kind}__screen32__{label}" / "samples").glob("*.json"))
                rows = [json.loads(path.read_text()) for path in paths]
                if {r["id"] for r in rows} != {s.id for s in splits["screen"]} or len(rows) != 32:
                    raise RuntimeError("cannot select an incomplete screen condition")
                candidates[label] = {"stage": stage, "spec": payload, "nll": nll(rows)}
        result[kind] = candidates
    return result


def matched_control(spec, label, top_layers, seed=42):
    if label.startswith("restore_"):
        count = sum(v is None for v in spec.layer_bits.values())
        pool = [i for i in range(36) if i not in top_layers]
        generator = torch.Generator().manual_seed(seed)
        chosen = [pool[i] for i in torch.randperm(len(pool), generator=generator)[:count].tolist()]
        return Intervention(layer_bits={i: None for i in chosen}, seed=seed)
    if label == "repair_channels":
        repairs = {}
        for layer, channels in spec.channel_repairs.items():
            pool = [i for i in range(11008) if i not in channels]
            rng = np.random.default_rng(seed)
            repairs[layer] = tuple(map(int, rng.choice(pool, len(channels), replace=False)))
        return dataclasses.replace(spec, channel_repairs=repairs, seed=seed)
    if label.startswith("gdn_"):
        return Intervention(bits=None, gdn_groups=spec.gdn_groups, carry=spec.carry, recurrent=True)
    return dataclasses.replace(spec, direction="gaussian", seed=seed)


def lock_confirmation(config, audit, splits):
    local_summary, dose_summary, propagation_summary = (
        load_summary(config, s) for s in ("local", "dose", "propagation")
    )
    candidates = screen_candidates(config, splits, local_summary, dose_summary, propagation_summary)
    base_rows = [
        json.loads(p.read_text()) for p in (config.output_dir / "dose/infinitevl__screen32__W/samples").glob("*.json")
    ]
    w_nll = nll(base_rows)
    eligible = [label for label in candidates["infinitevl"] if label in candidates["qwen"] or label.startswith("gdn_")]
    ordered = sorted(eligible, key=lambda label: (candidates["infinitevl"][label]["nll"], label))
    selected = [label for label in ordered if candidates["infinitevl"][label]["nll"] < w_nll][:2]
    locked = {
        "format_version": 1,
        "selection_metric": "screen32 InfiniteVL token-weighted absolute NLL recovery",
        "selected": selected,
        "confirm_used_for_selection": False,
        "audit_digest": json_digest(audit),
        "historical_reuse": True,
        "models": {},
        "upstream_summaries": {
            s: file_sha256(config.output_dir / s / "summary.json") for s in ("ratio", "local", "dose", "propagation")
        },
    }
    for kind in MODELS:
        conditions = {}
        for label in selected:
            candidate = candidates[kind].get(label)
            if candidate is None:
                if not label.startswith("gdn_"):
                    raise RuntimeError(f"missing cross-model counterpart: {label}")
                continue
            spec = Intervention.from_payload(candidate["spec"])
            if label.startswith("transport_"):
                counterpart = "transport_absolute" if label.endswith("relative") else "transport_relative"
                control = Intervention.from_payload(candidates[kind][counterpart]["spec"])
            else:
                control = matched_control(spec, label, local_summary["models"][kind]["top3_layers"])
            conditions[label] = {
                "spec": spec.payload(),
                "control": control.payload(),
                "screen_nll": candidate["nll"],
                "generate": not spec.recurrent,
                "primary_direction": "degradation" if kind == "qwen" and label == "transport_relative" else "recovery",
            }
        locked["models"][kind] = conditions
    locked = json.loads(json.dumps(locked, ensure_ascii=False))
    path = config.output_dir / "confirm/locked_interventions.json"
    if path.exists() and json.loads(path.read_text()) != locked:
        raise RuntimeError("confirmation lock changed after selection")
    atomic_json(path, locked)
    return locked


def confirm(config, audit, splits, progress, budget):
    locked = lock_confirmation(config, audit, splits)
    all_effects, pvalues, recovery_pvalues = {}, {}, {}
    for kind, engine in engines(config, audit, progress, budget):
        # Run QA for B/W regardless of whether the two selected interventions are oracle-only.
        rows = run_matrix(
            engine, splits["confirm"], "confirm192", {"B": Intervention(bits=None), "W": Intervention()}, generate=True
        )
        effects = {}
        for label, entry in locked["models"][kind].items():
            spec, control = Intervention.from_payload(entry["spec"]), Intervention.from_payload(entry["control"])
            variants = {label: spec, label + "__control": control}
            candidate_rows = run_matrix(engine, splits["confirm"], "confirm192", variants, generate=entry["generate"])
            baseline, quantized = rows["B"], rows["W"]
            if spec.recurrent:
                recurrence = run_matrix(
                    engine,
                    splits["confirm"],
                    "confirm192",
                    {
                        "recurrent_B": Intervention(bits=None, recurrent=True),
                        "recurrent_W": Intervention(recurrent=True),
                    },
                )
                baseline, quantized = recurrence["recurrent_B"], recurrence["recurrent_W"]
            effect = paired_effect(baseline, quantized, candidate_rows[label], samples=config.bootstrap_samples)
            effect["control"] = paired_effect(
                baseline, quantized, candidate_rows[label + "__control"], samples=config.bootstrap_samples
            )
            if spec.recurrent:
                # This is a baseline self-repair control, not a competing quantized intervention.
                effect["control_role"] = "reference_self_repair_identity"
                identity = {r["id"]: r for r in candidate_rows[label + "__control"]}
                base = {r["id"]: r for r in baseline}
                if max(abs(identity[i]["target_text"]["nll"] - base[i]["target_text"]["nll"]) for i in base) > 1e-4:
                    raise ArithmeticError("confirm reference repair failed identity check")
            else:
                effect["versus_matched_control"] = paired_effect(
                    baseline,
                    candidate_rows[label + "__control"],
                    candidate_rows[label],
                    samples=config.bootstrap_samples,
                )
            reverse = entry["primary_direction"] == "degradation"
            effect["primary_direction"] = "degradation" if reverse else "recovery"
            effect["p_primary_one_sided"] = (
                effect["p_degradation_one_sided"] if reverse else effect["p_recovery_one_sided"]
            )
            pvalues[kind + ":" + label] = effect["p_primary_one_sided"]
            recovery_pvalues[kind + ":" + label] = effect["p_recovery_one_sided"]
            effects[label] = effect
        all_effects[kind] = effects
    adjusted = holm(pvalues)
    recovery_adjusted = holm(recovery_pvalues)
    for kind, effects in all_effects.items():
        for label, effect in effects.items():
            effect["p_primary_holm"] = adjusted[kind + ":" + label]
            effect["p_recovery_holm"] = recovery_adjusted[kind + ":" + label]
    # Paired cross-model differences use sample IDs, but never align token positions across tokenizers.
    cross_model = {}
    for label in locked["selected"]:
        if label in all_effects["infinitevl"] and label in all_effects["qwen"]:
            cross_model[label] = cross_model_effect(config, progress, label, splits["confirm"])
    return {
        "lock": locked,
        "effects": all_effects,
        "cross_model": cross_model,
        "mechanism_evidence": mechanism_evidence(all_effects, locked),
        "holm_family": list(pvalues),
        "selection_is_exploratory": True,
        "confirmation_history": "historically_reused_evaluation",
        "no_speed_or_memory_claim": True,
        "oracle_QA_claim": False,
    }


def mechanism_evidence(effects, locked):
    components, mixed_precision = {}, []
    for label, effect in effects.get("infinitevl", {}).items():
        ci = effect["ci95"]["recovery_absolute"]
        recovered = bool(ci and ci[0] > 0 and effect["p_primary_holm"] <= 0.05)
        control = effect.get("versus_matched_control", {}).get("ci95", {}).get("recovery_absolute")
        superior = bool(control and control[0] > 0)
        if label.startswith("gdn_"):
            interpretation = "downstream_mediation_supported" if recovered else "downstream_mediation_unresolved"
        elif label.startswith("restore_") or label == "repair_channels":
            interpretation = "targeted_localization_supported" if recovered and superior else "localization_unresolved"
        else:
            reverse = effects.get("qwen", {}).get(label)
            reverse_ci = reverse["ci95"]["recovery_absolute"] if reverse else None
            bidirectional = bool(reverse_ci and reverse_ci[1] < 0 and reverse["p_primary_holm"] <= 0.05)
            interpretation = (
                "relative_amplitude_component_supported"
                if label == "transport_relative" and recovered and bidirectional and superior
                else "amplitude_intervention_only"
            )
        components[label] = {
            "interpretation": interpretation,
            "recovery_confirmed": recovered,
            "superior_to_matched_control": superior,
            "explained_fraction": effect["point"]["explained_fraction"],
            "explained_fraction_ci95": effect["ci95"]["explained_fraction"],
        }
        if label.startswith("restore_") and recovered and superior:
            spec = Intervention.from_payload(locked["models"]["infinitevl"][label]["spec"])
            mixed_precision.append(
                {
                    "condition": label,
                    "reference_ffn_layers": sorted(k for k, v in spec.layer_bits.items() if v is None),
                    "quality_simulation_only": True,
                }
            )
    minimum = (
        min(mixed_precision, key=lambda c: (len(c["reference_ffn_layers"]), c["condition"])) if mixed_precision else None
    )
    return {
        "components": components,
        "minimum_among_confirmed_mixed_precision_candidates": minimum,
        "minimum_is_global_optimum": False,
        "limits": [
            "amplitude scaling does not identify residual scale as a root cause",
            "nonsignificant direction differences do not prove equivalence",
            "multiple explained fractions must not be added without interaction analysis",
            "historically reused confirmation is not fresh external validation",
        ],
    }


def cross_model_effect(config, progress, label, samples):
    by_model = {}
    for kind in MODELS:
        by_model[kind] = {
            condition: {r["id"]: r for r in complete_rows(progress, f"{kind}__confirm192__{condition}", samples)}
            for condition in ("B", "W", label)
        }
    ids = [s.id for s in samples]
    groups = defaultdict(list)
    for i, sample in enumerate(samples):
        groups[sample.dataset].append(i)
    rng = np.random.default_rng(42)
    boot = np.concatenate([rng.choice(g, (config.bootstrap_samples, len(g)), replace=True) for g in groups.values()], 1)

    def evaluate(index):
        values = {}
        for kind, conditions in by_model.items():
            for condition, rows in conditions.items():
                loss = np.array([rows[i]["target_text"]["total_nll"] for i in ids])
                tokens = np.array([rows[i]["target_text"]["tokens"] for i in ids])
                values[kind, condition] = loss[index].sum(-1) / tokens[index].sum(-1)
        before = (values["infinitevl", "W"] - values["infinitevl", "B"]) - (values["qwen", "W"] - values["qwen", "B"])
        after = (values["infinitevl", label] - values["infinitevl", "B"]) - (values["qwen", label] - values["qwen", "B"])
        log_before = np.log(values["infinitevl", "W"] / values["infinitevl", "B"]) - np.log(
            values["qwen", "W"] / values["qwen", "B"]
        )
        log_after = np.log(values["infinitevl", label] / values["infinitevl", "B"]) - np.log(
            values["qwen", label] / values["qwen", "B"]
        )
        return {
            "absolute_excess_gap_before": before,
            "absolute_excess_gap_after": after,
            "absolute_gap_reduction": before - after,
            "log_nll_ratio_gap_before": log_before,
            "log_nll_ratio_gap_after": log_after,
            "log_ratio_gap_reduction": log_before - log_after,
        }

    point, distribution = evaluate(np.arange(len(ids))), evaluate(boot)
    return {
        "point": {k: float(v) for k, v in point.items()},
        "ci95": {k: np.quantile(v, [0.025, 0.975]).tolist() for k, v in distribution.items()},
        "paired_samples": len(ids),
        "bootstrap_samples": config.bootstrap_samples,
        "interpretation": "baseline-adjusted descriptive contrast; not architecture identification",
    }
