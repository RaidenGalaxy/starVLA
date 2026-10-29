from __future__ import annotations

import contextlib
import dataclasses
import math
import time

import torch

from experiments.ffn_quant_mechanism.gdn import capture_gdn_reference, intervene_gdn
from experiments.infinitevl_quant.evaluation import _compare_logits, _target_metrics, _teacher_inputs
from experiments.infinitevl_quant.inventory import decoder_layers
from experiments.infinitevl_quant.progress import file_sha256, safe_id
from experiments.infinitevl_quant_ffn_bf16_4x64.scoring import SCORER_ID, score_prediction

from .evidence import atomic_torch, input_digest
from .interventions import Intervention, QuantBank, capture_path, intervene
from .numerics import finite, geometry


class Engine:
    def __init__(self, bundle, config, progress, budget):
        self.bundle, self.config, self.progress, self.budget = bundle, config, progress, budget
        if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in bundle.model.parameters()):
            raise ValueError("the reference protocol requires FP32 CPU parameters")
        self.layers = list(decoder_layers(bundle.model))
        self.bank = QuantBank(self.layers, config.group_size)
        self.input_cache = {}

    def inputs(self, sample):
        if sample.id in self.input_cache:
            return self.input_cache[sample.id]
        prompt, teacher, labels, positions = _teacher_inputs(
            self.bundle.processor, sample, self.config.min_pixels, self.config.max_pixels
        )
        value = (
            prompt,
            teacher,
            labels,
            positions,
            input_digest({**teacher, "labels": labels, "prompt_ids": prompt["input_ids"]}),
        )
        self.input_cache[sample.id] = value
        return value

    def check(self, boundary, variant=None, sample_id=None):
        self.budget.check(stage=self.progress.root.name, boundary=boundary, variant=variant, sample_id=sample_id)

    def reference_path(self, split, sample):
        return (
            self.config.output_dir
            / "references"
            / self.progress.root.name
            / self.bundle.kind
            / split
            / f"{safe_id(sample.id)}.pt"
        )

    def forward(self, sample, teacher, spec, *, trace=False, observer=None):
        reference = None
        if spec.recurrent or spec.carry or spec.gdn_groups:
            if self.bundle.kind != "infinitevl":
                raise ValueError("GDN mediation is InfiniteVL-only")
            self.check("before_gdn_reference", sample_id=sample.id)
            with (
                torch.inference_mode(),
                intervene(self.layers, self.bank, Intervention(bits=None), sample_id=sample.id),
                capture_gdn_reference(self.bundle.model) as ref,
            ):
                self.bundle.model(**teacher, use_cache=False, return_dict=True)
            reference = ref
        path = None
        with contextlib.ExitStack() as stack:
            stack.enter_context(torch.inference_mode())
            stack.enter_context(intervene(self.layers, self.bank, spec, observer=observer, sample_id=sample.id))
            if trace:
                path = stack.enter_context(capture_path(self.layers))
            gdn = (
                stack.enter_context(
                    intervene_gdn(self.bundle.model, reference, repair_groups=spec.gdn_groups, carry_clamp=spec.carry)
                )
                if reference is not None
                else None
            )
            self.check("before_model_forward", sample_id=sample.id)
            output = self.bundle.model(**teacher, use_cache=False, return_dict=True)
        return output, path, gdn.state_audits if gdn else None

    def evaluate(self, sample, inputs, split, label, spec, *, generate=False, observer=None):
        prompt, teacher, labels, positions, digest = inputs
        variant = f"{self.bundle.kind}__{split}__{label}"
        unit = {"spec": spec.payload(), "generate": generate, "input_digest": digest}
        previous = self.progress.completed(variant, sample.id, unit)
        reference_path = self.reference_path(split, sample)
        if previous is not None and (label != "B" or reference_path.is_file()):
            if label == "B" and previous.get("reference_sha256") != file_sha256(reference_path):
                raise RuntimeError("reference logits checksum mismatch")
            return previous
        self.check("before_cell", variant, sample.id)
        started = time.perf_counter()
        output, _path, state_audits = self.forward(sample, teacher, spec, observer=observer)
        target = _target_metrics(output.logits, labels)
        if not all(math.isfinite(target[k]) for k in ("nll", "total_nll")):
            raise FloatingPointError("nonfinite target loss")
        rows = torch.tensor([p - 1 for p in positions], dtype=torch.long)
        logits = finite(output.logits[0, rows]).detach().float().cpu()
        teacher_seconds = time.perf_counter() - started
        reference_sha = None
        if label == "B":
            atomic_torch(
                reference_path,
                {
                    "format_version": 1,
                    "audit_digest": self.progress.inputs["audit"],
                    "input_digest": digest,
                    "id": sample.id,
                    "logits": logits,
                },
            )
            reference_sha = file_sha256(reference_path)
            comparison = {}
        else:
            reference = torch.load(reference_path, map_location="cpu", weights_only=True)
            if (
                reference.get("audit_digest") != self.progress.inputs["audit"]
                or reference.get("input_digest") != digest
                or reference.get("id") != sample.id
            ):
                raise RuntimeError("reference logits identity mismatch")
            comparison = _compare_logits(logits, reference["logits"])
        shifted_targets = labels[0, positions].long()
        target_log_prob = torch.log_softmax(logits, -1).gather(-1, shifted_targets[:, None]).squeeze(-1)
        top = logits.topk(2, dim=-1).values
        competitors = logits.clone()
        competitors.scatter_(-1, shifted_targets[:, None], float("-inf"))
        target_margin = logits.gather(-1, shifted_targets[:, None]).squeeze(-1) - competitors.max(-1).values
        backend_parity = None
        if spec.recurrent and not spec.carry and not spec.gdn_groups:
            native, _, _ = self.forward(sample, teacher, dataclasses.replace(spec, recurrent=False))
            native_logits = native.logits[0, rows].detach().float().cpu()
            delta = logits.double() - native_logits.double()
            native_norm = float(native_logits.double().norm())
            backend_parity = {
                "max_abs": float(delta.abs().max()),
                "relative_rms": float(delta.norm()) / native_norm if native_norm else None,
                "target_nll_absolute_difference": abs(target["nll"] - _target_metrics(native.logits, labels)["nll"]),
            }
            if (
                backend_parity["max_abs"] > 1e-3
                or (backend_parity["relative_rms"] or 0) > 2e-5
                or backend_parity["target_nll_absolute_difference"] > 1e-4
            ):
                raise ArithmeticError(f"recurrent/chunk backend parity failed: {backend_parity}")
        prediction, scoring, generation_seconds, generated_count = "", None, 0.0, 0
        if generate:
            if spec.recurrent or spec.carry or spec.gdn_groups:
                raise ValueError("oracle GDN experiments are teacher-forced only")
            self.check("before_generation", variant, sample.id)
            start = time.perf_counter()
            with torch.inference_mode(), intervene(self.layers, self.bank, spec, sample_id=sample.id):
                generated = self.bundle.model.generate(
                    **prompt, do_sample=False, max_new_tokens=self.config.max_new_tokens, use_cache=True
                )
            generation_seconds = time.perf_counter() - start
            new_ids = generated[:, prompt["input_ids"].shape[-1] :]
            generated_count = int(new_ids.shape[-1])
            prediction = self.bundle.processor.batch_decode(
                new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
            scored = score_prediction(sample, prediction)
            scoring = {"scorer": SCORER_ID, "rule": scored.rule, "extracted": scored.extracted}
            score = float(scored.score)
        else:
            score = math.nan
        row = {
            "dataset": sample.dataset,
            "model_kind": self.bundle.kind,
            "split": split,
            "intervention": spec.payload(),
            "input_digest": digest,
            "target_text": target,
            "score": score,
            "scoring": scoring,
            "prediction": prediction,
            "comparison_to_bf16": comparison,
            "mean_target_log_prob": float(target_log_prob.mean()),
            "mean_top1_top2_margin": float((top[:, 0] - top[:, 1]).mean()),
            "mean_target_vs_best_other_margin": float(target_margin.mean()),
            "backend_parity": backend_parity,
            "teacher_seconds": teacher_seconds,
            "generation_seconds": generation_seconds,
            "generation_tokens": generated_count,
            "generation_truncated": generated_count >= self.config.max_new_tokens,
            "sequence_tokens": int(teacher["input_ids"].shape[-1]),
            "state_audits": state_audits,
            "reference_sha256": reference_sha,
            "compute": "float32_cpu_fake_quant",
            "elapsed_seconds": time.perf_counter() - started,
        }
        self.progress.commit(variant, sample.id, unit, row)
        print(
            f"[{self.progress.root.name}/{variant}] {sample.id} "
            f"NLL={target['nll']:.6f} seconds={row['elapsed_seconds']:.1f}",
            flush=True,
        )
        self.check("after_cell_committed", variant, sample.id)
        return self.progress.completed(variant, sample.id, unit)


def path_difference(reference, candidate, *, linear=None):
    result = {}
    if reference.keys() != candidate.keys():
        raise RuntimeError("propagation layer alignment changed")
    for layer in reference:
        row = {}
        for key, ref in reference[layer].items():
            error = candidate[layer][key] - ref
            record = geometry(ref, torch.zeros_like(ref), error)
            if linear is not None:
                tangent = linear[layer][key]
                record["nonlinear_remainder_energy"] = float((error.double() - tangent.double()).square().sum())
                record["actual_error_energy"] = float(error.double().square().sum())
            row[key] = record
        result[str(layer)] = row
    return result
