from __future__ import annotations

import contextlib
import gc
import importlib.metadata
import json
import math
import platform
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from experiments.ffn_quant_causal.adapters import ModelBundle, build_ffn_targets
from experiments.ffn_quant_causal.budget import WallclockBudget, WallclockBudgetExhausted
from experiments.ffn_quant_causal.protocol import build_protocol_splits
from experiments.ffn_quant_rootcause.evidence import EvidenceProgress, implementation_identity, input_digest
from experiments.ffn_quant_rootcause.interventions import Intervention, QuantBank, intervene
from experiments.infinitevl_quant.evaluation import _target_metrics, _teacher_inputs
from experiments.infinitevl_quant.inventory import decoder_layers
from experiments.infinitevl_quant.loader import load_infinitevl_cpu
from experiments.infinitevl_quant.progress import atomic_json, file_sha256, json_digest
from experiments.vlm_quant_benchmark.data import load_manifest, sample_manifest_digest
from experiments.vlm_quant_benchmark.runner import load_reference

from .config import MODELS, ROOT
from .prepare import start_clock
from .scoring import SCORER_ID, score_prediction


def frozen_json(path, value):
    canonical = json.loads(json.dumps(value))
    if path.exists() and json.loads(path.read_text()) != canonical:
        raise RuntimeError(f"frozen identity changed: {path}; use a new output directory")
    atomic_json(path, canonical)
    return canonical


class Budget:
    def __init__(self, config):
        self.config = config
        started = start_clock(config)["started_unix"]
        self.actual = WallclockBudget.open_or_create(
            config.output_dir / "budget", wallclock_hours=240, config_digest=config.digest(), create=True, now=started
        )
        self.deadline_unix = self.actual.deadline_unix
        self.inference = True

    def remaining_seconds(self):
        return self.actual.remaining_seconds() - (self.config.reserve_hours * 3600 if self.inference else 0)

    def check(self, **kwargs):
        self.actual.check(**kwargs)
        observed = time.time()
        second = observed >= self.actual.started_unix + self.config.round_hours * 3600
        round_path = self.config.output_dir / "budget/round.json"
        if not round_path.exists() or json.loads(round_path.read_text())["round"] != 1 + int(second):
            atomic_json(
                round_path,
                {
                    "round": 1 + int(second),
                    "first_round_deadline_unix": self.actual.started_unix + 120 * 3600,
                    "total_deadline_unix": self.deadline_unix,
                    "transition": "automatic_checkpoint_no_budget_reset",
                },
            )
        if self.remaining_seconds() <= 0:
            raise WallclockBudgetExhausted("inference stopped to preserve 12-hour analysis reserve")


def progress(config, stage, audit, design=None):
    return EvidenceProgress(
        config.output_dir,
        stage,
        config_digest=config.digest(),
        inputs={"audit": json_digest(audit), "design": json_digest(design) if design else None},
    )


def load_splits(config):
    data = {
        name: load_manifest(config.output_dir / "prepare" / f"{name}.jsonl")
        for name in ("pilot8", "screen32", "confirm128", "qa32")
    }
    old, _ = build_protocol_splits(config.calibration_manifest, config.evaluation_manifest)
    data.update(oldscreen32=old["screen"], diagnostic8=old["diagnostic8"], smoke4=old["smoke4"])
    return data


def model_identity(kind, path):
    size, hidden, intermediate = MODELS[kind]
    cfg = json.loads((path / "config.json").read_text())
    cfg = cfg.get("text_config", cfg)
    for field, expected in (("num_hidden_layers", size), ("hidden_size", hidden), ("intermediate_size", intermediate)):
        if cfg[field] != expected:
            raise RuntimeError(f"{kind} config mismatch for {field}")
    mapping = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    expected = {
        f"model.layers.{i}.mlp.{p}_proj.weight": [hidden, intermediate] if p == "down" else [intermediate, hidden]
        for i in range(size)
        for p in ("gate", "up", "down")
    }
    actual = {name for name in mapping if name.startswith("model.layers.") and ".mlp." in name}
    if actual != set(expected):
        raise RuntimeError(f"{kind} FFN index inventory differs")
    shapes = {}
    for shard in sorted(set(mapping.values())):
        with safe_open(path / shard, framework="pt", device="cpu") as handle:
            for name in expected:
                if mapping[name] == shard:
                    shapes[name] = handle.get_slice(name).get_shape()
                    if shapes[name] != expected[name] or handle.get_slice(name).get_dtype() != "BF16":
                        raise RuntimeError(f"{kind} unexpected shape/dtype {name}")
    files = {
        p.name: file_sha256(p)
        for p in sorted(path.iterdir())
        if p.is_file() and p.suffix in (".safetensors", ".json", ".txt", ".model", ".py")
    }
    if kind == "qwen25vl_7b":
        for shard in sorted(set(mapping.values())):
            expected_blob = (path / shard).resolve().name
            if len(expected_blob) != 64 or files[shard] != expected_blob:
                raise RuntimeError("official 7B weight digest differs from pinned Hugging Face LFS blob identity")
    return {
        "path": str(path),
        "files": files,
        "content_sha256": json_digest(files),
        "ffn_shapes": shapes,
        "layers": size,
        "ffn_parameters": 3 * size * hidden * intermediate,
        "tokenizer_sha256": files.get("tokenizer.json"),
    }


def audit_identity(config, paths, splits):
    historical = json.loads((config.history_dir / "audit/audit.json").read_text())
    model_paths = {
        "infinitevl": Path(historical["model_paths"]["infinitevl"]),
        "qwen25vl_3b": Path(historical["model_paths"]["qwen"]),
        "qwen25vl_7b": Path(paths["model7b"]),
    }
    models = {kind: model_identity(kind, path) for kind, path in model_paths.items()}
    root = Path(__file__).parent
    source = {str(p.relative_to(ROOT)): file_sha256(p) for p in sorted(root.glob("*.py"))}
    payload = {
        "config": config.payload(),
        "models": models,
        "driver_files": source,
        "implementation": implementation_identity(),
        "splits": {k: sample_manifest_digest(v) for k, v in splits.items()},
        "data_identity": file_sha256(config.output_dir / "prepare/data.json"),
        "scorer": SCORER_ID,
        "scorer_reference_sha256": file_sha256(ROOT / "third_party/VLMEvalKit/vlmeval/dataset/utils/vqa_eval.py"),
        "history_audit_sha256": file_sha256(config.history_dir / "audit/audit.json"),
        "history_design_sha256": file_sha256(config.history_dir / "audit/locked_design.json"),
        "history_evidence_files": {
            str(p): file_sha256(p)
            for name in ("identity.json", ".complete", "summary.json", "summary.sha256.json")
            for p in (config.history_dir / "sweep" / name,)
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "cpu_capability": torch.backends.cpu.get_cpu_capability(),
            "mkldnn_enabled": torch.backends.mkldnn.enabled,
            "packages": {
                n: importlib.metadata.version(n)
                for n in (
                    "transformers",
                    "numpy",
                    "Pillow",
                    "tokenizers",
                    "safetensors",
                    "qwen-vl-utils",
                    "scipy",
                    "pyarrow",
                    "huggingface-hub",
                    "matplotlib",
                )
            },
        },
    }
    return frozen_json(config.output_dir / "audit/audit.json", payload)


def verify_frozen_sources(config, audit):
    for relative, digest in audit["driver_files"].items():
        if file_sha256(ROOT / relative) != digest:
            raise RuntimeError("driver source changed during execution")
    if implementation_identity() != audit["implementation"]:
        raise RuntimeError("inherited numerical source changed during execution")
    if (
        file_sha256(ROOT / "third_party/VLMEvalKit/vlmeval/dataset/utils/vqa_eval.py")
        != audit["scorer_reference_sha256"]
    ):
        raise RuntimeError("public scorer reference changed during execution")
    for name, digest in audit["history_evidence_files"].items():
        if file_sha256(Path(name)) != digest:
            raise RuntimeError("historical completion evidence changed during execution")
    data = json.loads((config.output_dir / "prepare/data.json").read_text())
    if file_sha256(config.output_dir / "prepare/data.json") != audit["data_identity"]:
        raise RuntimeError("dataset identity changed during execution")
    for name, digest in data["files"].items():
        if file_sha256(Path(name)) != digest:
            raise RuntimeError("image or manifest changed during execution")
    for model in audit["models"].values():
        for name, digest in model["files"].items():
            if file_sha256(Path(model["path"]) / name) != digest:
                raise RuntimeError("checkpoint changed during execution")


@contextlib.contextmanager
def engine_for(config, kind, audit, evidence, budget):
    path = Path(audit["models"][kind]["path"])
    budget.check(stage=evidence.root.name, boundary="before_model_load", variant=kind)
    model, processor, loaded = load_infinitevl_cpu(path) if kind == "infinitevl" else load_reference(str(path))
    bundle = engine = None
    try:
        targets, inventory = build_ffn_targets(model, strict=False)
        size, hidden, intermediate = MODELS[kind]
        if inventory["linear_count"] != size * 3 or inventory["weight_count"] != 3 * size * hidden * intermediate:
            raise RuntimeError("dynamic runtime FFN inventory mismatch")
        for target in targets:
            expected = (hidden, intermediate) if target.name.endswith("down_proj") else (intermediate, hidden)
            if tuple(target.module.weight.shape) != expected:
                raise RuntimeError("runtime FFN tensor differs from audited shape")
        bundle = ModelBundle(kind, model, processor, targets, loaded, inventory["decoder_prefix"])
        engine = Engine(bundle, config, evidence, budget, audit)
        yield engine
    finally:
        if engine is not None:
            engine.input_cache.clear()
            engine.bank.cache.clear()
            engine.bank.sources.clear()
            engine.layers.clear()
        if bundle is not None:
            bundle.ffn_targets.clear()
            del bundle.model, bundle.processor
        del model, processor
        gc.collect()


class Engine:
    def __init__(self, bundle, config, evidence, budget, audit):
        self.bundle, self.config, self.progress, self.budget, self.audit = bundle, config, evidence, budget, audit
        self.layers = list(decoder_layers(bundle.model))
        self.bank = QuantBank(self.layers, 128)
        self.input_cache = {}
        if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in bundle.model.parameters()):
            raise RuntimeError("all parameters must use CPU FP32")

    def inputs(self, sample):
        if sample.id not in self.input_cache:
            prompt, teacher, labels, positions = _teacher_inputs(
                self.bundle.processor, sample, self.config.min_pixels, self.config.max_pixels
            )
            digest = input_digest({**teacher, "labels": labels, "prompt_ids": prompt["input_ids"]})
            self.input_cache[sample.id] = prompt, teacher, labels, positions, digest
        return self.input_cache[sample.id]

    def forward(self, sample, spec, native=False):
        _, teacher, labels, positions, _ = self.inputs(sample)
        mask_positions = labels[0].ne(-100).nonzero().flatten().tolist()
        if list(positions) != mask_positions or not positions or positions[0] == 0:
            raise RuntimeError("target-token positions differ from label mask")
        with (
            torch.inference_mode(),
            contextlib.nullcontext() if native else intervene(self.layers, self.bank, spec, sample_id=sample.id),
        ):
            output = self.bundle.model(**teacher, use_cache=False, return_dict=True)
        values = output.logits[0, torch.tensor(positions) - 1].float()
        targets = labels[0, positions].long()
        loss = F.cross_entropy(values, targets, reduction="none")
        # Match the historical per-token loss then FP32 sum. The fused sum
        # reduction can differ by multiple ulps despite identical token losses.
        total = float(loss.sum())
        old = _target_metrics(output.logits, labels)
        if abs(old["total_nll"] - total) > 1e-6:
            raise RuntimeError("target-token selection differs from historical loss path")
        if not torch.isfinite(loss).all():
            raise FloatingPointError("nonfinite target loss")
        return values.detach(), {
            "tokens": len(positions),
            "total_nll": total,
            "nll": total / len(positions),
            "token_losses": loss.tolist(),
            "token_ids": targets.tolist(),
            "positions": positions,
            "fp64_total_nll": math.fsum(loss.tolist()),
        }

    def evaluate(self, sample, split, label, spec, condition=None):
        _, teacher, _, _, digest = self.inputs(sample)
        variant = f"{self.bundle.kind}__{split}__{label}"
        unit = {"spec": spec.payload(), "input_digest": digest, "operation": "teacher"}
        previous = self.progress.completed(variant, sample.id, unit)
        if previous is not None:
            return previous
        self.budget.check(stage=self.progress.root.name, boundary="before_teacher", variant=variant, sample_id=sample.id)
        start = time.perf_counter()
        _, metrics = self.forward(sample, spec)
        row = {
            "dataset": sample.dataset,
            "model_kind": self.bundle.kind,
            "split": split,
            "intervention": spec.payload(),
            "condition": condition,
            "input_digest": digest,
            "target_text": metrics,
            "target": sample.target,
            "score": math.nan,
            "sequence_tokens": int(teacher["input_ids"].shape[-1]),
            "teacher_seconds": time.perf_counter() - start,
            "model_content_sha256": self.audit["models"][self.bundle.kind]["content_sha256"],
            "compute": "bf16_value_float32_cpu_fake_quant",
        }
        self.progress.commit(variant, sample.id, unit, row)
        print(
            f"[{self.progress.root.name}/{variant}] {sample.id} "
            f"NLL={metrics['nll']:.6f} seconds={row['teacher_seconds']:.1f}",
            flush=True,
        )
        self.budget.check(stage=self.progress.root.name, boundary="after_teacher", variant=variant, sample_id=sample.id)
        return self.progress.completed(variant, sample.id, unit)

    def generate(self, sample, label, condition, teacher_row, split="qa32"):
        prompt, _, _, _, digest = self.inputs(sample)
        if digest != teacher_row["input_digest"] or teacher_row["id"] != sample.id:
            raise RuntimeError("QA teacher record does not match its current input")
        spec = Intervention.from_payload(condition["spec"])
        variant = f"{self.bundle.kind}__{split}__{label}"
        unit = {
            "spec": spec.payload(),
            "input_digest": digest,
            "operation": "generate",
            "max_new_tokens": 32,
            "teacher_row_sha256": teacher_row["row_sha256"],
            "scorer": SCORER_ID,
        }
        previous = self.progress.completed(variant, sample.id, unit)
        if previous is not None:
            return previous
        self.budget.check(stage="qa", boundary="before_generation", variant=variant, sample_id=sample.id)
        start = time.perf_counter()
        with torch.inference_mode(), intervene(self.layers, self.bank, spec, sample_id=sample.id):
            output = self.bundle.model.generate(**prompt, do_sample=False, max_new_tokens=32, use_cache=True)
        suffix = output[:, prompt["input_ids"].shape[-1] :]
        prediction = self.bundle.processor.batch_decode(
            suffix, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        row = {
            "dataset": sample.dataset,
            "model_kind": self.bundle.kind,
            "condition": condition,
            "input_digest": digest,
            "prediction": prediction,
            "target": sample.target,
            "score": score_prediction(sample, prediction),
            "scorer": SCORER_ID,
            "generation_tokens": int(suffix.shape[-1]),
            "generation_truncated": suffix.shape[-1] >= 32,
            "generation_seconds": time.perf_counter() - start,
            "teacher_row_sha256": teacher_row["row_sha256"],
            "target_text": teacher_row["target_text"],
        }
        self.progress.commit(variant, sample.id, unit, row)
        print(f"[qa/{variant}] {sample.id} score={row['score']}", flush=True)
        return self.progress.completed(variant, sample.id, unit)


def rows(evidence, kind, split, label, samples, required=True):
    values = evidence.rows(f"{kind}__{split}__{label}")
    expected = {s.id for s in samples}
    datasets = {s.id: s.dataset for s in samples if hasattr(s, "dataset")}
    if len(values) != len({r["id"] for r in values}) or any(r["id"] not in expected for r in values):
        raise RuntimeError("duplicate or foreign sample in condition")
    for row in values:
        if row["id"] in datasets and row["dataset"] != datasets[row["id"]]:
            raise RuntimeError("sample dataset label differs from its locked manifest")
        if row["row_sha256"] != json_digest({k: v for k, v in row.items() if k != "row_sha256"}):
            raise RuntimeError("sample row checksum mismatch")
        if row["provenance"]["stage_identity"] != evidence.identity_digest:
            raise RuntimeError("sample belongs to a different stage identity")
    if len(values) != len(samples):
        if required:
            raise RuntimeError(f"incomplete condition {kind}/{split}/{label}: {len(values)}/{len(samples)}")
        return None
    return values
