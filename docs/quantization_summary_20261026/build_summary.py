#!/usr/bin/env python3
"""Read existing local artifacts and build a traceable Chinese QA/NLL inventory.

No model inference and no writes to experimental artifacts. Run with .venv/bin/python.
"""
from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
DEST = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DATASETS = ["DocVQA_VAL", "MMMU_DEV_VAL", "ScienceQA_TEST", "TextVQA_VAL"]
SHORT = dict(zip(DATASETS, ["DocVQA", "MMMU", "ScienceQA", "TextVQA"]))
ROWS: list[dict] = []
SOURCES: dict[str, dict] = {}
ISSUES: list[dict] = []
READS: set[Path] = set()


def rel(p):
    return str(Path(p).relative_to(ROOT))


def read(p):
    p = Path(p)
    READS.add(p)
    return json.loads(p.read_text())


def number(v):
    try:
        n = float(v)
        return n if math.isfinite(n) else None
    except (ValueError, TypeError):
        return None


def avg(v):
    v = [number(x) for x in v]
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def fmt(v, digits=4):
    n = number(v)
    return f"{n:.{digits}f}" if n is not None else "—"


def infer_method(label):
    s = str(label).lower()
    for tag in ["vg_qig_mbq", "vg_uniform_mbq", "hybrid_awq", "smoothquant", "squeezellm", "omniquant", "vectorwise_w8", "llm_int8", "official_qwen_awq", "spqr", "gptq", "tender", "mbq", "qig", "awq", "rtn"]:
        if tag in s:
            return tag
    if s in {"b", "bf16", "bf16_reference", "original", "original_reference", "fp16_weight_reference", "fp16_rounded", "bf16_value_fp32_compute"} or s.endswith("__bf16"):
        return "reference"
    return "RTN诊断/干预"


def precision(label, config=None):
    c = config or {}
    s = str(label).lower()
    if "fp16" in s:
        return "FP16值/FP32算"
    if "bf16" in s or s in {"b", "reference"}:
        return "BF16值/FP32算"
    if s == "fp" or s.startswith("fp_k"):
        return "BF16值/FP32算"
    if "original" in s:
        return "原始参考精度"
    w = number(c.get("weight_bits"))
    a = number(c.get("activation_bits"))
    m = re.search(r"w([2348])(?:[_-]?(?:asym|sym_channel))?[_-]?a(16|8|4)", s)
    if m:
        w, a = int(m[1]), int(m[2])
    if w is None:
        m = re.search(r"w([2348])", s)
        if m:
            w = int(m[1])
    if a is None:
        m = re.search(r"(?:_|\b)a(16|8|4)(?:_|\b)", s)
        if m:
            a = int(m[1])
    if "a8_only" in s:
        return "W参考/A8"
    if "fp32" in s or c.get("compute_mode") == "dequant_fp32":
        a = 32
    if w is not None:
        return f"W{int(w)}A{int(a)}" if a else f"W{int(w)}/A参考(FP32算)"
    return "见配置/混合精度"


def nearest_config(directory):
    for parent in [directory, *directory.parents]:
        if parent == OUT or parent == ROOT:
            break
        p = parent / "config.json"
        if p.exists():
            o = read(p)
            if isinstance(o, dict):
                return o
    return {}


def add(row, paths):
    paths = sorted(set(Path(p) for p in paths if Path(p).exists()))
    identity = row.get("identity", row.get("condition", ""))
    key = "|".join([*(rel(p) for p in paths), str(identity), row.get("batch", "")])
    rid = "R" + hashlib.sha256(key.encode()).hexdigest()[:10]
    assert rid not in SOURCES, rid
    row["id"] = rid
    row.setdefault("model", "Qwen2.5-VL-3B-Instruct")
    row.setdefault("family", "")
    row.setdefault("batch", "")
    row.setdefault("method", infer_method(row.get("condition", "")))
    row["method"] = {"rtn_w4": "rtn", "rtn_w8": "rtn", "awq_w4": "awq", "bf16_reference": "reference", "bf16_value_fp32_compute": "reference", "original": "reference"}.get(row["method"], row["method"])
    row.setdefault("format", precision(row.get("condition", "")))
    if row["format"] == "见配置/混合精度" and row["method"] == "reference":
        row["format"] = "BF16值/FP32算"
    row.setdefault("scope", "见条件及原始配置")
    row.setdefault("qa_metric", "")
    row.setdefault("nll_definition", "答案token加权")
    row.setdefault("notes", "")
    row.setdefault("datasets", {})
    row["datasets"] = {
        k: {kk: number(vv) if kk in {"qa", "nll", "nll_sample_mean", "ppl"} else vv for kk, vv in v.items()}
        for k, v in row["datasets"].items()
    }
    row["qa"] = number(row.get("qa"))
    row["nll"] = number(row.get("nll"))
    row["status"] = row.get("status") or ("QA+NLL" if row["qa"] is not None and row["nll"] is not None else "仅QA" if row["qa"] is not None else "仅NLL")
    row["source"] = rel(paths[0]) if paths else ""
    SOURCES[rid] = {"condition": row.get("condition", ""), "files": [rel(p) for p in paths]}
    ROWS.append(row)
    return row


def metric_row(o, **kw):
    ds = {}
    for name, v in o.get("datasets", {}).items():
        ds[name] = {"qa": v.get("score"), "nll": v.get("target_nll_token_weighted", v.get("nll")), "nll_sample_mean": v.get("target_nll_sample_mean"), "samples": v.get("samples")}
    return dict(samples=o.get("samples"), tokens=o.get("target_tokens"), qa=o.get("mean_score"), nll=o.get("target_nll"), ppl=o.get("target_ppl"), datasets=ds, **kw)


def flat_metrics():
    corrected_path = OUT / "infinitevl_quant_ffn_bf16_4x64_v2/analyze/analysis_summary.json"
    corrected = read(corrected_path)
    for p in sorted(OUT.rglob("metrics.json")):
        if not any(tag in str(p) for tag in ["/infinitevl_quant", "/ffn_quant_causal_", "/ffn_quant_mechanism_", "/vlm_quant_benchmark/"]):
            continue
        o = read(p)
        if not isinstance(o, dict):
            continue
        family = p.parts[len(OUT.parts)]
        condition = p.parent.name
        sources = [p]
        notes = ""
        if "datasets" in o and "target_nll" in o:
            model = "InfiniteVL-4B" if "infinitevl" in family or condition.startswith("infinitevl") else "Qwen2.5-VL-3B-Instruct"
            scorer = o.get("scorer", "legacy_local_scorer")
            if family == "infinitevl_quant_ffn_bf16_4x64_v2" and p.parent.parent.name == "evaluate":
                o = corrected["metrics"]["full256"][condition]
                scorer = "local_deterministic_v2"
                sources.append(corrected_path)
                notes = "QA采用analyze内v2重评分；原metrics为旧评分。"
            if scorer == "legacy_local_scorer":
                notes += "历史评分器；仅同表对比，勿替代v2。"
            if family.startswith("infinitevl_quant_ffn_bf16"):
                notes += "FFN保持参考精度；仅相应attention/GDN范围量化。"
            batch = rel(p.parent.parent)
            format_ = precision(condition)
            if family == "infinitevl_quant_cpu_v2" and p.parent.parent.name == "confirm" and condition != "bf16":
                format_ = "W4A16"
                notes += "采用screen筛出的W4A16 frontier。"
            if family == "ffn_quant_mechanism_v1":
                format_ = "W4/参考轨迹干预,A参考"
                notes += "依赖参考轨迹的诊断干预。"
            add(metric_row(o, model=model, family=family, condition=condition, batch=batch, format=format_, qa_metric=scorer, notes=notes), sources)
        elif "exact_accuracy" in o:
            c = nearest_config(p.parent)
            diag = o.get("target_text_diagnostic", o.get("conditional_answer", {}))
            ds = {}
            gp = p.parent / "generations.jsonl"
            if gp.exists():
                grouped = defaultdict(list)
                with gp.open() as f:
                    for line in f:
                        v = json.loads(line)
                        grouped[v.get("dataset", "local_manifest")].append(v)
                for dataset, values in grouped.items():
                    diags = [v.get("target_text_diagnostic", v.get("conditional_answer", {})) for v in values]
                    tok = sum(v.get("tokens", 0) for v in diags)
                    loss = sum(v.get("total_nll", 0) for v in diags)
                    exact = avg(v.get("exact_match") for v in values)
                    ds[dataset] = {"qa": 100*exact if exact is not None else None, "nll": loss/tok if tok else None, "tokens": tok, "samples": len(values), "anls": avg(v.get("anls") for v in values), "edit_similarity": avg(v.get("normalized_edit_similarity") for v in values)}
                sources.append(gp)
            add(dict(family=family, condition=condition, batch=rel(p.parent.parent), method=c.get("method", infer_method(condition)), format=precision(c.get("scheme_id", condition), c), scope=c.get("scope", "decoder_canonical"), samples=o.get("samples"), tokens=diag.get("tokens"), qa=100 * o["exact_accuracy"], qa_metric="local Exact(%)", nll=diag.get("nll"), ppl=diag.get("ppl"), datasets=ds, anls=number(o.get("anls")), edit_similarity=number(o.get("normalized_edit_similarity")), notes="本地Exact受选择题回答格式影响，不等同官方QA。"), sources)
    for split in ["legacy128", "unseen_holdout128"]:
        for condition, o in corrected["metrics"][split].items():
            add(metric_row(o, model="InfiniteVL-4B", family="infinitevl_quant_ffn_bf16_4x64_v2", condition=condition, batch="infinitevl_ffn_reference/" + split, qa_metric="local_deterministic_v2", notes="与full256有包含关系，不是额外独立重复；FFN参考精度。"), [corrected_path])


def csv_early():
    for family in ["ffn_quant", "full_model_quant", "packed_model_quant", "native_model_quant"]:
        for p in sorted((OUT / family).rglob("summary.csv")):
            # Matrix root files are exact concatenations of the leaf result files.
            if p.parent.name == "matrix":
                continue
            c = nearest_config(p.parent)
            READS.add(p)
            with p.open(newline="") as f:
                for item in csv.DictReader(f):
                    name = item["name"]
                    fmt_ = precision(name)
                    mp = p.parent / "metrics.json"
                    metrics = {}
                    sources = [p]
                    if mp.exists():
                        raw = read(mp)
                        metrics = raw.get(name, raw).get("metrics", {})
                        sources.append(mp)
                    gen = metrics.get("generation", {})
                    qa = number(item.get("generation_target_accuracy", gen.get("target_exact_accuracy")))
                    qa = 100 * qa if qa is not None else None
                    samples = len(c.get("evaluation_samples", [])) or gen.get("samples")
                    ds = {"local_manifest": {"qa": qa, "nll": item.get("target_nll"), "samples": samples}}
                    add(dict(family=family, condition=name, batch=rel(p.parent), method=item.get("method", infer_method(name)), format=fmt_, scope=item.get("scope", name), samples=samples, tokens=metrics.get("logits", {}).get("target_tokens"), qa=qa, qa_metric="本地目标Exact(%)" if qa is not None else "未运行QA", nll=item.get("target_nll"), ppl=item.get("target_perplexity", metrics.get("logits", {}).get("target_perplexity")), datasets=ds, notes="早期本地manifest；top1_agreement是logits一致率，非QA。" + ("真实packed/native路径；按compute_mode/backend区分。" if family in {"packed_model_quant", "native_model_quant"} else ""), details={k: item[k] for k in ["compute_mode", "backend", "top1_agreement", "generation_reference_agreement"] if k in item}), sources)


def official_scores():
    groups = defaultdict(dict)
    for root in [OUT / "vlm_quant_benchmark", OUT / "vlm_quant_layer_pruning"]:
        for name in ["official_score.json", "local_score.json"]:
            for p in sorted(root.rglob(name)):
                o = read(p)
                ds = o.get("dataset", p.parent.name)
                if name == "local_score.json" and (p.parent / "official_score.json").exists():
                    continue
                groups[p.parent.parent][ds] = (o, p, name)
    for directory, items in sorted(groups.items()):
        c = nearest_config(directory)
        result_path = directory / "official_result.json"
        meta = read(result_path) if result_path.exists() else c
        condition = directory.name
        scheme = meta.get("scheme_id", c.get("scheme_id", condition))
        method = meta.get("method", c.get("method", infer_method(scheme)))
        ds, paths, notes = {}, [], []
        for name, (o, p, kind) in items.items():
            paths.append(p)
            score = number(o.get("score"))
            if kind == "local_score.json":
                score = number(o.get("mean_local_score"))
                score = score * 100 if score is not None else None
                metric = "local_proxy_v1"
                notes.append("含旧协议local_proxy；非官方可比")
            else:
                metric = o.get("metric", "legacy_untyped")
                if score is None:
                    score = number(o.get("mean_official_score"))
                    # Untyped old DocVQA scores are distances, not QA accuracy.
                    if name == "DocVQA_VAL":
                        notes.append("旧DocVQA距离未作为QA纳入")
                        score = None
                    elif score is not None:
                        score *= 100
                        notes.append("历史未标度评分按0–1转百分制")
            ds[name] = {"qa": score, "samples": o.get("samples"), "metric": metric}
        if result_path.exists():
            paths.append(result_path)
        legacy = any(x in str(directory) for x in ["archive_pre_fix", "formal_pre_mbq", "__legacy", "archive_pre_selective"])
        if legacy:
            notes.append("历史归档；不用于当前排名")
        metrics = sorted(set(v.get("metric", "") for v in ds.values()))
        add(dict(family=directory.parts[len(OUT.parts)], condition=condition, batch=rel(directory.parent), method=method, format=precision(scheme, meta), scope=meta.get("scope", c.get("scope", "见目录配置")), samples=sum(int(v.get("samples") or 0) for v in ds.values()), qa=avg(v["qa"] for v in ds.values()), qa_metric=" / ".join(metrics), datasets=ds, notes="；".join(sorted(set(notes))), status="历史归档QA" if legacy else None), paths)


def corpora():
    for p in sorted((OUT / "vlm_quant_benchmark").rglob("corpus_metrics.json")):
        o, c = read(p), nearest_config(p.parent)
        ds = {}
        for name, v in o.items():
            if isinstance(v, dict) and "nll" in v:
                ds[name] = {"nll": v["nll"], "ppl": v.get("ppl"), "tokens": v.get("tokens"), "samples": v.get("windows")}
        if not ds:
            continue
        add(dict(family="vlm_quant_benchmark", condition=p.parent.name, batch=rel(p.parent.parent), method=c.get("method", infer_method(p.parent.name)), format=precision(c.get("scheme_id", p.parent.name), c), scope=c.get("scope", "decoder_canonical"), samples=sum(int(v.get("samples") or 0) for v in ds.values()), qa_metric="无QA：纯文本语料", nll_definition="各语料独立token加权；不合并成答案NLL", datasets=ds, status="语料NLL/PPL", notes="样本列为窗口总数；数据集明细保留token数与PPL。"), [p])


def existing_report_tables():
    """Some later rescored QA results were saved only as Markdown tables."""
    vg_root = OUT / "vlm_quant_benchmark/vg_qig_mbq_v1"
    p = vg_root / "UNIFIED_SCORE_TABLE.md"
    READS.add(p)
    section = p.read_text().split("## 统一结果", 1)[1].split("粗体", 1)[0]
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip().replace("**", "") for c in line.strip().strip("|").split("|")]
        if len(cells) != 7 or not cells[1].startswith("W"):
            continue
        method = {"MBQ": "mbq", "VG-Uniform-MBQ": "vg_uniform_mbq", "VG-QIG-MBQ": "vg_qig_mbq"}[cells[0]]
        label = method + "__" + cells[1].lower()
        mp = vg_root / "runs" / label / "metrics.json"
        raw = read(mp)
        local = next(r for r in ROWS if r["batch"] == rel(vg_root / "runs") and r["condition"] == label)
        ds = {name: dict(local["datasets"].get(name, {}), qa=float(score), metric=metric) for name, score, metric in zip(DATASETS, cells[2:6], ["ANLS", "MMMU本地确定性解析", "选项Accuracy", "TextVQA consensus"])}
        add(dict(family="vlm_quant_benchmark", batch=rel(vg_root) + "/unified_task_scoring", condition=label, method=method, format=cells[1], scope="decoder_canonical", samples=256, qa=float(cells[6]), qa_metric="统一任务评分；MMMU本地解析", nll=local["nll"], tokens=local.get("tokens"), datasets=ds, reported_rounded=True, notes="优先使用的后续重评分；QA仅以报告两位小数保存，NLL为原始metrics全精度；非外部API judge。"), [p, mp, mp.parent / "generations.jsonl"])
    # Retain a clearly separated earlier, non-preregistered choice-parser diagnostic.
    p = vg_root / "EXPERIMENT_ANALYSIS_REPORT.md"
    READS.add(p)
    section = p.read_text().split("### 5.4 分数据集 Exact", 1)[1].split("### 5.5", 1)[0]
    local_parts = {}
    method_names = {"W3 MBQ": "mbq__w3a16", "W3 VG-Uniform": "vg_uniform_mbq__w3a16", "W3 VG-QIG": "vg_qig_mbq__w3a16", "W4 MBQ": "mbq__w4a16", "W4 VG-QIG": "vg_qig_mbq__w4a16"}
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        c = [s.strip().replace("**", "").replace("%", "") for s in line.strip().strip("|").split("|")]
        if c[0] in method_names:
            local_parts[c[0]] = [float(v) for v in c[1:5]]
    section = p.read_text().split("| 方法 | ScienceQA 首字母准确率", 1)[1].split("这个 post-hoc", 1)[0]
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        c = [s.strip().replace("**", "").replace("%", "") for s in line.strip().strip("|").split("|")]
        if c[0] not in method_names:
            continue
        label = method_names[c[0]]
        doc, text, _, _ = local_parts[c[0]]
        science, mmmu, macro = map(float, c[1:4])
        values = [doc, mmmu, science, text]
        ds = {d: dict(samples=64, qa=v, metric="post-hoc leading-letter diagnostic") for d, v in zip(DATASETS, values)}
        add(dict(family="vlm_quant_benchmark", batch=rel(vg_root) + "/historical_posthoc_parser", condition=label, method=infer_method(label), format=precision(label), scope="decoder_canonical", samples=256, qa=macro, qa_metric="历史post-hoc首字母诊断", datasets=ds, reported_rounded=True, notes="原分析报告临时首字母抽取；后续已有统一评分，不进入当前结论。"), [p])


def sample_aggregate(rows):
    grouped = defaultdict(list)
    for o in rows:
        grouped[o["dataset"]].append(o)
    ds = {}
    for name, values in grouped.items():
        tokens = sum(int(o["target_text"].get("tokens", 0)) for o in values)
        loss = sum(float(o["target_text"].get("total_nll", 0)) for o in values)
        scores = [number(o.get("score")) for o in values]
        scored = [s for s in scores if s is not None]
        ds[name] = {"samples": len(values), "qa_samples": len(scored), "qa": 100 * avg(scored) if scored else None, "nll": loss / tokens if tokens else None, "tokens": tokens}
    tokens = sum(v["tokens"] for v in ds.values())
    loss = sum(float(o["target_text"].get("total_nll", 0)) for o in rows)
    return ds, loss / tokens if tokens else None, tokens


def sample_only_families():
    for family in ["ffn_quant_rootcause_v1", "ffn_quant_compression_v1"]:
        for directory in sorted((OUT / family).glob("*/*/samples")):
            files, values = [], []
            for p in sorted(directory.glob("*.json")):
                o = read(p)
                if "target_text" not in o:
                    continue
                values.append(o)
                files.append(p)
            if not values:
                continue
            ds, nll, tokens = sample_aggregate(values)
            first, condition = values[0], directory.parent.name
            intervention = first.get("intervention", {})
            bits = intervention.get("bits")
            layer_bits, projection_bits = intervention.get("layer_bits", {}), intervention.get("projection_bits", {})
            bitset = sorted(set(v for v in [bits, *layer_bits.values(), *projection_bits.values()] if v is not None))
            label = "+".join("W" + str(x) for x in bitset) + "/参考混合,A参考" if bitset else "BF16值/FP32算"
            notes = "按逐样本loss总和/token总数复算；QA按数据集宏平均。"
            if first.get("split") == "confirm192":
                notes += "历史已用评测样本，非全新独立集。"
            if family == "ffn_quant_rootcause_v1" and condition.split("__")[-1] not in {"B", "W", "W3"}:
                notes += "诊断干预，部分条件需参考轨迹；不等同可部署量化器。"
            compression = first.get("compression")
            details = {"intervention": intervention}
            if compression:
                details["compression"] = compression
                condition += " [q=" + str(compression["q"]) + ";" + ",".join(x["strategy"] for x in compression.get("allocations", [])) + "]"
                label = "W4/参考混合,A参考" if compression["q"] else "BF16值/FP32算"
            scores = [v["qa"] for v in ds.values() if v["qa"] is not None]
            add(dict(model="InfiniteVL-4B" if first.get("model_kind") == "infinitevl" else "Qwen2.5-VL-3B-Instruct", family=family, batch=rel(directory.parent.parent) + "/" + str(first.get("split", "")), condition=condition, method="RTN混合精度" if compression else "RTN诊断/干预", format=label, scope="decoder FFN；详见intervention", samples=len(values), tokens=tokens, qa=avg(scores), qa_metric=(first.get("scoring") or {}).get("scorer", "local_deterministic_v2"), nll=nll, datasets=ds, notes=notes, details=details), files)


def sensitivity_jsonl():
    root = OUT / "vlm_quant_benchmark/sensitivity_analysis_v1"
    for p in sorted(root.rglob("*results.jsonl")):
        groups = defaultdict(dict)
        with p.open() as f:
            for line in f:
                if not line.strip():
                    continue
                o = json.loads(line)
                if "target_nll" not in o or "variant" not in o:
                    continue
                groups[(o.get("evaluation_split", ""), o["variant"])][o["id"]] = o
        for (split, variant), mapped in sorted(groups.items()):
            values = list(mapped.values())
            ds = {}
            for dataset in sorted(set(o["dataset"] for o in values)):
                part = [o for o in values if o["dataset"] == dataset]
                tok = sum(int(o["answer_tokens"]) for o in part)
                nll = sum(o["target_nll"] * o["answer_tokens"] for o in part) / tok
                ds[dataset] = {"samples": len(part), "tokens": tok, "nll": nll, "nll_sample_mean": avg(o["target_nll"] for o in part)}
            tok = sum(int(o["answer_tokens"]) for o in values)
            nll = sum(o["target_nll"] * o["answer_tokens"] for o in values) / tok
            method = "reference" if variant == "bf16" else "SqueezeLLM消融" if "squeeze" in variant else "MBQ消融"
            add(dict(family="vlm_quant_benchmark", batch=rel(p.parent) + "/" + split, condition=variant, method=method, format="W4A16" if variant != "bf16" else "BF16值/FP32算", scope="decoder_canonical", samples=len(values), tokens=tok, nll=nll, datasets=ds, qa_metric="未运行QA", notes="按answer_tokens加权汇总；另存样本均值，二者不可混用。"), [p])


def pruning_jsonl():
    for p in sorted((OUT / "vlm_quant_layer_pruning").glob("*.jsonl")):
        groups = defaultdict(dict)
        with p.open() as f:
            for line in f:
                if not line.strip():
                    continue
                o = json.loads(line)
                if "target_nll" not in o.get("metrics", {}):
                    continue
                key = (o["quant_scheme"], o["operation"], o["layer_set_id"])
                groups[key][o["sample_id"]] = o
        for (scheme, operation, layer_set), mapped in sorted(groups.items()):
            values = list(mapped.values())
            ds = {}
            for name in sorted(set(o["dataset"] for o in values)):
                part = [o for o in values if o["dataset"] == name]
                ds[name] = {"samples": len(part), "nll": avg(o["metrics"]["target_nll"] for o in part)}
            format_ = "W3A16" if scheme == "mbq_w3" else "W4A16" if scheme == "mbq_w4" else precision(scheme)
            add(dict(family="vlm_quant_layer_pruning", batch=rel(p), condition=f"{scheme}/{operation}/{layer_set}", method="MBQ+层/视觉token裁剪" if "mbq" in scheme else "参考精度+裁剪", format=format_, scope="decoder层/视觉token裁剪，见operation", samples=len(values), nll=avg(o["metrics"]["target_nll"] for o in values), datasets=ds, nll_definition="逐样本NLL算术均值", qa_metric="teacher forcing，无生成QA", notes="联合裁剪诊断；不可归因为单纯量化损失。", details={"layer_indices": values[0].get("layer_indices", [])}), [p])


def missing():
    p = OUT / "infinitevl_quant_cpu_v1/INFINITEVL_CPU_QUANT_REPORT_ZH.md"
    READS.add(p)
    ISSUES.append(dict(run="infinitevl_quant_cpu_v1", status="阶段缺失/NO_GO", source=rel(p), detail="旧报告缺少preflight、smoke、scope、screen、qig、confirm、state_probe；没有完整QA/NLL。"))
    vg = OUT / "vlm_quant_benchmark/vg_qig_mbq_v1/w2a16_state.json"
    o = read(vg)
    for item in o["runs"]:
        if item.get("returncode") != 0:
            ISSUES.append(dict(run=item["label"], status="被终止/无完整指标", source=rel(vg), detail=f"returncode={item['returncode']}；不得用VG-QIG的W2成绩代替MBQ。"))
    for p in sorted(OUT.rglob("ABORTED.json")):
        ISSUES.append(dict(run=rel(p.parent), status="ABORTED", source=rel(p), detail=str(read(p))[:500]))
    for p in sorted((OUT / "vlm_quant_benchmark").rglob("status.json")):
        o = read(p)
        status = str(o.get("status", ""))
        if status not in {"complete", "completed", "scored", "ok", ""}:
            ISSUES.append(dict(run=rel(p.parent), status=status, source=rel(p), detail=str(o)[:500]))
    p = OUT / "ffn_quant_compression_v1/sweep/summary.json"
    o = read(p)
    ISSUES.append(dict(run="FFN compression random NLL controls", status="预算省略", source=rel(p), detail=json.dumps(o["budgeted_omissions"], ensure_ascii=False)))


def flat_row(r):
    d = {k: v for k, v in r.items() if k not in {"datasets", "details", "identity"}}
    for name, v in r["datasets"].items():
        for metric, value in v.items():
            d[name + "/" + metric] = value
    if "details" in r:
        d["details_json"] = json.dumps(r["details"], ensure_ascii=False, sort_keys=True)
    return d


def write_csv(path, values):
    fields = list(dict.fromkeys(k for v in values for k in v))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


BLOCKS = []


def para(text):
    BLOCKS.append(("p", text))


def heading(text, level=2):
    BLOCKS.append(("h" + str(level), text))


def table(headers, values):
    BLOCKS.append(("table", (headers, [[str(v) for v in row] for row in values])))


def qa_table(rows):
    headers = ["方法/条件", "位宽", "总样本", *[SHORT[d] for d in DATASETS], "QA均分", "答案NLL", "记录ID"]
    values = []
    for r in rows:
        values.append([r["condition"], r["format"], r.get("samples") or "—", *[fmt(r["datasets"].get(d, {}).get("qa"), 2) for d in DATASETS], fmt(r.get("qa"), 2), fmt(r.get("nll"), 6), r["id"]])
    table(headers, values)


def document():
    now = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S %Z")
    heading("Qwen / InfiniteVL 全量化方法 QA 与 NLL 结果汇总", 1)
    para(f"环境快照：{now}。工作目录：{ROOT}。本次仅整理既有结果，未重新运行模型或改写原实验数据。")
    para("阅读说明：正文按评测协议分表，附录列出所有已提取的配置级结果。表中的记录ID可在配套Excel/CSV与source_manifest.json中定位原文件。相同配置在不同批次、样本集或计算路径上的记录分别保留，条目数不代表独立实验次数。")
    heading("1. 覆盖范围与指标口径")
    counts = Counter(r["family"] for r in ROWS)
    para(f"共整理{len(ROWS)}条配置/批次结果，覆盖{len(counts)}个结果目录系列。模型主要为Qwen2.5-VL-3B-Instruct与hustvl/InfiniteVL（本地标为4B，revision 3d705618aabb1ac60bae75fab6ad0e4679aff58d）。流式视频选帧、VLA动作敏感度及硬件矩阵误差结果不属于本次量化QA/NLL汇总；量化与层/token裁剪的联合实验单列附录。")
    table(["指标/标签", "含义与使用规则"], [
        ["QA（0–100，越高越好）", "当前四项为DocVQA ANLS、MMMU选项准确率、ScienceQA选项准确率、TextVQA共识分；QA均分为数据集宏平均。"],
        ["答案NLL（越低越好）", "teacher forcing对目标答案计算的负对数似然；主口径为loss总和/答案token总数。裁剪附录另标样本均值。"],
        ["语料NLL / PPL", "C4、PTB、WikiText-2的纯语言建模指标；PPL=exp(NLL)。不与多模态答案NLL混为一列。"],
        ["WnAm", "权重量化位宽/激活量化或保留精度标签。A16标签在多数CPU实验中仍由FP32执行，不表示原生FP16内核。"],
        ["BF16 reference", "通常为官方BF16 checkpoint值扩展为FP32存储与计算；早期FP16舍入参考另标。"],
        ["—", "未运行、未记录、非有限值或该指标不适用；不是0，也不是通过。"],
        ["历史/代理/诊断", "旧评分、local Exact、oracle轨迹干预与小样本扫描仅在各自协议中解释。"],
    ])
    table(["结果系列", "配置/批次条目", "说明"], [[g, n, "包括该系列历史、smoke与诊断结果；原始来源见附表"] for g, n in sorted(counts.items())])
    heading("2. 方法与位宽覆盖")
    methods = defaultdict(set)
    for r in ROWS:
        methods[r["method"]].add(r["format"])
    table(["方法/实验类别", "已发现位宽/精度标签"], [[m, "；".join(sorted(v))] for m, v in sorted(methods.items())])
    para("方法概述：RTN为直接舍入基线；AWQ/Hybrid-AWQ利用激活统计选择缩放；GPTQ使用二阶近似；SmoothQuant与LLM.int8处理激活范围或异常值；OmniQuant、SpQR、SqueezeLLM分别包含局部参数优化、稀疏高精度残差、非均匀码本；MBQ平衡不同模态的误差，VG-QIG-MBQ进一步选层并使用token重要性。这里多为本地CPU研究实现，不能直接称作作者官方内核复现。")
    heading("3. Qwen 当前四数据集筛选与确认")
    para("四数据集固定顺序为DocVQA、MMMU、ScienceQA、TextVQA。注意4×256是每项256、总计1024条；4×64是总计256条。该协议的QA文件没有多模态答案NLL，语言健康检查另看下一节语料NLL；不从PPL倒填答案NLL。")
    for name, prefix in [("3.1 Top3确认：每数据集256条（总1024）", "outputs/vlm_quant_benchmark/top3_protocol_v4/confirm"), ("3.2 Top4筛选：每数据集64条（总256）", "outputs/vlm_quant_benchmark/top4_protocol_v3/vlm_screen")]:
        heading(name, 3)
        qa_table([r for r in ROWS if r["batch"] == prefix])
    top3 = {r["condition"]: r for r in ROWS if r["batch"] == "outputs/vlm_quant_benchmark/top3_protocol_v4/confirm"}
    scores = [fmt(top3[k]["qa"], 4) for k in ["bf16_value_fp32_compute", "squeezellm_w4a16_g128_asym", "mbq_w4a16_g128_asym", "omniquant_w4a16_g128_asym"]]
    para(f"简要说明：Top3确认中BF16 QA均分{scores[0]}；W4 SqueezeLLM、MBQ、OmniQuant分别为{scores[1]}、{scores[2]}、{scores[3]}。W3退化更明显。这里的均值排序不能替代配对统计；原报告中接近方法的置信区间存在重叠。")
    heading("4. Qwen 全方法纯文本语料 NLL / PPL")
    para("下表采用corrected/corpus_screen及top4_protocol_v3/corpus_screen中的直接落盘结果，默认decoder范围。每个单元为NLL / PPL；同一方法跨批次保留两行。它们是固定窗口语料筛选结果，并非完整语料测试；每语料的窗口与token数在Excel明细中。")
    corp = [r for r in ROWS if r["status"] == "语料NLL/PPL" and r["batch"] in {"outputs/vlm_quant_benchmark/corrected/corpus_screen", "outputs/vlm_quant_benchmark/top4_protocol_v3/corpus_screen"}]
    table(["方法/方案", "位宽", "批次", "C4 NLL/PPL", "PTB NLL/PPL", "WikiText2 NLL/PPL", "ID"], [[r["condition"], r["format"], r["batch"].split("/")[-2], *[fmt(r["datasets"].get(d, {}).get("nll"), 6) + " / " + fmt(r["datasets"].get(d, {}).get("ppl"), 3) for d in ["c4_validation", "ptb_test", "wikitext2_test"]], r["id"]] for r in corp])
    heading("5. Qwen 早期四数据集全方法 QA（历史协议）")
    para("此表覆盖GPTQ、RTN、SmoothQuant、LLM.int8、Tender、SpQR、SqueezeLLM、OmniQuant、MBQ、官方AWQ等历史全方法结果。DocVQA读取修复后的ANLS；MathVista、MMMU、MMBench在此批次使用local_deterministic_proxy_v1，不等同当前正式协议。旧four_dataset_comparison.csv中的DocVQA距离与混合平均已弃用。")
    hist = [r for r in ROWS if r["batch"] == "outputs/vlm_quant_benchmark/corrected/vlm_screen"]
    old_ds = ["DocVQA_VAL", "MathVista_MINI", "MMMU_DEV_VAL", "MMBench_DEV_EN"]
    table(["方法/方案", "位宽", "范围", "DocVQA ANLS", "MathVista代理", "MMMU代理", "MMBench代理", "ID"], [[r["condition"], r["format"], r["scope"], *[fmt(r["datasets"].get(d, {}).get("qa"), 2) for d in old_ds], r["id"]] for r in hist])
    heading("6. Qwen VG-QIG-MBQ：W2 / W3 / W4 与 A8")
    para("此批次为总计256个多模态样本。已找到后续UNIFIED_SCORE_TABLE.md，采用任务对应评分、MMMU本地确定性解析；它替代早期跨任务字符串Exact作为本批次QA主结果。该统一表只保存两位小数，以下QA保留报告精度，答案NLL来自各配置原始metrics.json。")
    heading("6.1 已有统一任务评分（优先）", 3)
    qa_table([r for r in ROWS if r["batch"] == "outputs/vlm_quant_benchmark/vg_qig_mbq_v1/unified_task_scoring"])
    para("简要说明：同次MBQ对照下，统一QA均分在W3由66.37升至68.44（+2.07分），W4由69.93升至70.10（+0.17分），W4A8由67.61升至69.61（+2.00分）。W4收益主要由MMMU改善抵消OCR下降；W2均分仅5.86且NLL 9.484594，仍明显失效。统一评分结果不用于反向改写以旧Exact预注册的replacement gate。")
    heading("6.2 原始local Exact诊断（保留追溯）", 3)
    para("选择题输出含解释文字时Exact可能低估正确率；ANLS与Edit Similarity也为本地跨任务代理。下表不与6.1的统一任务评分混用。")
    vg = [r for r in ROWS if r["batch"] == "outputs/vlm_quant_benchmark/vg_qig_mbq_v1/runs"]
    table(["方法/条件", "位宽", "总样本", "Exact %", "ANLS×100", "Edit×100", "答案NLL", "ID"], [[r["condition"], r["format"], r["samples"], fmt(r["qa"], 4), fmt(100*r["anls"] if r.get("anls") is not None else None, 4), fmt(100*r["edit_similarity"] if r.get("edit_similarity") is not None else None, 4), fmt(r["nll"], 6), r["id"]] for r in vg])
    para("旧Exact口径下VG-QIG在W3有小幅改善，W4/W4A8的Exact下降，方向与统一QA部分不同。MBQ W2进程被终止，没有完整指标，不能补为0或引用VG-QIG的成绩。原分析报告中的post-hoc首字母诊断也作为独立历史条目收录附录。")
    heading("7. InfiniteVL 初始量化与恢复方法")
    para("本节为infinitevl_quant_cpu_v2旧评分协议，screen/confirm各总计32条，非每数据集32条。decoder_safe保留GDN a/b dynamics等模块；W4A8使用对称按输出通道W4与动态A8，与W4A16的非对称g128量化也有粒度差异。")
    for phase in ["screen", "confirm"]:
        heading("7." + ("1" if phase == "screen" else "2") + " " + phase, 3)
        qa_table([r for r in ROWS if r["batch"] == "outputs/infinitevl_quant_cpu_v2/" + phase])
    para("简要说明：初始RTN W2/W3与W4A8质量明显下降，W4A16较稳定；MBQ/QIG恢复实验最终标记NO_GO，因为预设的QIG增益与状态漂移门槛未通过。该结论与后续只量化attention、保留FFN的实验范围不同。")
    heading("8. InfiniteVL：FFN参考精度、attention/GDN量化（修复评分）")
    para("优先使用4×64扩展实验的local_deterministic_v2：总256条，FFN保持参考精度。attn_main为SWA+GDN主投影，attn_all还包括GDN dynamics；它们均不是全FFN/全模型W3或W4。原evaluate/metrics.json中的旧QA由analyze/analysis_summary.json内v2分数替换，NLL保持一致。")
    for title, batch in [("8.1 全256条", "outputs/infinitevl_quant_ffn_bf16_4x64_v2/evaluate"), ("8.2 扩展实验内未用holdout128（每项32）", "infinitevl_ffn_reference/unseen_holdout128")]:
        heading(title, 3)
        qa_table([r for r in ROWS if r["batch"] == batch])
    para("简要说明：全256条中BF16均分72.0647、NLL 0.756236；W4 attention均分71.8370，NLL约0.746，W8更接近参考分布。W3均分约62.70–63.38，NLL约增加40%。原扩展报告最终选择W8 attn_all；W4虽通过点估计门槛，仍标为POINT_USABLE_UNCERTAIN。")
    heading("9. Qwen / InfiniteVL 匹配 FFN 量化与 MBQ/QIG 恢复")
    para("以下来自ffn_quant_causal_v3，均为FFN专项，QA使用v2。screen32为总32条；within_protocol_holdout192为总192条（每项48），这组样本历史上已使用，不能称为新的外部独立确认。")
    selected = [r for r in ROWS if r["family"] == "ffn_quant_causal_v3" and ("__screen32__" in r["condition"] and r["condition"].endswith(("bf16", "rtn_w8a16_ffn", "rtn_w4a16_ffn", "rtn_w3a16_ffn")) or "/confirm" in r["batch"])]
    for model in ["InfiniteVL-4B", "Qwen2.5-VL-3B-Instruct"]:
        heading("9：" + model, 3)
        qa_table([r for r in selected if r["model"] == model])
    heading("10. FFN W4压缩比例与诊断干预")
    para("10.1 使用最终curve_points.csv中的confirm192逐点结果；相同端点在两种排序策略中重复出现，用于展示两条完整曲线。W4量化的FFN层数q从0到36变化，未量化层保持参考精度，理论节省率排除元数据。")
    cp = OUT / "ffn_quant_compression_v1/report/curve_points.csv"
    READS.add(cp)
    with cp.open(newline="") as f:
        points = [o for o in csv.DictReader(f) if o["cohort"] == "confirm192"]
    table(["模型", "层分配策略", "W4层数/36", "理论节省%", "QA均分", "NLL", "样本"], [[p["model"], p["policy"], p["q"], fmt(p["compression_percent"], 2), fmt(p["qa_macro"], 4), fmt(p["nll"], 6), p["samples"]] for p in points])
    para("简要说明：InfiniteVL全FFN W4在confirm192上QA由72.2334降至60.6390、NLL由0.849485升至1.315304；只保留零起始层35参考精度时，QA为69.2531、NLL为0.986823。Qwen同批次全FFN W4 QA约69.7605，接近其参考69.6804。层分配影响明显，压缩率不能单独解释质量。")
    heading("10.2 参考轨迹诊断：保留为机制证据", 3)
    diag = [r for r in ROWS if r["family"] == "ffn_quant_rootcause_v1" and "/confirm/" in r["batch"]]
    table(["模型", "诊断条件", "精度", "总样本", "QA均分", "NLL", "ID"], [[r["model"], r["condition"], r["format"], r["samples"], fmt(r["qa"], 4), fmt(r["nll"], 6), r["id"]] for r in diag])
    para("误差×0.25/0.5、Gaussian及GDN修复等条件可能需要参考FFN输出或轨迹，属于诊断性干预，不是独立可部署的量化方法。完整W2/W3/W4/W8、单层/投影、混合精度与NLL扫描见附录。")
    heading("11. 早期 FFN / 全Linear / Packed / Native 结果")
    para("早期本地manifest通常总16条、smoke总2条；它们的目标Exact与正式VQA不同。部分结果generation_target_accuracy为0，这只表示该小型manifest的目标匹配，不代表模型在所有QA任务准确率为0。logits top1一致率没有当作QA填入。")
    early = [r for r in ROWS if r["family"] in {"ffn_quant", "full_model_quant", "packed_model_quant", "native_model_quant"}]
    table(["批次", "条件", "位宽/激活", "目标Exact %", "答案NLL", "ID"], [[r["batch"].removeprefix("outputs/"), r["condition"], r["format"], fmt(r["qa"], 2), fmt(r["nll"], 6), r["id"]] for r in early])
    heading("12. 缺失结果与解释边界")
    para("目前未找到的完整成绩不补算。MBQ W2被终止；InfiniteVL cpu_v1及ffn_causal_v1保留中止记录；部分历史数据集仅有预测或受外部judge阻塞，没有官方分。compression新增随机NLL对照因预算省略。全部状态详情见uncompleted_runs.csv/Excel。")
    table(["事项", "处理"], [
        ["DocVQA历史距离", "正文仅引用修复ANLS；未经修复的旧距离不纳入QA。"],
        ["InfiniteVL旧选择题scorer", "修复版优先；旧版作为单独历史批次保留。"],
        ["仅NLL扫描", "QA留空，不能由NLL推导任务正确率。"],
        ["跨模型/跨批次", "baseline、样本、scorer和量化范围须匹配，不能直接对不同表的绝对值排名。"],
        ["fake quantization", "绝大多数为FP32 CPU解量化质量模拟；只有packed/native系列单独使用对应存储/执行路径。"],
        ["独立性", "同一批样本的screen/confirm切片、历史复用与重复归档不构成额外独立证据。"],
    ])
    heading("附录A：全配置 QA / NLL 总表")
    para("下表保留每个原始批次的全部配置，不按成绩筛选。QA与NLL的详细定义、每数据集分数及来源见配套Excel/CSV。同一方法在多个路径中的条目可能复用同一批预测。语料行在本表的NLL为—，三语料独立NLL在第4节及Excel数据集明细中。")
    groups = defaultdict(list)
    for r in ROWS:
        groups[(r["family"], r["batch"], r["model"])].append(r)
    for (family, batch, model), rows in sorted(groups.items()):
        heading(model + " | " + batch.removeprefix("outputs/"), 3)
        table(["条件", "位宽", "样本", "QA", "NLL", "记录ID"], [[r["condition"], r["format"], r.get("samples") or "—", fmt(r["qa"], 4), fmt(r["nll"], 6), r["id"]] for r in rows])
    heading("附录B：来源与复核方式")
    para("all_experiments.csv为配置/批次宽表，dataset_results.csv为逐数据集长表，quantization_results.xlsx包含两者及缺失状态。source_manifest.json记录每条记录对应的源文件与SHA256，source_files.csv保存完整文件哈希。所有数值CSV保留原始精度，文档表格仅四舍五入展示。build_summary.py可在同一环境重建文档。")


def exports():
    flat = [flat_row(r) for r in ROWS]
    dataset_rows = []
    for r in ROWS:
        for name, values in r["datasets"].items():
            dataset_rows.append(dict(id=r["id"], model=r["model"], family=r["family"], batch=r["batch"], condition=r["condition"], method=r["method"], format=r["format"], scope=r["scope"], dataset=name, qa_metric=values.get("metric", r["qa_metric"]), nll_definition=r["nll_definition"], **values))
    write_csv(DEST / "all_experiments.csv", flat)
    write_csv(DEST / "dataset_results.csv", dataset_rows)
    write_csv(DEST / "uncompleted_runs.csv", ISSUES)
    (DEST / "all_experiments.json").write_text(json.dumps(ROWS, ensure_ascii=False, indent=2, allow_nan=False))
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    wb = Workbook()
    wb.remove(wb.active)
    for name, values in [("配置总表", flat), ("数据集明细", dataset_rows), ("未完成及缺失", ISSUES)]:
        ws = wb.create_sheet(name)
        fields = list(dict.fromkeys(k for o in values for k in o))
        ws.append(fields)
        for o in values:
            ws.append([o.get(k) for k in fields])
        ws.freeze_panes = "E2"
        ws.auto_filter.ref = ws.dimensions
        for cell in ws[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="193B58")
        for i, key in enumerate(fields, 1):
            ws.column_dimensions[get_column_letter(i)].width = 20 if key not in {"source", "condition", "batch", "notes", "details_json"} else 48
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=False)
    wb.save(DEST / "quantization_results.xlsx")
    hashes = []
    paths = {ROOT / name for item in SOURCES.values() for name in item["files"]}
    paths.update(READS)
    for p in sorted(paths):
        data = p.read_bytes()
        hashes.append(dict(path=rel(p), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)))
    write_csv(DEST / "source_files.csv", hashes)
    digest = {o["path"]: o["sha256"] for o in hashes}
    for v in SOURCES.values():
        v["aggregate_sources_sha256"] = hashlib.sha256("\n".join(name + ":" + digest[name] for name in v["files"]).encode()).hexdigest()
    manifest = dict(snapshot_beijing=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), workspace=str(ROOT), row_count=len(ROWS), dataset_rows=len(dataset_rows), source_files=len(hashes), records=SOURCES)
    (DEST / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return len(dataset_rows), len(hashes)


def render():
    markdown = []
    body = []
    from docx import Document
    from docx.shared import Cm, Pt, RGBColor
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(29.7), Cm(21)
    sec.top_margin = sec.bottom_margin = Cm(1.4)
    sec.left_margin = sec.right_margin = Cm(1.3)
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(9)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    normal.paragraph_format.space_after = Pt(5)
    for style in ["Heading 1", "Heading 2", "Heading 3"]:
        doc.styles[style].font.color.rgb = RGBColor.from_string("193B58")
        doc.styles[style].element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    footer = sec.footer.paragraphs[0]
    footer.add_run("Qwen / InfiniteVL 量化QA与NLL汇总 · 2026-10-26     ")
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)
    for kind, value in BLOCKS:
        if kind.startswith("h"):
            level = int(kind[1])
            markdown.extend(["#" * level + " " + value, ""])
            body.append(f"<h{level}>{html.escape(value)}</h{level}>")
            doc.add_heading(value, level)
        elif kind == "p":
            markdown.extend([value, ""])
            body.append("<p>" + html.escape(value) + "</p>")
            doc.add_paragraph(value)
        else:
            headers, values = value
            escape = lambda s: s.replace("|", "\\|").replace("\n", " ")
            markdown.extend(["| " + " | ".join(map(escape, headers)) + " |", "| " + " | ".join("---" for _ in headers) + " |"])
            markdown.extend("| " + " | ".join(map(escape, row)) + " |" for row in values)
            markdown.append("")
            body.append("<div class='table-wrap'><table><thead><tr>" + "".join("<th>" + html.escape(x) + "</th>" for x in headers) + "</tr></thead><tbody>" + "".join("<tr>" + "".join("<td>" + html.escape(x) + "</td>" for x in row) + "</tr>" for row in values) + "</tbody></table></div>")
            t = doc.add_table(rows=1, cols=len(headers))
            t.style = "Light Shading Accent 1"
            for cell, text in zip(t.rows[0].cells, headers):
                cell.text = text
            repeat = OxmlElement("w:tblHeader")
            t.rows[0]._tr.get_or_add_trPr().append(repeat)
            for row in values:
                for cell, text in zip(t.add_row().cells, row):
                    cell.text = text
                    for paragraph in cell.paragraphs:
                        paragraph.paragraph_format.space_after = Pt(1)
                        for run in paragraph.runs:
                            run.font.size = Pt(7)
            doc.add_paragraph()
    (DEST / "Qwen_InfiniteVL_量化QA_NLL汇总.md").write_text("\n".join(markdown))
    css = "body{font:14px/1.6 system-ui,sans-serif;max-width:1500px;margin:36px auto;padding:0 28px;color:#172b3a}h1,h2,h3{color:#193b58}h2{margin-top:36px}h3{overflow-wrap:anywhere}table{border-collapse:collapse;width:100%;font-size:12px;margin:16px 0}th,td{border:1px solid #cbd5df;padding:6px 9px;text-align:left;overflow-wrap:anywhere}th{background:#193b58;color:white}tbody tr:nth-child(even){background:#f1f6fa}.table-wrap{overflow-x:auto}p{max-width:1200px}@media print{body{max-width:none;padding:0;font-size:10px}thead{display:table-header-group}tr{break-inside:avoid}h2,h3{break-after:avoid}}"
    (DEST / "Qwen_InfiniteVL_量化QA_NLL汇总.html").write_text("<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>Qwen / InfiniteVL量化QA与NLL汇总</title><style>" + css + "</style><body>" + "\n".join(body) + "</body></html>")
    doc.save(DEST / "Qwen_InfiniteVL_量化QA_NLL汇总.docx")


def validate():
    assert len(ROWS) == len(SOURCES)
    for r in ROWS:
        for val in [r.get("qa"), *(v.get("qa") for v in r["datasets"].values())]:
            assert val is None or 0 <= val <= 100, (r["id"], val)
        if r.get("qa") is not None and r["datasets"]:
            values = [v.get("qa") for v in r["datasets"].values()]
            mean = avg(values)
            if mean is not None:
                tolerance = 0.011 if r.get("reported_rounded") else 1e-7
                assert abs(r["qa"] - mean) < tolerance, (r["id"], r["qa"], mean)
    # The repaired InfiniteVL scoring must not regress to old choice extraction.
    r = next(r for r in ROWS if r["batch"] == "outputs/infinitevl_quant_ffn_bf16_4x64_v2/evaluate" and r["condition"] == "bf16")
    assert abs(r["qa"] - 72.0646504766011) < 1e-10
    r = next(r for r in ROWS if r["batch"] == "outputs/vlm_quant_benchmark/top3_protocol_v4/confirm" and r["condition"] == "bf16_value_fp32_compute")
    assert r["samples"] == 1024
    assert abs(r["datasets"]["DocVQA_VAL"]["qa"] - 93.88818827321866) < 1e-10


def main():
    flat_metrics()
    csv_early()
    official_scores()
    corpora()
    existing_report_tables()
    sample_only_families()
    sensitivity_jsonl()
    pruning_jsonl()
    missing()
    validate()
    n_ds, n_files = exports()
    document()
    render()
    result = dict(rows=len(ROWS), dataset_rows=n_ds, source_files=n_files, issue_rows=len(ISSUES), family_counts=dict(Counter(r["family"] for r in ROWS)), validation="passed", no_new_model_inference=True)
    (DEST / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
