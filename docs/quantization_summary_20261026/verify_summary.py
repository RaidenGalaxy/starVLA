#!/usr/bin/env python3
"""Independently verify exported tables against saved source artifacts."""
import csv
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

from openpyxl import load_workbook

DEST = Path(__file__).resolve().parent
ROOT = DEST.parents[1]
rows = json.loads((DEST / "all_experiments.json").read_text())
manifest = json.loads((DEST / "source_manifest.json").read_text())
checks = defaultdict(int)
max_diff = defaultdict(float)


def compare(a, b, kind, tolerance=1e-9):
    if a is None or b is None:
        return
    delta = abs(float(a) - float(b))
    assert delta <= tolerance, (kind, a, b, delta)
    max_diff[kind] = max(max_diff[kind], delta)
    checks[kind] += 1


for row in rows:
    files = [ROOT / p for p in manifest["records"][row["id"]]["files"]]
    if row["family"] in {"ffn_quant_rootcause_v1", "ffn_quant_compression_v1"}:
        # A second implementation: aggregate each scalar with math.fsum.
        samples = [json.loads(p.read_text()) for p in files]
        loss = math.fsum(float(o["target_text"]["total_nll"]) for o in samples)
        tokens = sum(int(o["target_text"]["tokens"]) for o in samples)
        compare(row["nll"], loss / tokens, "raw_samples_nll", 1e-11)
        by_dataset = defaultdict(list)
        for o in samples:
            score = o.get("score")
            if score is not None and math.isfinite(float(score)):
                by_dataset[o["dataset"]].append(float(score))
        scores = [100 * math.fsum(v) / len(v) for v in by_dataset.values()]
        if scores:
            compare(row["qa"], math.fsum(scores) / len(scores), "raw_samples_qa", 1e-10)
        assert len(samples) == row["samples"]
        checks["raw_sample_conditions"] += 1
    for p in files:
        if p.name == "official_score.json":
            o = json.loads(p.read_text())
            if "score" in o:
                name = o.get("dataset", p.parent.name)
                compare(row["datasets"][name]["qa"], o["score"], "official_qa")
        if p.name == "corpus_metrics.json":
            o = json.loads(p.read_text())
            for dataset, values in row["datasets"].items():
                compare(values.get("nll"), o[dataset].get("nll"), "corpus_nll")
                compare(values.get("ppl"), o[dataset].get("ppl"), "corpus_ppl")
        if p.name == "metrics.json":
            o = json.loads(p.read_text())
            if "target_nll" in o:
                compare(row["nll"], o["target_nll"], "direct_target_nll")
            elif "target_text_diagnostic" in o:
                compare(row["nll"], o["target_text_diagnostic"]["nll"], "direct_local_nll")
            elif "conditional_answer" in o:
                compare(row["nll"], o["conditional_answer"]["nll"], "direct_local_nll")
            else:
                o = o.get(row["condition"], o)
                compare(row["nll"], o.get("metrics", {}).get("logits", {}).get("target_nll"), "early_target_nll")

wb = load_workbook(DEST / "quantization_results.xlsx", read_only=True, data_only=True)
assert wb["配置总表"].max_row - 1 == len(rows)
assert wb["数据集明细"].max_row - 1 == manifest["dataset_rows"]
checks["excel_tables"] = 2
with (DEST / "all_experiments.csv").open(encoding="utf-8-sig", newline="") as f:
    assert len(list(csv.DictReader(f))) == len(rows)
with zipfile.ZipFile(DEST / "Qwen_InfiniteVL_量化QA_NLL汇总.docx") as z:
    assert z.testzip() is None
    document = ET.fromstring(z.read("word/document.xml"))
    text = "".join(document.itertext())
    for row in rows:
        assert row["id"] in text, row["id"]
checks["docx_all_record_ids"] = len(rows)
markdown = (DEST / "Qwen_InfiniteVL_量化QA_NLL汇总.md").read_text()
assert "UNIFIED_SCORE_TABLE.md" in markdown
assert "68.44" in markdown and "9.484594" in markdown
assert "72.0647" in markdown

result = dict(status="passed", new_model_inference=False, checks=dict(checks), maximum_absolute_differences=dict(max_diff), configuration_rows=len(rows), dataset_rows=manifest["dataset_rows"])
(DEST / "independent_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps(result, ensure_ascii=False, indent=2))
