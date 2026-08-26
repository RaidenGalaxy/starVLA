"""Aggregate and rank the active BF16/W4A16/W3A16 four-dataset protocol."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

from .dataset_protocol import ACTIVE_DATASET_NAMES, DATASET_SPECS, PROTOCOL_VERSION
from .reporting import atomic_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Top4 W4/W3 Chinese report")
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--full-root", type=Path)
    parser.add_argument("--legacy-corpus-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_ci(values: list[float], seed: int = 42, repeats: int = 10000) -> list[float]:
    rng = random.Random(seed)
    means = []
    for _ in range(repeats):
        means.append(sum(rng.choice(values) for _ in values) / len(values) * 100.0)
    return [percentile(means, 0.025), percentile(means, 0.975)]


def wilson_ci(values: list[float], z: float = 1.959963984540054) -> list[float]:
    count = len(values)
    probability = sum(values) / count
    denominator = 1 + z * z / count
    center = (probability + z * z / (2 * count)) / denominator
    margin = z * math.sqrt(
        probability * (1 - probability) / count + z * z / (4 * count * count)
    ) / denominator
    return [100 * (center - margin), 100 * (center + margin)]


def score_ci(dataset: str, values: list[float]) -> tuple[str, list[float]]:
    if dataset in {"MMMU_DEV_VAL", "ScienceQA_TEST"} and set(values) <= {0.0, 1.0}:
        return "wilson_95", wilson_ci(values)
    return "bootstrap_95", bootstrap_ci(values)


LEGACY_CORPUS_ALIASES = {
    "rtn_w4a16_g128_asym": "naive_w4a16_g128_asym",
    "omniquant_w4a16_g128_asym": "local_omniquant_w4a16_g128_asym",
    "squeezellm_w4a16_g128_asym": "local_squeezellm_w4a16_g128_asym",
}


def load_ppl(root: Path | None, scheme: str) -> dict | None:
    if root is None:
        return None
    directory = "bf16" if scheme == "bf16_value_fp32_compute" else scheme
    candidates = (directory, LEGACY_CORPUS_ALIASES.get(directory))
    for candidate in candidates:
        if candidate is None:
            continue
        path = root / candidate / "corpus_metrics.json"
        if path.is_file():
            return json.loads(path.read_text())
    return None


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_table(rows: list[dict]) -> list[str]:
    headers = ["method", "format", "DocVQA", "MMMU", "ScienceQA", "TextVQA", "avg", "worst-ret", "mean-ret", "PPL×", "BPW"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        lines.append("| " + " | ".join([
            row["method"], row["format"],
            f'{row["DocVQA_VAL"]:.2f}', f'{row["MMMU_DEV_VAL"]:.2f}',
            f'{row["ScienceQA_TEST"]:.2f}', f'{row["TextVQA_VAL"]:.2f}',
            f'{row["raw_average"]:.2f}', f'{row["worst_retention"]:.2f}%',
            f'{row["mean_retention"]:.2f}%',
            "-" if not math.isfinite(row["ppl_inflation"]) else f'{row["ppl_inflation"]:.3f}',
            f'{row["logical_bits_per_weight"]:.3f}',
        ]) + " |")
    return lines


def format_ci_table(rows: list[dict]) -> list[str]:
    lines = [
        "| method | format | DocVQA 95% CI | MMMU 95% CI | ScienceQA 95% CI | TextVQA 95% CI |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        cells = []
        for dataset in ACTIVE_DATASET_NAMES:
            item = row["confidence_intervals"][dataset]
            low, high = item["interval"]
            cells.append(f"{item['type']} [{low:.2f}, {high:.2f}]")
        lines.append(
            f"| {row['method']} | {row['format']} | " + " | ".join(cells) + " |"
        )
    return lines


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for result_path in sorted(args.screen_root.glob("*/official_result.json")):
        payload = json.loads(result_path.read_text())
        if payload.get("status") != "complete":
            continue
        if set(payload.get("datasets", {})) != set(ACTIVE_DATASET_NAMES):
            continue
        sample_map = {name: [] for name in ACTIVE_DATASET_NAMES}
        for sample in payload.get("samples", []):
            if sample.get("dataset") in sample_map:
                sample_map[sample["dataset"]].append(float(sample["official_score"]))
        if any(len(values) != 64 for values in sample_map.values()):
            continue
        scores = {
            name: 100.0 * sum(values) / len(values)
            for name, values in sample_map.items()
        }
        cis = {}
        for name, values in sample_map.items():
            ci_type, interval = score_ci(name, values)
            cis[name] = {"type": ci_type, "interval": interval}
        runs.append({
            "run_dir": result_path.parent.name,
            "method": payload["method"],
            "scheme_id": payload["scheme_id"],
            "scope": payload["scope"],
            "weight_bits": payload.get("weight_bits"),
            "activation_bits": payload.get("activation_bits"),
            "logical_bits_per_weight": float(payload["logical_bits_per_weight"]),
            "scores": scores,
            "confidence_intervals": cis,
        })
    full_confirmation = []
    if args.full_root is not None and args.full_root.is_dir():
        for result_path in sorted(args.full_root.glob("*/official_result.json")):
            payload = json.loads(result_path.read_text())
            if payload.get("status") != "complete":
                continue
            if set(payload.get("datasets", {})) != set(ACTIVE_DATASET_NAMES):
                continue
            sample_map = {name: [] for name in ACTIVE_DATASET_NAMES}
            for sample in payload.get("samples", []):
                if sample.get("dataset") in sample_map:
                    sample_map[sample["dataset"]].append(float(sample["official_score"]))
            if any(not values for values in sample_map.values()):
                continue
            bits = payload.get("weight_bits")
            full_confirmation.append({
                "method": payload["method"],
                "scheme_id": payload["scheme_id"],
                "scope": payload["scope"],
                "weight_bits": bits,
                "activation_bits": payload.get("activation_bits"),
                "format": "BF16" if bits is None else f"W{bits}A{payload.get('activation_bits') or 16}",
                "sample_counts": {key: len(value) for key, value in sample_map.items()},
                "scores": {
                    key: 100.0 * sum(value) / len(value)
                    for key, value in sample_map.items()
                },
            })

    references = [row for row in runs if row["method"] == "bf16_value_fp32_compute"]
    if len(references) != 1:
        raise ValueError(f"expected one complete BF16 run, found {len(references)}")
    reference = references[0]
    bf_ppl = load_ppl(args.corpus_root, reference["scheme_id"])
    if bf_ppl is None:
        bf_ppl = load_ppl(args.legacy_corpus_root, reference["scheme_id"])

    rows = []
    for run in runs:
        scores = run["scores"]
        retentions = [scores[name] / reference["scores"][name] * 100 for name in ACTIVE_DATASET_NAMES]
        ppl = load_ppl(args.corpus_root, run["scheme_id"])
        if ppl is None:
            ppl = load_ppl(args.legacy_corpus_root, run["scheme_id"])
        ratios = []
        if ppl is not None and bf_ppl is not None:
            for name in sorted(set(ppl) & set(bf_ppl)):
                ratios.append(float(ppl[name]["ppl"]) / float(bf_ppl[name]["ppl"]))
        bits = run["weight_bits"]
        row = {
            "run_dir": run["run_dir"],
            "method": run["method"],
            "scheme_id": run["scheme_id"],
            "scope": run["scope"],
            "weight_bits": bits,
            "activation_bits": run["activation_bits"],
            "format": "BF16" if bits is None else f"W{bits}A{run['activation_bits'] or 16}",
            **scores,
            "raw_average": sum(scores.values()) / len(scores),
            "worst_retention": min(retentions),
            "mean_retention": sum(retentions) / len(retentions),
            "ocr_retention": (retentions[0] + retentions[3]) / 2,
            "reasoning_retention": (retentions[1] + retentions[2]) / 2,
            "ppl_inflation": sum(ratios) / len(ratios) if ratios else math.inf,
            "logical_bits_per_weight": run["logical_bits_per_weight"],
            "confidence_intervals": run["confidence_intervals"],
        }
        rows.append(row)

    def rank_key(row: dict) -> tuple:
        return (
            -row["worst_retention"], -row["mean_retention"],
            row["ppl_inflation"], row["logical_bits_per_weight"], row["method"],
        )

    selections = {}
    for bits in (4, 3):
        candidates = sorted([row for row in rows if row["weight_bits"] == bits], key=rank_key)
        if candidates:
            selections[f"w{bits}a16"] = candidates[0]
    serializable = []
    for row in rows:
        serializable.append(row | {
            "ppl_inflation": None if not math.isfinite(row["ppl_inflation"]) else row["ppl_inflation"],
        })
    atomic_json(args.output_dir / "summary.json", {
        "status": "complete", "protocol_version": PROTOCOL_VERSION,
        "datasets": list(ACTIVE_DATASET_NAMES), "runs": serializable,
        "full_confirmation": full_confirmation,
    })
    serializable_selections = {
        key: value | {
            "ppl_inflation": (
                None if not math.isfinite(value["ppl_inflation"])
                else value["ppl_inflation"]
            ),
        }
        for key, value in selections.items()
    }
    atomic_json(args.output_dir / "selection.json", {
        "ranking": "worst_retention_desc,mean_retention_desc,ppl_inflation_asc,logical_bpw_asc",
        "winners": serializable_selections,
    })
    csv_rows = [{key: value for key, value in row.items() if key != "confidence_intervals"} for row in serializable]
    write_csv(args.output_dir / "comparison.csv", csv_rows)

    bf = next(row for row in rows if row["method"] == "bf16_value_fp32_compute")
    w4 = sorted([row for row in rows if row["weight_bits"] == 4], key=rank_key)
    w3 = sorted([row for row in rows if row["weight_bits"] == 3], key=rank_key)
    lines = [
        "# Qwen2.5-VL Top4 W4A16/W3A16 本地评测报告", "",
        "> 活动协议：DocVQA_VAL、MMMU_DEV_VAL、ScienceQA-IMG/test、TextVQA_VAL；每项固定 64 样本。", "",
        "## DocVQA 指标修复", "",
        "- DocVQA 满分为 100：答案与预测先小写、去首尾空白并合并空白，取多参考答案最小归一 Levenshtein 距离；相似度低于 0.5 记 0，否则保留相似度。",
        "- 历史 `0.0936668` 是平均归一编辑距离（越低越好），曾被 `eval_score` 字段语义衔接错误当作正式高优分；现已禁止距离字段进入 ANLS 聚合。",
        f'- 修复后的未量化 BF16 64 样本成绩为 **{bf["DocVQA_VAL"]:.4f}/100**；完整集结果仅在完整集确认表中引用。', "",
        "## BF16", "",
    ]
    lines.extend(format_table([bf]))
    lines.extend(["", "## Top4 W4A16", ""]); lines.extend(format_table(w4))
    lines.extend(["", "## Top4 W3A16", ""]); lines.extend(format_table(w3))
    if full_confirmation:
        lines.extend(["", "## 完整集确认（与 64 样本筛选分表）", ""])
        lines.append("| method | format | DocVQA | MMMU | ScienceQA | TextVQA | 样本数 |")
        lines.append("|---|---|---:|---:|---:|---:|---|")
        for row in full_confirmation:
            score = row["scores"]
            counts = row["sample_counts"]
            lines.append(
                f"| {row['method']} | {row['format']} | {score['DocVQA_VAL']:.2f} | "
                f"{score['MMMU_DEV_VAL']:.2f} | {score['ScienceQA_TEST']:.2f} | "
                f"{score['TextVQA_VAL']:.2f} | "
                + ", ".join(f"{key}={value}" for key, value in counts.items()) + " |"
            )
    lines.extend(["", "## 64 样本 95% 置信区间", ""])
    lines.extend(format_ci_table([bf, *w4, *w3]))
    lines.extend(["", "## OCR 与推理敏感性", ""])
    lines.append("| method | format | OCR 平均保留率 | 推理平均保留率 | 差值（OCR-推理） |")
    lines.append("|---|---|---:|---:|---:|")
    for row in [*w4, *w3]:
        lines.append(
            f"| {row['method']} | {row['format']} | {row['ocr_retention']:.2f}% | "
            f"{row['reasoning_retention']:.2f}% | "
            f"{row['ocr_retention'] - row['reasoning_retention']:+.2f}pp |"
        )
    lines.extend(["", "## W4 → W3 配对变化", ""])
    lines.append("| method | DocVQA Δ | MMMU Δ | ScienceQA Δ | TextVQA Δ | 平均保留率 Δ |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    by_method = {(row["method"], row["weight_bits"]): row for row in rows}
    for method in ("mbq", "rtn", "omniquant", "squeezellm"):
        if (method, 4) not in by_method or (method, 3) not in by_method:
            continue
        left, right = by_method[(method, 4)], by_method[(method, 3)]
        changes = [right[name] - left[name] for name in ACTIVE_DATASET_NAMES]
        lines.append(
            f"| {method} | {changes[0]:+.2f} | {changes[1]:+.2f} | "
            f"{changes[2]:+.2f} | {changes[3]:+.2f} | "
            f"{right['mean_retention'] - left['mean_retention']:+.2f}pp |"
        )
    lines.extend(["", "## 对比分析", ""])
    if w4:
        lines.append(f'- W4A16 筛选第一名：{w4[0]["method"]}，最差保留率 {w4[0]["worst_retention"]:.2f}%。')
    if w3:
        lines.append(f'- W3A16 筛选第一名：{w3[0]["method"]}，最差保留率 {w3[0]["worst_retention"]:.2f}%。')
    if w4 and w3:
        lines.append(
            f"- 当前质量推荐分别为 W4A16 `{w4[0]['method']}` 与 W3A16 `{w3[0]['method']}`；"
            "最终部署选择应在完整集确认后结合 PPL 膨胀和逻辑 BPW 决定。"
        )
    lines.extend([
        "- DocVQA/TextVQA 代表 OCR 与文本理解，MMMU/ScienceQA 代表视觉推理；可按两类平均退化判断量化对识别和推理的差异敏感性。",
        "- 各数据集固定种子 bootstrap/Wilson 95% 区间完整记录在 summary.json。",
        "- W3/W4 当前均为解量化后的 CPU 质量模拟，不代表真实打包显存或 GPU 加速。", "",
    ])
    (args.output_dir / "report_zh.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
