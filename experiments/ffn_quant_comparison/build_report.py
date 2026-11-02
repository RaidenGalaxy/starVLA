from __future__ import annotations

# Long report paragraphs intentionally remain single literals.
# ruff: noqa: E501, RUF001
import base64
import csv
import hashlib
import html
import io
import json
import math
import shutil
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import rankdata

from experiments.ffn_quant_compression.statistics import Bootstrap, interval, spearman_curves
from experiments.infinitevl_quant.progress import atomic_json, atomic_text, file_sha256, json_digest

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "outputs/ffn_quant_compression_v1"
NEW = ROOT / "outputs/ffn_quant_generalization_v1"
OUT = ROOT / "outputs/ffn_quant_cross_experiment_comparison_v1"
MODELS = ("infinitevl", "qwen25vl_3b", "qwen25vl_7b")
NAMES = ("InfiniteVL", "Qwen2.5-VL-3B", "Qwen2.5-VL-7B")
POLICIES = ("sensitive_first", "sensitive_protected")
COLORS = ("#C44E52", "#4C72B0")
SCOPES = ("old_pooled", "overall", "ChartQA_TEST", "GQA_TESTDEV_BALANCED")
PLOT_SCOPES = ("old_pooled", "ChartQA_TEST", "GQA_TESTDEV_BALANCED")
TITLES = (
    "Old four-task mixture\nscreen32, 8/task",
    "ChartQA\nnew screen32, n=16",
    "GQA\nnew screen32, n=16",
)


def read(path):
    return json.loads(path.read_text())


def write_csv(path, rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, stream.getvalue())


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (a, b)


def render_text(lines):
    result = []
    table = False
    for line in lines:
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(set(c) <= set("-:") for c in cells):
                continue
            first = not table
            if first:
                result.append("<table>")
                table = True
            tag = "th" if first else "td"
            result.append("<tr>" + "".join(f"<{tag}>{html.escape(c)}</{tag}>" for c in cells) + "</tr>")
            continue
        if table:
            result.append("</table>")
            table = False
        if line.startswith("# "):
            result.append("<h1>" + html.escape(line[2:]) + "</h1>")
        elif line.startswith("## "):
            result.append("<h2>" + html.escape(line[3:]) + "</h2>")
        elif line:
            result.append("<p>" + html.escape(line) + "</p>")
    if table:
        result.append("</table>")
    return "".join(result)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source_paths = [
        OLD / "report/analysis_snapshot.json",
        OLD / "report/curve_points.csv",
        OLD / "verification.json",
        NEW / "report/analysis.json",
        NEW / "report/curve_points.csv",
        NEW / "audit/locked_design.json",
        NEW / "verification.json",
        ROOT / "experiments/ffn_quant_compression/statistics.py",
    ]
    sources = {str(p): file_sha256(p) for p in source_paths}
    old, new = read(source_paths[0]), read(source_paths[3])
    design = read(NEW / "audit/locked_design.json")
    assert not old["partial"] and new["core_complete"] and not new["partial"]
    assert new["verification"]["checked_rows"] == 11104 and new["verification"]["qa_rescored"] == 480
    assert read(NEW / "report/final_identity.json")["analysis_sha256"] == file_sha256(NEW / "analyze/summary.json")
    old_export = read(OLD / "report/figure_export.json")
    for name, digest in old_export["files"].items():
        assert file_sha256(OLD / "report" / name) == digest
    for name, digest in read(NEW / "report/artifacts.json").items():
        assert file_sha256(NEW / "report" / name) == digest

    old_csv = list(csv.DictReader((OLD / "report/curve_points.csv").open()))
    new_csv = list(csv.DictReader((NEW / "report/curve_points.csv").open()))
    for row in old_csv:
        assert int(row["samples"]) == (32 if row["cohort"] == "screen32" else 192)
        close(float(row["compression_percent"]), 75 * int(row["q"]) / 36)
        close(float(row["nll"]), float(row["total_nll"]) / int(row["target_tokens"]))
    for row in new_csv:
        layers = new["models"][row["model"]]["layers"]
        close(float(row["coverage_percent"]), 100 * int(row["q"]) / layers)
        close(float(row["bf16_saving_percent"]), 0.75 * float(row["coverage_percent"]))
        close(float(row["nll"]), float(row["total_nll"]) / int(row["target_tokens"]))

    bridge = {(r["model"], r["scope"], r["mask_id"]): r for r in new["old_bridge"]}
    curves, stats, points = {}, [], []
    for kind in MODELS:
        model = design["models"][kind]
        layers = new["models"][kind]["layers"]
        for policy in POLICIES:
            old_values = [bridge[kind, "overall", model["aliases"][policy][str(q)]] for q in range(layers + 1)]
            curves[kind, "old_pooled", policy] = [p["nll"] for p in old_values]
            if kind != "qwen25vl_7b":
                legacy = "qwen" if kind == "qwen25vl_3b" else kind
                original = old["models"][legacy]["cohorts"]["screen32"]
                v = original["correlations"][policy + ":nll"]
                stats.append(
                    {
                        "model": kind,
                        "cohort": "old_screen32",
                        "scope": "old_pooled",
                        "strategy": policy,
                        "rho": v["rho"],
                        "ci95_low": v["ci95"][0],
                        "ci95_high": v["ci95"][1],
                        "history_reused": True,
                    }
                )
                for q, p in enumerate(original["policy_points"][policy]):
                    close(p["metrics"]["nll"], old_values[q]["nll"])
                    matches = [
                        r
                        for r in old_csv
                        if r["model"] == legacy
                        and r["cohort"] == "screen32"
                        and r["policy"] == policy
                        and int(r["q"]) == q
                    ]
                    assert len(matches) == 1
                    close(float(matches[0]["nll"]), old_values[q]["nll"])
            for scope in SCOPES[1:]:
                current = new["models"][kind]["cohorts"]["screen32"]["scopes"][scope]["curves"][policy]
                assert [x["q"] for x in current["points"]] == list(range(layers + 1))
                curves[kind, scope, policy] = [p["metrics"]["nll"] for p in current["points"]]
            for cohort, cohort_data in new["models"][kind]["cohorts"].items():
                for scope, result in cohort_data["scopes"].items():
                    v = result["curves"][policy]["spearman"]
                    stats.append(
                        {
                            "model": kind,
                            "cohort": cohort,
                            "scope": scope,
                            "strategy": policy,
                            "rho": v["rho"],
                            "ci95_low": v["ci95"][0],
                            "ci95_high": v["ci95"][1],
                            "history_reused": False,
                        }
                    )
            for scope in SCOPES:
                y = curves[kind, scope, policy]
                assert len(y) == layers + 1 and np.isfinite(y).all()
                for q, value in enumerate(y):
                    points.append(
                        {
                            "model": kind,
                            "cohort": "old_screen32" if scope == "old_pooled" else "new_screen32",
                            "scope": scope,
                            "strategy": policy,
                            "q": q,
                            "layers": layers,
                            "coverage_percent": 100 * q / layers,
                            "bf16_saving_percent": 75 * q / layers,
                            "nll": value,
                            "reference_nll": y[0],
                            "delta_nll": value - y[0],
                        }
                    )

    # The 7B old-data curves were newly measured; use paired image draws, not layers, for their CIs.
    kind = "qwen25vl_7b"
    rows_by_label = {}
    stage_ids = {stage: json_digest(read(NEW / stage / "identity.json")) for stage in ("bridge", "old_supplement")}
    for label in design["models"][kind]["main"]:
        stage = "bridge" if label in design["models"][kind]["bridge"] else "old_supplement"
        values = [read(p) for p in (NEW / stage / f"{kind}__oldscreen32__{label}/samples").glob("*.json")]
        assert len(values) == len({x["id"] for x in values}) == 32
        for row in values:
            assert row["provenance"]["stage_identity"] == stage_ids[stage]
            assert row["row_sha256"] == json_digest({k: v for k, v in row.items() if k != "row_sha256"})
            t = row["target_text"]
            assert math.isclose(math.fsum(t["token_losses"]), t["total_nll"], rel_tol=1e-6, abs_tol=1e-6)
        rows_by_label[label] = values
    baseline_inputs = {r["id"]: r for r in rows_by_label["B"]}
    for values in rows_by_label.values():
        assert {r["id"] for r in values} == set(baseline_inputs)
        for row in values:
            baseline = baseline_inputs[row["id"]]
            assert row["input_digest"] == baseline["input_digest"]
            for field in ("positions", "token_ids"):
                assert row["target_text"][field] == baseline["target_text"][field]
    boot = Bootstrap(rows_by_label["B"], 10000)
    dist = {label: boot.summarize(values)[1]["nll"] for label, values in rows_by_label.items()}
    for policy in POLICIES:
        labels = [design["models"][kind]["aliases"][policy][str(q)] for q in range(29)]
        values = np.stack([dist[label] for label in labels], axis=1)
        rho = spearman_curves(curves[kind, "old_pooled", policy])[0]
        draws = spearman_curves(values)
        ranks = rankdata(np.round(values, 12), axis=1)
        ranks -= ranks.mean(1, keepdims=True)
        x = np.arange(29) - 14
        oracle = (ranks * x).sum(1) / np.sqrt((ranks * ranks).sum(1) * (x * x).sum())
        assert np.allclose(interval(draws), interval(oracle), atol=1e-12, rtol=0)
        ci = interval(draws)
        stats.append(
            {
                "model": kind,
                "cohort": "old_screen32",
                "scope": "old_pooled",
                "strategy": policy,
                "rho": float(rho),
                "ci95_low": ci[0],
                "ci95_high": ci[1],
                "history_reused": False,
            }
        )

    endpoints = []
    contributions = []
    for kind in MODELS:
        for scope in SCOPES:
            y = curves[kind, scope, POLICIES[0]]
            endpoints.append(
                {
                    "model": kind,
                    "cohort": "old_screen32" if scope == "old_pooled" else "new_screen32",
                    "scope": scope,
                    "reference_nll": y[0],
                    "w4_nll": y[-1],
                    "delta_nll": y[-1] - y[0],
                    "relative_change_percent": 100 * (y[-1] / y[0] - 1),
                }
            )
        for cohort in ("screen32", "confirm128"):
            scopes = new["models"][kind]["cohorts"][cohort]["scopes"]
            pooled = scopes["overall"]["points"]
            total = pooled["B"]["target_tokens"]
            acc = 0
            for scope in ("ChartQA_TEST", "GQA_TESTDEV_BALANCED"):
                p = scopes[scope]["points"]
                weight = p["B"]["target_tokens"] / total
                delta = p["W"]["nll"] - p["B"]["nll"]
                contributions.append(
                    {
                        "model": kind,
                        "cohort": cohort,
                        "scope": scope,
                        "target_tokens": p["B"]["target_tokens"],
                        "total_tokens": total,
                        "token_weight": weight,
                        "dataset_delta_nll": delta,
                        "weighted_contribution": weight * delta,
                    }
                )
                acc += weight * delta
            close(acc, pooled["W"]["nll"] - pooled["B"]["nll"])
    assert len(points) == 824
    write_csv(OUT / "unified_curve_points.csv", points)
    write_csv(OUT / "correlations_with_ci.csv", stats)
    write_csv(OUT / "endpoint_comparison.csv", endpoints)
    write_csv(OUT / "pooling_decomposition.csv", contributions)

    pdf = PdfPages(OUT / "figures_and_comparison.pdf")
    for delta, filename in ((True, "comparison_delta_nll"), (False, "comparison_nll")):
        fig, axes = plt.subplots(3, len(PLOT_SCOPES), figsize=(15, 11), sharex=True, sharey="col" if delta else False)
        for i, kind in enumerate(MODELS):
            layers = new["models"][kind]["layers"]
            for j, scope in enumerate(PLOT_SCOPES):
                ax = axes[i, j]
                for policy, color, label in zip(
                    POLICIES, COLORS, ("Sensitive first", "Sensitive protected"), strict=True
                ):
                    y = np.array(curves[kind, scope, policy])
                    baseline = y[0]
                    ax.plot(
                        np.arange(layers + 1) * 100 / layers,
                        y - baseline if delta else y,
                        color=color,
                        marker="o",
                        ms=2.6,
                        lw=1.2,
                        label=label,
                    )
                baseline = curves[kind, scope, POLICIES[0]][0]
                ax.axhline(0 if delta else baseline, ls=":", color="black", lw=1)
                if scope != "old_pooled":
                    random = new["models"][kind]["cohorts"]["screen32"]["scopes"][scope]["random"]
                    for item in random.values():
                        shift = baseline if delta else 0
                        ax.plot(item["coverage_percent"], item["mean"] - shift, "s", ms=4, color="#777777")
                        ax.vlines(
                            item["coverage_percent"], *[v - shift for v in item["seed_range"]], color="#777777", lw=2.3
                        )
                ax.set(xlim=(-1, 101), xticks=(0, 25, 50, 75, 100))
                ax.grid(alpha=0.2)
                if i == 0:
                    ax.set_title(TITLES[j], fontsize=10)
                if j == 0:
                    ax.set_ylabel(NAMES[i] + "\n" + ("Delta NLL (nats/token)" if delta else "NLL (nats/token)"))
                if i == 2:
                    ax.set_xlabel("FFN quantization coverage (%)")
                if i == 2 and j == 0:
                    ax.text(0.03, 0.93, "7B old cohort: newly measured", transform=ax.transAxes, fontsize=8, va="top")
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(
            "Old vs new W4 FFN experiments | "
            + ("loss change from each panel's reference" if delta else "original target NLL"),
            fontsize=15,
        )
        fig.text(
            0.5,
            0.008,
            "Old axis rescaled: coverage = BF16 saving / 0.75. Point estimates only; no smoothing. Gray bars: 3-seed range, not sample CI.\n"
            + ("Y scales shared within columns. " if delta else "Y scales vary by panel. ")
            + "Old mixture: DocVQA/MMMU/ScienceQA/TextVQA; two original models reused. See companion CI and independent-confirmation results.",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.055, 1, 0.94))
        for ext in ("png", "svg"):
            fig.savefig(OUT / f"{filename}.{ext}", dpi=170, bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, cohort in zip(axes, ("screen32", "confirm128"), strict=True):
        sums = []
        for i, kind in enumerate(MODELS):
            vals = [x["weighted_contribution"] for x in contributions if x["model"] == kind and x["cohort"] == cohort]
            for value, color, label, offset in zip(
                vals, ("#D79B46", "#549A88"), ("ChartQA contribution", "GQA contribution"), (-0.16, 0.16), strict=True
            ):
                ax.bar(i + offset, value, width=0.3, color=color, label=label if i == 0 else None)
                ax.text(
                    i + offset,
                    value + (0.025 if value >= 0 else -0.03),
                    f"{value:+.3f}",
                    ha="center",
                    va="bottom" if value >= 0 else "top",
                    fontsize=8,
                )
            sums.append(sum(vals))
        ax.plot(range(3), sums, "kD", ms=5, label="Pooled delta NLL")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(3), NAMES, rotation=10)
        ax.set_title(cohort + " | actual target-token weights")
        ax.set_ylabel("Contribution to full-W4 minus reference NLL")
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Why a pooled NLL can hide opposite task effects")
    fig.text(
        0.5,
        0.01,
        "Pooled delta = sum of dataset delta x its target-token fraction. Positive means higher loss. Point estimates; cohorts kept separate.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"pooling_decomposition.{ext}", dpi=180, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    originals = [
        (OLD, "nll_vs_compression", "Old NLL: the two original model panels"),
        (NEW, "nll_layer_sweep", "New layerwise NLL: 37 / 37 / 29 points"),
        (NEW, "nll_independent_confirmation", "Independent confirmation: 128 image-disjoint samples"),
        (NEW, "marginal_layer_damage", "New marginal damage: adding one W4 layer"),
        (NEW, "qa_sanity_check", "New QA32: directional check only"),
        (OLD, "qa_vs_compression", "Old QA192: a different cohort and scoring mix"),
    ]
    for src, name, title in originals:
        prefix = "old" if src == OLD else "new"
        for ext in ("png", "svg"):
            p = src / "report" / f"{name}.{ext}"
            sources[str(p)] = file_sha256(p)
            shutil.copy2(p, OUT / f"{prefix}_{name}.{ext}")
        fig, ax = plt.subplots(figsize=(13, 10))
        ax.imshow(plt.imread(src / "report" / f"{name}.png"))
        ax.axis("off")
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        pdf.savefig(fig)
        plt.close(fig)
    pdf.close()

    inf = new["models"]["infinitevl"]["cohorts"]["confirm128"]["scopes"]
    dc = inf["ChartQA_TEST"]["points"]["W"]["nll"] - inf["ChartQA_TEST"]["points"]["B"]["nll"]
    dg = inf["GQA_TESTDEV_BALANCED"]["points"]["W"]["nll"] - inf["GQA_TESTDEV_BALANCED"]["points"]["B"]["nll"]
    text = [
        "# 新旧FFN W4实验图表联合分析",
        "",
        "所有横轴统一为FFN量化覆盖率P=100q/L（0–100%）。旧图横轴为BF16理论权重节省率C=0.75P（0–75%），只是线性换算，不改变NLL、相关系数或原有结论。所有实验均为CPU FP32上的W4 asymmetric RTN、g128 fake-quant。",
        "",
        "## 结论",
        "",
        "1. 旧任务组合上的正相关没有成为跨任务统一规律。InfiniteVL在ChartQA复现NLL上升，在GQA则稳定下降；Qwen 3B的ChartQA筛选下降趋势区间跨0且独立确认转为上升；7B确认集的相关区间跨0。应区分相关方向、效应幅度和样本不确定性。",
        "",
        "2. 总体平均可能掩盖相反的任务变化。InfiniteVL旧screen32全W4的总体ΔNLL为+0.39020，新screen32为−0.74316。独立confirm128中ChartQA的ΔNLL为+%.5f、GQA为%.5f；按225/317、92/317目标token加权，贡献分别为+%.5f、%.5f，总体为%.5f。总体NLL下降并不意味着两个任务同时改善。"
        % (dc, dg, dc * 225 / 317, dg * 92 / 317, dc * 225 / 317 + dg * 92 / 317),
        "",
        "3. 层分配效应在InfiniteVL/ChartQA复现，但不是跨任务通用保护方案。旧确认集4个比例的NLL保护优势均显著；新ChartQA的25/50/75/97.22%也均通过36项Holm校正（p=0.00360），GQA的25/50/75%却是保护更差。新两个Qwen均未检出校正后分配差异；未显著不等于等效。旧3B仅保护最敏感1层的NLL分配优势曾显著，新任务未复现这一显著性。",
        "",
        "4. 非单调与背景交互是可复现现象。旧3B在50%覆盖率的NLL为0.68969，全W4反而为0.64879。新各模型也都有新增一层后NLL下降的步骤。InfiniteVL/ChartQA在97.22%覆盖率时，仅保护最敏感第35层的NLL比敏感先压更低；新确认值为0.55482对0.84152，差0.28670（95%CI[0.22128,0.35378]）。这支持已测背景中的层选择作用，不是单层固有敏感度或全局最优证明。",
        "",
        "5. 随机对照补上了旧实验的证据缺口，但收益仍不普遍。新screen32中，InfiniteVL/ChartQA在75%覆盖率的保护相对随机均值收益CI为[0.09612,0.31895]，25/50%区间跨0；GQA三个比例方向相反。该随机比较是筛选证据。",
        "",
        "6. NLL下降不等于QA提高。同一QA核验子集上，InfiniteVL/GQA的NLL由6.37939降至3.78034，参考与全W4的QA分数均为50%。目标token概率改善可以仍不足以改变自由生成答案；QA样本较小，不能把未检出变化当作等效。旧QA192与新QA32的样本和评分任务不同，百分数不能直接横向比较。",
        "",
        "7. 7B在已测检查点上变化较小，但不能归因为纯模型规模因果效应。同一旧screen32上，3B全W4 ΔNLL为+0.07366，7B为−0.04559；新确认总体分别为+0.09815、−0.00506。7B确认相关区间跨0，仍需更多独立样本判断稳定性。",
        "",
        "## 三模型端点对比",
        "",
        "下表均为screen32的全W4减各自参考的绝对NLL变化；正值为损失增加。旧列是四任务混合，新总体是ChartQA/GQA混合，不能把旧总体和新总体当作相同任务。",
        "",
        "| 模型 | 旧四任务总体ΔNLL | 新两任务总体ΔNLL | 新ChartQA ΔNLL | 新GQA ΔNLL |",
        "|---|---:|---:|---:|---:|",
    ]
    for kind, name in zip(MODELS, NAMES, strict=True):
        vals = [
            next(x["delta_nll"] for x in endpoints if x["model"] == kind and x["scope"] == scope) for scope in SCOPES
        ]
        text.append("| " + name + " | " + " | ".join(f"{v:+.5f}" for v in vals) + " |")
    text += [
        "",
        "## 相关性与不确定性",
        "",
        "以下并列敏感先压/敏感保护的Spearman及95%配对bootstrap区间。旧7B为本次在旧数据上的新补测，并非历史已有曲线；旧数据包含排序样本。",
        "",
        "| 模型 | 数据范围 | 策略 | rho | 95%CI |",
        "|---|---|---|---:|---|",
    ]
    for r in stats:
        if r["cohort"] == "confirm128":
            continue
        text.append(
            f"| {NAMES[MODELS.index(r['model'])]} | {r['cohort']}/{r['scope']} | {r['strategy']} | {r['rho']:.4f} | [{r['ci95_low']:.4f}, {r['ci95_high']:.4f}] |"
        )
    text += [
        "",
        "## 核算与边界",
        "",
        "所有原始实验目录只读；824个汇总曲线点与原报告逐点核对，其中三列对照图显示618个点，仅保留旧四任务、ChartQA、GQA；新两任务总体数据保留于CSV及加权分解图。旧横轴换算、层数、分子分母及跨图端点一致性均通过。旧7B的56×32条补测记录验证了逐token损失、缓存身份和配对样本，并用10,000次分层配对bootstrap及独立秩实现复核相关性区间。旧两模型曲线是明确标记的历史复用，不视为新增独立证据。",
        "",
        "旧16项与新36项Holm校正属于不同预设检验家族；不比较p值大小来判断哪轮证据更强。新主扫描每任务16图，新独立确认每任务64图；不能把每层或曲线点当独立样本。结论限定为这些模型、任务、冻结排序和掩码，不解释为量化提升通用能力、原生低比特速度或内存收益。",
        "",
        "图表：comparison_delta_nll.png统一显示相对参考的损失变化（同列同尺度），comparison_nll.png保留原NLL（面板尺度不同）；图中为点估计，原图与CSV另有不确定性。pooling_decomposition.png解释总体平均。figures_and_comparison.pdf汇总三幅新比较图及六组原图；index.html为离线总览。",
    ]
    atomic_text(OUT / "comparison_zh.md", "\n".join(text) + "\n")
    paragraphs = render_text(text[: text.index("## 相关性与不确定性")])

    def img(path):
        return '<img loading="lazy" src="data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode() + '">'

    body = (
        paragraphs
        + "<h2>统一后的损失变化</h2>"
        + img(OUT / "comparison_delta_nll.png")
        + "<h2>总体NLL分解</h2>"
        + img(OUT / "pooling_decomposition.png")
        + "<details><summary>原始NLL量纲对照</summary>"
        + img(OUT / "comparison_nll.png")
        + "</details><details><summary>六组原图汇总</summary>"
    )
    for src, name, title in originals:
        prefix = "old" if src == OLD else "new"
        body += "<h3>" + html.escape(title) + "</h3>" + img(OUT / f"{prefix}_{name}.png")
    body += "</details><p>完整论证和核算边界见comparison_zh.md，原始数值见同目录CSV。</p>"
    atomic_text(
        OUT / "index.html",
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>FFN新旧实验对照</title><style>body{max-width:1500px;margin:36px auto;padding:0 24px;font:17px/1.7 system-ui;color:#17202a}img{width:100%;height:auto;margin:20px 0}h1,h2{line-height:1.3}summary{cursor:pointer;font-weight:bold}details{border:1px solid #ddd;padding:16px;margin:16px 0}p{max-width:1100px}table{border-collapse:collapse;margin:20px 0;font-variant-numeric:tabular-nums}td,th{border:1px solid #ccd;padding:8px 16px;text-align:right}td:first-child,th:first-child{text-align:left}</style>'
        + body
        + "</html>",
    )
    for p, digest in sources.items():
        assert file_sha256(Path(p)) == digest, p
    verification = {
        "passed": True,
        "new_inference": False,
        "unified_points": len(points),
        "plotted_scopes": list(PLOT_SCOPES),
        "plotted_points": sum(row["scope"] in PLOT_SCOPES for row in points),
        "old_csv_points_checked": len(old_csv),
        "new_csv_points_checked": len(new_csv),
        "old_7b_rows_verified": 1792,
        "old_7b_bootstrap": 10000,
        "axis": "P=100*q/L; old C=0.75*P",
        "original_sources_unchanged": True,
        "sources_sha256": sources,
        "code_sha256": file_sha256(Path(__file__)),
    }
    atomic_json(OUT / "verification.json", verification)
    files = {
        p.name: file_sha256(p)
        for p in OUT.iterdir()
        if p.is_file() and p.name not in ("comparison_bundle.zip", "artifacts.json")
    }
    atomic_json(OUT / "artifacts.json", files)
    with zipfile.ZipFile(OUT / "comparison_bundle.zip", "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.iterdir()):
            if p.is_file() and p.name != "comparison_bundle.zip":
                z.write(p, p.name)
    with zipfile.ZipFile(OUT / "comparison_bundle.zip") as z:
        for name, digest in files.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest
    print(
        json.dumps(
            {
                "output": str(OUT),
                "verification": {k: v for k, v in verification.items() if k != "sources_sha256"},
                "old_7b_correlations": [
                    r for r in stats if r["cohort"] == "old_screen32" and r["model"] == "qwen25vl_7b"
                ],
                "bundle_bytes": (OUT / "comparison_bundle.zip").stat().st_size,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
