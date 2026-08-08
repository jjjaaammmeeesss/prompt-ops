# @persistent — Track B：用 test-agent SystemComparator 对 v3.1 vs v2.0 做统计检验
"""Track B：消费 compare_v20_v31.py 的共享生成产物，做统计显著性检验。

复用 test-agent 的 SystemComparator.compare_from_reports()（bootstrap 95% CI +
Wilcoxon signed-rank + 配对 t 近似），对 GLM 五维分做逐维度显著性裁决。

说明（口径透明）：
- 这 10 个商务案例不在 test-agent 的 golden bank（golden bank 全是亲子案例），
  故 SystemComparator.compare()（校标锚定）无法直接跑；这里用 compare_from_reports()
  快速路径，吃两份由 GLM 五维分构造的 RunReport。
- 单弹窗场景 V5（序列节奏）、V6（去重）结构性不适用：V5 置 None（自动排除），V6 置 0.0 占位。
- 五维 → V1..V4 映射：V1=insight/5, V2=third_party/5, V3=language/5, V4=evidence/5。
- 分数归一化到 0-1（÷5），使 pass_rate(≥0.5) 与 interpretation 语义自洽。

Usage:
    python trackB_v20_v31.py --from-gen <共享生成JSON> [--n-judge 5]
输出：results/compare_tests/trackB_v20_v31_{ts}.json + .md
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results" / "compare_tests"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── 路径接线 ──
PROMPT_OPS_DIR = HERE
TEST_AGENT_ROOT = HERE.parent.parent.parent.parent / "星灵-soul-手搓" / "亲子沟通洞见" / "测试智能体"
for p in (str(PROMPT_OPS_DIR), str(TEST_AGENT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from compare_v20_v31 import (  # noqa: E402
    DIMS, VERSIONS, judge_popup, _resolve_keys,
)
from src.models import (  # noqa: E402
    EvalSummary, RunMetadata, RunReport,
)
from src.system_comparator import SystemComparator  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("trackB")

MAX_WORKERS = int(os.environ.get("GEN_WORKERS", "4"))

# 五维 → V1..V4 映射（V5/V6 不适用）
DIM_TO_V = {d: f"V{i}" for i, d in enumerate(("insight", "third_party", "language", "evidence"), start=1)}


def _judge_run(dialogue: str, popup: str | None, judge_key: str) -> dict | None:
    """单次评审；popup 为 None（未弹）时返回 None 表示该 run 失分。"""
    if popup is None:
        return None
    try:
        er = judge_popup(dialogue, popup, judge_key)
        if er.get("soft"):
            return er["soft"]
        return None
    except Exception as e:
        logger.error("评审失败: %s", e)
        return None


def build_summaries(gen: dict, judge_key: str) -> tuple[dict[str, EvalSummary], dict[str, dict]]:
    """对每个 case × 版本评审全部 run，返回 (summaries_by_version, per_case_dim_means)。"""
    # (ver, case_id, dialogue, popup)
    tasks = []
    for case_id, g in gen.items():
        for ver in ("v2.0", "v3.1"):
            for run in g[ver]:
                tasks.append((ver, case_id, g["dialogue"], run["popup"]))

    judged: dict[tuple, list] = {}
    def _one(t):
        ver, cid, dia, popup = t
        return (ver, cid), _judge_run(dia, popup, judge_key)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = [pool.submit(_one, t) for t in tasks]
        for fut in as_completed(futs):
            key, soft = fut.result()
            judged.setdefault(key, []).append(soft)

    summaries = {"v2.0": [], "v3.1": []}
    per_case = {"v2.0": {}, "v3.1": {}}

    for case_id in gen:
        for ver in ("v2.0", "v3.1"):
            softs = [s for s in judged.get((ver, case_id), []) if s is not None]
            n_runs = len(judged.get((ver, case_id), []))
            if not softs:
                # 全部未弹/失败 → 该 case 各维失分
                means = {d: 0.0 for d in DIMS}
            else:
                means = {d: round(sum(s.get(d, 3) for s in softs) / len(softs) / 5, 4)
                         for d in DIMS}
            per_case[ver][case_id] = means

            summaries[ver].append(EvalSummary(
                case_id=case_id,
                total_runs=max(1, n_runs),
                mean_v1=means["insight"],
                mean_v2=means["third_party"],
                mean_v3=means["language"],
                mean_v4=means["evidence"],
                mean_v5=None,      # 序列节奏：单弹窗不适用
                mean_v6=0.0,       # 去重：单弹窗占位
                case_score=round(sum(means[d] for d in DIMS) / len(DIMS), 4),
            ))

    return summaries, per_case


def build_report(metadata: dict, summaries: dict, per_case: dict) -> RunReport:
    return RunReport(
        metadata=RunMetadata(
            run_id=metadata["run_id"], timestamp=datetime.now(),
            sut_version=metadata["sut_version"], total_cases=len(per_case),
            runtime_seconds=metadata["elapsed_seconds"],
        ),
        summaries=summaries,
    )


def render_md(comp, per_case) -> str:
    lines = []
    lines.append("# Track B · v3.1 vs v2.0 统计检验\n")
    lines.append(f"> 生成 DeepSeek v4-pro（对应执行器）| 评审 GLM 5.2 | bootstrap 95% CI + Wilcoxon")
    lines.append(f"> 裁决：**{comp.winner}**（置信度 {comp.confidence}）\n")
    lines.append(comp.verdict_summary + "\n")
    lines.append("| 维度 | 语义 | v2.0 均值 | v3.1 均值 | Δ(v3.1−v2.0) | 95%CI | 显著 | Wilcoxon p |")
    lines.append("|------|------|:--:|:--:|:--:|:--:|:--:|:--:|")
    labels = {"V1": "洞察", "V2": "第三方立场", "V3": "语言", "V4": "证据", "V5": "序列节奏(N/A)", "V6": "去重(N/A)"}
    for dim in ("v1_comparison", "v2_comparison", "v3_comparison", "v4_comparison", "v5_comparison", "v6_comparison"):
        dc = getattr(comp, dim)
        key = dim.split("_")[0].upper()
        if dc is None:
            lines.append(f"| {key} | {labels[key]} | — | — | — | — | 无数据 | — |")
            continue
        sig = "✅" if dc.significant else "—"
        wp = f"{dc.wilcoxon_p_value:.4f}" if dc.wilcoxon_p_value is not None else "—"
        lines.append(
            f"| {key} | {labels[key]} | {dc.system_a_mean:.3f} | {dc.system_b_mean:.3f} "
            f"| {dc.delta:+.3f} | [{dc.ci_lower:+.3f},{dc.ci_upper:+.3f}] | {sig} | {wp} |"
        )
    lines.append("")
    lines.append("## 逐案例（归一化 0-1 均分）\n")
    lines.append("| case | v2.0 | v3.1 | Δ |")
    lines.append("|------|:--:|:--:|:--:|")
    for cid in sorted(per_case["v2.0"]):
        a = round(sum(per_case["v2.0"][cid].values()) / 4, 3)
        b = round(sum(per_case["v3.1"][cid].values()) / 4, 3)
        lines.append(f"| {cid} | {a:.3f} | {b:.3f} | {b - a:+.3f} |")
    lines.append("")
    lines.append("> 注：V5/V6 单弹窗结构性不适用。分数 = 五维/5 归一化。")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Track B：SystemComparator 统计检验")
    parser.add_argument("--from-gen", required=True, help="共享生成产物 JSON 路径")
    args = parser.parse_args()

    gen_path = Path(args.from_gen)
    if not gen_path.exists():
        logger.error("找不到共享生成产物: %s", gen_path)
        sys.exit(1)
    gen = json.loads(gen_path.read_text(encoding="utf-8"))

    _, judge_key = _resolve_keys()
    if not judge_key:
        logger.error("缺少评审 key")
        sys.exit(1)

    t0 = time.perf_counter()
    summaries, per_case = build_summaries(gen, judge_key)

    report_a = build_report(
        {"run_id": "v2.0", "sut_version": f"v2.0 (executor {VERSIONS['v2.0']['executor_version']})",
         "elapsed_seconds": round(time.perf_counter() - t0, 1)},
        summaries["v2.0"], per_case["v2.0"],
    )
    report_b = build_report(
        {"run_id": "v3.1", "sut_version": f"v3.1 (executor {VERSIONS['v3.1']['executor_version']})",
         "elapsed_seconds": round(time.perf_counter() - t0, 1)},
        summaries["v3.1"], per_case["v3.1"],
    )

    comparator = SystemComparator(golden_bank=None, quality_evaluator=None)
    comp = comparator.compare_from_reports(
        report_a, report_b,
        system_a_name="v2.0", system_b_name="v3.1",
        ci_level=0.95, n_bootstrap=2000,
    )

    elapsed = time.perf_counter() - t0
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_out = RESULTS_DIR / f"trackB_v20_v31_{ts}.json"
    json_out.write_text(comp.model_dump_json(indent=2), encoding="utf-8")

    md = render_md(comp, per_case)
    md_out = RESULTS_DIR / f"trackB_v20_v31_{ts}.md"
    md_out.write_text(md, encoding="utf-8")

    print("\n" + "=" * 64)
    print("  Track B · v3.1 vs v2.0 统计检验")
    print("=" * 64)
    print(f"  裁决: {comp.winner} (置信度 {comp.confidence})")
    print(f"  {comp.verdict_summary}")
    print()
    labels = {"V1": "洞察", "V2": "第三方立场", "V3": "语言", "V4": "证据", "V5": "序列(N/A)", "V6": "去重(N/A)"}
    for dim in ("v1_comparison", "v2_comparison", "v3_comparison", "v4_comparison", "v5_comparison", "v6_comparison"):
        dc = getattr(comp, dim)
        if dc is None:
            print(f"  {dim.split('_')[0].upper()} ({labels[dim.split('_')[0].upper()]}): 无数据")
            continue
        sig = "显著" if dc.significant else "不显著"
        print(f"  {dim.split('_')[0].upper()} ({labels[dim.split('_')[0].upper()]}): "
              f"v2.0={dc.system_a_mean:.3f} v3.1={dc.system_b_mean:.3f} Δ={dc.delta:+.3f} "
              f"CI[{dc.ci_lower:+.3f},{dc.ci_upper:+.3f}] {sig} wilcoxon_p={dc.wilcoxon_p_value}")
    print(f"\n  耗时 {elapsed:.0f}s")
    print(f"  JSON: {json_out}")
    print(f"  MD:   {md_out}")
    print("=" * 64)


if __name__ == "__main__":
    main()
