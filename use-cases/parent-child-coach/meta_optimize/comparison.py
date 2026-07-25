"""Level 2 机制比较：四轮协议 + test-set 终局对比。

四轮协议（来自 Bilevel Autoresearch）:
  Round 1: Explore — 读三方 trace，逐 case 对比
  Round 2: Critique — 交叉验证 + 攻击性验证
  Round 3: Specify — 推荐方案 / 条件路由 / 融合方案
  Round 4: Generate — test-set 终局对比 + 对比矩阵报告
"""

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from candidate_store import (
    CANDIDATES_DIR, TraceEntry, MetricsSnapshot,
    load_trace, load_metrics, get_best_candidate,
)
from dataset_split import load_split_manifest, get_eval_entries_for_split


# ═══════════════════════════════════════════════════════════════
# 对比结果数据结构
# ═══════════════════════════════════════════════════════════════

@dataclass
class StrategyComparison:
    """单个策略的对比快照。"""
    strategy: str
    best_candidate_id: str = ""
    overall_score: float = 0.0
    m5_match: float = 0.0
    m6_insight: float = 0.0
    m7_safety: float = 0.0
    m5_failure_cases: list[str] = field(default_factory=list)
    m5_only_success_cases: list[str] = field(default_factory=list)
    # 成本估算
    estimated_llm_calls_per_window: float = 0.0


@dataclass
class CaseLevelComparison:
    """逐 case 对比矩阵中的一行。"""
    case_id: str
    senate_tone: str = ""
    senate_pass: bool = False
    teacher_student_tone: str = ""
    teacher_student_pass: bool = False
    saga_tone: str = ""
    saga_pass: bool = False
    all_pass: bool = False
    none_pass: bool = False
    gold_tone: str = ""


@dataclass
class ComparisonReport:
    """L2 终局对比报告。"""
    timestamp: str = ""
    strategies: list[StrategyComparison] = field(default_factory=list)
    case_matrix: list[CaseLevelComparison] = field(default_factory=list)
    recommendation: str = ""
    condition_routing: dict = field(default_factory=dict)
    fusion_proposal: str = ""


# ═══════════════════════════════════════════════════════════════
# Round 1: Explore — 读三方 trace
# ═══════════════════════════════════════════════════════════════

def explore_traces(strategies: list[str]) -> dict[str, list[TraceEntry]]:
    """读取各策略最佳候选的 trace，返回 {strategy: trace_entries}。

    这个是本地分析步骤，不需要 LLM。
    """
    all_traces: dict[str, list[TraceEntry]] = {}
    for s in strategies:
        best = get_best_candidate(s)
        # 无候选时回退到 baseline
        if not best:
            baseline = CANDIDATES_DIR / s / "baseline"
            if baseline.exists() and (baseline / "metrics.json").exists():
                best = baseline
        if best:
            all_traces[s] = load_trace(best)
            print(f"[Explore] {s}: {len(all_traces[s])} trace entries "
                  f"from {best.name}")
        else:
            all_traces[s] = []
            print(f"[Explore] {s}: 无候选")
    return all_traces


# ═══════════════════════════════════════════════════════════════
# 逐 case 对比矩阵
# ═══════════════════════════════════════════════════════════════

def build_case_matrix(
    all_traces: dict[str, list[TraceEntry]],
) -> list[CaseLevelComparison]:
    """构建逐 case M5 对比矩阵。

    对于每个 case_id，检查三策略是否都 M5 正确。
    """
    # 收集所有 case_id
    all_case_ids = set()
    for entries in all_traces.values():
        for e in entries:
            all_case_ids.add(e.case_id)

    # 构建每策略的 case → pass 映射
    pass_map: dict[str, dict[str, dict]] = {}
    for s, entries in all_traces.items():
        pass_map[s] = {}
        for e in entries:
            pass_map[s][e.case_id] = {
                "pass": e.m5_tone_match == 1.0,
                "sys_tone": e.sys_tone,
                "gold_tone": e.gold_tone,
            }

    matrix = []
    for cid in sorted(all_case_ids):
        row = CaseLevelComparison(case_id=cid)

        s_info = pass_map.get("senate", {}).get(cid, {})
        ts_info = pass_map.get("teacher_student", {}).get(cid, {})
        saga_info = pass_map.get("saga", {}).get(cid, {})

        row.senate_pass = s_info.get("pass", False)
        row.senate_tone = s_info.get("sys_tone", "-")
        row.teacher_student_pass = ts_info.get("pass", False)
        row.teacher_student_tone = ts_info.get("sys_tone", "-")
        row.saga_pass = saga_info.get("pass", False)
        row.saga_tone = saga_info.get("sys_tone", "-")
        row.gold_tone = (s_info.get("gold_tone") or ts_info.get("gold_tone") or
                         saga_info.get("gold_tone") or "-")

        row.all_pass = row.senate_pass and row.teacher_student_pass and row.saga_pass
        row.none_pass = (not row.senate_pass and not row.teacher_student_pass
                         and not row.saga_pass)

        matrix.append(row)

    return matrix


# ═══════════════════════════════════════════════════════════════
# Round 2: Critique — 交叉验证
# ═══════════════════════════════════════════════════════════════

CRITIQUE_SYSTEM_PROMPT = """你是多智能体架构评估专家。基于三个候选策略的逐 case trace 对比，做攻击性验证：

1. 哪些 case 只有一个策略能解决？（策略特异性优势）
2. 哪些 case 所有策略都失败？（共同天花板——可能是 judge/dataset/model 问题）
3. 每个策略最脆弱的场景是什么？（预测它在哪些 case 上最容易失败）
4. 交叉验证：如果策略 A 的 harness 用在策略 B 的失败 case 上，会改善吗？（还是架构问题？）
"""


def build_critique_prompt(
    matrix: list[CaseLevelComparison],
    strategy_summaries: dict[str, MetricsSnapshot],
) -> str:
    """构建 Round 2 Critique 的 prompt。"""
    # 统计
    only_senate = [r.case_id for r in matrix if r.senate_pass and not r.teacher_student_pass and not r.saga_pass]
    only_ts = [r.case_id for r in matrix if not r.senate_pass and r.teacher_student_pass and not r.saga_pass]
    only_saga = [r.case_id for r in matrix if not r.senate_pass and not r.teacher_student_pass and r.saga_pass]
    all_fail = [r.case_id for r in matrix if r.none_pass]
    all_pass = [r.case_id for r in matrix if r.all_pass]

    prompt = f"""## 三策略对比矩阵

| case | Senate | Teacher-Student | SAGA | Gold | 模式 |
|------|--------|-----------------|------|------|------|
"""
    for r in matrix:
        s = "✓" if r.senate_pass else "✗"
        ts = "✓" if r.teacher_student_pass else "✗"
        sg = "✓" if r.saga_pass else "✗"
        pattern = ""
        if r.all_pass:
            pattern = "全过"
        elif r.none_pass:
            pattern = "全败"
        elif r.senate_pass and not r.teacher_student_pass and not r.saga_pass:
            pattern = "仅Senate"
        elif not r.senate_pass and r.teacher_student_pass and not r.saga_pass:
            pattern = "仅T-S"
        elif not r.senate_pass and not r.teacher_student_pass and r.saga_pass:
            pattern = "仅SAGA"
        prompt += f"| {r.case_id} | {s} | {ts} | {sg} | {r.gold_tone} | {pattern} |\n"

    prompt += f"""
## 统计摘要

- 全过 case: {len(all_pass)} — {', '.join(all_pass[:10])}
- 全败 case: {len(all_fail)} — {', '.join(all_fail[:10])}
- 仅 Senate 过: {len(only_senate)} — {', '.join(only_senate[:10])}
- 仅 Teacher-Student 过: {len(only_ts)} — {', '.join(only_ts[:10])}
- 仅 SAGA 过: {len(only_saga)} — {', '.join(only_saga[:10])}

## 各策略指标

"""
    for s_name, m in strategy_summaries.items():
        prompt += (f"- **{s_name}**: overall={m.overall_score:.3f}, "
                   f"M5={m.aggregate_m5:.1%}, "
                   f"M6={m.aggregate_m6:.2f}, M7={m.aggregate_m7:.2f}\n")

    prompt += """
请分析：
1. 哪些 case 的失败是 prompt 可修的，哪些是架构级天花板？
2. 三个策略是否存在互补关系（A 适合 X 场景、B 适合 Y 场景）？
3. 交叉验证预测：如果让 Senate 的 Speaker 接收 T-S 的 Validator 反馈，哪些 case 可能改善？
"""
    return prompt


# ═══════════════════════════════════════════════════════════════
# 综合报告生成
# ═══════════════════════════════════════════════════════════════

def generate_comparison_report(
    all_traces: dict[str, list[TraceEntry]],
    matrix: list[CaseLevelComparison],
    critique_result: str = "",
) -> ComparisonReport:
    """生成 L2 终局对比报告。

    Args:
        all_traces: {strategy: trace entries from best candidate}
        matrix: 逐 case 对比矩阵
        critique_result: Round 2 Critique 的 LLM 输出（可选，没有时只做统计）

    Returns:
        ComparisonReport
    """
    report = ComparisonReport(
        timestamp=datetime.now().isoformat(),
    )

    # ── 策略级对比 ──────────────────────────────
    for strategy, entries in all_traces.items():
        m5_pass = [e for e in entries if e.m5_tone_match == 1.0]
        m5_fail = [e for e in entries if e.m5_tone_match == 0.0]
        m6_scores = [e.m6_insight_score for e in entries if e.m6_insight_score is not None]
        m7_scores = [e.m7_safety_score for e in entries if e.m7_safety_score is not None]

        # 找到哪些 case 仅此策略通过
        only_this = []
        for r in matrix:
            if strategy == "senate" and r.senate_pass and not r.teacher_student_pass and not r.saga_pass:
                only_this.append(r.case_id)
            elif strategy == "teacher_student" and not r.senate_pass and r.teacher_student_pass and not r.saga_pass:
                only_this.append(r.case_id)
            elif strategy == "saga" and not r.senate_pass and not r.teacher_student_pass and r.saga_pass:
                only_this.append(r.case_id)

        # 成本估算
        cost_map = {
            "senate": 5.0,          # 3专家 + Speaker + Production = 5 calls
            "teacher_student": 4.0,  # Teacher + Student + Validator + (avg 1 retry) = 4 calls
            "saga": 5.0,            # Fast Path (3) + Deep Review (2) = 5 calls (同步版)
        }

        report.strategies.append(StrategyComparison(
            strategy=strategy,
            best_candidate_id="",
            overall_score=round(len(m5_pass) / max(len(entries), 1), 3),
            m5_match=round(len(m5_pass) / max(len(entries), 1), 3),
            m6_insight=round(sum(m6_scores) / max(len(m6_scores), 1), 2) if m6_scores else 0.0,
            m7_safety=round(sum(m7_scores) / max(len(m7_scores), 1), 2) if m7_scores else 0.0,
            m5_failure_cases=[e.case_id for e in m5_fail],
            m5_only_success_cases=only_this,
            estimated_llm_calls_per_window=cost_map.get(strategy, 3.0),
        ))

    report.case_matrix = matrix

    # ── 自动推荐 ──────────────────────────────
    if report.strategies:
        best = max(report.strategies, key=lambda s: s.m5_match)
        worst = min(report.strategies, key=lambda s: s.m5_match)

        # 检查是否显著差异
        gap = best.m5_match - worst.m5_match
        if gap > 0.05:
            report.recommendation = (
                f"推荐 **{best.strategy}** 作为主策略。M5={best.m5_match:.1%}，"
                f"领先第二名 {gap:.1%}。"
            )
        else:
            # 差异不大，做条件路由
            report.recommendation = (
                f"三策略 M5 差距 < 5pp，建议条件路由：\n"
                f"- 需要最高准确率时用 Senate（M5 最佳）\n"
                f"- 需要低延迟时用 Teacher-Student（+0.5s）\n"
                f"- 有跨窗口上下文时用 SAGA（长期收益）"
            )

        # 全败 case 分析
        all_fail = [r.case_id for r in matrix if r.none_pass]
        if all_fail:
            report.recommendation += (
                f"\n\n⚠️ {len(all_fail)} 个 case 三策略全败: {', '.join(all_fail[:8])}。"
                f"这些可能是 judge/dataset/model 问题，非架构可修。"
            )

    return report


def save_comparison_report(report: ComparisonReport, path: str | None = None) -> Path:
    """保存对比报告为 JSON。"""
    p = Path(path) if path else CANDIDATES_DIR / "comparison_report.json"

    # 简化序列化
    out = {
        "timestamp": report.timestamp,
        "strategies": [
            {
                "strategy": s.strategy,
                "overall_score": s.overall_score,
                "m5_match": s.m5_match,
                "m6_insight": s.m6_insight,
                "m7_safety": s.m7_safety,
                "m5_only_success_cases": s.m5_only_success_cases,
                "estimated_llm_calls_per_window": s.estimated_llm_calls_per_window,
            }
            for s in report.strategies
        ],
        "case_matrix": [
            {
                "case_id": r.case_id,
                "gold_tone": r.gold_tone,
                "senate": r.senate_pass,
                "teacher_student": r.teacher_student_pass,
                "saga": r.saga_pass,
                "all_pass": r.all_pass,
                "none_pass": r.none_pass,
            }
            for r in report.case_matrix
        ],
        "recommendation": report.recommendation,
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


# ═══════════════════════════════════════════════════════════════
# 便捷入口：生成对比 markdown 报告
# ═══════════════════════════════════════════════════════════════

def render_markdown_report(report: ComparisonReport) -> str:
    """将 ComparisonReport 渲染为 Markdown。"""
    lines = [
        "# 三策略自进化对比报告",
        f"生成时间: {report.timestamp}",
        "",
        "## 策略级对比",
        "",
        "| 策略 | M5 | M6 | M7 | LLM调用/窗 | 独有正确 |",
        "|------|-----|----|----|-----------|--------|",
    ]
    for s in report.strategies:
        lines.append(
            f"| {s.strategy} | {s.m5_match:.1%} | {s.m6_insight:.2f} | "
            f"{s.m7_safety:.2f} | {s.estimated_llm_calls_per_window:.0f} | "
            f"{len(s.m5_only_success_cases)} |"
        )

    lines.extend([
        "",
        "## 逐 case 对比矩阵",
        "",
        "| case | Gold | Senate | T-S | SAGA | 模式 |",
        "|------|------|--------|-----|------|------|",
    ])
    for r in report.case_matrix:
        s = "✓" if r.senate_pass else "✗"
        ts = "✓" if r.teacher_student_pass else "✗"
        sg = "✓" if r.saga_pass else "✗"
        pattern = ""
        if r.all_pass:
            pattern = "全过"
        elif r.none_pass:
            pattern = "全败"
        elif r.senate_pass and not r.teacher_student_pass and not r.saga_pass:
            pattern = "仅Senate"
        elif not r.senate_pass and r.teacher_student_pass and not r.saga_pass:
            pattern = "仅T-S"
        elif not r.senate_pass and not r.teacher_student_pass and r.saga_pass:
            pattern = "仅SAGA"
        lines.append(f"| {r.case_id} | {r.gold_tone} | {s} | {ts} | {sg} | {pattern} |")

    lines.extend([
        "",
        "## 推荐",
        "",
        report.recommendation,
    ])

    return "\n".join(lines)
