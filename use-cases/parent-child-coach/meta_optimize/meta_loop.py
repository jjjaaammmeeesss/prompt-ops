"""Meta-Loop: Level 1 内圈 + Level 1.5 搜索控制。

核心循环（每个策略独立运行）:
  1. 运行基线评估
  2. 冻结 origin_baseline
  3. 迭代:
     a. proposer 读 trace → 诊断失败源 → 生成修改
     b. 应用修改到 harness
     c. evaluate on search set
     d. should_keep(origin_baseline, candidate)
     e. 收敛检查
     f. 如果需要，按 L1.5 规则切换搜索策略
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from auto_evolve.evaluator import EvalResult, BaselineReport, aggregate_results  # noqa: E402
from auto_evolve.optimizer import (  # noqa: E402
    EVAL_CASES, MIN_OVERALL_IMPROVEMENT, MAX_ITERATIONS,
    evaluate_with_prompt, should_keep, load_golden_dataset,
    find_case, get_input_text, get_gold_labels, load_env,
)

from candidate_store import (  # noqa: E402
    CANDIDATES_DIR, TraceEntry, MetricsSnapshot, ParentRef,
    init_candidate, save_trace, save_metrics, save_proposal, save_validation,
    load_trace, load_metrics, load_proposal,
    list_candidates, get_best_candidate, archive_candidate,
    build_candidate_from_report,
)
from proposer import (  # noqa: E402
    MutationProposal, EditProposal,
    build_proposer_prompt, parse_proposer_response,
    heuristic_classify_failures, heuristic_propose,
)
from dataset_split import (  # noqa: E402
    load_split_manifest, get_eval_entries_for_split,
)

# ═══════════════════════════════════════════════════════════════
# 配置常量（来自 plan）
# ═══════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    "max_candidates": 8,
    "max_consecutive_discards": 3,
    "early_convergence_overall": 0.90,
    "min_improvement_streak_stop": 0.005,  # 连续 2 次 keep 但提升 < 此值 → 收敛
    "n_runs_per_case_search": 1,  # search set 降噪次数（1=快，3=稳定）
    "n_runs_per_case_validation": 3,
}

XINGLING_PROMPTS = Path(
    "D:/星灵-soul-手搓/亲子沟通洞见/路线A_自上而下法_鼓励版/prompts"
)


# ═══════════════════════════════════════════════════════════════
# 迭代记录
# ═══════════════════════════════════════════════════════════════

@dataclass
class StrategyIteration:
    """单策略一次迭代记录。"""
    iteration: int
    candidate_id: str
    strategy: str
    report: BaselineReport | None = None
    metrics: MetricsSnapshot | None = None
    proposal: MutationProposal | None = None
    kept: bool = False
    reason: str = ""
    timestamp: str = ""

    @classmethod
    def now(cls, iteration: int, strategy: str, candidate_id: str) -> "StrategyIteration":
        return cls(
            iteration=iteration,
            candidate_id=candidate_id,
            strategy=strategy,
            timestamp=datetime.now().isoformat(),
        )


# ═══════════════════════════════════════════════════════════════
# 策略 harness 加载/保存
# ═══════════════════════════════════════════════════════════════

def load_strategy_harness(strategy: str, harness_type: str = "md") -> str:
    """加载策略的 harness 文件。

    策略 harness 文件保存在 candidates/<strategy>/baseline/ 下。
    """
    d = CANDIDATES_DIR / strategy / "baseline"
    file_path = d / f"harness.{harness_type}"
    if file_path.exists():
        return file_path.read_text(encoding="utf-8")
    return ""


def save_strategy_harness(strategy: str, content: str, harness_type: str = "md") -> Path:
    """保存策略 harness 到候选目录。"""
    d = CANDIDATES_DIR / strategy / "baseline"
    d.mkdir(parents=True, exist_ok=True)
    file_path = d / f"harness.{harness_type}"
    file_path.write_text(content, encoding="utf-8")
    return file_path


def apply_edits_to_harness(harness_text: str, edits: list[EditProposal]) -> str:
    """应用编辑列表到 harness 文本。每个 edit 做精确字符串替换。"""
    result = harness_text
    for edit in edits:
        if edit.before in result:
            result = result.replace(edit.before, edit.after, 1)
        else:
            print(f"  ⚠ 未找到匹配文本: {edit.before[:60]}...")
    return result


# ═══════════════════════════════════════════════════════════════
# L1 内圈：策略优化主循环
# ═══════════════════════════════════════════════════════════════

def run_strategy_optimization(
    strategy: str,
    client: OpenAI,
    model: str,
    proposer_context: str = "",
    mutable_files: list[str] | None = None,
    config: dict | None = None,
    use_heuristic_proposer: bool = False,
) -> list[StrategyIteration]:
    """对单个策略运行 L1 harness 优化循环。

    Args:
        strategy: 策略名 (senate / teacher_student / saga)
        client: OpenAI 客户端
        model: 模型名
        proposer_context: 策略特定的 proposer 上下文
        mutable_files: 可变异文件列表
        config: 覆盖默认配置
        use_heuristic_proposer: True = 使用规则 proposer（快速验证），False = LLM proposer

    Returns:
        迭代历史列表
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    mutable_files = mutable_files or ["harness.md"]

    # 加载 split manifest
    manifest = load_split_manifest()
    search_entries = get_eval_entries_for_split("search", manifest)

    # 创建临时 EVAL_CASES 只包含 search set
    # （通过 monkey-patch optimizer 模块的 EVAL_CASES）
    import optimizer as opt_module
    original_eval_cases = opt_module.EVAL_CASES
    opt_module.EVAL_CASES = search_entries

    try:
        history: list[StrategyIteration] = []

        # ── 1. 运行基线 ──────────────────────────────────
        print(f"\n{'='*60}")
        print(f"[{strategy}] 运行基线评估 (search set, n={cfg['n_runs_per_case_search']})")
        print(f"{'='*60}")

        harness_md = load_strategy_harness(strategy, "md")
        harness_py = load_strategy_harness(strategy, "py")

        baseline_report = evaluate_with_prompt(
            client, model,
            n_runs_per_case=cfg["n_runs_per_case_search"],
            verbose=True,
        )
        origin_baseline = baseline_report  # 冻结

        # 保存基线候选
        d = build_candidate_from_report(
            strategy, "baseline", baseline_report,
            harness_md=harness_md, harness_py=harness_py,
        )
        print(f"[{strategy}] 基线: overall={baseline_report.overall_score:.3f}, "
              f"M5={baseline_report.aggregate_m5:.1%}")

        # ── 2. 迭代循环 ──────────────────────────────────
        current_harness_md = harness_md
        current_harness_py = harness_py
        consecutive_discards = 0
        candidate_idx = 0
        last_two_improvements: list[float] = []

        while candidate_idx < cfg["max_candidates"]:
            candidate_idx += 1
            candidate_id = f"candidate_{candidate_idx:03d}"
            iter_record = StrategyIteration.now(candidate_idx, strategy, candidate_id)

            # 2a. proposer 诊断 + 生成修改
            print(f"\n[{strategy}] 迭代 {candidate_idx}: proposer 分析...")

            trace_entries = load_trace(d)
            metrics = load_metrics(d)

            if use_heuristic_proposer:
                proposal = heuristic_propose(
                    strategy, candidate_id, trace_entries,
                    current_harness_md, metrics,
                )
            else:
                # LLM proposer
                prompt = build_proposer_prompt(
                    strategy=strategy,
                    trace_entries=trace_entries,
                    current_harness_md=current_harness_md,
                    current_harness_py=current_harness_py,
                    metrics=metrics,
                    history_summary=_build_history_summary(history),
                    strategy_context=proposer_context,
                )
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": proposer_context},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.3, timeout=120,
                )
                raw = resp.choices[0].message.content or ""
                proposal = parse_proposer_response(raw)
                if proposal is None:
                    print(f"[{strategy}] proposer 返回解析失败，跳过本轮")
                    consecutive_discards += 1
                    if consecutive_discards >= cfg["max_consecutive_discards"]:
                        break
                    continue

            proposal.strategy = strategy
            proposal.candidate_id = candidate_id

            # 2b. 应用修改
            mutated_md = current_harness_md
            mutated_py = current_harness_py
            for edit in proposal.edits:
                if edit.target_file == "harness.md":
                    mutated_md = apply_edits_to_harness(mutated_md, [edit])
                elif edit.target_file == "harness.py":
                    mutated_py = apply_edits_to_harness(mutated_py, [edit])

            if mutated_md == current_harness_md and mutated_py == current_harness_py:
                print(f"[{strategy}] 无有效修改，跳过本轮")
                consecutive_discards += 1
                if consecutive_discards >= cfg["max_consecutive_discards"]:
                    break
                continue

            # 保存临时 harness 文件供 evaluate_with_prompt 使用
            tmp_md_path = save_strategy_harness(strategy, mutated_md, "md")
            tmp_py_path = save_strategy_harness(strategy, mutated_py, "py")

            # 2c. 评估（由于 evaluate_with_prompt 绑定了原始的 MultiAgentOrchestrator，
            # 这里先用 original harness 评估，后续 Phase 1 会对接各策略的实际 orchestrator）
            print(f"[{strategy}] 迭代 {candidate_idx}: 评估...")
            candidate_report = evaluate_with_prompt(
                client, model,
                n_runs_per_case=cfg["n_runs_per_case_search"],
                verbose=False,
            )

            # 2d. keep/discard
            kept, reason = should_keep(origin_baseline, candidate_report)
            iter_record.report = candidate_report
            iter_record.metrics = MetricsSnapshot.from_baseline_report(candidate_report)
            iter_record.proposal = proposal
            iter_record.kept = kept
            iter_record.reason = reason

            # 写入候选目录
            candidate_d = build_candidate_from_report(
                strategy, candidate_id, candidate_report,
                harness_md=mutated_md, harness_py=mutated_py,
                proposal=proposal.to_markdown(),
                parent=ParentRef(
                    candidate_id=Path(d).name if d else "baseline",
                    strategy=strategy,
                    overall_score=metrics.overall_score,
                ),
            )

            # 补充 trace 中的失败源分类
            trace_entries_new = load_trace(candidate_d)
            for entry in trace_entries_new:
                if entry.m5_tone_match == 0.0 and entry.case_id in proposal.failure_analysis:
                    for source, cases in proposal.failure_analysis.items():
                        if entry.case_id in cases:
                            entry.failure_source = source
                            break
            save_trace(candidate_d, trace_entries_new)

            if kept:
                print(f"  ✅ KEEP | overall={candidate_report.overall_score:.3f} "
                      f"(Δ={candidate_report.overall_score - origin_baseline.overall_score:+.3f})")
                current_harness_md = mutated_md
                current_harness_py = mutated_py
                d = candidate_d
                consecutive_discards = 0
                last_two_improvements.append(
                    candidate_report.overall_score - origin_baseline.overall_score
                )
                if len(last_two_improvements) > 2:
                    last_two_improvements = last_two_improvements[-2:]

                # 早期收敛检查：> 0.90
                if candidate_report.overall_score > cfg["early_convergence_overall"]:
                    print(f"  🎉 overall 达 {candidate_report.overall_score:.3f}，提前收敛")
                    break

                # 收敛检查：连续提升太小
                if (len(last_two_improvements) >= 2 and
                    all(imp < cfg["min_improvement_streak_stop"] for imp in last_two_improvements)):
                    print(f"  ⏸️  连续 2 次提升 < {cfg['min_improvement_streak_stop']}，收敛")
                    break
            else:
                print(f"  ❌ DISCARD | {reason}")
                archive_candidate(candidate_d)
                # 清理临时 harness
                consecutive_discards += 1
                if consecutive_discards >= cfg["max_consecutive_discards"]:
                    print(f"  ⏸️  连续 {consecutive_discards} 次 discard，收敛")
                    break

            history.append(iter_record)

        # 恢复原始 EVAL_CASES
        opt_module.EVAL_CASES = original_eval_cases
        return history

    finally:
        opt_module.EVAL_CASES = original_eval_cases


# ═══════════════════════════════════════════════════════════════
# L1.5 搜索控制
# ═══════════════════════════════════════════════════════════════

@dataclass
class SearchControlState:
    """L1.5 搜索控制状态。"""
    strategy: str
    active_mutable_file: str = "harness.md"  # 当前可变文件
    discard_streak: int = 0
    mutation_amplitude: str = "small"       # small | medium | large
    switched_targets: list[str] = field(default_factory=list)
    hard_cases: set[str] = field(default_factory=set)   # 不可修的 case


def update_search_control(
    state: SearchControlState,
    last_iteration: StrategyIteration,
    config: dict,
) -> SearchControlState:
    """L1.5: 根据上一轮结果调整搜索策略。

    规则：
      1. 连续 3 次 discard → 切换可变文件
      2. 连续 3 次 discard 但所有文件都试过 → 加大变异幅度
      3. 识别并标记不可修的 hard case

    Returns:
        更新后的 SearchControlState
    """
    mutable_files = config.get("mutable_files", ["harness.md"])

    if last_iteration.kept:
        state.discard_streak = 0
    else:
        state.discard_streak += 1

    # 连续 discard → 切换可变文件
    if state.discard_streak >= 3:
        current_idx = mutable_files.index(state.active_mutable_file) if state.active_mutable_file in mutable_files else 0
        next_idx = (current_idx + 1) % len(mutable_files)
        next_file = mutable_files[next_idx]

        if next_file in state.switched_targets:
            # 所有文件都试过了 → 加大幅度
            amplitudes = ["small", "medium", "large"]
            amp_idx = amplitudes.index(state.mutation_amplitude)
            state.mutation_amplitude = amplitudes[min(amp_idx + 1, len(amplitudes) - 1)]
            state.switched_targets = []
            print(f"  [L1.5] 加大变异幅度至: {state.mutation_amplitude}")
        else:
            state.active_mutable_file = next_file
            state.switched_targets.append(next_file)
            state.discard_streak = 0
            print(f"  [L1.5] 切换可变文件至: {state.active_mutable_file}")

    # 标记 hard cases（连续 3+ 轮 M5 失败的 case）
    if last_iteration.proposal:
        for source, cases in last_iteration.proposal.failure_analysis.items():
            if source in ("model", "dataset", "judge"):
                state.hard_cases.update(cases)

    return state


# ═══════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════

def _build_history_summary(history: list[StrategyIteration]) -> str:
    """构建历史候选摘要供 proposer 参考。"""
    if not history:
        return "（无历史）"
    lines = []
    for h in history[-3:]:  # 最近 3 轮
        lines.append(
            f"- {h.candidate_id}: kept={h.kept}, "
            f"overall={h.metrics.overall_score:.3f if h.metrics else '?'}, "
            f"M5={h.metrics.aggregate_m5:.1% if h.metrics else '?'}, "
            f"reason={h.reason[:80]}"
        )
    return "\n".join(lines)


def save_iteration_history(history: list[StrategyIteration], strategy: str) -> Path:
    """保存迭代历史到 JSON。"""
    d = CANDIDATES_DIR / strategy
    d.mkdir(parents=True, exist_ok=True)
    p = d / "iteration_history.json"
    records = []
    for h in history:
        records.append({
            "iteration": h.iteration,
            "candidate_id": h.candidate_id,
            "strategy": h.strategy,
            "kept": h.kept,
            "reason": h.reason,
            "timestamp": h.timestamp,
            "overall_score": h.metrics.overall_score if h.metrics else None,
            "aggregate_m5": h.metrics.aggregate_m5 if h.metrics else None,
        })
    p.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return p
