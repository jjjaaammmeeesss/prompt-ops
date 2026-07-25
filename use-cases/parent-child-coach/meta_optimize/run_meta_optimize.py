"""Meta-Optimizer 主入口：三策略自进化 + 对比。

用法:
  # Phase 2a: 单策略 3 轮内圈验证
  python -m meta_optimize.run_meta_optimize --strategy senate --max-iter 3 --dry-run

  # Phase 2b: 三策略全流程（串行）
  python -m meta_optimize.run_meta_optimize --all-strategies --max-iter 3

  # 仅对比（假设各策略已有候选）
  python -m meta_optimize.run_meta_optimize --compare-only
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from openai import OpenAI

# 路径设置
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

XINGLING_ROOT = Path("D:/星灵-soul-手搓/亲子沟通洞见/路线A_自上而下法_鼓励版")
sys.path.insert(0, str(XINGLING_ROOT))

from dotenv import load_dotenv

# 加载环境变量
DOTENV_PATH = Path("D:/星灵-soul-手搓/.env")
load_dotenv(DOTENV_PATH)

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")

from auto_evolve.evaluator import (
    EvalResult, BaselineReport,
    compute_m1_trigger, compute_m5_tone,
    build_m6_prompt, build_m7_prompt,
    parse_llm_judge_response, aggregate_results,
)
from auto_evolve.optimizer import (
    EVAL_CASES, MIN_OVERALL_IMPROVEMENT,
    load_golden_dataset, find_case, get_input_text, get_gold_labels,
)

from meta_optimize.candidate_store import (
    CANDIDATES_DIR, TraceEntry, MetricsSnapshot, ParentRef,
    init_candidate, save_trace, save_metrics, save_proposal,
    load_trace, load_metrics, get_best_candidate,
    build_candidate_from_report, archive_candidate,
)
from meta_optimize.proposer import (
    MutationProposal, heuristic_propose,
    build_proposer_prompt, parse_proposer_response,
    parse_proposer_text_response, PROPOSER_TEXT_SYSTEM_PROMPT,
)
from meta_optimize.meta_loop import (
    DEFAULT_CONFIG, StrategyIteration,
    apply_edits_to_harness, save_iteration_history,
)
from meta_optimize.comparison import (
    explore_traces, build_case_matrix,
    generate_comparison_report, render_markdown_report,
    save_comparison_report,
)
from meta_optimize.strategy_adapters import get_adapter


# ═══════════════════════════════════════════════════════════════
# 策略感知的评估函数
# ═══════════════════════════════════════════════════════════════

def evaluate_with_strategy(
    client: OpenAI,
    model: str,
    strategy: str,
    harness_dir: str,
    eval_entries: list[tuple[str, int | None]] | None = None,
    n_runs_per_case: int = 1,
    verbose: bool = False,
) -> BaselineReport:
    """用指定策略适配器运行评估。

    Args:
        client: OpenAI 客户端
        model: 模型名
        strategy: 策略名
        harness_dir: harness 目录路径
        eval_entries: 要评估的 case 列表，默认 EVAL_CASES
        n_runs_per_case: 每 case 跑 N 次去噪
        verbose: 是否打印详情

    Returns:
        BaselineReport
    """
    dataset = load_golden_dataset()
    entries = eval_entries if eval_entries is not None else EVAL_CASES

    # 创建策略适配器
    adapter = get_adapter(
        strategy=strategy,
        llm_client=client,
        model=model,
        harness_dir=harness_dir,
    )

    all_results: list[EvalResult] = []
    n_runs = max(1, int(n_runs_per_case))

    for case_id, win_idx in entries:
        case = find_case(dataset, case_id)
        if not case:
            continue

        gold = get_gold_labels(case, win_idx)
        input_text = get_input_text(case, win_idx)

        # 调用策略适配器的 process_window
        try:
            result = adapter.process_window(input_text, family=case_id)
        except Exception as e:
            # 出错时创建空的 EvalResult
            er = EvalResult(
                case_id=case_id,
                window_index=win_idx if win_idx else 0,
                error=f"{type(e).__name__}: {str(e)[:100]}",
            )
            all_results.append(er)
            if verbose:
                print(f"  {case_id:10s} ERROR: {er.error}")
            continue

        # 计算 M1/M5
        m1 = compute_m1_trigger(result.should_popup, gold.get("should_popup"))
        m5 = compute_m5_tone(result.tone, gold.get("tone", ""))

        # M6/M7 LLM judge
        m6_score = None
        m7_score = None
        m6_raw = ""
        m7_raw = ""

        if result.popup_text and gold.get("reference_popup"):
            m6_prompt = build_m6_prompt(
                input_text, gold["reference_popup"], result.popup_text,
                sys_direction=result.tone, sys_contradiction=result.main_contradiction,
            )
            m6_raw = _call_judge(client, model, m6_prompt)
            m6_score, _ = parse_llm_judge_response(m6_raw, "M6")

        if result.popup_text and gold.get("forbid_checklist"):
            m7_prompt = build_m7_prompt(
                input_text, gold["forbid_checklist"], result.popup_text,
            )
            m7_raw = _call_judge(client, model, m7_prompt)
            m7_score, _ = parse_llm_judge_response(m7_raw, "M7")

        er = EvalResult(
            case_id=case_id,
            window_index=win_idx if win_idx else 0,
            sys_should_popup=result.should_popup,
            sys_tone=result.tone,
            sys_popup_text=result.popup_text,
            sys_main_contradiction=result.main_contradiction,
            gold_should_popup=gold.get("should_popup"),
            gold_tone=gold.get("tone", ""),
            gold_reference_popup=gold.get("reference_popup", ""),
            gold_score=gold.get("score"),
            m1_trigger_match=m1,
            m5_tone_match=m5,
            m6_insight_score=m6_score,
            m7_safety_score=m7_score,
            m6_judge_raw=m6_raw,
            m7_judge_raw=m7_raw,
        )
        all_results.append(er)

        if verbose:
            m1_str = f"{m1:.0f}" if m1 is not None else "-"
            m5_str = f"{m5:.0f}" if m5 is not None else "-"
            m6_str = f"{m6_score:.1f}" if m6_score is not None else "-"
            print(f"  {case_id:10s} tone={result.tone:11s} (gold={gold.get('tone', '-'):11s}) "
                  f"M1={m1_str} M5={m5_str} M6={m6_str}")

    return aggregate_results(all_results)


def _call_judge(client: OpenAI, model: str, prompt: str, max_retries: int = 2) -> str:
    """调用 LLM judge，含指数退避重试。"""
    import time
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是评估专家。只输出严格 JSON。"},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1, timeout=60,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            return json.dumps({"error": f"{type(e).__name__}: {str(e)[:100]}"})
    return "{}"


# ═══════════════════════════════════════════════════════════════
# 单策略优化
# ═══════════════════════════════════════════════════════════════

def optimize_strategy(
    strategy: str,
    client: OpenAI,
    model: str,
    max_iterations: int = 3,
    use_heuristic: bool = True,
    eval_entries: list | None = None,
) -> list[StrategyIteration]:
    """对单个策略运行 L1 优化循环。

    Args:
        strategy: 策略名
        client: OpenAI 客户端
        model: 模型名
        max_iterations: 最大迭代次数
        use_heuristic: True=规则 proposer（无需 LLM proposer），False=LLM proposer
        eval_entries: 评估用的 case 列表

    Returns:
        迭代历史
    """
    from meta_optimize.dataset_split import load_split_manifest, get_eval_entries_for_split

    harness_dir = CANDIDATES_DIR / strategy / "baseline"
    if not (harness_dir / "harness.md").exists():
        print(f"❌ {strategy} baseline harness 不存在: {harness_dir / 'harness.md'}")
        return []

    # 加载 search set
    manifest = load_split_manifest()
    search_entries = eval_entries if eval_entries else get_eval_entries_for_split("search", manifest)
    print(f"[{strategy}] search set: {len(search_entries)} eval entries")

    # 1. 基线评估
    print(f"\n{'='*50}")
    print(f"[{strategy}] 基线评估...")
    print(f"{'='*50}")

    baseline_report = evaluate_with_strategy(
        client, model, strategy, str(harness_dir),
        eval_entries=search_entries,
        verbose=True,
    )
    origin_baseline = baseline_report

    # 保存基线候选
    harness_md = (harness_dir / "harness.md").read_text(encoding="utf-8")
    harness_py = ""
    py_path = harness_dir / "harness.py"
    if py_path.exists():
        harness_py = py_path.read_text(encoding="utf-8")

    d = build_candidate_from_report(
        strategy, "baseline", baseline_report,
        harness_md=harness_md, harness_py=harness_py,
    )
    print(f"[{strategy}] 基线: overall={baseline_report.overall_score:.3f}, "
          f"M5={baseline_report.aggregate_m5:.1%}, "
          f"M6={baseline_report.aggregate_m6:.2f}, "
          f"M7={baseline_report.aggregate_m7:.2f}")

    # 2. 迭代
    history: list[StrategyIteration] = []
    current_harness_md = harness_md
    consecutive_discards = 0
    candidate_idx = 0

    for iteration in range(1, max_iterations + 1):
        candidate_idx += 1
        candidate_id = f"candidate_{candidate_idx:03d}"

        # 2a. proposer
        trace_entries = load_trace(d)
        metrics = load_metrics(d)

        if use_heuristic:
            proposal = heuristic_propose(
                strategy, candidate_id, trace_entries, current_harness_md, metrics,
            )
        else:
            prompt = build_proposer_prompt(
                strategy=strategy,
                trace_entries=trace_entries,
                current_harness_md=current_harness_md,
                current_harness_py=harness_py,
                metrics=metrics,
            )
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": PROPOSER_TEXT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3, timeout=120,
            )
            raw = resp.choices[0].message.content or ""
            proposal = parse_proposer_text_response(raw)
            if proposal is None:
                # fallback: try JSON parser
                proposal = parse_proposer_response(raw, harness_md=current_harness_md)
            if proposal is None:
                consecutive_discards += 1
                if consecutive_discards >= 3:
                    break
                continue

        proposal.strategy = strategy
        proposal.candidate_id = candidate_id

        # 2b. 应用修改
        mutated_md = current_harness_md
        for edit in proposal.edits:
            if edit.target_file == "harness.md":
                mutated_md = apply_edits_to_harness(mutated_md, [edit])

        if mutated_md == current_harness_md:
            consecutive_discards += 1
            if consecutive_discards >= 3:
                break
            continue

        # 写入临时 harness
        tmp_harness_dir = CANDIDATES_DIR / strategy / candidate_id
        tmp_harness_dir.mkdir(parents=True, exist_ok=True)
        (tmp_harness_dir / "harness.md").write_text(mutated_md, encoding="utf-8")

        # 2c. 评估
        print(f"\n[{strategy}] 迭代 {iteration}: 评估 {candidate_id}...")
        candidate_report = evaluate_with_strategy(
            client, model, strategy, str(tmp_harness_dir),
            eval_entries=search_entries,
            verbose=False,
        )

        # 2d. keep/discard
        from auto_evolve.optimizer import should_keep
        kept, reason = should_keep(origin_baseline, candidate_report)

        iter_record = StrategyIteration(
            iteration=iteration,
            candidate_id=candidate_id,
            strategy=strategy,
            report=candidate_report,
            metrics=MetricsSnapshot.from_baseline_report(candidate_report),
            proposal=proposal,
            kept=kept,
            reason=reason,
            timestamp=datetime.now().isoformat(),
        )

        if kept:
            print(f"  ✅ KEEP | overall={candidate_report.overall_score:.3f} "
                  f"(Δ={candidate_report.overall_score - origin_baseline.overall_score:+.3f})")
            current_harness_md = mutated_md
            d = build_candidate_from_report(
                strategy, candidate_id, candidate_report,
                harness_md=mutated_md,
                proposal=proposal.to_markdown(),
                parent=ParentRef(
                    candidate_id=Path(d).name,
                    strategy=strategy,
                    overall_score=metrics.overall_score,
                ),
            )
            consecutive_discards = 0
        else:
            print(f"  ❌ DISCARD | {reason}")
            archive_candidate(tmp_harness_dir)
            consecutive_discards += 1
            if consecutive_discards >= 3:
                print(f"  ⏸️  连续 {consecutive_discards} 次 discard，收敛")
                break

        history.append(iter_record)

    # 保存历史
    save_iteration_history(history, strategy)
    return history


# ═══════════════════════════════════════════════════════════════
# 全流程入口
# ═══════════════════════════════════════════════════════════════

def run_all_strategies(
    max_iterations: int = 3,
    use_heuristic: bool = True,
    skip_optimization: bool = False,
    eval_entries: list | None = None,
):
    """运行三策略全流程：优化 → 对比 → 报告。"""
    if not API_KEY:
        print("❌ DEEPSEEK_API_KEY 未设置")
        return

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    strategies = ["senate", "teacher_student", "saga"]

    if not skip_optimization:
        # Phase 1: 各策略优化
        all_histories = {}
        for s in strategies:
            history = optimize_strategy(
                s, client, MODEL,
                max_iterations=max_iterations,
                use_heuristic=use_heuristic,
                eval_entries=eval_entries,
            )
            all_histories[s] = history

            best = get_best_candidate(s)
            if best:
                m = load_metrics(best)
                print(f"\n[{s}] 最佳候选: {best.name}, overall={m.overall_score:.3f}")

    # Phase 2: L2 对比
    print(f"\n{'='*50}")
    print("Level 2: 三策略对比")
    print(f"{'='*50}")

    all_traces = explore_traces(strategies)
    matrix = build_case_matrix(all_traces)

    # 汇总各策略指标
    summaries = {}
    for s in strategies:
        best = get_best_candidate(s)
        if best:
            summaries[s] = load_metrics(best)

    report = generate_comparison_report(all_traces, matrix)
    md_report = render_markdown_report(report)

    report_path = CANDIDATES_DIR / "comparison_report.md"
    report_path.write_text(md_report, encoding="utf-8")
    save_comparison_report(report)

    print(md_report)
    print(f"\n报告已保存: {report_path}")

    return report


# ═══════════════════════════════════════════════════════════════
# Smoke test 用例
# ═══════════════════════════════════════════════════════════════

SMOKE_CASES: list[tuple[str, int | None]] = [
    ("C10-001", None),   # diagnostic, score=9
    ("C10-004", None),   # empowering, score=8
    ("C11-006", 1),      # empowering, 挑战 case
]


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Meta-Optimizer: 三策略自进化对比",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  # 环境验证（无 API 调用）\n"
            "  python -m meta_optimize.run_meta_optimize --dry-run\n\n"
            "  # 烟雾测试（3 个 case，快速验证 API）\n"
            "  python -m meta_optimize.run_meta_optimize --strategy senate --smoke --max-iter 1\n\n"
            "  # 单策略正式运行\n"
            "  python -m meta_optimize.run_meta_optimize --strategy senate --max-iter 3\n\n"
            "  # 三策略全流程\n"
            "  python -m meta_optimize.run_meta_optimize --all-strategies --max-iter 3\n\n"
            "  # 仅对比已有候选\n"
            "  python -m meta_optimize.run_meta_optimize --compare-only"
        ),
    )
    parser.add_argument("--strategy", choices=["senate", "teacher_student", "saga"],
                        help="单策略优化")
    parser.add_argument("--all-strategies", action="store_true",
                        help="三策略全流程")
    parser.add_argument("--compare-only", action="store_true",
                        help="仅对比已有候选")
    parser.add_argument("--max-iter", type=int, default=3,
                        help="最大迭代次数 (默认: 3)")
    parser.add_argument("--smoke", action="store_true",
                        help="烟雾测试：仅用 3 个 case 快速验证")
    parser.add_argument("--dry-run", action="store_true",
                        help="跳过 LLM 调用（验证循环逻辑）")
    parser.add_argument("--no-heuristic", action="store_true",
                        help="使用 LLM proposer 而非规则 proposer")

    args = parser.parse_args()

    # smoke 模式覆盖 eval_entries
    eval_entries_override = SMOKE_CASES if args.smoke else None

    if args.dry_run:
        print("🔍 Dry-run 模式：验证循环逻辑（不调 LLM）")
        # 快速验证：打印配置和预期流程
        from meta_optimize.dataset_split import load_split_manifest, get_eval_entries_for_split
        manifest = load_split_manifest()
        for split_name in ("search", "validation", "test"):
            entries = get_eval_entries_for_split(split_name, manifest)
            info = manifest["splits"][split_name]
            print(f"  {split_name}: {info['n_case_ids']} case_ids, "
                  f"{len(entries)} eval entries")

        print("\n策略 harness 状态:")
        for s in ["senate", "teacher_student", "saga"]:
            harness = CANDIDATES_DIR / s / "baseline" / "harness.md"
            status = "✅" if harness.exists() else "❌"
            print(f"  {status} {s}: {harness}")

        print("\n策略适配器可用性:")
        for s in ["senate", "teacher_student", "saga"]:
            try:
                get_adapter(s, None, model="dry-run", harness_dir="")
                print(f"  ✅ {s} 适配器可导入")
            except Exception as e:
                print(f"  ❌ {s}: {e}")

        if args.smoke:
            print(f"\n💨 Smoke 模式将使用以下 case: {[c[0] for c in SMOKE_CASES]}")
    elif args.compare_only:
        if not API_KEY:
            print("❌ DEEPSEEK_API_KEY 未设置，请在 D:/星灵-soul-手搓/.env 中配置")
        else:
            client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
            run_all_strategies(skip_optimization=True)
    elif args.all_strategies:
        if not API_KEY:
            print("❌ DEEPSEEK_API_KEY 未设置，请在 D:/星灵-soul-手搓/.env 中配置")
        else:
            client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
            run_all_strategies(
                max_iterations=args.max_iter,
                use_heuristic=not args.no_heuristic,
                eval_entries=eval_entries_override,
            )
    elif args.strategy:
        if not API_KEY:
            print("❌ DEEPSEEK_API_KEY 未设置，请在 D:/星灵-soul-手搓/.env 中配置")
        else:
            client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
            optimize_strategy(
                args.strategy, client, MODEL,
                max_iterations=args.max_iter,
                use_heuristic=not args.no_heuristic,
                eval_entries=eval_entries_override,
            )
    else:
        parser.print_help()
