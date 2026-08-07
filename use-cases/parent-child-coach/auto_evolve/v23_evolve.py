r"""v2.3 Co-evolution Loop —— 6-Phase 自动进化主循环。

被测系统: system_prompt_v2.3.txt（路线 A 单智能体）
Task model: 百度千帆 DeepSeek-v4-pro
Judge model: 星鸾 Claude Opus 4.7
测试池: golden case + 对抗用例（每轮生成并预验证）

6 个 Phase:
  1. AUDIT — 冷启动，编码已知 failure profile，跑基线评估
  2. GEN_ATTACK — 根据 failure_profile 生成对抗用例
  3. MUTATE — LLM 分析失败 → 变异 v2.3 prompt
  4. EVALUATE — 双模型评估（千帆生成 + 星鸾裁判）
  5. DECIDE — keep/discard + 更新 failure_profile
  6. CHECK — 收敛判断 → Phase 2 或停止

运行方式:
  python -m auto_evolve.v23_evolve [--max-rounds 10] [--resume]

后台常驻:
  Start-Process pwsh -ArgumentList "-NoProfile","-Command",
    "python D:/prompt-ops/use-cases/parent-child-coach/auto_evolve/v23_evolve.py 2>&1 |
     Tee-Object D:/prompt-ops/use-cases/parent-child-coach/results/v23_evolve/run.log"
    -WindowStyle Hidden
"""

import json
import re
import sys
import time
import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
from datetime import datetime

# 路径设置
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "D:/星灵-soul-手搓/亲子沟通洞见/路线A_自上而下法_鼓励版")

from auto_evolve.dual_client import init_clients, call_judge_claude
from auto_evolve.v23_runner import run_v23_once, read_v23_prompt
from auto_evolve.evaluator import (
    EvalResult, BaselineReport, aggregate_results,
    compute_m1_trigger, compute_m5_tone,
    build_m6_prompt_claude, build_m7_prompt_claude,
    parse_claude_judge_response,
)
from auto_evolve.adversarial_gen import (
    FAILURE_MODES,
    generate_adversarial_cases, pre_validate_cases,
    load_adversarial_pool, save_adversarial_round,
)
from auto_evolve.optimizer import (
    load_golden_dataset, find_case, get_input_text, get_gold_labels,
    should_keep, EVAL_CASES, MIN_OVERALL_IMPROVEMENT,
)
from auto_evolve.prompt_mutator import (
    build_failure_report, build_mutator_prompt, apply_mutation,
    MUTATOR_SYSTEM_PROMPT, FailureReport, FailureCase, PromptMutation,
    extract_version, bump_version,
)

# json_repair 可选依赖
try:
    from json_repair import loads as _repair_loads
except ImportError:
    _repair_loads = None

# ═══════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════

V23_PROMPT_PATH = ROOT / "prompts_archive" / "system_prompt_v2.3.txt"
RESULTS_DIR = ROOT / "results" / "v23_evolve"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_RUNS = 1          # 每案例降噪次数（Windows 后台不稳定，降为 1 提速）
MAX_ROUNDS = 10     # 最大轮数
MIN_ADVERSARIAL_PER_MODE = 3  # 每个失败模式至少生成 3 个对抗用例

# 收敛阈值
CONVERGE_OVERALL = 0.92
CONVERGE_M5 = 0.70
MAX_CONSECUTIVE_DISCARD = 3
MAX_DRY_ADVERSARIAL_ROUNDS = 2


# ═══════════════════════════════════════════════════════════════
# Phase 1: AUDIT — 冷启动 failure profile
# ═══════════════════════════════════════════════════════════════

def build_initial_failure_profile() -> list[dict]:
    """根据 v2.3 审计文档构建初始 failure_profile。"""
    return [
        {
            "mode": "tone_blindspot_diagnostic_bias",
            "priority": "critical",
            "evidence": "M5=0.400, ~60% tone mismatch. System defaults to diagnostic.",
            "examples": [],
        },
        {
            "mode": "tone_blindspot_empowering_bias",
            "priority": "high",
            "evidence": "System occasionally outputs empowering when diagnostic needed.",
            "examples": [],
        },
        {
            "mode": "generalization_gap",
            "priority": "critical",
            "evidence": "12 new cases 0.625 vs 70 calibration 0.883 (Δ=-0.258).",
            "examples": [],
        },
        {
            "mode": "being_seen_weak",
            "priority": "medium",
            "evidence": "Blind 50: being_seen=4.66, weakest of 5 dimensions.",
            "examples": [],
        },
        {
            "mode": "insight_depth_insufficient",
            "priority": "medium",
            "evidence": "M6=3.8/5 (baseline 5 cases), insight quality moderate.",
            "examples": [],
        },
        {
            "mode": "edge_case_low_score",
            "priority": "high",
            "evidence": "C5-001 ~0.463, DS_001 ~0.513 — responsibility+failure combos.",
            "examples": [],
        },
    ]


# ═══════════════════════════════════════════════════════════════
# 对抗用例 → 评估兼容层
# ═══════════════════════════════════════════════════════════════

def _adversarial_gold_labels(adv_case: dict) -> dict:
    """从对抗用例元数据构造 gold labels（兼容 get_gold_labels 返回格式）。"""
    return {
        "should_popup": True,  # 对抗用例都应该弹窗
        "tone": adv_case.get("expected_tone", ""),
        "reference_popup": adv_case.get("reference_popup", ""),
        "score": None,
        "hit_checklist": [],
        "forbid_checklist": [],
    }


# ═══════════════════════════════════════════════════════════════
# Phase 4: EVALUATE — 评估单个 case
# ═══════════════════════════════════════════════════════════════

def evaluate_case_with_prompts(
    task_client, task_model: str,
    system_prompt: str,
    case_id: str,
    win_idx: int | None,
    gold: dict,
    input_text: str,
) -> EvalResult:
    """对单个 case 做一次评估（由 Claude judge 打分）。"""

    # 用千帆生成弹窗
    # v2.6+ 不需要 type/contradiction 字段，自动切换 JSON 输出指令
    from auto_evolve.v23_runner import _JSON_OUTPUT_INSTRUCTION_V26
    is_v26 = "v2.6" in system_prompt or "# Prompt · v2.6" in system_prompt or "二选一" in system_prompt
    json_inst = _JSON_OUTPUT_INSTRUCTION_V26 if is_v26 else None
    result = run_v23_once(task_client, task_model, system_prompt, input_text,
                          json_output_instruction=json_inst)
    if result["error"]:
        return EvalResult(
            case_id=case_id, window_index=win_idx or 1,
            error=result["error"],
        )

    m1 = compute_m1_trigger(result["should_popup"], gold["should_popup"])
    m5 = compute_m5_tone(result["tone"], gold["tone"])

    # M6: Claude judge
    m6_score = None
    m6_raw = ""
    ref_popup = gold.get("reference_popup", "")
    if ref_popup.strip() and "内容标注" not in ref_popup:
        m6_prompt = build_m6_prompt_claude(
            dialogue=input_text,
            reference_popup=ref_popup,
            sys_popup=result["popup_text"],
            sys_direction=result["tone"],
            sys_contradiction=result["contradiction"],
        )
        m6_raw = call_judge_claude(m6_prompt)
        m6_score, _ = parse_claude_judge_response(m6_raw, "m6")

    # M7: Claude judge
    m7_score = None
    m7_raw = ""
    forbid = gold.get("forbid_checklist", [])
    if forbid:
        m7_prompt = build_m7_prompt_claude(
            dialogue=input_text,
            forbid_checklist=forbid,
            sys_popup=result["popup_text"],
        )
        m7_raw = call_judge_claude(m7_prompt)
        m7_score, _ = parse_claude_judge_response(m7_raw, "m7")

    return EvalResult(
        case_id=case_id, window_index=win_idx or 1,
        sys_should_popup=result["should_popup"],
        sys_tone=result["tone"],
        sys_popup_text=result["popup_text"],
        sys_main_contradiction=result["contradiction"],
        gold_should_popup=gold.get("should_popup"),
        gold_tone=gold.get("tone", ""),
        gold_reference_popup=ref_popup,
        gold_score=gold.get("score"),
        gold_hit_checklist=gold.get("hit_checklist", []),
        gold_forbid_checklist=forbid,
        m1_trigger_match=m1,
        m5_tone_match=m5,
        m6_insight_score=m6_score,
        m7_safety_score=m7_score,
        m6_judge_raw=m6_raw,
        m7_judge_raw=m7_raw,
    )


def _denoise_case_runs(runs: list[EvalResult]) -> EvalResult:
    """n=3 降噪聚合（majority vote for tone/should_popup, mean for M6/M7）。"""
    ok = [r for r in runs if not r.error]
    if not ok:
        return runs[0]

    valid_tones = {"diagnostic", "empowering"}
    tones = [r.sys_tone for r in ok if r.sys_tone in valid_tones]
    final_tone = Counter(tones).most_common(1)[0][0] if tones else ok[0].sys_tone

    shoulds = [r.sys_should_popup for r in ok if r.sys_should_popup is not None]
    final_should = Counter(shoulds).most_common(1)[0][0] if shoulds else ok[0].sys_should_popup

    rep = next((r for r in ok if r.sys_tone == final_tone), ok[0])

    m6s = [r.m6_insight_score for r in ok if r.m6_insight_score is not None]
    m7s = [r.m7_safety_score for r in ok if r.m7_safety_score is not None]

    return EvalResult(
        case_id=rep.case_id, window_index=rep.window_index,
        sys_should_popup=final_should, sys_tone=final_tone,
        sys_popup_text=rep.sys_popup_text,
        sys_main_contradiction=rep.sys_main_contradiction,
        gold_should_popup=rep.gold_should_popup, gold_tone=rep.gold_tone,
        gold_reference_popup=rep.gold_reference_popup,
        gold_score=rep.gold_score,
        m1_trigger_match=compute_m1_trigger(final_should, rep.gold_should_popup),
        m5_tone_match=compute_m5_tone(final_tone, rep.gold_tone),
        m6_insight_score=sum(m6s) / len(m6s) if m6s else None,
        m7_safety_score=sum(m7s) / len(m7s) if m7s else None,
    )


# ═══════════════════════════════════════════════════════════════
# 评估运行
# ═══════════════════════════════════════════════════════════════

def evaluate_full(
    task_client, task_model: str,
    system_prompt: str,
    eval_cases: list[tuple[str, int | None]],
    adversarial_pool: list[dict] | None = None,
    n_runs: int = N_RUNS,
    verbose: bool = True,
) -> BaselineReport:
    """用当前 prompt 跑完整评估（golden case + 对抗用例）。

    Args:
        eval_cases: [(case_id, window_index), ...] — golden case
        adversarial_pool: 对抗用例池（dict 列表，含 dialogue/expected_tone/reference_popup/case_id）
    """
    dataset = load_golden_dataset()
    all_results = []

    # 构建对抗用例索引（按 case_id 快速查找）
    adv_index = {}
    if adversarial_pool:
        adv_index = {c["case_id"]: c for c in adversarial_pool}

    total = len(eval_cases)
    for i, (case_id, win_idx) in enumerate(eval_cases):
        # 判断来源：golden 还是 adversarial
        if case_id.startswith("adv_") and case_id in adv_index:
            adv = adv_index[case_id]
            case = {"case_id": case_id, "windows": [], "dialogue": adv.get("dialogue", "")}
            gold = _adversarial_gold_labels(adv)
            input_text = adv.get("dialogue", "")
        else:
            case = find_case(dataset, case_id)
            if not case:
                if verbose:
                    print(f"  ⚠️ [{i+1}/{total}] {case_id}: not found in dataset, skip")
                continue
            gold = get_gold_labels(case, win_idx)
            input_text = get_input_text(case, win_idx)

        t_case = time.time()
        if n_runs == 1:
            r = evaluate_case_with_prompts(
                task_client, task_model, system_prompt,
                case_id, win_idx, gold, input_text)
            all_results.append(r)
        else:
            runs = []
            for _ in range(n_runs):
                runs.append(evaluate_case_with_prompts(
                    task_client, task_model, system_prompt,
                    case_id, win_idx, gold, input_text))
            all_results.append(_denoise_case_runs(runs))
        el_case = time.time() - t_case

        if verbose:
            r = all_results[-1]
            status = "❌" if r.error else "✓"
            m6_str = f"{r.m6_insight_score:.2f}" if r.m6_insight_score is not None else "-"
            m7_str = f"{r.m7_safety_score:.2f}" if r.m7_safety_score is not None else "-"
            print(f"  [{i+1}/{total}] {case_id} {status} "
                  f"M1={r.m1_trigger_match} M5={r.m5_tone_match} "
                  f"M6={m6_str} M7={m7_str} "
                  f"({el_case:.0f}s)")

    return aggregate_results(all_results)


# ═══════════════════════════════════════════════════════════════
# 状态持久化
# ═══════════════════════════════════════════════════════════════

def load_state() -> dict | None:
    """加载 loop 状态（支持 resume）。"""
    state_path = RESULTS_DIR / "state.json"
    if state_path.exists():
        return json.loads(state_path.read_text(encoding="utf-8"))
    return None


def save_state(state: dict) -> None:
    """保存 loop 状态。"""
    state["last_saved"] = datetime.now().isoformat()
    (RESULTS_DIR / "state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════
# Prompt 变异（低层级接口，不依赖 Route B 的 PROMPT_FILES）
# ═══════════════════════════════════════════════════════════════

def _mutate_prompt_v23(
    client,
    model: str,
    current_text: str,
    current_version: str,
    failure_report: FailureReport,
    previous_attempts: list[dict] | None = None,
) -> PromptMutation | None:
    """v2.3 prompt 变异 —— 直接使用 build_mutator_prompt + LLM 调用，
    而非 propose_mutation（后者内部 write_prompt_variant 依赖 Route B 文件路径）。

    Returns:
        PromptMutation on success, None on failure.
    """
    old_version = current_version or extract_version(current_text)
    new_version = bump_version(old_version)

    user_prompt = build_mutator_prompt(
        current_text, failure_report,
        prompt_name=f"system_prompt_{current_version}",
        previous_attempts=previous_attempts,
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": MUTATOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.4,
            timeout=120,
        )
        raw = resp.choices[0].message.content or ""
    except Exception as e:
        print(f"  ❌ LLM 变异调用失败: {e}")
        return None

    # 解析 JSON（复用 propose_mutation 的容错逻辑）
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        if _repair_loads is not None:
            try:
                data = _repair_loads(raw)
            except Exception:
                print(f"  ❌ JSON 解析失败 (raw): {raw[:300]}")
                return None
        else:
            print(f"  ❌ JSON 解析失败 (raw): {raw[:300]}")
            return None

    edits = data.get("edits", [])
    analysis = data.get("analysis", data.get("root_cause", ""))
    expected = data.get("expected_improvement", "")

    if not edits:
        print(f"  ⚠️ 变异器未提出任何修改")
        return None

    modified_text = apply_mutation(current_text, edits, new_version)

    return PromptMutation(
        version_from=old_version,
        version_to=new_version,
        target_prompt="v23_system",
        edit_description=[f"{e.get('reason', '')}" for e in edits],
        modified_text=modified_text,
        rationale=f"Analysis: {analysis}\nExpected: {expected}",
    )


# ═══════════════════════════════════════════════════════════════
# 主循环
# ═══════════════════════════════════════════════════════════════

def main(max_rounds: int = MAX_ROUNDS, resume: bool = False):
    """主入口。"""
    # 初始化客户端
    print("🔌 初始化双模型客户端...")
    task_client, task_model, judge_client, judge_model = init_clients()
    print(f"   Task: 千帆 {task_model}")
    print(f"   Judge: 星鸾 {judge_model}")

    # 读取 v2.3 prompt
    v23_prompt = read_v23_prompt()
    print(f"   Prompt: {len(v23_prompt)} 字符")

    # 加载或初始化状态
    state = load_state() if resume else None
    if state:
        print(f"\n📂 从 round {state['current_round']} 恢复...")
        start_round = state["current_round"]
        current_prompt = state.get("current_prompt_text", v23_prompt)
        current_version = state.get("current_version", "v2.3")
        current_best_score = state.get("current_best_score", 0.0)
        failure_profile = state.get("failure_profile", build_initial_failure_profile())
        history = state.get("history", [])
        adversarial_pool = state.get("adversarial_pool", [])
        golden_eval_cases = [
            tuple(x) for x in state.get("golden_eval_cases", EVAL_CASES)
        ]
        last_phase = state.get("last_phase", "AUDIT")
    else:
        print("\n🚀 冷启动 v2.3 Co-evolution Loop")
        start_round = 1
        current_prompt = v23_prompt
        current_version = "v2.3"
        current_best_score = 0.0
        failure_profile = build_initial_failure_profile()
        history = []
        adversarial_pool = []
        golden_eval_cases = list(EVAL_CASES)
        last_phase = "AUDIT"

    # Phase 1: AUDIT（冷启动时执行）
    if last_phase == "AUDIT" or (start_round == 1 and not state):
        print(f"\n{'='*70}")
        print(f"📋 Phase 1: AUDIT — 冷启动 failure profile")
        print(f"{'='*70}")
        for fp in failure_profile:
            mode_def = FAILURE_MODES.get(fp["mode"], {})
            print(f"  🔴 [{fp['priority']}] {mode_def.get('name', fp['mode'])}")
            print(f"     {fp['evidence']}")

        # 跑基线评估（用 golden case 确定 starting point）
        print(f"\n  🏃 跑基线评估 (n={N_RUNS})...")
        t0 = time.time()
        baseline = evaluate_full(
            task_client, task_model, current_prompt,
            golden_eval_cases, adversarial_pool=None,
            n_runs=N_RUNS, verbose=True,
        )
        el = (time.time() - t0) / 60
        current_best_score = baseline.overall_score
        print(f"  ⏱ {el:.1f}min | M1={baseline.aggregate_m1:.1%} "
              f"M5={baseline.aggregate_m5:.1%} "
              f"M6={baseline.aggregate_m6:.2f} M7={baseline.aggregate_m7:.2f} "
              f"overall={baseline.overall_score:.3f}")

        # 保存基线
        save_full_report(baseline, RESULTS_DIR / "baseline_round_000.json",
                        meta={"version": current_version, "round": 0})
        last_phase = "GEN_ATTACK"

        # 持久化初始状态
        save_state({
            "current_round": 1,
            "current_version": current_version,
            "current_prompt_text": current_prompt,
            "current_best_score": current_best_score,
            "failure_profile": failure_profile,
            "history": history,
            "adversarial_pool": adversarial_pool,
            "golden_eval_cases": golden_eval_cases,
            "last_phase": "GEN_ATTACK",
        })

    # 主循环
    consecutive_discard = 0
    dry_adversarial_rounds = 0

    # Resume shortcut: 如果上次在 EVALUATE 阶段中断，直接跳到评估
    if last_phase == "EVALUATE":
        round_num = start_round
        print(f"\n{'='*70}")
        print(f"🔄 Round {round_num}/{max_rounds} | 版本: {current_version} | "
              f"best_overall={current_best_score:.3f}")
        print(f"{'='*70}")
        print(f"  📂 Resume: 跳过 Phases 2-3，直接进入 Phase 4 EVALUATE")

        all_eval_cases = list(golden_eval_cases)
        seen = set((c, w) for c, w in golden_eval_cases)
        for adv in adversarial_pool:
            key = (adv["case_id"], None)
            if key not in seen:
                all_eval_cases.append(key)
                seen.add(key)
        print(f"  测试池: {len(golden_eval_cases)} golden + "
              f"{len(all_eval_cases) - len(golden_eval_cases)} adversarial = "
              f"{len(all_eval_cases)} total")

        print(f"\n📊 Phase 4: EVALUATE — 双模型评估 (n={N_RUNS})")
        t0 = time.time()
        try:
            report = evaluate_full(
                task_client, task_model, current_prompt,
                all_eval_cases, adversarial_pool=adversarial_pool,
                n_runs=N_RUNS, verbose=True,
            )
        except Exception as e:
            print(f"  ❌ 评估异常: {e}")
            import traceback; traceback.print_exc()
            save_state({
                "current_round": round_num, "current_version": current_version,
                "current_prompt_text": current_prompt,
                "current_best_score": current_best_score,
                "failure_profile": failure_profile, "history": history,
                "adversarial_pool": adversarial_pool,
                "golden_eval_cases": golden_eval_cases,
                "last_phase": "EVALUATE",
            })
            return

        el = (time.time() - t0) / 60
        print(f"  ⏱ {el:.1f}min")
        print(f"  M1={report.aggregate_m1:.1%} M5={report.aggregate_m5:.1%} "
              f"M6={report.aggregate_m6:.2f} M7={report.aggregate_m7:.2f} "
              f"overall={report.overall_score:.3f}")

        save_full_report(report, RESULTS_DIR / f"round_{round_num:03d}_eval.json",
                        meta={"version": current_version, "round": round_num,
                              "n_adversarial": len(adversarial_pool)})

        # DECIDE — 始终与原 baseline 比较，防止基准漂移
        baseline_path = RESULTS_DIR / "baseline_round_000.json"
        if not baseline_path.exists():
            print("  ❌ baseline_round_000.json 不存在，无法 DECIDE")
            return
        prev_data = json.loads(baseline_path.read_text(encoding="utf-8"))
        prev_report = _report_from_dict(prev_data)
        prev_score = prev_data.get("aggregate", {}).get("overall_score", current_best_score)
        kept, reason = should_keep(prev_report, report)
        if kept:
            print(f"  ✅ KEEP: {reason}")
            current_best_score = report.overall_score
            history.append({"round": round_num, "version": current_version,
                          "overall": report.overall_score, "kept": True, "reason": reason})
            consecutive_discard = 0
        else:
            print(f"  ❌ DISCARD: {reason}")
            current_prompt = read_v23_prompt()
            current_version = "v2.3"
            history.append({"round": round_num, "version": current_version,
                          "overall": report.overall_score, "kept": False, "reason": reason})
            consecutive_discard += 1

        # CHECK
        if report.overall_score >= CONVERGE_OVERALL and report.aggregate_m5 >= CONVERGE_M5:
            print(f"\n🎉 达成收敛目标!")
            save_state({"current_round": round_num + 1, "current_version": current_version,
                       "current_prompt_text": current_prompt, "current_best_score": current_best_score,
                       "failure_profile": failure_profile, "history": history,
                       "adversarial_pool": adversarial_pool, "golden_eval_cases": golden_eval_cases,
                       "last_phase": "DONE"})
            return
        if consecutive_discard >= MAX_CONSECUTIVE_DISCARD:
            print(f"\n⏹ 连续 {consecutive_discard} 轮 discard，停止")
            return

        # 推进到下一轮
        start_round = round_num + 1
        last_phase = "GEN_ATTACK"
        if start_round > max_rounds:
            # 本轮未完成，退出让主循环正常结束
            pass
            return

    for round_num in range(start_round, max_rounds + 1):
        print(f"\n{'='*70}")
        print(f"🔄 Round {round_num}/{max_rounds} | 版本: {current_version} | "
              f"best_overall={current_best_score:.3f}")
        print(f"{'='*70}")

        # ——— Phase 2: GEN_ATTACK ———
        print(f"\n🎯 Phase 2: GEN_ATTACK — 对抗用例生成")

        # 整理已有对话（去重参考）
        existing_dialogues = [c.get("dialogue", "") for c in adversarial_pool
                             if c.get("dialogue")]

        # 只对 active（未攻克的）失败模式生成用例
        active_failures = [fp for fp in failure_profile
                          if not fp.get("conquered", False)]

        new_cases = generate_adversarial_cases(
            judge_client, judge_model,
            failure_profile=active_failures,
            existing_dialogues=existing_dialogues,
            n_per_mode=MIN_ADVERSARIAL_PER_MODE,
            verbose=True,
        )

        if new_cases:
            # 预验证
            print(f"\n  🔍 预验证 {len(new_cases)} 个对抗用例...")
            validated = pre_validate_cases(
                task_client, task_model,
                judge_client, judge_model,
                current_prompt, new_cases, verbose=True,
            )
            print(f"  ✅ {len(validated)}/{len(new_cases)} 通过预验证")

            if validated:
                adversarial_pool.extend(validated)
                save_adversarial_round(validated, round_num, RESULTS_DIR / "adversarial")
                dry_adversarial_rounds = 0
            else:
                dry_adversarial_rounds += 1
                print(f"  ⚠️ 本轮无有效对抗用例 (连续 {dry_adversarial_rounds} 轮)")
        else:
            dry_adversarial_rounds += 1
            print(f"  ⚠️ 对抗生成无新用例 (连续 {dry_adversarial_rounds} 轮)")

        # 构建完整评估 case list（golden + 对抗）
        all_eval_cases = list(golden_eval_cases)
        seen = set((c, w) for c, w in golden_eval_cases)
        for adv in adversarial_pool:
            key = (adv["case_id"], None)
            if key not in seen:
                all_eval_cases.append(key)
                seen.add(key)

        print(f"  测试池: {len(golden_eval_cases)} golden + "
              f"{len(all_eval_cases) - len(golden_eval_cases)} adversarial = "
              f"{len(all_eval_cases)} total")

        # ——— Phase 3: MUTATE ———
        print(f"\n🧬 Phase 3: MUTATE — Prompt 进化")

        # 用上一轮的评估结果构建失败报告
        if round_num == 1:
            prev_eval_path = RESULTS_DIR / "baseline_round_000.json"
        else:
            prev_eval_path = RESULTS_DIR / f"round_{round_num - 1:03d}_eval.json"

        if prev_eval_path.exists():
            failure_report = build_failure_report(str(prev_eval_path))
        else:
            failure_report = FailureReport()

        if failure_report.failures:
            print(f"  失败案例: {len(failure_report.failures)}")
            for p in failure_report.top_patterns[:3]:
                print(f"  📋 {p[:120]}")

            # 构建之前 discard 的尝试列表（供 mutator 避免重复方向）
            previous_attempts = [
                {
                    "version": h.get("version", "?"),
                    "overall": h.get("overall", 0.0),
                    "reason": h.get("reason", "")[:80],
                    "edit_descriptions": h.get("edit_descriptions", []),
                }
                for h in history if not h.get("kept")
            ]

            mutation = _mutate_prompt_v23(
                task_client, task_model,
                current_text=current_prompt,
                current_version=current_version,
                failure_report=failure_report,
                previous_attempts=previous_attempts,
            )

            if mutation and mutation.modified_text:
                new_version = mutation.version_to
                print(f"  📝 {mutation.version_from} → {new_version}")
                for desc in mutation.edit_description[:3]:
                    print(f"     - {desc[:100]}")

                # 保存变体（v2.3 路线 A 文件在 prompts_archive 下）
                variant_path = ROOT / "prompts_archive" / f"system_prompt_{new_version}.txt"
                variant_path.write_text(mutation.modified_text, encoding="utf-8")
                print(f"  💾 已保存: {variant_path.name}")
            else:
                print(f"  ❌ 变异失败: {mutation.rationale if mutation else 'unknown'}")
                mutation = None
                new_version = current_version
                variant_path = None
        else:
            print(f"  ✅ 无失败案例，跳过变异")
            mutation = None
            new_version = current_version
            variant_path = None

        # ——— Phase 4: EVALUATE ———
        print(f"\n📊 Phase 4: EVALUATE — 双模型评估 (n={N_RUNS})")

        eval_prompt = mutation.modified_text if mutation else current_prompt
        eval_version = new_version if mutation else current_version

        # 评估前持久化 —— 防止中途崩溃
        save_state({
            "current_round": round_num,
            "current_version": eval_version,
            "current_prompt_text": eval_prompt,
            "current_best_score": current_best_score,
            "failure_profile": failure_profile,
            "history": history,
            "adversarial_pool": adversarial_pool,
            "golden_eval_cases": golden_eval_cases,
            "last_phase": "EVALUATE",
        })

        t0 = time.time()
        try:
            report = evaluate_full(
                task_client, task_model, eval_prompt,
                all_eval_cases, adversarial_pool=adversarial_pool,
                n_runs=N_RUNS, verbose=True,
            )
        except Exception as e:
            print(f"  ❌ 评估阶段异常: {e}")
            import traceback
            traceback.print_exc()
            # 保存当前进度然后退出，允许 resume
            print(f"  💾 已保存状态，可用 --resume 恢复")
            save_state({
                "current_round": round_num,
                "current_version": eval_version,
                "current_prompt_text": eval_prompt,
                "current_best_score": current_best_score,
                "failure_profile": failure_profile,
                "history": history,
                "adversarial_pool": adversarial_pool,
                "golden_eval_cases": golden_eval_cases,
                "last_phase": "EVALUATE",
            })
            return
        el = (time.time() - t0) / 60

        print(f"  ⏱ {el:.1f}min")
        print(f"  M1={report.aggregate_m1:.1%} M5={report.aggregate_m5:.1%} "
              f"M6={report.aggregate_m6:.2f} M7={report.aggregate_m7:.2f} "
              f"overall={report.overall_score:.3f}")

        # 保存评估报告
        save_full_report(report, RESULTS_DIR / f"round_{round_num:03d}_eval.json",
                        meta={"version": eval_version, "round": round_num,
                              "n_adversarial": len(adversarial_pool)})

        # ——— Phase 5: DECIDE ———
        if mutation and variant_path:
            print(f"\n⚖️  Phase 5: DECIDE")

            # 构建 baseline 报告（始终与原点比较，防止基准漂移）
            baseline_path = RESULTS_DIR / "baseline_round_000.json"
            if baseline_path.exists():
                prev_data = json.loads(baseline_path.read_text(encoding="utf-8"))
                prev_report = _report_from_dict(prev_data)
            else:
                prev_report = report  # fallback（不应发生）

            keep, reason = should_keep(prev_report, report)

            if keep:
                print(f"  ✅ KEEP — {reason}")
                current_prompt = mutation.modified_text
                current_version = new_version
                current_best_score = report.overall_score
                consecutive_discard = 0
                history.append({
                    "round": round_num, "version": new_version,
                    "overall": report.overall_score, "kept": True,
                    "reason": reason,
                    "edit_descriptions": mutation.edit_description,
                })
            else:
                print(f"  ❌ DISCARD — {reason}")
                consecutive_discard += 1
                if variant_path.exists():
                    variant_path.unlink()
                history.append({
                    "round": round_num, "version": new_version,
                    "overall": report.overall_score, "kept": False,
                    "reason": reason,
                    "edit_descriptions": mutation.edit_description,
                })
        else:
            # 无变异，仅更新 best_score
            current_best_score = report.overall_score

        # ——— 更新 failure_profile ———
        for fp in failure_profile:
            mode_key = fp["mode"]
            has_failures = False

            for r in report.results:
                if r.error:
                    continue
                if mode_key == "tone_blindspot_diagnostic_bias":
                    if (r.m5_tone_match == 0.0 and r.sys_tone == "diagnostic"
                            and r.gold_tone == "empowering"):
                        has_failures = True
                        break
                elif mode_key == "tone_blindspot_empowering_bias":
                    if (r.m5_tone_match == 0.0 and r.sys_tone == "empowering"
                            and r.gold_tone == "diagnostic"):
                        has_failures = True
                        break
                elif mode_key == "being_seen_weak":
                    if r.m6_insight_score is not None and r.m6_insight_score < 4.0:
                        has_failures = True
                        break
                elif mode_key in ("generalization_gap", "insight_depth_insufficient",
                                  "edge_case_low_score"):
                    if r.m6_insight_score is not None and r.m6_insight_score < 3.5:
                        has_failures = True
                        break

            if not has_failures:
                fp["clean_rounds"] = fp.get("clean_rounds", 0) + 1
                if fp["clean_rounds"] >= 2:
                    fp["conquered"] = True
            else:
                fp["clean_rounds"] = 0
                fp["conquered"] = False

        conquered = sum(1 for fp in failure_profile if fp.get("conquered"))
        print(f"\n  📊 失败模式: {conquered}/{len(failure_profile)} 已攻克")

        # ——— Phase 6: CHECK ———
        print(f"\n🔍 Phase 6: CHECK — 收敛判断")

        should_stop = False
        stop_reason = ""

        if report.overall_score >= CONVERGE_OVERALL and report.aggregate_m5 >= CONVERGE_M5:
            should_stop = True
            stop_reason = (
                f"达成目标: overall={report.overall_score:.3f}≥{CONVERGE_OVERALL}, "
                f"M5={report.aggregate_m5:.1%}≥{CONVERGE_M5}"
            )

        if dry_adversarial_rounds >= MAX_DRY_ADVERSARIAL_ROUNDS:
            should_stop = True
            stop_reason = f"对抗生成枯竭: 连续 {dry_adversarial_rounds} 轮无新有效用例"

        if consecutive_discard >= MAX_CONSECUTIVE_DISCARD:
            should_stop = True
            stop_reason = f"优化收益归零: 连续 {consecutive_discard} 轮全部 discard"

        if should_stop:
            print(f"  🛑 {stop_reason}")
        else:
            print(f"  ✅ 继续下一轮")

        # 保存状态（支持 resume）
        save_state({
            "current_round": round_num + 1,
            "current_version": current_version,
            "current_prompt_text": current_prompt,
            "current_best_score": current_best_score,
            "failure_profile": failure_profile,
            "history": history,
            "adversarial_pool": adversarial_pool,
            "golden_eval_cases": golden_eval_cases,
            "last_phase": "GEN_ATTACK",
        })

        if should_stop:
            break

    # ——— 最终报告 ———
    print(f"\n{'='*70}")
    print(f"🏁 Co-evolution 完成")
    print(f"{'='*70}")
    print(f"  起始版本: v2.3")
    print(f"  最终版本: {current_version}")
    print(f"  最终综合分: {current_best_score:.3f}")
    print(f"  总轮数: {round_num}")
    print(f"  对抗用例池: {len(adversarial_pool)} 个")
    print(f"  已攻克失败模式: "
          f"{sum(1 for fp in failure_profile if fp.get('conquered'))}/{len(failure_profile)}")

    kept = sum(1 for h in history if h.get("kept"))
    print(f"  Keep: {kept} | Discard: {len(history) - kept}")
    for h in history:
        status = "✅" if h.get("kept") else "❌"
        print(f"    R{h['round']} {h['version']} {status} overall={h['overall']:.3f}")

    # 保存最终报告
    final_report = {
        "start_version": "v2.3",
        "final_version": current_version,
        "final_score": current_best_score,
        "total_rounds": round_num,
        "adversarial_pool_size": len(adversarial_pool),
        "failure_modes_conquered": sum(
            1 for fp in failure_profile if fp.get("conquered")),
        "history": history,
    }
    (RESULTS_DIR / "final_report.json").write_text(
        json.dumps(final_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n📄 最终报告: {RESULTS_DIR / 'final_report.json'}")


# ═══════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════

def save_full_report(report: BaselineReport, path: Path,
                     meta: dict | None = None) -> None:
    """保存完整评估报告为 JSON。"""
    data = {
        "aggregate": {
            "m1_trigger_accuracy": report.aggregate_m1,
            "m5_tone_match": report.aggregate_m5,
            "m6_insight_quality": report.aggregate_m6,
            "m7_safety_score": report.aggregate_m7,
            "overall_score": report.overall_score,
        },
        "per_case": [
            {
                "case_id": r.case_id,
                "window_index": r.window_index,
                "sys_tone": r.sys_tone,
                "sys_popup_text": (r.sys_popup_text or "")[:300],
                "sys_main_contradiction": r.sys_main_contradiction,
                "gold_tone": r.gold_tone,
                "gold_reference_popup": r.gold_reference_popup,
                "gold_score": r.gold_score,
                "m1_trigger_match": r.m1_trigger_match,
                "m5_tone_match": r.m5_tone_match,
                "m6_insight_score": r.m6_insight_score,
                "m7_safety_score": r.m7_safety_score,
                "error": r.error,
            }
            for r in report.results
        ],
    }
    if meta:
        data["meta"] = meta
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _report_from_dict(data: dict) -> BaselineReport:
    """从保存的报告 JSON 重建 BaselineReport。"""
    results = []
    for c in data.get("per_case", []):
        results.append(EvalResult(
            case_id=c.get("case_id", ""),
            window_index=c.get("window_index", 0),
            sys_tone=c.get("sys_tone", ""),
            sys_popup_text=c.get("sys_popup_text", ""),
            sys_main_contradiction=c.get("sys_main_contradiction", ""),
            gold_tone=c.get("gold_tone", ""),
            gold_reference_popup=c.get("gold_reference_popup", ""),
            gold_score=c.get("gold_score"),
            m1_trigger_match=c.get("m1_trigger_match"),
            m5_tone_match=c.get("m5_tone_match"),
            m6_insight_score=c.get("m6_insight_score"),
            m7_safety_score=c.get("m7_safety_score"),
        ))
    return aggregate_results(results)


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="v2.3 Co-evolution Loop")
    parser.add_argument("--max-rounds", type=int, default=MAX_ROUNDS,
                       help=f"最大轮数 (默认: {MAX_ROUNDS})")
    parser.add_argument("--resume", action="store_true",
                       help="从上次中断处恢复")
    args = parser.parse_args()

    main(max_rounds=args.max_rounds, resume=args.resume)
