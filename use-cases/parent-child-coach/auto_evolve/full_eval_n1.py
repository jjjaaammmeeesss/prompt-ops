"""全量校标 n=1 探路：在 golden_dataset 全部 59 cases / 128 windows 上跑 v3.0 多智能体。

目标：发现 v3.0 在 EVAL_CASES 12 案之外的盲区。
- 不做降噪（n=1），快
- 重点关注：
  1. tone 规则引擎在新对话上是否误判
  2. should_popup=False 的窗口（EVAL_CASES 全是 True）
  3. 多窗口案例的 tone 切换是否稳定
  4. 哪些未见过的 case M6/M7 退化严重
"""
import json
import os
import sys
import time
from pathlib import Path
from openai import OpenAI

sys.path.insert(0, "D:/prompt-ops/use-cases/parent-child-coach")
sys.path.insert(0, "D:/星灵-soul-手搓/亲子沟通洞见/路线A_自上而下法_鼓励版")

from auto_evolve.optimizer import (
    load_env, load_golden_dataset, find_case, get_input_text, get_gold_labels,
    _evaluate_case_once, EVAL_CASES, aggregate_results,
)
from src.multi_agent_orchestrator import MultiAgentOrchestrator

RESULTS = Path("D:/prompt-ops/use-cases/parent-child-coach/results")
RESULTS.mkdir(parents=True, exist_ok=True)


def main():
    load_env()
    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

    dataset = load_golden_dataset()
    eval_case_ids = {c for c, _ in EVAL_CASES}

    orch = MultiAgentOrchestrator(
        llm_client=client,
        model=model,
        prompt_path_master="D:/星灵-soul-手搓/亲子沟通洞见/路线A_自上而下法_鼓励版/prompts/prompt_总控_v3.1.md",
        prompt_path_perception="D:/星灵-soul-手搓/亲子沟通洞见/路线A_自上而下法_鼓励版/prompts/prompt_感知层_v3.1.md",
        prompt_path_production="D:/星灵-soul-手搓/亲子沟通洞见/路线A_自上而下法_鼓励版/prompts/prompt_生产层_v3.1.md",
    )

    all_results = []
    skipped_empty = []
    print(f"🚀 全量校标 n=1 开始 — {len(dataset)} cases, start: {time.strftime('%H:%M:%S')}")
    print("=" * 100)

    for case in dataset:
        case_id = case["case_id"]
        windows = case.get("windows", [])
        if not windows:
            continue

        in_eval = case_id in eval_case_ids
        tag = "[EVAL]" if in_eval else "[NEW ]"

        for w in windows:
            win_idx = w.get("window_index", 1)
            gold = get_gold_labels(case, win_idx)
            input_text = get_input_text(case, win_idx)

            # 跳过空 dialogue 的占位案例
            if not input_text.strip():
                skipped_empty.append((case_id, win_idx))
                continue

            r = _evaluate_case_once(client, model, orch, case_id, win_idx, case, gold, input_text)
            all_results.append(r)

            m1 = f"{r.m1_trigger_match:.0f}" if r.m1_trigger_match is not None else "-"
            m5 = f"{r.m5_tone_match:.0f}" if r.m5_tone_match is not None else "-"
            m6 = f"{r.m6_insight_score:.1f}" if r.m6_insight_score is not None else "-"
            m7 = f"{r.m7_safety_score:.1f}" if r.m7_safety_score is not None else "-"
            tone = r.sys_tone or "-"
            gold_tone = r.gold_tone or "-"
            gold_should = "T" if r.gold_should_popup else "F"
            err = "ERR" if r.error else ""

            print(f"  {tag} {case_id:10s} w{win_idx} | gold_should={gold_should} gold_tone={gold_tone[:6]:6s} | sys_tone={tone[:11]:11s} M1={m1} M5={m5} M6={m6} M7={m7} {err}")

    if skipped_empty:
        print(f"\n  ⚠️ 跳过 {len(skipped_empty)} 个空 dialogue 窗口: {[c for c,_ in skipped_empty[:5]]}...")

    # 汇总
    report = aggregate_results(all_results)

    # 拆分：EVAL_CASES 12 案 vs 未见过的
    eval_results = [r for r in all_results if r.case_id in eval_case_ids]
    new_results = [r for r in all_results if r.case_id not in eval_case_ids]
    eval_report = aggregate_results(eval_results)
    new_report = aggregate_results(new_results)

    print(f"\n{'=' * 100}")
    print(f"📊 全量校标 n=1 结果")
    print(f"{'=' * 100}")
    print(f"  {'子集':16s} {'n':>5s} {'M1触发':>8s} {'M5口吻':>8s} {'M6洞察':>8s} {'M7安全':>8s} {'overall':>8s}")
    print(f"  {'-'*60}")
    for label, rep, n in [
        ("全部", report, len(all_results)),
        ("EVAL_CASES(已知)", eval_report, len(eval_results)),
        ("未见过的(NEW)", new_report, len(new_results)),
    ]:
        m1 = f"{rep.aggregate_m1:.1%}" if rep.aggregate_m1 is not None else "-"
        m5 = f"{rep.aggregate_m5:.1%}" if rep.aggregate_m5 is not None else "-"
        m6 = f"{rep.aggregate_m6:.2f}" if rep.aggregate_m6 is not None else "-"
        m7 = f"{rep.aggregate_m7:.2f}" if rep.aggregate_m7 is not None else "-"
        ov = f"{rep.overall_score:.3f}" if rep.overall_score is not None else "-"
        print(f"  {label:16s} {n:>5d} {m1:>8s} {m5:>8s} {m6:>8s} {m7:>8s} {ov:>8s}")

    # 盲区清单
    print(f"\n{'=' * 100}")
    print(f"🚨 盲区清单（NEW 案例中 M6<3 或 M5 mismatch 或 M7<4 或 error）")
    print(f"{'=' * 100}")
    blind = []
    for r in new_results:
        issues = []
        if r.error:
            issues.append(f"error:{r.error[:50]}")
        if r.m5_tone_match == 0 and r.gold_should_popup:
            issues.append(f"tone_mismatch(sys={r.sys_tone},gold={r.gold_tone})")
        if r.m6_insight_score is not None and r.m6_insight_score < 3:
            issues.append(f"M6_low={r.m6_insight_score:.1f}")
        if r.m7_safety_score is not None and r.m7_safety_score < 4:
            issues.append(f"M7_low={r.m7_safety_score:.1f}")
        if r.m1_trigger_match == 0:
            issues.append(f"M1_mismatch(sys_should={r.sys_should_popup},gold={r.gold_should_popup})")
        if issues:
            blind.append((r, issues))
            print(f"  {r.case_id:10s} w{r.window_index} | " + " | ".join(issues))

    print(f"\n  盲区总数: {len(blind)} / {len(new_results)} NEW windows = {len(blind)/max(1,len(new_results)):.1%}")

    # 保存详细
    out = {
        "n_runs": 1,
        "summary_all": _report_to_dict(report),
        "summary_eval": _report_to_dict(eval_report),
        "summary_new": _report_to_dict(new_report),
        "blind_count": len(blind),
        "blind_rate": len(blind) / max(1, len(new_results)),
        "per_window": [_result_to_dict(r) for r in all_results],
    }
    out_path = RESULTS / "full_eval_n1.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  详细结果: {out_path.name}")


def _report_to_dict(rep):
    return {
        "n": len(rep.results) if hasattr(rep, "results") else 0,
        "aggregate_m1": rep.aggregate_m1,
        "aggregate_m5": rep.aggregate_m5,
        "aggregate_m6": rep.aggregate_m6,
        "aggregate_m7": rep.aggregate_m7,
        "overall_score": rep.overall_score,
    }


def _result_to_dict(r):
    return {
        "case_id": r.case_id,
        "window_index": r.window_index,
        "sys_should_popup": r.sys_should_popup,
        "sys_tone": r.sys_tone,
        "sys_popup_text": (r.sys_popup_text or "")[:300],
        "sys_main_contradiction": (r.sys_main_contradiction or "")[:200],
        "gold_should_popup": r.gold_should_popup,
        "gold_tone": r.gold_tone,
        "gold_reference_popup": (r.gold_reference_popup or "")[:200],
        "gold_score": r.gold_score,
        "m1_trigger_match": r.m1_trigger_match,
        "m5_tone_match": r.m5_tone_match,
        "m6_insight_score": r.m6_insight_score,
        "m7_safety_score": r.m7_safety_score,
        "error": r.error or "",
    }


if __name__ == "__main__":
    main()
