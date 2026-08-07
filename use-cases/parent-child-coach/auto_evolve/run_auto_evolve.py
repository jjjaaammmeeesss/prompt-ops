"""自动迭代 runner —— 基于星灵多智能体 v3.1，自动变异 + 评估 + keep/discard。

复用 optimizer.py 的 evaluate_with_prompt / should_keep 和 prompt_mutator.py 的 propose_mutation。
适配星灵多智能体 v3.1（三智能体 + 规则引擎）。

变异目标：production（生产层）—— M6 是当前短板，且生产层改动不影响 tone 判定（规则层决定）。
如果 production 连续 3 次 discard，切换到 master。
"""
import json
import os
import sys
import time
from pathlib import Path
from openai import OpenAI

sys.path.insert(0, "D:/星灵-soul-手搓/亲子沟通洞见/路线A_自上而下法_鼓励版")
sys.path.insert(0, "D:/prompt-ops/use-cases/parent-child-coach")

from auto_evolve.optimizer import (
    load_env, evaluate_with_prompt, should_keep, EVAL_CASES,
)
from auto_evolve.prompt_mutator import (
    propose_mutation, build_failure_report, read_prompt,
)
from auto_evolve.evaluator import aggregate_results, EvalResult, BaselineReport

P = "D:/星灵-soul-手搓/亲子沟通洞见/路线A_自上而下法_鼓励版/prompts"
RESULTS = Path("D:/prompt-ops/use-cases/parent-child-coach/results")
RESULTS.mkdir(parents=True, exist_ok=True)

# 当前基线 prompt 路径（星灵多智能体 v3.1）
BASELINE_PROMPTS = {
    "master": f"{P}/prompt_总控_v3.1.md",
    "perception": f"{P}/prompt_感知层_v3.1.md",
    "production": f"{P}/prompt_生产层_v3.1.md",
}

N_RUNS = 3
MAX_ITERATIONS = 3
MAX_DISCARD_STREAK = 3  # 连续 3 次 discard 切换目标


def save_full_report(report: BaselineReport, path: Path, meta: dict = None) -> None:
    """保存完整字段报告（含 sys_popup_text 供失败分析）。"""
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
                "sys_should_popup": r.sys_should_popup,
                "sys_tone": r.sys_tone,
                "sys_popup_text": r.sys_popup_text,
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


def load_full_report(path: Path) -> BaselineReport:
    """从完整字段报告重建 BaselineReport。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    results = []
    for c in data.get("per_case", []):
        results.append(EvalResult(
            case_id=c["case_id"],
            window_index=c.get("window_index", 0),
            sys_should_popup=c.get("sys_should_popup", False),
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


def run_eval(client, model, prompt_overrides: dict, n_runs: int = N_RUNS) -> BaselineReport:
    """跑评估，prompt_overrides 指定哪些 prompt 用自定义路径。"""
    kwargs = {
        "prompt_path_master": prompt_overrides.get("master", BASELINE_PROMPTS["master"]),
        "prompt_path_perception": prompt_overrides.get("perception", BASELINE_PROMPTS["perception"]),
        "prompt_path_production": prompt_overrides.get("production", BASELINE_PROMPTS["production"]),
        "n_runs_per_case": n_runs,
        "verbose": True,
    }
    return evaluate_with_prompt(client, model, **kwargs)


def _variant_path_for(target: str, version: str) -> Path:
    """根据 target 和 version 返回变体文件路径。"""
    prefix_map = {
        "master": "prompt_总控",
        "perception": "prompt_感知层",
        "production": "prompt_生产层",
    }
    prefix = prefix_map.get(target, "prompt_总控")
    return Path(f"{P}/{prefix}_{version}.md")


def _load_coach_env():
    r"""从 parent-child-coach\.env 加载（修复：load_env 读的是 星灵\.env）。"""
    env_path = Path("D:/prompt-ops/use-cases/parent-child-coach/.env")
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ[key.strip()] = val.strip().strip('"').strip("'")


def main():
    _load_coach_env()  # 先加载正确的 key（coach .env）
    load_env()          # setdefault 不会覆盖已设的值
    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        max_retries=1,
    )
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

    # 1. 加载或跑基线评估
    baseline_path = RESULTS / "auto_baseline_v25_full.json"
    if baseline_path.exists():
        print(f"加载已有基线: {baseline_path.name}")
        baseline = load_full_report(baseline_path)
    else:
        print(f"🏃 跑基线评估 (星灵多智能体 v3.1, n={N_RUNS})...")
        print(f"   start: {time.strftime('%H:%M:%S')}")
        t0 = time.time()
        baseline = run_eval(client, model, {}, n_runs=N_RUNS)
        el = time.time() - t0
        print(f"   elapsed {el/60:.1f}min")
        print(f"   M1={baseline.aggregate_m1:.1%} M5={baseline.aggregate_m5:.1%} M6={baseline.aggregate_m6:.2f} M7={baseline.aggregate_m7:.2f} overall={baseline.overall_score:.3f}")
        save_full_report(baseline, baseline_path, meta={"prompts": BASELINE_PROMPTS, "n_runs": N_RUNS})
        print(f"   saved: {baseline_path.name}")

    print(f"\n🚀 自动迭代启动")
    print(f"   基线综合分: {baseline.overall_score:.3f}")
    print(f"   M1={baseline.aggregate_m1:.1%} M5={baseline.aggregate_m5:.1%} M6={baseline.aggregate_m6:.2f} M7={baseline.aggregate_m7:.2f}")
    print(f"   最大迭代: {MAX_ITERATIONS}")
    print(f"   降噪: n_runs={N_RUNS}")
    print("=" * 70)

    # 当前最佳状态
    current_best = baseline
    origin_baseline = baseline  # 冻结原点，should_keep 始终与此比较
    current_prompts = dict(BASELINE_PROMPTS)
    current_versions = {"master": "v3.1", "perception": "v3.1", "production": "v3.1"}
    current_texts = {
        "master": open(BASELINE_PROMPTS["master"], encoding="utf-8").read(),
        "perception": open(BASELINE_PROMPTS["perception"], encoding="utf-8").read(),
        "production": open(BASELINE_PROMPTS["production"], encoding="utf-8").read(),
    }

    # 变异目标顺序：perception → master → production
    # production 放最后（已验证为局部最优）
    targets = ["perception", "master", "production"]
    target_idx = 0
    discard_streak = 0
    history = []
    # 每个目标的失败尝试记忆（避免重复方向）
    failed_attempts: dict[str, list[dict]] = {t: [] for t in targets}
    # 统一版本计数器（v3.1 → v3.2 → v3.3 ...，不按层分别计数）
    unified_counter = 0

    for iteration in range(1, MAX_ITERATIONS + 1):
        target = targets[target_idx]
        unified_counter += 1
        new_version = f"v3.{1 + unified_counter}"  # v3.2, v3.3, ...
        cur_ver = current_versions[target]

        print(f"\n{'─' * 60}")
        print(f"🔄 迭代 {iteration}/{MAX_ITERATIONS} | 统一版本 {cur_ver} → {new_version} | 目标: {target}")
        print(f"{'─' * 60}")

        # 2. 构建失败报告
        iter_baseline_path = RESULTS / f"auto_iter_{iteration-1:02d}_baseline.json"
        save_full_report(current_best, iter_baseline_path)
        failure_report = build_failure_report(str(iter_baseline_path))

        if not failure_report.failures:
            print("✅ 无失败案例，优化完成！")
            break

        print(f"  失败案例: {len(failure_report.failures)}")
        for p in failure_report.top_patterns[:3]:
            print(f"  📋 {p[:120]}")

        if failed_attempts[target]:
            print(f"  📚 已有 {len(failed_attempts[target])} 条失败尝试记忆")

        # 3. LLM 提议变异（带重试 + 尝试记忆）
        print(f"\n  🧬 变异中... (based on {cur_ver})")
        mutation = None
        for attempt in range(3):
            mutation = propose_mutation(
                client, model, target, failure_report,
                current_text=current_texts[target],
                current_version=cur_ver,
                previous_attempts=failed_attempts[target],
                new_version_override=new_version,
            )
            if mutation.modified_text:
                break
            print(f"  ⚠️ 变异尝试 {attempt+1} 失败: {mutation.rationale[:80]}")
            if attempt < 2:
                time.sleep(2)
        if not mutation or not mutation.modified_text:
            print(f"  ❌ 变异连续 3 次失败，跳过本轮")
            # 记录为失败尝试
            failed_attempts[target].append({
                "version": new_version,
                "overall": 0.0,
                "reason": "mutation JSON parse failed",
                "edit_descriptions": [],
            })
            continue

        print(f"  📝 版本: {mutation.version_from} → {mutation.version_to}")
        for desc in mutation.edit_description[:3]:
            print(f"     - {desc[:100]}")

        # 4. 评估变体
        variant_path = Path(_variant_path_for(target, mutation.version_to))
        eval_overrides = dict(current_prompts)
        eval_overrides[target] = str(variant_path)

        print(f"\n  🏃 跑评估: {variant_path.name} (n={N_RUNS})")
        t0 = time.time()
        candidate = run_eval(client, model, eval_overrides, n_runs=N_RUNS)
        el = time.time() - t0
        print(f"  ⏱ {el/60:.1f}min | M1={candidate.aggregate_m1:.0%} M5={candidate.aggregate_m5:.0%} M6={candidate.aggregate_m6:.1f} M7={candidate.aggregate_m7:.1f} | 综合={candidate.overall_score:.3f}")

        # 5. Keep/Discard
        keep, reason = should_keep(origin_baseline, candidate)
        history.append({
            "iteration": iteration,
            "target": target,
            "version": mutation.version_to,
            "overall": candidate.overall_score,
            "kept": keep,
            "reason": reason,
        })

        if keep:
            print(f"\n  ✅ KEEP — {reason}")
            current_best = candidate
            current_prompts[target] = str(variant_path)
            current_versions[target] = mutation.version_to
            current_texts[target] = mutation.modified_text
            discard_streak = 0
            # 保存完整报告供下一轮失败分析
            save_full_report(candidate, RESULTS / f"auto_iter_{iteration:02d}_{target}_{mutation.version_to}.json",
                             meta={"target": target, "version": mutation.version_to, "kept": True})
        else:
            print(f"\n  ❌ DISCARD — {reason}")
            discard_streak += 1
            # 记入失败尝试记忆，避免下轮重复
            failed_attempts[target].append({
                "version": mutation.version_to,
                "overall": candidate.overall_score,
                "reason": reason,
                "edit_descriptions": mutation.edit_description,
            })
            # 删除变体文件
            if variant_path.exists():
                variant_path.unlink()
            save_full_report(candidate, RESULTS / f"auto_iter_{iteration:02d}_{target}_{mutation.version_to}_discard.json",
                             meta={"target": target, "version": mutation.version_to, "kept": False, "reason": reason})

        # 收敛检查
        if candidate.overall_score > 0.92:
            print(f"\n🎉 综合分达 {candidate.overall_score:.3f}，提前收敛！")
            break

        if discard_streak >= MAX_DISCARD_STREAK:
            print(f"\n⏸️  {target} 连续 {discard_streak} 次 discard，切换目标")
            target_idx += 1
            discard_streak = 0
            if target_idx >= len(targets):
                print("所有目标都收敛，停止迭代。")
                break

    # 最终摘要
    print(f"\n{'=' * 70}")
    print(f"📊 自动迭代结果")
    print(f"{'=' * 70}")
    kept = sum(1 for h in history if h["kept"])
    print(f"  迭代: {len(history)} | 采纳: {kept} | 丢弃: {len(history) - kept}")
    print(f"  基线综合分: {baseline.overall_score:.3f} → 最终: {current_best.overall_score:.3f} (Δ={current_best.overall_score - baseline.overall_score:+.3f})")
    for h in history:
        status = "✅" if h["kept"] else "❌"
        print(f"  #{h['iteration']} {h['target']} {h['version']} {status} | 综合={h['overall']:.3f} | {h['reason'][:70]}")

    # 保存历史
    (RESULTS / "auto_evolve_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  最终 prompt 版本: {current_versions}")
    print(f"  历史保存: results/auto_evolve_history.json")


if __name__ == "__main__":
    main()
