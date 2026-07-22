"""[历史实验] 星灵多智能体降噪评估，对比旧基线。

用法: python -m auto_evolve.run_v23_v37
结果存: results/v23_v37_denoised_n3.json
"""
import json
import sys
import time
from pathlib import Path
from openai import OpenAI

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"

import os
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "D:/星灵-soul-手搓/亲子沟通洞见/路线A_自上而下法_鼓励版")

from auto_evolve.optimizer import (
    load_env, evaluate_with_prompt, EVAL_CASES,
)

PROMPTS = "D:/星灵-soul-手搓/亲子沟通洞见/路线A_自上而下法_鼓励版/prompts"


def main(n_runs: int = 3):
    load_env()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = "deepseek-chat"
    client = OpenAI(api_key=api_key, base_url=base_url)

    print(f"🧪 星灵多智能体降噪评估 (n_runs={n_runs})")
    print(f"   prompt: 感知层 + 总控")
    print(f"   cases: {len(EVAL_CASES)} × {n_runs} runs = {len(EVAL_CASES) * n_runs} calls")
    print(f"   开始: {time.strftime('%H:%M:%S')}")
    print("=" * 70)

    t0 = time.time()
    report = evaluate_with_prompt(
        client, model,
        prompt_path_master=f"{PROMPTS}/prompt_总控_v3.1.md",
        prompt_path_perception=f"{PROMPTS}/prompt_感知层_v3.1.md",
        n_runs_per_case=n_runs,
        verbose=True,
    )
    elapsed = time.time() - t0

    print("=" * 70)
    print(f"⏱ 总耗时: {elapsed/60:.1f} 分钟")
    print(f"\n📊 评估结果 (n_runs={n_runs})")
    print(f"   M1 触发准确率: {report.aggregate_m1:.1%}")
    print(f"   M5 口吻匹配:   {report.aggregate_m5:.1%}")
    print(f"   M6 洞察质量:   {report.aggregate_m6:.2f} / 5.0")
    print(f"   M7 安全性:     {report.aggregate_m7:.2f} / 5.0")
    print(f"   综合分:        {report.overall_score:.3f}")

    # 对比基线
    baseline_path = RESULTS_DIR / "denoised_baseline_n3.json"
    if baseline_path.exists():
        with open(baseline_path, encoding="utf-8") as f:
            base = json.load(f)
        print(f"\n📈 对比旧基线:")
        print(f"   {'指标':12s} {'基线':>8s} {'新版':>8s} {'Δ':>8s}")
        ba, ca = base["aggregate"], {
            "m1": report.aggregate_m1, "m5": report.aggregate_m5,
            "m6": report.aggregate_m6, "m7": report.aggregate_m7,
            "overall": report.overall_score,
        }
        for key, label in [("m1","M1触发"),("m5","M5口吻"),("m6","M6洞察"),("m7","M7安全"),("overall","综合")]:
            b = ba.get({"m1":"m1_trigger_accuracy","m5":"m5_tone_match","m6":"m6_insight_quality","m7":"m7_safety_score","overall":"overall_score"}[key], 0)
            c = ca[key]
            delta = c - b
            arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
            print(f"   {label:12s} {b:8.3f} {c:8.3f} {arrow}{abs(delta):7.3f}")

    # 逐案例详情
    print(f"\n📋 逐案例详情:")
    print(f"   {'case_id':10s} {'sys_tone':11s} {'gold_tone':11s} {'M1':>3s} {'M5':>3s} {'M6':>5s} {'M7':>5s}  denoise_info")
    for r in report.results:
        m1 = f"{r.m1_trigger_match:.0f}" if r.m1_trigger_match is not None else "-"
        m5 = f"{r.m5_tone_match:.0f}" if r.m5_tone_match is not None else "-"
        m6 = f"{r.m6_insight_score:.1f}" if r.m6_insight_score is not None else "-"
        m7 = f"{r.m7_safety_score:.1f}" if r.m7_safety_score is not None else "-"
        print(f"   {r.case_id:10s} {r.sys_tone:11s} {r.gold_tone:11s} {m1:>3s} {m5:>3s} {m6:>5s} {m7:>5s}  {r.error[:60]}")

    # 存盘
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"v23_v37_denoised_n{n_runs}.json"
    detailed = {
        "n_runs_per_case": n_runs,
        "prompts": {"master": "prompt_总控_v3.1.md", "perception": "prompt_感知层_v3.1.md"},
        "elapsed_seconds": elapsed,
        "aggregate": {
            "m1_trigger_accuracy": report.aggregate_m1,
            "m5_tone_match": report.aggregate_m5,
            "m6_insight_quality": report.aggregate_m6,
            "m7_safety_score": report.aggregate_m7,
            "overall_score": report.overall_score,
        },
        "per_case": [
            {
                "case_id": r.case_id, "sys_tone": r.sys_tone, "gold_tone": r.gold_tone,
                "m1_trigger_match": r.m1_trigger_match, "m5_tone_match": r.m5_tone_match,
                "m6_insight_score": r.m6_insight_score, "m7_safety_score": r.m7_safety_score,
                "denoise_info": r.error,
            }
            for r in report.results
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(detailed, f, ensure_ascii=False, indent=2)
    print(f"\n💾 已存: {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-runs", type=int, default=3)
    args = parser.parse_args()
    main(n_runs=args.n_runs)
