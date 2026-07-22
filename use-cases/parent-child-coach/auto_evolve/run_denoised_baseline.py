"""跑 n_runs=3 降噪基线评估，验证方向5（评估降噪）假设。

用法: python -m auto_evolve.run_denoised_baseline
结果存: results/denoised_baseline_n3.json
"""
import json
import sys
import time
from pathlib import Path
from openai import OpenAI

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"

import os
sys.path.insert(0, str(ROOT.parent.parent))  # 让 auto_evolve 包可被 import
sys.path.insert(0, str(ROOT.parent))  # 让 use-cases/parent-child-coach 可被 import

# 实际上 auto_evolve 在 use-cases/parent-child-coach 下
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "D:/星灵-soul-手搓/亲子沟通洞见/路线A_自上而下法_鼓励版")

from auto_evolve.optimizer import (
    load_env, evaluate_with_prompt, EVAL_CASES,
)
from auto_evolve.evaluator import aggregate_results

PROMPTS = "D:/星灵-soul-手搓/亲子沟通洞见/路线A_自上而下法_鼓励版/prompts"


def main(n_runs: int = 3):
    load_env()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = "deepseek-chat"
    client = OpenAI(api_key=api_key, base_url=base_url)

    print(f"🧪 降噪基线评估 (n_runs={n_runs})")
    print(f"   prompt: 统一 v3.1 (感知+总控+生产)")
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
    print(f"\n📊 降噪基线结果 (n_runs={n_runs})")
    print(f"   M1 触发准确率: {report.aggregate_m1:.1%}")
    print(f"   M5 口吻匹配:   {report.aggregate_m5:.1%}")
    print(f"   M6 洞察质量:   {report.aggregate_m6:.2f} / 5.0")
    print(f"   M7 安全性:     {report.aggregate_m7:.2f} / 5.0")
    print(f"   综合分:        {report.overall_score:.3f}")

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
    out_path = RESULTS_DIR / f"denoised_baseline_n{n_runs}.json"
    detailed = {
        "n_runs_per_case": n_runs,
        "prompts": {
            "master": "prompt_总控_v3.1.md",
            "perception": "prompt_感知层_v3.1.md",
        },
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
                "case_id": r.case_id,
                "sys_tone": r.sys_tone,
                "gold_tone": r.gold_tone,
                "m1_trigger_match": r.m1_trigger_match,
                "m5_tone_match": r.m5_tone_match,
                "m6_insight_score": r.m6_insight_score,
                "m7_safety_score": r.m7_safety_score,
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
