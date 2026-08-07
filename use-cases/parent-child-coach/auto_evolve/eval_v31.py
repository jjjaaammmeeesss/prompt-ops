"""跑 v3.1 评估（规则引擎改动后）对比 v3.0 基线"""
import json
import sys
import time
from pathlib import Path
from openai import OpenAI
import os

sys.path.insert(0, "D:/prompt-ops/use-cases/parent-child-coach")
sys.path.insert(0, "D:/星灵-soul-手搓/亲子沟通洞见/路线A_自上而下法_鼓励版")

from auto_evolve.optimizer import load_env, evaluate_with_prompt
from auto_evolve.run_auto_evolve import save_full_report

P = "D:/星灵-soul-手搓/亲子沟通洞见/路线A_自上而下法_鼓励版/prompts"
RESULTS = Path("D:/prompt-ops/use-cases/parent-child-coach/results")

BASELINE_PROMPTS = {
    "master": f"{P}/prompt_总控_v3.1.md",
    "perception": f"{P}/prompt_感知层_v3.1.md",
    "production": f"{P}/prompt_生产层_v3.1.md",
}

def main():
    load_env()
    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

    # v3.0 基线
    v30_path = RESULTS / "auto_baseline_v25_full.json"
    v30 = json.load(open(v30_path, encoding='utf-8'))
    v30_agg = v30['aggregate']
    print(f"=== v3.0 基线 (规则引擎旧版) ===")
    print(f"  M1={v30_agg['m1_trigger_accuracy']:.1%} M5={v30_agg['m5_tone_match']:.1%} M6={v30_agg['m6_insight_quality']:.2f} M7={v30_agg['m7_safety_score']:.2f} overall={v30_agg['overall_score']:.3f}")
    print()

    # 跑 v3.1
    print(f"=== 跑 v3.1 评估 (规则引擎改动: 移除规则8 + 加 genuine_transformation 覆写) ===")
    print(f"  start: {time.strftime('%H:%M:%S')}")
    t0 = time.time()
    v31 = evaluate_with_prompt(
        client, model,
        prompt_path_master=BASELINE_PROMPTS["master"],
        prompt_path_perception=BASELINE_PROMPTS["perception"],
        prompt_path_production=BASELINE_PROMPTS["production"],
        n_runs_per_case=3,
        verbose=True,
    )
    el = time.time() - t0
    print(f"  elapsed {el/60:.1f}min")
    print(f"  M1={v31.aggregate_m1:.1%} M5={v31.aggregate_m5:.1%} M6={v31.aggregate_m6:.2f} M7={v31.aggregate_m7:.2f} overall={v31.overall_score:.3f}")
    print()

    # 对比
    delta = v31.overall_score - v30_agg['overall_score']
    print(f"=== v3.0 → v3.1 对比 ===")
    print(f"  overall: {v30_agg['overall_score']:.3f} → {v31.overall_score:.3f} (Δ={delta:+.3f})")
    print(f"  M1:      {v30_agg['m1_trigger_accuracy']:.1%} → {v31.aggregate_m1:.1%} (Δ={v31.aggregate_m1 - v30_agg['m1_trigger_accuracy']:+.1%})")
    print(f"  M5:      {v30_agg['m5_tone_match']:.1%} → {v31.aggregate_m5:.1%} (Δ={v31.aggregate_m5 - v30_agg['m5_tone_match']:+.1%})")
    print(f"  M6:      {v30_agg['m6_insight_quality']:.2f} → {v31.aggregate_m6:.2f} (Δ={v31.aggregate_m6 - v30_agg['m6_insight_quality']:+.2f})")
    print(f"  M7:      {v30_agg['m7_safety_score']:.2f} → {v31.aggregate_m7:.2f} (Δ={v31.aggregate_m7 - v30_agg['m7_safety_score']:+.2f})")

    # 保存
    save_full_report(v31, RESULTS / "v31_rule_engine_eval.json",
                     meta={"version": "v3.1", "change": "remove rule8 + add genuine_transformation override", "n_runs": 3})

    # keep/discard
    if delta > 0.003:
        print(f"\n✅ KEEP — overall 提升 {delta:+.3f} > 0.003 阈值")
    elif delta > 0:
        print(f"\n⚠️  marginal — overall 提升 {delta:+.3f} 但未达 0.003 阈值")
    else:
        print(f"\n❌ DISCARD — overall 下降 {delta:+.3f}")

if __name__ == "__main__":
    main()
