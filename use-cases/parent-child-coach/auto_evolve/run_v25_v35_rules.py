"""[历史实验] 星灵多智能体降噪评估，对比基线。

感知层输出 6 个信号字段，orchestrator 的规则层基于信号覆写 tone。
生产层重新定义 empowering——高冲突场景写"带退路的鼓励"。
"""
import json
import os
import sys
import time
from pathlib import Path
from openai import OpenAI

sys.path.insert(0, "D:/星灵-soul-手搓/亲子沟通洞见/路线A_自上而下法_鼓励版")
sys.path.insert(0, "D:/prompt-ops/use-cases/parent-child-coach")

from auto_evolve.optimizer import load_env, evaluate_with_prompt, EVAL_CASES

P = "D:/星灵-soul-手搓/亲子沟通洞见/路线A_自上而下法_鼓励版/prompts"

load_env()
client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)

print(f"星灵多智能体 + rules + 生产层 n=3, cases={len(EVAL_CASES)}")
print(f"start: {time.strftime('%H:%M:%S')}")
t0 = time.time()
report = evaluate_with_prompt(
    client, "deepseek-chat",
    prompt_path_master=f"{P}/prompt_总控_v3.1.md",
    prompt_path_perception=f"{P}/prompt_感知层_v3.1.md",
    prompt_path_production=f"{P}/prompt_生产层_v3.1.md",
    n_runs_per_case=3, verbose=True,
)
el = time.time() - t0

print("=" * 70)
print(f"elapsed {el/60:.1f}min")
print(f"M1={report.aggregate_m1:.1%} M5={report.aggregate_m5:.1%} M6={report.aggregate_m6:.2f} M7={report.aggregate_m7:.2f} overall={report.overall_score:.3f}")

# 对比基线
base_path = Path("results/denoised_baseline_n3.json")
if base_path.exists():
    base = json.loads(base_path.read_text(encoding="utf-8"))
    ba = base["aggregate"]
    print(f"\n对比旧基线:")
    print(f"  {'指标':10s} {'基线':>8s} {'新版':>8s} {'Δ':>8s}")
    for key, label in [("m1","M1"),("m5","M5"),("m6","M6"),("m7","M7"),("overall","综合")]:
        b = ba.get({"m1":"m1_trigger_accuracy","m5":"m5_tone_match","m6":"m6_insight_quality","m7":"m7_safety_score","overall":"overall_score"}[key], 0)
        c = {"m1":report.aggregate_m1,"m5":report.aggregate_m5,"m6":report.aggregate_m6,"m7":report.aggregate_m7,"overall":report.overall_score}[key]
        d = c - b
        arrow = "↑" if d > 0 else ("↓" if d < 0 else "→")
        print(f"  {label:10s} {b:8.3f} {c:8.3f} {arrow}{abs(d):7.3f}")

# 对比 rules（无生产层）
prev_path = Path("results/v25_v35_rules_denoised_n3.json")
if prev_path.exists():
    prev = json.loads(prev_path.read_text(encoding="utf-8"))
    pa = prev["aggregate"]
    print(f"\n对比 rules（无生产层）:")
    print(f"  {'指标':10s} {'无生产层':>8s} {'有生产层':>8s} {'Δ':>8s}")
    for key, label in [("m1","M1"),("m5","M5"),("m6","M6"),("m7","M7"),("overall","综合")]:
        p = pa.get({"m1":"m1","m5":"m5","m6":"m6","m7":"m7","overall":"overall"}[key], 0)
        c = {"m1":report.aggregate_m1,"m5":report.aggregate_m5,"m6":report.aggregate_m6,"m7":report.aggregate_m7,"overall":report.overall_score}[key]
        d = c - p
        arrow = "↑" if d > 0 else ("↓" if d < 0 else "→")
        print(f"  {label:10s} {p:8.3f} {c:8.3f} {arrow}{abs(d):7.3f}")

# 存盘
out = Path("results/v25_v35_rules_v32prod_denoised_n3.json")
out.parent.mkdir(parents=True, exist_ok=True)
d = {
    "n_runs": 3, "prompts": {"system": "v3.1", "rules": "enabled"}, "elapsed": el,
    "aggregate": {
        "m1": report.aggregate_m1, "m5": report.aggregate_m5,
        "m6": report.aggregate_m6, "m7": report.aggregate_m7, "overall": report.overall_score,
    },
    "per_case": [
        {"case_id": r.case_id, "sys_tone": r.sys_tone, "gold_tone": r.gold_tone,
         "m1": r.m1_trigger_match, "m5": r.m5_tone_match,
         "m6": r.m6_insight_score, "m7": r.m7_safety_score, "info": r.error}
        for r in report.results
    ],
}
out.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nsaved: {out}")
