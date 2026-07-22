"""多 seed 复测 C11-009：跑 8 次，看 M6 分布。

目的：确认 M6=3.33 是噪声区间还是真实退化。
baseline M6=4.33（diagnostic tone，错配），新版 M6=3.33（empowering tone，匹配）。
如果 8 次都在 3.0~4.0，说明是 empowering 弹窗的固有评分水平（length bias）。
如果有时到 4.0+，说明是噪声。
"""
import os, sys, time, json
from openai import OpenAI
sys.path.insert(0, "D:/星灵-soul-手搓/亲子沟通洞见/路线A_自上而下法_鼓励版")
sys.path.insert(0, "D:/prompt-ops/use-cases/parent-child-coach")

from auto_evolve.optimizer import load_env, find_case, get_input_text, get_gold_labels, load_golden_dataset, _evaluate_case_once
from src.multi_agent_orchestrator import MultiAgentOrchestrator

P = "D:/星灵-soul-手搓/亲子沟通洞见/路线A_自上而下法_鼓励版/prompts"
load_env()
client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))

orch = MultiAgentOrchestrator(
    llm_client=client, model="deepseek-chat",
    prompt_path_master=f"{P}/prompt_总控_v3.1.md",
    prompt_path_perception=f"{P}/prompt_感知层_v3.1.md",
    prompt_path_production=f"{P}/prompt_生产层_v3.1.md",
)

dataset = load_golden_dataset()
case_id, win_idx = "C11-009", 2
case = find_case(dataset, case_id)
gold = get_gold_labels(case, win_idx)
input_text = get_input_text(case, win_idx)

print(f"=== C11-009 多 seed 复测 (8 runs) ===")
print(f"gold tone={gold['tone']}, gold score={gold['score']}")
print(f"start: {time.strftime('%H:%M:%S')}")
print()

results = []
for i in range(8):
    t0 = time.time()
    orch.reset_family(case_id)
    r = _evaluate_case_once(client, "deepseek-chat", orch, case_id, win_idx, case, gold, input_text)
    el = time.time() - t0
    tone = r.sys_tone or "?"
    m6 = r.m6_insight_score if r.m6_insight_score is not None else 0
    m7 = r.m7_safety_score if r.m7_safety_score is not None else 0
    m5 = "✓" if r.m5_tone_match else "✗"
    print(f"  run{i+1}: tone={tone:12s} M5={m5} M6={m6:.1f} M7={m7:.1f} ({el:.0f}s)")
    results.append({"run": i+1, "tone": tone, "m6": m6, "m7": m7, "m5": r.m5_tone_match})

print()
m6s = [r["m6"] for r in results if r["m6"] > 0]
m7s = [r["m7"] for r in results if r["m7"] > 0]
print(f"M6: min={min(m6s):.1f} max={max(m6s):.1f} mean={sum(m6s)/len(m6s):.2f} std={((sum((x-sum(m6s)/len(m6s))**2 for x in m6s)/len(m6s))**0.5):.2f}")
print(f"M7: min={min(m7s):.1f} max={max(m7s):.1f} mean={sum(m7s)/len(m7s):.2f}")
print(f"M5 match: {sum(1 for r in results if r['m5'])}/{len(results)}")
print(f"tone distribution: ", end="")
from collections import Counter
for t, c in Counter(r["tone"] for r in results).most_common():
    print(f"{t}={c} ", end="")
print()

# 判断
m6_mean = sum(m6s)/len(m6s)
if m6_mean < 3.5:
    print(f"\n结论: M6 均值 {m6_mean:.2f} < 3.5，empowering 弹窗 M6 系统性低于 diagnostic（length bias 确认）")
elif max(m6s) - min(m6s) > 1.0:
    print(f"\n结论: M6 range {max(m6s)-min(m6s):.1f} > 1.0，噪声成分大，-1.00 退化含噪声")
else:
    print(f"\n结论: M6 稳定在 {m6_mean:.2f}，退化可能是真实的")

# 存盘
out = {
    "case_id": case_id, "n_runs": 8,
    "gold_tone": gold["tone"], "gold_score": gold["score"],
    "prompts": {"system": "v3.1", "rules": "enabled"},
    "results": results,
    "m6_stats": {"min": min(m6s), "max": max(m6s), "mean": sum(m6s)/len(m6s)},
    "m7_stats": {"min": min(m7s), "max": max(m7s), "mean": sum(m7s)/len(m7s)},
}
out_path = "results/c11_009_multi_seed.json"
from pathlib import Path
Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nsaved: {out_path}")
