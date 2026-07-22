"""冒烟测试：用 Kimi API 跑 2 个案例确认全链路通"""
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, r"D:\星灵-soul-手搓\亲子沟通洞见\路线A_自上而下法_鼓励版")
sys.path.insert(0, r"D:\prompt-ops\use-cases\parent-child-coach")

from auto_evolve.optimizer import evaluate_with_prompt, EVAL_CASES
import auto_evolve.optimizer as opt

load_dotenv(r"D:\prompt-ops\use-cases\parent-child-coach\.env")

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url=os.environ["DEEPSEEK_BASE_URL"],
)
model = os.environ["DEEPSEEK_MODEL"]
print(f"🔑 API: {os.environ['DEEPSEEK_BASE_URL']}")
print(f"🤖 Model: {model}")
print(f"📋 EVAL_CASES total: {len(EVAL_CASES)}, smoke test: 2")

orig_cases = opt.EVAL_CASES
opt.EVAL_CASES = EVAL_CASES[:2]

import time
print(f"\n开始评估 {len(opt.EVAL_CASES)} 个案例...", flush=True)
t0 = time.time()

try:
    report = evaluate_with_prompt(
        client=client,
        model=model,
        n_runs_per_case=1,
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"✅ 冒烟测试通过 (耗时 {elapsed:.0f}s)")
    print(f"   overall: {report.overall_score:.4f}")
    print(f"   M1: {report.aggregate_m1:.4f}")
    print(f"   M5: {report.aggregate_m5:.4f}")
    print(f"   M6: {report.aggregate_m6:.3f}")
    print(f"   M7: {report.aggregate_m7:.3f}")
    print(f"   cases: {len(report.results)}")
    for r in report.results:
        m6 = r.m6_insight_score if r.m6_insight_score is not None else -1
        print(f"   - {r.case_id}/w{r.window_index}: tone={r.sys_tone} (gold={r.gold_tone}) M5={r.m5_tone_match} M6={m6:.2f}")
        print(f"     error: {r.error}")
        if r.m6_judge_raw:
            print(f"     m6_raw: {r.m6_judge_raw[:200]}")
        if r.sys_popup_text:
            print(f"     popup: {r.sys_popup_text[:200]}")
finally:
    opt.EVAL_CASES = orig_cases
