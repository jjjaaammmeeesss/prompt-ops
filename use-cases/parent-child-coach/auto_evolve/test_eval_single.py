"""用 evaluate_with_prompt 跑 C10-001 单案 n=3，复现基线问题"""
import os, sys, time
from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, r"D:\星灵-soul-手搓\亲子沟通洞见\路线A_自上而下法_鼓励版")
sys.path.insert(0, r"D:\prompt-ops\use-cases\parent-child-coach")

load_dotenv(r"D:\prompt-ops\use-cases\parent-child-coach\.env")
client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=os.environ["DEEPSEEK_BASE_URL"], max_retries=1)
model = os.environ["DEEPSEEK_MODEL"]

import auto_evolve.optimizer as opt
from auto_evolve.optimizer import evaluate_with_prompt

# 只跑 C10-001
opt.EVAL_CASES = [("C10-001", None)]

P = r"D:\星灵-soul-手搓\亲子沟通洞见\路线A_自上而下法_鼓励版\prompts"

t0 = time.time()
report = evaluate_with_prompt(
    client=client,
    model=model,
    prompt_path_master=f"{P}/prompt_总控_v3.1.md",
    prompt_path_perception=f"{P}/prompt_感知层_v3.1.md",
    prompt_path_production=f"{P}/prompt_生产层_v3.1.md",
    n_runs_per_case=3,
    verbose=True,
)
t1 = time.time()
print(f"\n=== 总耗时 {t1-t0:.0f}s ===", flush=True)
print(f"overall: {report.overall_score:.4f}", flush=True)
print(f"M1: {report.aggregate_m1:.4f} M5: {report.aggregate_m5:.4f}", flush=True)
for r in report.results:
    print(f"  case={r.case_id} should_popup={r.sys_should_popup} tone={r.sys_tone!r} M5={r.m5_tone_match} M6={r.m6_insight_score}", flush=True)
    print(f"  error: {r.error}", flush=True)
    print(f"  popup: {(r.sys_popup_text or '')[:150]!r}", flush=True)
