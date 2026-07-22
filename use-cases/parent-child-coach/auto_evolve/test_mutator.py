"""直接测 propose_mutation 是否报 402"""
import os, sys
from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, r"D:\星灵-soul-手搓\亲子沟通洞见\路线A_自上而下法_鼓励版")
sys.path.insert(0, r"D:\prompt-ops\use-cases\parent-child-coach")

load_dotenv(r"D:\prompt-ops\use-cases\parent-child-coach\.env")
print(f"key: {os.environ['DEEPSEEK_API_KEY'][:15]}...")
print(f"base_url: {os.environ['DEEPSEEK_BASE_URL']}")
print(f"model: {os.environ['DEEPSEEK_MODEL']}")

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url=os.environ["DEEPSEEK_BASE_URL"],
    max_retries=1,
)
model = os.environ["DEEPSEEK_MODEL"]

# 先测最简单的调用
print("\n=== 直接调用 deepseek-chat ===", flush=True)
try:
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "说'可用'"}],
        max_tokens=10,
        temperature=0.4,
    )
    print(f"✅ 成功: {r.choices[0].message.content!r}", flush=True)
except Exception as e:
    print(f"❌ 失败: {e}", flush=True)

# 再测 propose_mutation
print("\n=== 测 propose_mutation ===", flush=True)
from auto_evolve.prompt_mutator import propose_mutation, build_failure_report, read_prompt
from pathlib import Path

RESULTS = Path(r"D:\prompt-ops\use-cases\parent-child-coach\results")
# 用最新基线
baseline_path = RESULTS / "auto_iter_00_baseline.json"
if not baseline_path.exists():
    # 复制 v25_full 作为 baseline
    import shutil
    shutil.copy(RESULTS / "auto_baseline_v25_full.json", baseline_path)

failure_report = build_failure_report(str(baseline_path))
print(f"失败案例数: {len(failure_report.failures)}", flush=True)

P = r"D:\星灵-soul-手搓\亲子沟通洞见\路线A_自上而下法_鼓励版\prompts"
current_text = open(f"{P}/prompt_感知层_v3.1.md", encoding="utf-8").read()

try:
    mutation = propose_mutation(
        client, model, "perception", failure_report,
        current_text=current_text,
        current_version="v2.5",
        previous_attempts=[],
        new_version_override="v2.6",
    )
    print(f"✅ 变异成功: {mutation.version_to}", flush=True)
    print(f"   rationale: {mutation.rationale[:150]}", flush=True)
    print(f"   modified_text len: {len(mutation.modified_text) if mutation.modified_text else 0}", flush=True)
except Exception as e:
    print(f"❌ 变异失败: {e}", flush=True)
