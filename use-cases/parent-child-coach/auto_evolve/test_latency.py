"""详细检查 Kimi 返回"""
import os, time, json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(r"D:\prompt-ops\use-cases\parent-child-coach\.env")
client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=os.environ["DEEPSEEK_BASE_URL"])
model = os.environ["DEEPSEEK_MODEL"]

long_msg = "请分析以下对话片段，输出 JSON:\n妈妈: 你怎么又没写作业\n孩子: 我写了\n\n输出格式: {\"signal\": true/false}"
t0 = time.time()
r = client.chat.completions.create(
    model=model,
    messages=[{"role":"system","content":"你是助手，只输出JSON"},{"role":"user","content":long_msg}],
    temperature=1.0,
    max_tokens=500,
)
t1 = time.time()
c = r.choices[0]
print(f"耗时: {t1-t0:.1f}s", flush=True)
print(f"finish_reason: {c.finish_reason}", flush=True)
print(f"message.content: {c.message.content!r}", flush=True)
print(f"message.role: {c.message.role}", flush=True)
print(f"usage: prompt={r.usage.prompt_tokens} completion={r.usage.completion_tokens} total={r.usage.total_tokens}", flush=True)
# 检查是否有 reasoning_content
if hasattr(c.message, 'reasoning_content') and c.message.reasoning_content:
    print(f"reasoning_content (前200): {c.message.reasoning_content[:200]!r}", flush=True)
