"""debug: 看 v1.7 实际输出 JSON 的 keys"""
import json, os, sys
sys.path.insert(0, "D:/prompt-ops/use-cases/parent-child-coach")
sys.path.insert(0, "D:/星灵-soul-手搓/亲子沟通洞见/路线A_自上而下法_鼓励版")
from openai import OpenAI
from auto_evolve.optimizer import load_env, load_golden_dataset, find_case, get_input_text
from auto_evolve.h2h_v17_vs_v30 import parse_v17_prompt

load_env()
client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=os.environ.get("DEEPSEEK_BASE_URL","https://api.deepseek.com"))
sys_p, usr_p = parse_v17_prompt("D:/星灵-soul-手搓/亲子沟通洞见/路线A_自上而下法_鼓励版/prompts/prompt_A轨_v1.7_修复感知版.md")
ds = load_golden_dataset()
case = find_case(ds, "C10-001")
dialogue = case.get("dialogue","")
user = usr_p.replace("{user_input}", dialogue).replace("{profile_context}","").replace("{context_block}","")
resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role":"system","content":sys_p},{"role":"user","content":user}],
    response_format={"type":"json_object"},
    temperature=0.4, timeout=120,
)
raw = resp.choices[0].message.content
try:
    data = json.loads(raw)
except Exception:
    from json_repair import loads
    data = loads(raw)
print("KEYS:", list(data.keys()))
print("tone:", repr(data.get("tone")))
print("should_popup:", repr(data.get("should_popup")))
for k,v in data.items():
    if isinstance(v, dict):
        print(f"  {k} keys:", list(v.keys())[:10])
        if "tone" in v:
            print(f"    {k}.tone:", repr(v["tone"]))
# 打印完整 raw
print("\n--- RAW (first 2000 chars) ---")
print(raw[:2000])
