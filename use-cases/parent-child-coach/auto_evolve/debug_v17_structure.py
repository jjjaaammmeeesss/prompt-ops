"""debug: dump v1.7 raw response on C10-001"""
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
dialogue = get_input_text(case, None)
user = usr_p.replace("{user_input}", dialogue).replace("{profile_context}","").replace("{context_block}","")
resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role":"system","content":sys_p},{"role":"user","content":user}],
    response_format={"type":"json_object"},
    temperature=0.4, timeout=120,
)
raw = resp.choices[0].message.content
# 保存 raw 到文件（避免终端 mojibake）
out = "D:/prompt-ops/use-cases/parent-child-coach/results/debug_v17_c10-001_raw.json"
open(out, "w", encoding="utf-8").write(raw)
print(f"saved to {out}")
print(f"length: {len(raw)} chars")
# 解析并 dump 结构
try:
    data = json.loads(raw)
except Exception as e:
    print(f"JSON parse failed: {e}")
    from json_repair import loads
    data = loads(raw)

def walk(obj, prefix="", depth=0):
    if depth > 6:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str):
                vs = v[:60].replace("\n"," ")
                print(f"{prefix}{k}: str({len(v)}) = {vs!r}")
            elif isinstance(v, (dict, list)):
                print(f"{prefix}{k}: {type(v).__name__}({len(v)})")
                walk(v, prefix+"  ", depth+1)
            else:
                print(f"{prefix}{k}: {v!r}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj[:3]):
            print(f"{prefix}[{i}]:")
            walk(item, prefix+"  ", depth+1)

walk(data)
