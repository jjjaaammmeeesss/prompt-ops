"""debug: 为什么 GB_001 sys_tone 为空"""
import os, sys, json
sys.path.insert(0, "D:/prompt-ops/use-cases/parent-child-coach")
sys.path.insert(0, "D:/星灵-soul-手搓/亲子沟通洞见/路线A_自上而下法_鼓励版")
from openai import OpenAI
from auto_evolve.optimizer import load_env, load_golden_dataset, find_case, get_input_text, get_gold_labels
from src.multi_agent_orchestrator import MultiAgentOrchestrator

load_env()
client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=os.environ.get("DEEPSEEK_BASE_URL","https://api.deepseek.com"))
ds = load_golden_dataset()
case = find_case(ds, "GB_001")
dialogue = get_input_text(case, None)
print(f"GB_001 dialogue (first 500):\n{dialogue[:500]}")
print(f"\nGB_001 keys: {list(case.keys())}")
print(f"windows: {case.get('windows')}")
gold = get_gold_labels(case, None)
print(f"gold: should_popup={gold['should_popup']}, tone={gold['tone']}, ref_popup={gold['reference_popup'][:100] if gold['reference_popup'] else None}")

orch = MultiAgentOrchestrator(
    llm_client=client,
    model="deepseek-chat",
    prompt_path_master="D:/星灵-soul-手搓/亲子沟通洞见/路线A_自上而下法_鼓励版/prompts/prompt_总控_v3.1.md",
    prompt_path_perception="D:/星灵-soul-手搓/亲子沟通洞见/路线A_自上而下法_鼓励版/prompts/prompt_感知层_v3.1.md",
    prompt_path_production="D:/星灵-soul-手搓/亲子沟通洞见/路线A_自上而下法_鼓励版/prompts/prompt_生产层_v3.1.md",
)
orch.reset_family("GB_001")
result = orch.process_window(dialogue, family="GB_001")
print(f"\nresult.should_popup={result.should_popup}, tone={result.tone!r}")
print(f"result.popup_text={result.popup_text[:200] if result.popup_text else None!r}")
print(f"result.main_contradiction={result.main_contradiction[:200] if result.main_contradiction else None!r}")
