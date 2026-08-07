"""[历史实验] 调试单案例：看感知层和总控的实际 LLM 输出。"""
import json
import os
import sys
from pathlib import Path
from openai import OpenAI

sys.path.insert(0, "D:/星灵-soul-手搓/亲子沟通洞见/路线A_自上而下法_鼓励版")
sys.path.insert(0, "D:/prompt-ops/use-cases/parent-child-coach")

from auto_evolve.optimizer import load_env, find_case, get_input_text, get_gold_labels, load_golden_dataset
from src.perception_agent import PerceptionAgent
from src.master_agent import MasterAgent
from src.case_memory import CaseMemory

PROMPTS = "D:/星灵-soul-手搓/亲子沟通洞见/路线A_自上而下法_鼓励版/prompts"

load_env()
client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)

# 测 C10-002（tone 为空的案例之一）
dataset = load_golden_dataset()
case = find_case(dataset, "C10-002")
gold = get_gold_labels(case, None)
input_text = get_input_text(case, None)

print(f"=== C10-002 调试 ===")
print(f"gold_tone: {gold['tone']}")
print(f"input_text (first 300): {input_text[:300]}")
print()

# 感知层
print(f"--- 感知层 v3.1 ---")
pagent = PerceptionAgent(client, model="deepseek-chat",
    prompt_path=f"{PROMPTS}/prompt_感知层_v3.1.md")
pr = pagent.perceive(input_text)
print(f"emotion_track: {pr.emotion_track}")
print(f"belief_diagnosis: {pr.belief_diagnosis}")
print(f"child_state: {pr.child_state}")
print(f"relation_pattern: {pr.relation_pattern}")
print(f"positive_moment: {pr.positive_moment}")
print(f"response_need: {pr.response_need!r}")
print()

# 总控
print(f"--- 总控 v3.1 ---")
magent = MasterAgent(client, model="deepseek-chat",
    prompt_path=f"{PROMPTS}/prompt_总控_v3.1.md")
memory = CaseMemory()
md = magent.decide(pr, memory, input_text)
print(f"route_a_insight: {md.route_a_insight}")
print(f"route_b_insight: {md.route_b_insight}")
print(f"main_contradiction: {md.main_contradiction}")
print(f"direction: {md.direction!r}")
print(f"should_popup: {md.should_popup}")
print(f"contradiction_flag: {md.contradiction_flag}")
