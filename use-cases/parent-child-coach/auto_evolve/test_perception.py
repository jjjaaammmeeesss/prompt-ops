"""直接测 perception agent 返回"""
import os, sys, time
from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, r"D:\星灵-soul-手搓\亲子沟通洞见\路线A_自上而下法_鼓励版")
sys.path.insert(0, r"D:\prompt-ops\use-cases\parent-child-coach")

load_dotenv(r"D:\prompt-ops\use-cases\parent-child-coach\.env")
client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=os.environ["DEEPSEEK_BASE_URL"], max_retries=1)
model = os.environ["DEEPSEEK_MODEL"]

from auto_evolve.optimizer import load_golden_dataset, find_case, get_input_text
from src.perception_agent import PerceptionAgent
from src.master_agent import MasterAgent
from src.production_agent import ProductionAgent

dataset = load_golden_dataset()
case = find_case(dataset, "C10-001")
input_text = get_input_text(case, None)
print(f"Input text ({len(input_text)} chars): {input_text[:200]}...", flush=True)

prompt_path = r"D:\星灵-soul-手搓\亲子沟通洞见\路线A_自上而下法_鼓励版\prompts\prompt_感知层_v3.1.md"

# Test perception
t0 = time.time()
pa = PerceptionAgent(client, model, prompt_path=prompt_path)
report = pa.perceive(input_text)
t1 = time.time()
print(f"\n=== Perception ({t1-t0:.1f}s) ===", flush=True)
print(f"emotion_track: {report.emotion_track[:150]!r}", flush=True)
print(f"belief_diagnosis: {report.belief_diagnosis[:150]!r}", flush=True)
print(f"child_state: {report.child_state[:150]!r}", flush=True)
print(f"genuine_transformation: {report.genuine_transformation}", flush=True)
print(f"surface_compliance: {report.surface_compliance}", flush=True)
print(f"defensive_rationalization: {report.defensive_rationalization}", flush=True)
