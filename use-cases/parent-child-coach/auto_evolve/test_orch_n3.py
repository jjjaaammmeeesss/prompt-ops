"""测完整 orchestrator 单案例 n=3"""
import os, sys, time
from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, r"D:\星灵-soul-手搓\亲子沟通洞见\路线A_自上而下法_鼓励版")
sys.path.insert(0, r"D:\prompt-ops\use-cases\parent-child-coach")

load_dotenv(r"D:\prompt-ops\use-cases\parent-child-coach\.env")
client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=os.environ["DEEPSEEK_BASE_URL"], max_retries=1)
model = os.environ["DEEPSEEK_MODEL"]

from auto_evolve.optimizer import load_golden_dataset, find_case, get_input_text, get_gold_labels
from src.multi_agent_orchestrator import MultiAgentOrchestrator

dataset = load_golden_dataset()
case = find_case(dataset, "C10-001")
gold = get_gold_labels(case, None)
input_text = get_input_text(case, None)

P = r"D:\星灵-soul-手搓\亲子沟通洞见\路线A_自上而下法_鼓励版\prompts"
orch = MultiAgentOrchestrator(
    llm_client=client,
    model=model,
    prompt_path_master=f"{P}/prompt_总控_v3.1.md",
    prompt_path_perception=f"{P}/prompt_感知层_v3.1.md",
    prompt_path_production=f"{P}/prompt_生产层_v3.1.md",
)

for i in range(3):
    t0 = time.time()
    orch.reset_family("C10-001")
    rep = orch.process_window(input_text, family="C10-001")
    t1 = time.time()
    print(f"\n=== Run {i+1} ({t1-t0:.1f}s) ===", flush=True)
    print(f"  should_popup: {rep.should_popup}", flush=True)
    print(f"  tone: {rep.tone!r}", flush=True)
    print(f"  main_contradiction: {(rep.main_contradiction or '')[:150]!r}", flush=True)
    print(f"  popup_text: {(rep.popup_text or '')[:150]!r}", flush=True)
    if rep.decision:
        print(f"  decision.should_popup: {rep.decision.should_popup}", flush=True)
        print(f"  decision.direction: {rep.decision.direction!r}", flush=True)
        print(f"  decision.main_contradiction: {(rep.decision.main_contradiction or '')[:100]!r}", flush=True)
