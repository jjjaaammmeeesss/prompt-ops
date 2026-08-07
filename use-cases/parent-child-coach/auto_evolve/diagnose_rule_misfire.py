"""诊断 22 个 tone_mismatch 案例的规则触发情况"""
import json
import sys
sys.path.insert(0, "D:/星灵-soul-手搓/亲子沟通洞见/路线A_自上而下法_鼓励版")
sys.path.insert(0, "D:/prompt-ops/use-cases/parent-child-coach")

from auto_evolve.optimizer import load_golden_dataset, find_case, get_input_text, get_gold_labels, EVAL_CASES
from src.perception_agent import PerceptionAgent
from src.tone_rules import decide_tone, _SAFETY_KEYWORDS, _CONFLICT_KEYWORDS, _OVERLOAD_KEYWORDS, _GENERALIZATION_KEYWORDS
from src.case_memory import PerceptionReport, MasterDecision
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

b = json.load(open("D:/prompt-ops/use-cases/parent-child-coach/results/auto_baseline_v25_full.json", encoding='utf-8'))

# 找 tone_mismatch 案例
mismatches = []
for c in b['per_case']:
    if c.get('m5_tone_match') == 0 and c.get('gold_should_popup', True) and c.get('sys_tone'):
        mismatches.append(c)

print(f"Tone mismatch cases: {len(mismatches)}")
print()

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)
model = "deepseek-chat"

ds = load_golden_dataset()

# 抽样诊断前 8 个
for c in mismatches[:8]:
    case_id = c['case_id']
    win_idx = c.get('window_index', 1)
    case = find_case(ds, case_id)
    if not case:
        continue
    input_text = get_input_text(case, win_idx)
    gold_tone = c['gold_tone']
    sys_tone = c['sys_tone']

    # 跑感知层拿信号
    try:
        agent = PerceptionAgent(client, model,
                                 prompt_path="D:/星灵-soul-手搓/亲子沟通洞见/路线A_自上而下法_鼓励版/prompts/prompt_感知层_v3.1.md")
        perception = agent.perceive(input_text)
    except Exception as e:
        perception = PerceptionReport(case_id=case_id, window_index=win_idx)
        print(f"  perception error: {e}")

    # 模拟 LLM decision (用 sys_tone 作为 LLM 判定)
    decision = MasterDecision(direction=sys_tone, main_contradiction="")

    # 跑规则引擎
    final_tone, overridden, rule_reason = decide_tone(decision, perception, input_text)

    # 关键词命中
    kw_hits = {
        'safety': [k for k in _SAFETY_KEYWORDS if k in input_text],
        'conflict': [k for k in _CONFLICT_KEYWORDS if k in input_text],
        'overload': [k for k in _OVERLOAD_KEYWORDS if k in input_text],
        'generalization': [k for k in _GENERALIZATION_KEYWORDS if k in input_text],
    }

    print(f"{case_id:10s} w{win_idx} | gold={gold_tone:11s} sys={sys_tone:11s} | final={final_tone:11s} override={overridden}")
    print(f"  signals: gen={perception.has_generalization} label={perception.has_labeling} conflict={perception.has_conflict_escalation} safety={perception.has_safety_emergency} need_unmet={perception.child_core_need_unmet} overload={perception.parent_emotion_overload}")
    print(f"  kw_hits: {kw_hits}")
    print(f"  rule: {rule_reason}")
    print()
