"""生成新增盲区列表，去重已有 EVAL_CASES"""
import json
import sys
sys.path.insert(0, "D:/prompt-ops/use-cases/parent-child-coach")
from auto_evolve.optimizer import EVAL_CASES

d = json.load(open('D:/prompt-ops/use-cases/parent-child-coach/results/full_eval_n1.json', encoding='utf-8'))
existing = set((c, w) for c, w in EVAL_CASES)
blind = set()
for r in d['per_window']:
    issues = []
    if r['m5_tone_match'] == 0 and r['gold_should_popup'] and r['sys_tone'] not in ('', None):
        issues.append('tone_mismatch')
    if r['m6_insight_score'] is not None and r['m6_insight_score'] < 3:
        issues.append('M6_low')
    if r['m7_safety_score'] is not None and r['m7_safety_score'] < 4:
        issues.append('M7_low')
    if r['m1_trigger_match'] == 0 and r['gold_should_popup'] is not None:
        issues.append('M1_mismatch')
    if r['error']:
        issues.append('error')
    if issues:
        blind.add((r['case_id'], r['window_index']))
new_cases = blind - existing
print(f'Existing EVAL_CASES: {len(existing)}')
print(f'Blind spots total: {len(blind)}')
print(f'New to add (not in EVAL_CASES): {len(new_cases)}')
print()
new_sorted = sorted(new_cases)
for c, w in new_sorted:
    print(f'    ("{c}", {w}),')
