"""验证所有 EVAL_CASES 都有有效输入文本和 gold 标签"""
import sys
sys.path.insert(0, "D:/prompt-ops/use-cases/parent-child-coach")
from auto_evolve.optimizer import EVAL_CASES, load_golden_dataset, find_case, get_input_text, get_gold_labels

ds = load_golden_dataset()
problems = []
for case_id, win_idx in EVAL_CASES:
    case = find_case(ds, case_id)
    if not case:
        problems.append((case_id, win_idx, "case not found"))
        continue
    input_text = get_input_text(case, win_idx)
    gold = get_gold_labels(case, win_idx)
    if not input_text.strip():
        problems.append((case_id, win_idx, "empty input_text"))
    if not gold["tone"]:
        problems.append((case_id, win_idx, "empty gold tone"))
    if gold["should_popup"] is None:
        problems.append((case_id, win_idx, "gold should_popup is None"))

print(f"Total EVAL_CASES: {len(EVAL_CASES)}")
print(f"Problems: {len(problems)}")
for p in problems:
    print(f"  {p}")
if not problems:
    print("✅ All cases have valid input and gold labels")
