"""临时调试脚本：读取三个稳定错案例的对话内容 + 黄金标签。"""
import json
import sys
from pathlib import Path

DATA = Path("D:/prompt-ops/use-cases/parent-child-coach/data/golden_dataset.json")
with open(DATA, encoding="utf-8") as f:
    ds = json.load(f)

# 三个稳定错案例 + 对应的 window_index（来自 EVAL_CASES）
targets = {
    "C10-008": None,
    "C11-006": 1,
    "C11-009": 2,
}

for c in ds:
    if c["case_id"] not in targets:
        continue
    win_idx = targets[c["case_id"]]
    print("=" * 80)
    print(f"CASE: {c['case_id']}  window_index={win_idx}")
    print(f"case overall context: {c.get('case_summary', '')[:200]}")
    print()

    # 找对应 window
    win = None
    if win_idx is None:
        win = c.get("windows", [{}])[0] if c.get("windows") else {}
    else:
        for w in c.get("windows", []):
            if w.get("window_index") == win_idx:
                win = w
                break

    if not win:
        print("  !! no matching window found, dumping first window")
        win = c.get("windows", [{}])[0] if c.get("windows") else {}

    print(f"  expected_tone: {win.get('expected_tone')}")
    print(f"  overall_score: {win.get('overall_score')}")
    print(f"  should_popup: {win.get('should_popup')}")
    print()
    print(f"  reference_popup:")
    print(f"  {(win.get('reference_popup') or '')[:600]}")
    print()
    print(f"  hit_checklist: {win.get('hit_checklist', [])}")
    print(f"  forbid_checklist: {win.get('forbid_checklist', [])}")
    print()
    txt = win.get("window_text") or c.get("dialogue", "")
    print(f"  window_text ({len(txt)} chars):")
    print(f"  {txt[:1200]}")
    print()
