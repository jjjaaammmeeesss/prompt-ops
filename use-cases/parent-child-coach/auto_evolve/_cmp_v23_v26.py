"""对比 v2.3 vs v2.6 在用例 1/3/9 上的弹窗输出。"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from auto_evolve.dual_client import init_clients
from auto_evolve.v23_runner import run_v23_once, _JSON_OUTPUT_INSTRUCTION_V26, _TONE_MAP, _normalize_tone
from auto_evolve.optimizer import load_golden_dataset, find_case, get_input_text, get_gold_labels, EVAL_CASES

# 选 case #1, #3, #9（1-based → index 0, 2, 8）
SELECTED = [0, 2, 8]

PROMPT_V23 = Path("prompts_archive/system_prompt_v2.3.txt").read_text(encoding="utf-8")
PROMPT_V26 = Path("prompts_archive/system_prompt_v2.6.txt").read_text(encoding="utf-8")

def main():
    init_clients()
    from auto_evolve.dual_client import task_client, task_model

    dataset = load_golden_dataset()

    for idx in SELECTED:
        case_id, win_idx = EVAL_CASES[idx]
        case = find_case(dataset, case_id)
        gold = get_gold_labels(case, win_idx)
        dialogue = get_input_text(case, win_idx)

        print(f"{'='*70}")
        print(f"Case #{idx+1}: {case_id}" + (f" w{win_idx}" if win_idx else ""))
        print(f"  Gold tone: {gold['tone']} | should_popup: {gold['should_popup']}")
        ref = gold.get('reference_popup', '')
        if ref and '内容标注' not in ref:
            print(f"  Reference: {ref[:120]}")
        print(f"  对话:")
        for line in dialogue.strip().split('\n')[:15]:
            print(f"    {line}")
        if len(dialogue.strip().split('\n')) > 15:
            print(f"    ... ({len(dialogue.strip().split(chr(10)))} lines total)")
        print()

        for label, prompt, json_inst in [
            ("v2.3", PROMPT_V23, None),
            ("v2.6", PROMPT_V26, _JSON_OUTPUT_INSTRUCTION_V26),
        ]:
            t0 = time.time()
            r = run_v23_once(task_client, task_model, prompt, dialogue,
                            json_output_instruction=json_inst, temperature=0.3)
            el = time.time() - t0

            tone = r['tone'] or '(空)'
            popup = r['popup_text'] or '(空)'
            err = r['error']

            print(f"  [{label}] {el:.1f}s | tone={tone}")
            if err:
                print(f"    ❌ ERROR: {err}")
            else:
                print(f'    → "{popup}"')
        print()

if __name__ == "__main__":
    main()
