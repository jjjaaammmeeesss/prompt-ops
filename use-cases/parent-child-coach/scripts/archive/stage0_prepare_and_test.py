"""
Stage 0 Diagnostic: Steps 1-4
- Step 1: Create tiny test dataset from 5 test case files
- Step 2: Generate answers using v1.7 via DeepSeek API
- Step 3: Create a deliberately degraded prompt variant
- Step 4: Test metric sensitivity
"""

import json
import os
import re
import sys
import time

# Add src to path for prompt_ops imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')))

# === Config ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TEST_CASES_DIR = r"C:\Users\h\Downloads\7.6 test_cases\test_cases"
DATA_DIR = os.path.join(BASE_DIR, "data")
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
SYSTEM_PROMPT_PATH = os.path.join(BASE_DIR, "system_prompt.txt")
FEW_SHOTS_PATH = os.path.join(RESULTS_DIR, "config_20260708_161156.json")

# Files to pick (one per category)
FILE_PICKS = {
    "A": "A优秀对话/A1-考试·比赛取得好成绩.txt",
    "B": "B日常对话/B-亲子日常对话.txt",
    "C": "C日常摩擦/C2-电子产品问题.txt",
    "D": "D严重摩擦/D1-情绪崩溃问题.txt",
    "E": "E极端场景/E1-家长情绪失控.txt",
}


def read_stripped_dialogue(filepath):
    """Read a test case file and strip metadata header lines (starting with #)."""
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # Strip lines starting with # (metadata) and blank lines
    dialogue_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        dialogue_lines.append(line.rstrip("\n"))
    return "\n".join(dialogue_lines)


def step1_create_dataset():
    """Step 1: Create the 5-question dataset."""
    print("=" * 60)
    print("Step 1: Creating tiny test dataset")
    print("=" * 60)

    questions = []
    files_picked = []

    for cat_label, rel_path in FILE_PICKS.items():
        full_path = os.path.join(TEST_CASES_DIR, rel_path)
        dialogue = read_stripped_dialogue(full_path)
        questions.append({"question": dialogue, "category": cat_label})
        files_picked.append(os.path.basename(full_path))
        print(f"  [{cat_label}] Loaded: {os.path.basename(full_path)} ({len(dialogue)} chars)")

    os.makedirs(DATA_DIR, exist_ok=True)

    # Save questions-only for step 1 reference
    dataset_path = os.path.join(DATA_DIR, "stage0_5_questions.json")
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    print(f"\n  Saved {len(questions)} questions to {dataset_path}")
    return questions, files_picked


def step2_generate_answers(questions):
    """Step 2: Generate answers using v1.7 via DeepSeek API."""
    print("\n" + "=" * 60)
    print("Step 2: Generating answers via DeepSeek API")
    print("=" * 60)

    # Load system prompt
    with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        system_prompt = f.read()
    print(f"  System prompt loaded: {len(system_prompt)} chars")

    # Load few-shot examples
    with open(FEW_SHOTS_PATH, "r", encoding="utf-8") as f:
        few_shots_data = json.load(f)
    few_shots = few_shots_data.get("few_shots", [])
    print(f"  Few-shot examples loaded: {len(few_shots)}")

    # Load API key from .env
    env_path = os.path.join(BASE_DIR, ".env")
    api_key = None
    with open(env_path, "r") as f:
        for line in f:
            if line.startswith("DEEPSEEK_API_KEY="):
                api_key = line.strip().split("=", 1)[1]
                break
    if not api_key:
        print("  ERROR: DEEPSEEK_API_KEY not found in .env")
        return None, False

    import requests

    results = []
    success = True

    for i, q in enumerate(questions):
        dialogue = q["question"]
        category = q["category"]
        print(f"\n  [{i+1}/5] Category {category}: Sending to DeepSeek...")

        # Build messages: system prompt + few-shot examples + current question
        messages = [{"role": "system", "content": system_prompt}]

        # Add few-shot examples
        for shot in few_shots:
            messages.append({"role": "user", "content": shot["question"]})
            messages.append({"role": "assistant", "content": shot["answer"]})

        # Add current question
        messages.append({"role": "user", "content": dialogue})

        try:
            resp = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "deepseek-chat",
                    "messages": messages,
                    "temperature": 0.0,
                    "max_tokens": 1024,
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            answer_text = data["choices"][0]["message"]["content"]

            # Try to parse JSON from the response
            # DeepSeek may wrap in ```json ... ``` or return raw JSON
            answer_json = extract_json(answer_text)
            if answer_json:
                print(f"    Parsed JSON successfully: should_popup={answer_json.get('should_popup')}, tone={answer_json.get('tone')}")
                results.append({
                    "question": dialogue,
                    "answer": json.dumps(answer_json, ensure_ascii=False),
                    "category": category,
                })
            else:
                print(f"    WARNING: Could not parse JSON from response. Raw: {answer_text[:200]}...")
                results.append({
                    "question": dialogue,
                    "answer": answer_text,
                    "category": category,
                })

        except Exception as e:
            print(f"    ERROR: {e}")
            success = False
            results.append({
                "question": dialogue,
                "answer": "{}",
                "category": category,
                "error": str(e),
            })

        # 1s delay between calls
        if i < len(questions) - 1:
            time.sleep(1)

    # Save results
    output_path = os.path.join(DATA_DIR, "stage0_5_examples.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved {len(results)} examples to {output_path}")

    return results, success


def extract_json(text):
    """Extract valid JSON from model response."""
    if not text:
        return None
    # Try raw parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to extract from ```json ... ``` code block
    json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(json_block_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    # Try to find JSON object in the text
    brace_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(brace_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    return None


def step3_create_degraded_prompt():
    """Step 3: Create a deliberately degraded prompt by removing methodology sections."""
    print("\n" + "=" * 60)
    print("Step 3: Creating degraded prompt variant")
    print("=" * 60)

    with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        full_prompt = f.read()

    # Remove the methodology section: everything about 三维六格, 盲区二分, 七层结构, CBT
    # We want to keep the values layer but remove the methodology layer
    # Specifically, remove from "## 盲区二分" to "## 系统默认站位" (the entire methodology block)
    # And also remove the detailed tables

    # Strategy: find and remove the key methodology blocks
    degraded = full_prompt

    # Remove 盲区二分 section (from "## 盲区二分" through end of table)
    degraded = re.sub(
        r'## 盲区二分.*?(?=\n## |\n# )',
        '',
        degraded,
        flags=re.DOTALL
    )

    # Remove 三维六格 section
    degraded = re.sub(
        r'## 三维六格.*?(?=\n## |\n# )',
        '',
        degraded,
        flags=re.DOTALL
    )

    # Remove 七层结构 section
    degraded = re.sub(
        r'## 七层结构.*?(?=\n## |\n# )',
        '',
        degraded,
        flags=re.DOTALL
    )

    # Remove 系统默认站位 section
    degraded = re.sub(
        r'## 系统默认站位.*?(?=\n## |\n# )',
        '',
        degraded,
        flags=re.DOTALL
    )

    # Remove 流程总览 if it exists (the last section)
    degraded = re.sub(
        r'# 流程总览.*$',
        '',
        degraded,
        flags=re.DOTALL
    )

    # Clean up excessive blank lines
    degraded = re.sub(r'\n{3,}', '\n\n', degraded)

    output_path = os.path.join(DATA_DIR, "stage0_degraded_prompt.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(degraded)

    print(f"  Original prompt: {len(full_prompt)} chars")
    print(f"  Degraded prompt: {len(degraded)} chars")
    print(f"  Removed: {len(full_prompt) - len(degraded)} chars of methodology")
    print(f"  Saved to: {output_path}")

    return degraded, True


def step4_test_metric_sensitivity(examples):
    """Step 4: Test metric sensitivity between v1.7 and degraded prompt predictions."""
    print("\n" + "=" * 60)
    print("Step 4: Testing metric sensitivity")
    print("=" * 60)

    if not examples:
        print("  ERROR: No examples to test with")
        return False, 0, 0

    # Load system prompt and degraded prompt
    with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        v17_prompt = f.read()

    degraded_prompt_path = os.path.join(DATA_DIR, "stage0_degraded_prompt.txt")
    with open(degraded_prompt_path, "r", encoding="utf-8") as f:
        degraded_prompt = f.read()

    # Load API key
    env_path = os.path.join(BASE_DIR, ".env")
    api_key = None
    with open(env_path, "r") as f:
        for line in f:
            if line.startswith("DEEPSEEK_API_KEY="):
                api_key = line.strip().split("=", 1)[1]
                break

    import requests

    v17_scores = []
    degraded_scores = []

    for i, ex in enumerate(examples):
        question = ex["question"]
        v17_answer_str = ex.get("answer", "{}")

        # Parse v17 answer as ground truth
        try:
            v17_answer = json.loads(v17_answer_str)
        except json.JSONDecodeError:
            v17_answer_str = extract_json_str(v17_answer_str)
            if v17_answer_str:
                try:
                    v17_answer = json.loads(v17_answer_str)
                except json.JSONDecodeError:
                    v17_answer = {}
            else:
                v17_answer = {}

        print(f"\n  [{i+1}/5] Testing example {i+1}...")

        # Generate prediction with degraded prompt
        try:
            degraded_pred = call_deepseek(question, degraded_prompt, api_key)
        except Exception as e:
            print(f"    ERROR calling degraded prompt: {e}")
            degraded_pred = {}

        # Compute scores using selected_fields_comparison
        fields = ["should_popup", "tone", "popup_insight"]
        v17_score = compute_field_score(v17_answer, v17_answer, fields)  # self-score
        degraded_score = compute_field_score(v17_answer, degraded_pred, fields)

        v17_scores.append(v17_score)
        degraded_scores.append(degraded_score)

        print(f"    v1.7 self-score: {v17_score:.3f}, degraded score: {degraded_score:.3f}")

        time.sleep(1)

    avg_v17 = sum(v17_scores) / len(v17_scores) if v17_scores else 0
    avg_degraded = sum(degraded_scores) / len(degraded_scores) if degraded_scores else 0

    metric_sensitive = avg_v17 > avg_degraded

    print(f"\n  Average v1.7 self-score: {avg_v17:.3f}")
    print(f"  Average degraded score:  {avg_degraded:.3f}")
    print(f"  Metric sensitive: {'YES (v1.7 > degraded)' if metric_sensitive else 'NO (scores identical or reversed)'}")

    return metric_sensitive, avg_v17, avg_degraded


def call_deepseek(dialogue, system_prompt, api_key):
    """Call DeepSeek API with system prompt and dialogue."""
    import requests

    # Load few-shot examples
    with open(FEW_SHOTS_PATH, "r", encoding="utf-8") as f:
        few_shots_data = json.load(f)
    few_shots = few_shots_data.get("few_shots", [])

    messages = [{"role": "system", "content": system_prompt}]
    for shot in few_shots:
        messages.append({"role": "user", "content": shot["question"]})
        messages.append({"role": "assistant", "content": shot["answer"]})
    messages.append({"role": "user", "content": dialogue})

    resp = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 1024,
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    answer_text = data["choices"][0]["message"]["content"]

    result = extract_json(answer_text)
    return result if result else {}


def compute_field_score(gold, pred, fields):
    """Compute selected_fields_comparison score between gold and pred JSON."""
    score = 0.0
    valid_fields = 0
    for field in fields:
        if field in gold:
            valid_fields += 1
            if field in pred and str(gold[field]) == str(pred[field]):
                score += 1.0
    return score / valid_fields if valid_fields > 0 else 0.0


def extract_json_str(text):
    """Extract JSON string from model response."""
    if not text:
        return None
    json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(json_block_pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    brace_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(brace_pattern, text, re.DOTALL)
    if matches:
        return matches[0]
    return None


def main():
    print("=" * 60)
    print("STAGE 0 DIAGNOSTIC: Steps 1-4")
    print("=" * 60)

    # Step 1
    questions, files_picked = step1_create_dataset()

    # Step 2
    examples, step2_success = step2_generate_answers(questions)

    # Step 3
    degraded_prompt, step3_success = step3_create_degraded_prompt()

    # Step 4
    if examples:
        metric_sensitive, v17_score, degraded_score = step4_test_metric_sensitivity(examples)
    else:
        metric_sensitive = False
        v17_score = 0
        degraded_score = 0
        print("\n  Step 4 SKIPPED: No examples available")

    # Summary
    print("\n" + "=" * 60)
    print("STEPS 1-4 SUMMARY")
    print("=" * 60)
    print(f"  Step 1 - Files picked: {files_picked}")
    print(f"  Step 2 - Answers generated: {step2_success}")
    print(f"  Step 3 - Degraded prompt created: {step3_success}")
    print(f"  Step 4 - Metric sensitive: {metric_sensitive}")
    print(f"  Step 4 - v1.7 self-score: {v17_score:.3f}")
    print(f"  Step 4 - Degraded score: {degraded_score:.3f}")

    # Save partial report for step 6
    partial_report = {
        "step1_files_picked": files_picked,
        "step2_answers_generated": step2_success,
        "step3_degraded_prompt_created": step3_success,
        "step4_metric_sensitive": metric_sensitive,
        "step4_v17_self_score": round(v17_score, 3),
        "step4_degraded_score": round(degraded_score, 3),
    }
    report_path = os.path.join(DATA_DIR, "stage0_partial_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(partial_report, f, ensure_ascii=False, indent=2)
    print(f"\n  Partial report saved to {report_path}")


if __name__ == "__main__":
    main()
