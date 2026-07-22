"""
Generate golden answers for all 82 test case questions using the v1.7 prompt via DeepSeek API.

Reuses extract_json() and API call pattern from stage0_prepare_and_test.py.
"""

import json
import os
import re
import sys
import time
import requests

# === Config ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
SYSTEM_PROMPT_PATH = os.path.join(BASE_DIR, "system_prompt.txt")
FEW_SHOTS_PATH = os.path.join(BASE_DIR, "results", "config_20260708_161156.json")
QUESTIONS_PATH = os.path.join(DATA_DIR, "test_cases_questions.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "test_cases_with_answers.json")
ENV_PATH = os.path.join(BASE_DIR, ".env")


def load_api_key():
    """Load DEEPSEEK_API_KEY from .env file."""
    if not os.path.exists(ENV_PATH):
        print(f"ERROR: .env not found at {ENV_PATH}")
        return None
    with open(ENV_PATH, "r") as f:
        for line in f:
            if line.startswith("DEEPSEEK_API_KEY="):
                return line.strip().split("=", 1)[1]
    print("ERROR: DEEPSEEK_API_KEY not found in .env")
    return None


def extract_json(text):
    """
    Extract valid JSON from model response.
    Copied from stage0_prepare_and_test.py.
    """
    if not text:
        return None
    # Try raw parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to extract from ```json ... ``` code block
    json_block_pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
    matches = re.findall(json_block_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    # Try to find JSON object in the text
    brace_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
    matches = re.findall(brace_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    return None


def call_deepseek(dialogue, system_prompt, api_key, few_shots):
    """
    Call DeepSeek API with system prompt + few-shot examples + current dialogue.
    Returns (answer_text, parsed_json or None, error_string or None).
    """
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
    parsed = extract_json(answer_text)
    return answer_text, parsed, None


def generate_all():
    """Main generation loop."""
    print("=" * 60)
    print("GENERATING GOLDEN ANSWERS FOR 82 TEST CASES")
    print("=" * 60)

    # Load questions
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"\nLoaded {len(questions)} questions from {QUESTIONS_PATH}")

    # Load system prompt
    with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        system_prompt = f.read()
    print(f"System prompt loaded: {len(system_prompt)} chars")

    # Load few-shot examples
    with open(FEW_SHOTS_PATH, "r", encoding="utf-8") as f:
        few_shots_data = json.load(f)
    few_shots = few_shots_data.get("few_shots", [])
    print(f"Few-shot examples loaded: {len(few_shots)}")

    # Load API key
    api_key = load_api_key()
    if not api_key:
        print("\nFATAL: Cannot proceed without API key.")
        return

    results = []
    errors = 0

    print(f"\nStarting generation of {len(questions)} answers...\n")

    for i, q in enumerate(questions):
        dialogue = q["question"]
        category = q.get("category", "")
        source = q.get("source", "")
        idx = i + 1

        try:
            answer_text, parsed, err = call_deepseek(
                dialogue, system_prompt, api_key, few_shots
            )

            if parsed and isinstance(parsed, dict):
                # Fill missing popup_suggestion with ""
                if "popup_suggestion" not in parsed:
                    parsed["popup_suggestion"] = ""
                results.append({
                    "question": dialogue,
                    "answer": json.dumps(parsed, ensure_ascii=False),
                    "category": category,
                    "source": source,
                })
                print(f"  [{idx:3d}/82] OK  | should_popup={parsed.get('should_popup')}, tone={parsed.get('tone')} | {source}")
            else:
                # Can't parse JSON, save raw
                results.append({
                    "question": dialogue,
                    "answer": json.dumps({}, ensure_ascii=False),
                    "category": category,
                    "source": source,
                })
                errors += 1
                raw_preview = answer_text[:120].replace("\n", " ") if answer_text else "(empty)"
                print(f"  [{idx:3d}/82] BAD | JSON parse failed | raw: {raw_preview}... | {source}")

        except Exception as e:
            error_preview = str(e)[:100].replace("\n", " ")
            results.append({
                "question": dialogue,
                "answer": json.dumps({}, ensure_ascii=False),
                "category": category,
                "source": source,
                "error": str(e),
            })
            errors += 1
            print(f"  [{idx:3d}/82] ERR | {error_preview} | {source}")

        # Progress report every 10 calls
        if idx % 10 == 0:
            print(f"  ---> {idx}/82 completed, {errors} errors so far")

        # Early abort if error rate exceeds 10% (8+ failures)
        if errors >= 8:
            print(f"\n!! ABORTED: Error count reached {errors} (>{10}% of 82). Stopping generation.")
            break

        # 1-second delay between calls (skip after last)
        if idx < len(questions):
            time.sleep(1)

    # === Final report ===
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)

    generated_count = len(results)
    parse_success_count = 0
    for r in results:
        try:
            ans = json.loads(r["answer"])
            if ans and "should_popup" in ans:
                parse_success_count += 1
        except (json.JSONDecodeError, TypeError):
            pass

    print(f"  Total generated:  {generated_count}")
    print(f"  Parse success:    {parse_success_count} (with 'should_popup' field)")
    print(f"  Errors:           {errors}")

    # Verification
    verification_pass = True
    if generated_count != 82 and errors < 8:
        print(f"  WARNING: Expected 82 entries but generated {generated_count}")
        verification_pass = False
    else:
        print(f"  Count check:      {'PASS' if generated_count == 82 else 'partial (early abort)'}")
    if parse_success_count < 75 and errors < 8:
        print(f"  WARNING: Only {parse_success_count}/82 parsed as valid JSON with should_popup")
        verification_pass = False
    else:
        print(f"  Parse rate check: {'PASS' if parse_success_count >= 75 else 'partial (early abort)'}")

    # Check every entry has non-empty answer
    empty_answers = sum(1 for r in results if not r.get("answer") or r["answer"] == "{}")
    if empty_answers > 0:
        print(f"  Empty answers:    {empty_answers} (these are errors or parse failures)")

    print(f"  Verification:     {'PASS' if verification_pass else 'see warnings above'}")

    # Save results with utf-8-sig encoding
    with open(OUTPUT_PATH, "w", encoding="utf-8-sig") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved to: {OUTPUT_PATH}")

    return results, generated_count, parse_success_count, errors, verification_pass


if __name__ == "__main__":
    generate_all()
