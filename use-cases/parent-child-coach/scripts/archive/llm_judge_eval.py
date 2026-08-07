"""
LLM Judge Evaluation Script
Uses Claude Opus 4.7 (primary) with GLM-5.2 (fallback) to independently evaluate
baseline (v1.7) vs optimized prompt quality.

Workflow:
1. Load the last 14 test examples from dataset.json
2. For each: generate answer with baseline prompt AND optimized prompt via DeepSeek
3. Send both answers to Claude Opus 4.7 for blind pairwise comparison (fallback to GLM-5.2)
4. Aggregate wins, dimension scores, and save results to data/llm_judge_results.json
"""

import json
import os
import re
import sys
import time
import glob as glob_mod
import requests

# === Paths ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATASET_PATH = os.path.join(BASE_DIR, "dataset.json")
SYSTEM_PROMPT_PATH = os.path.join(BASE_DIR, "system_prompt.txt")
OUTPUT_PATH = os.path.join(DATA_DIR, "llm_judge_results.json")
ENV_PATH = os.path.join(BASE_DIR, ".env")

# === Primary Judge: Claude Opus 4.7 (智创聚合 — most stable) ===
CLAUDE_OPUS_API_URL = "https://s.lconai.com/v1/messages"
CLAUDE_OPUS_API_KEY = "CLAUDE_API_KEY_PLACEHOLDER"
CLAUDE_OPUS_MODEL = "claude-opus-4-7"

# === Fallback Judge 1: Claude Opus 4.7 (星鸾) ===
CLAUDE_OPUS_FALLBACK_URL = "https://api.xingluan.vip/runningai/open/v1/messages"
CLAUDE_OPUS_FALLBACK_KEY = "06131c3816c4483c8a4da408102d52e3"

# === Fallback Judge 2: GLM-5.2 (智谱) ===
GLM_API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
GLM_API_KEY = "6152d59501254ab3916dfce0a6ff6092.Oo26YzT1AZy3xPAi"
GLM_MODEL = "glm-5.2"

# === Judge Prompt Template ===
JUDGE_PROMPT_TEMPLATE = """你是一名亲子沟通教练评估专家。下面是一段亲子对话，以及两个AI教练生成的弹窗洞见（A 和 B）。

请从以下维度对比两个回答：
1. 诊断准确度：是否准确识别了家长的盲区/模式？
2. 共情力：是否先看见家长的努力和人性？
3. 洞察深度：是否提升了家长的视角（从单维到多维）？
4. 行动启发性：是否暗示了新的可能性？

对话：
{dialogue}

回答 A：
{answer_a}

回答 B：
{answer_b}

请输出JSON：
{{"winner": "A"|"B"|"tie", "a_score": 1-5, "b_score": 1-5, "reason": "<一句话理由>", "diagnosis_accuracy": {{"a": 1-5, "b": 1-5}}, "empathy": {{"a": 1-5, "b": 1-5}}, "insight_depth": {{"a": 1-5, "b": 1-5}}, "actionability": {{"a": 1-5, "b": 1-5}}}}"""


def load_api_key():
    """Load DEEPSEEK_API_KEY from .env file."""
    if not os.path.exists(ENV_PATH):
        print(f"ERROR: .env not found at {ENV_PATH}")
        return None
    with open(ENV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("DEEPSEEK_API_KEY="):
                return line.strip().split("=", 1)[1]
    print("ERROR: DEEPSEEK_API_KEY not found in .env")
    return None


def find_optimized_prompt():
    """
    Find the latest config_*.json in results/ that contains an optimized prompt.
    The stage0 config (config_20260708_161156.json) is the baseline few-shot config,
    not the MIPROv2 output. We look for a newer config file.

    Returns (prompt_text, config_file_path) or (None, None) if not found.
    """
    pattern = os.path.join(RESULTS_DIR, "config_*.json")
    files = sorted(glob_mod.glob(pattern))

    if not files:
        print("No config files found in results/")
        return None, None

    # Sort by filename timestamp (descending), newest first
    files.sort(key=lambda f: os.path.basename(f), reverse=True)

    # The stage0 config is config_20260708_161156.json
    # If that's the only one, MIPROv2 hasn't produced output yet
    stage0_config = os.path.join(RESULTS_DIR, "config_20260708_161156.json")
    non_stage0 = [f for f in files if os.path.basename(f) != "config_20260708_161156.json"]

    if not non_stage0:
        print("Only stage0 config found (config_20260708_161156.json) — MIPROv2 output not yet available.")
        return None, None

    # Use the newest non-stage0 config
    target = non_stage0[0]
    print(f"Using optimized prompt from: {os.path.basename(target)}")

    with open(target, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompt = data.get("prompt")
    if not prompt:
        print(f"ERROR: 'prompt' field not found in {target}")
        return None, None

    return prompt, target


def load_test_examples():
    """Load the last 14 entries from dataset.json as test examples."""
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    test_examples = dataset[-14:]
    print(f"Loaded {len(test_examples)} test examples (last 14 of {len(dataset)} total)")
    return test_examples


def load_few_shots():
    """Load few-shot examples from the stage0 config."""
    config_path = os.path.join(RESULTS_DIR, "config_20260708_161156.json")
    if not os.path.exists(config_path):
        print(f"WARNING: Few-shot config not found at {config_path}")
        return []

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    few_shots = config.get("few_shots", [])
    print(f"Loaded {len(few_shots)} few-shot examples")
    return few_shots


def generate_answer(system_prompt, dialogue, api_key, few_shots):
    """
    Generate an answer using the given system prompt via DeepSeek API.
    Returns the raw answer text (JSON string from the model).
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
    return data["choices"][0]["message"]["content"]


def extract_popup_insight(answer_text):
    """
    Extract the popup_insight field from a model-generated JSON answer.
    Returns the insight string, or the raw text if parsing fails.
    """
    if not answer_text:
        return ""

    # Try raw parse
    try:
        parsed = json.loads(answer_text)
        # Support both raw dict and JSON-string format in dataset
        if isinstance(parsed, dict):
            return parsed.get("popup_insight", "")
    except (json.JSONDecodeError, TypeError):
        pass

    # Try to extract JSON from code block
    json_block_pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
    matches = re.findall(json_block_pattern, answer_text, re.DOTALL)
    for match in matches:
        try:
            parsed = json.loads(match.strip())
            if isinstance(parsed, dict):
                return parsed.get("popup_insight", "")
        except (json.JSONDecodeError, TypeError):
            continue

    # Try to find JSON object in text
    brace_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
    matches = re.findall(brace_pattern, answer_text, re.DOTALL)
    for match in matches:
        try:
            parsed = json.loads(match)
            if isinstance(parsed, dict):
                insight = parsed.get("popup_insight", "")
                if insight:
                    return insight
        except (json.JSONDecodeError, TypeError):
            continue

    # Fallback: return raw text
    return answer_text


def parse_judge_json(content):
    """
    Parse the JSON output from a judge model's response.
    Tries raw parse, code block extraction, and regex fallback.
    Returns parsed dict or None.
    """
    if not content:
        return None

    # Try raw parse
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        pass

    # Try code block
    json_block_pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
    matches = re.findall(json_block_pattern, content, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match.strip())
        except (json.JSONDecodeError, TypeError):
            continue

    # Try to find JSON object in text
    brace_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
    matches = re.findall(brace_pattern, content, re.DOTALL)
    for match in matches:
        try:
            result = json.loads(match)
            if "winner" in result:
                return result
        except (json.JSONDecodeError, TypeError):
            continue

    return None


def call_claude_opus_judge(dialogue, answer_a, answer_b):
    """
    Send a pairwise comparison request to Claude Opus 4.7 (primary judge).
    Uses Anthropic-compatible API format with system as top-level field.

    Returns (parsed_json, model_name) or raises on failure.
    """
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        dialogue=dialogue,
        answer_a=answer_a,
        answer_b=answer_b,
    )

    resp = requests.post(
        CLAUDE_OPUS_API_URL,
        headers={
            "x-api-key": CLAUDE_OPUS_API_KEY,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        },
        json={
            "model": CLAUDE_OPUS_MODEL,
            "max_tokens": 4096,
            "temperature": 0.0,
            "system": "你是一名亲子沟通教练评估专家。请严格按照用户要求的JSON格式输出评估结果，只输出JSON，不要输出其他内容。",
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    # Anthropic Messages API returns content as an array of blocks
    content = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")

    parsed = parse_judge_json(content)
    if parsed is None:
        print(f"  WARNING: Could not parse Claude Opus response: {content[:200]}...")
    return parsed, CLAUDE_OPUS_MODEL


def call_glm_judge(dialogue, answer_a, answer_b):
    """
    Send a pairwise comparison request to GLM-5.2 (fallback judge).

    Returns (parsed_json, model_name) or raises on failure.
    """
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        dialogue=dialogue,
        answer_a=answer_a,
        answer_b=answer_b,
    )

    messages = [{"role": "user", "content": prompt}]

    resp = requests.post(
        GLM_API_URL,
        headers={
            "Authorization": f"Bearer {GLM_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": GLM_MODEL,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 4096,
            "thinking": {"type": "enabled"},
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]

    parsed = parse_judge_json(content)
    if parsed is None:
        print(f"  WARNING: Could not parse GLM judge response: {content[:200]}...")
    return parsed, GLM_MODEL


def call_judge_with_fallback(dialogue, answer_a, answer_b):
    """
    Try Claude Opus 4.7 first; fall back to GLM-5.2 on any failure.
    Returns (parsed_json, model_used).
    """
    # Try primary judge: Claude Opus 4.7
    try:
        parsed, model = call_claude_opus_judge(dialogue, answer_a, answer_b)
        if parsed is not None:
            return parsed, model
        # Parsed was None (parse failure), try fallback
        print("  -> Claude Opus returned unparseable output, falling back to GLM-5.2...")
    except Exception as e:
        print(f"  -> Claude Opus failed ({e}), falling back to GLM-5.2...")

    # Fallback: GLM-5.2
    try:
        parsed, model = call_glm_judge(dialogue, answer_a, answer_b)
        if parsed is not None:
            return parsed, model
        print("  -> GLM-5.2 also returned unparseable output.")
    except Exception as e:
        print(f"  -> GLM-5.2 also failed ({e}).")

    return None, "none"


def aggregate_results(detailed_results):
    """
    Compute aggregate statistics from all judge results.
    Returns a summary dict.
    """
    baseline_wins = 0
    optimized_wins = 0
    ties = 0

    dim_scores = {
        "diagnosis_accuracy": {"a_total": 0, "b_total": 0, "count": 0},
        "empathy": {"a_total": 0, "b_total": 0, "count": 0},
        "insight_depth": {"a_total": 0, "b_total": 0, "count": 0},
        "actionability": {"a_total": 0, "b_total": 0, "count": 0},
    }
    a_scores = []
    b_scores = []

    for r in detailed_results:
        judge = r.get("judge_result")
        if not judge:
            continue

        winner = judge.get("winner", "").upper()
        if winner == "A":
            baseline_wins += 1
        elif winner == "B":
            optimized_wins += 1
        elif winner == "TIE":
            ties += 1

        a_score = judge.get("a_score", 0)
        b_score = judge.get("b_score", 0)
        if isinstance(a_score, (int, float)):
            a_scores.append(a_score)
        if isinstance(b_score, (int, float)):
            b_scores.append(b_score)

        for dim_key in dim_scores:
            dim_data = judge.get(dim_key, {})
            if isinstance(dim_data, dict):
                a_val = dim_data.get("a", 0)
                b_val = dim_data.get("b", 0)
                if isinstance(a_val, (int, float)) or isinstance(b_val, (int, float)):
                    dim_scores[dim_key]["a_total"] += a_val
                    dim_scores[dim_key]["b_total"] += b_val
                    dim_scores[dim_key]["count"] += 1

    # Compute averages
    avg_dim_scores = {}
    for dim_key, data in dim_scores.items():
        if data["count"] > 0:
            avg_dim_scores[dim_key] = {
                "a_avg": round(data["a_total"] / data["count"], 2),
                "b_avg": round(data["b_total"] / data["count"], 2),
            }
        else:
            avg_dim_scores[dim_key] = {"a_avg": 0, "b_avg": 0}

    avg_a = round(sum(a_scores) / len(a_scores), 2) if a_scores else 0
    avg_b = round(sum(b_scores) / len(b_scores), 2) if b_scores else 0

    # Determine verdict
    if baseline_wins > optimized_wins + 2:
        verdict = "baseline_better"
    elif optimized_wins > baseline_wins + 2:
        verdict = "optimized_better"
    else:
        verdict = "no_significant_difference"

    summary = {
        "baseline_wins": baseline_wins,
        "optimized_wins": optimized_wins,
        "ties": ties,
        "avg_overall_score": {"a": avg_a, "b": avg_b},
        "avg_dimension_scores": avg_dim_scores,
        "verdict": verdict,
    }

    return summary


def main():
    print("=" * 60)
    print("LLM JUDGE EVALUATION — Claude Opus 4.7 (primary) + GLM-5.2 (fallback)")
    print("=" * 60)

    # === Step 1: Find optimized prompt ===
    print("\n[1/5] Looking for optimized prompt...")
    optimized_prompt, optimized_config_path = find_optimized_prompt()

    if optimized_prompt is None:
        print("\nOptimized prompt not yet available — waiting for U5 to complete")
        return 0

    # === Step 2: Load resources ===
    print("\n[2/5] Loading resources...")

    # Baseline prompt (v1.7)
    if not os.path.exists(SYSTEM_PROMPT_PATH):
        print(f"ERROR: System prompt not found at {SYSTEM_PROMPT_PATH}")
        return 1
    with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        baseline_prompt = f.read()
    print(f"  Baseline prompt (v1.7): {len(baseline_prompt)} chars")

    print(f"  Optimized prompt:       {len(optimized_prompt)} chars")

    # Test examples (last 14)
    test_examples = load_test_examples()

    # Few-shot examples
    few_shots = load_few_shots()

    # DeepSeek API key
    api_key = load_api_key()
    if not api_key:
        print("FATAL: Cannot proceed without DEEPSEEK_API_KEY.")
        return 1

    # === Step 3: Generate answers for all test examples ===
    print(f"\n[3/5] Generating answers for {len(test_examples)} test examples...")

    pairs = []  # List of (dialogue, answer_a_text, answer_b_text)

    for i, example in enumerate(test_examples):
        dialogue = example["question"]
        idx = i + 1
        print(f"\n  Example {idx}/{len(test_examples)}:")

        # Generate with baseline prompt (Answer A)
        try:
            raw_a = generate_answer(baseline_prompt, dialogue, api_key, few_shots)
            insight_a = extract_popup_insight(raw_a)
            print(f"    Baseline (A): generated ({len(insight_a)} chars insight)")
        except Exception as e:
            print(f"    Baseline (A): ERROR - {e}")
            insight_a = f"[ERROR: {e}]"

        time.sleep(1)

        # Generate with optimized prompt (Answer B)
        try:
            raw_b = generate_answer(optimized_prompt, dialogue, api_key, few_shots)
            insight_b = extract_popup_insight(raw_b)
            print(f"    Optimized (B): generated ({len(insight_b)} chars insight)")
        except Exception as e:
            print(f"    Optimized (B): ERROR - {e}")
            insight_b = f"[ERROR: {e}]"

        pairs.append({
            "dialogue": dialogue,
            "answer_a": insight_a,
            "answer_b": insight_b,
            "raw_a": raw_a,
            "raw_b": raw_b,
        })

        time.sleep(1)

    # === Step 4: Pairwise judgment (Claude Opus primary, GLM-5.2 fallback) ===
    print(f"\n[4/5] Pairwise judging ({len(pairs)} comparisons)...")
    print(f"  Primary judge: {CLAUDE_OPUS_MODEL}")
    print(f"  Fallback judge: {GLM_MODEL}")

    detailed_results = []

    for i, pair in enumerate(pairs):
        idx = i + 1
        dialogue = pair["dialogue"]
        answer_a = pair["answer_a"]
        answer_b = pair["answer_b"]

        judge_result, judge_model = call_judge_with_fallback(dialogue, answer_a, answer_b)

        if judge_result:
            winner = judge_result.get("winner", "?")
            print(f"  Evaluating {idx}/14: winner={winner}  [judge: {judge_model}]")
        else:
            print(f"  Evaluating {idx}/14: winner=PARSE_ERROR  [judge: {judge_model}]")
            judge_result = {"winner": "error", "reason": "Both judges failed or returned unparseable output"}

        detailed_results.append({
            "dialogue": dialogue,
            "answer_a": answer_a,
            "answer_b": answer_b,
            "judge_result": judge_result,
            "judge_model_used": judge_model,
        })

        # 1-second delay between judge calls (skip after last)
        if idx < len(pairs):
            time.sleep(1)

    # === Step 5: Aggregate and save results ===
    print("\n[5/5] Aggregating results...")

    summary = aggregate_results(detailed_results)

    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Baseline wins (A):   {summary['baseline_wins']}")
    print(f"  Optimized wins (B):  {summary['optimized_wins']}")
    print(f"  Ties:                {summary['ties']}")
    print(f"  Avg overall A:       {summary['avg_overall_score']['a']}")
    print(f"  Avg overall B:       {summary['avg_overall_score']['b']}")
    print(f"  Dimension scores:")
    for dim_key, scores in summary["avg_dimension_scores"].items():
        print(f"    {dim_key}: A={scores['a_avg']}, B={scores['b_avg']}")
    print(f"  Verdict:             {summary['verdict']}")
    print(f"{'=' * 60}")

    # Full output
    output = {
        "summary": summary,
        "detailed_results": detailed_results,
        "config": {
            "baseline_prompt_path": SYSTEM_PROMPT_PATH,
            "optimized_prompt_path": optimized_config_path,
            "primary_judge_model": CLAUDE_OPUS_MODEL,
            "fallback_judge_model": GLM_MODEL,
            "num_test_examples": len(test_examples),
        },
    }

    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nDetailed results saved to: {OUTPUT_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
