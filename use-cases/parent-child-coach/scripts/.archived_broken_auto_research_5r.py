"""
Auto Research: 5-Round Prompt Optimization
==========================================
Task model: DeepSeek v4 pro (generating popups)
Judge: LLM Judge with DeepSeek v4 pro (scoring)
Starting point: v1.7 baseline (1924 chars)

Each round:
  1. Generate popups for 12 test dialogues using current best prompt
  2. Judge all popups (per-dimension + overall score)
  3. Analyze weakest dimensions and failure patterns
  4. Generate improved prompt variant targeting weaknesses
  5. Head-to-head comparison on 6 dialogues (current vs variant)
  6. Winner → next round's baseline

Output: results/auto_research_r{1-5}/*.json
"""

import json
import os
import re
import sys
import time
import hashlib
from datetime import datetime
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Tuple

import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results", "auto_research")
os.makedirs(RESULTS_DIR, exist_ok=True)

# === API Config ===
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY_PLACEHOLDER")

# Task model = DeepSeek v4 pro (user requirement)
TASK_MODEL = "deepseek-v4-pro"
# Judge model = DeepSeek v4 pro (consistent, stronger than deepseek-chat)
JUDGE_MODEL = "deepseek-v4-pro"

# === Load prompts ===
with open(os.path.join(BASE_DIR, "system_prompt_backup_v17.txt"), "r", encoding="utf-8") as f:
    PROMPT_V17 = f.read()

# === Load test set (12 diverse dialogues) ===
def load_test_set(n: int = 12) -> List[Dict]:
    """Load n diverse dialogues from merged dataset."""
    merged_path = os.path.join(BASE_DIR, "data", "dataset_merged_train.json")
    with open(merged_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Pick diverse set: evenly spaced to cover different categories
    step = max(1, len(data) // n)
    test_set = [data[i] for i in range(0, min(len(data), step * n), step)][:n]

    # Ensure all have dialogue
    valid = [item for item in test_set if item.get("question")]
    return valid[:n]


# === Task: Generate popup using DeepSeek v4 pro ===
def generate_popup(system_prompt: str, dialogue: str, model: str = TASK_MODEL) -> str:
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 2048,  # v4 pro: reasoning + output share this budget
        "temperature": 0.7,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"对话：\n{dialogue}\n\n请生成弹窗："},
        ],
    }
    resp = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=90)
    resp.raise_for_status()
    data = resp.json()
    return data.get("choices", [{}])[0].get("message", {}).get("content", "")


# === Judge: 7-dimension scoring using DeepSeek v4 pro ===
SCORING_PROMPT = """你是一名亲子沟通教练评估专家。下面是一段亲子对话，以及AI教练针对这段对话生成的弹窗洞见。

请从以下七个维度独立评分（1-5分，整数）：

## 有洞察

1. **发心承认（V4a）**：弹窗是否先看见了家长的发心和难处，而非只指出问题？
   - 1=只指出家长哪里不对，完全没看见发心
   - 5=深刻照见了家长的发心和挣扎

2. **洞察准确性（V4b）**：弹窗是否基于对话中的具体行为命中痛点，而非泛泛而谈或脑补推断？
   - 1=空洞泛泛，跟这段对话的具体内容没关系
   - 5=精准命中家长此刻最核心的盲区

3. **模式揭示（V4f）**：弹窗是否把单次事件连成了反复出现的模式？
   - 1=只描述了这一次的事
   - 5=揭示了跨场景反复出现的思维/行为模式

## 易懂说人话

4. **邀请感（V4g）**：弹窗是否用邀请、试探的语气（"也许""会不会是"），而非宣告式？
   - 1=全程宣告式，像老师打红叉
   - 5=全程假设式

5. **建议可操作性（V4c）**：如果有建议，是否具体可执行？（纯诊断型标记为 N/A）
   - 1=建议空洞无物
   - 5=建议具体到动作层面
   - N/A=纯诊断型，无具体建议

6. **措辞自然度（V4d）**：语言是否口语化、不爹味、不书面化、不煽情？
   - 1=充满术语/书面语/爹味说教/刻意煽情
   - 5=自然口语，像朋友聊天

7. **专一度（V4e）**：弹窗是否聚焦一个主要矛盾讲透？
   - 1=撒网式覆盖多个问题，重点涣散
   - 5=咬住一个核心矛盾，讲深讲透

---
对话：
{dialogue}

AI教练的回应：
{response}

请输出JSON（只输出JSON，不要其他文字）：
{{"acknowledgment": 1-5, "insight_accuracy": 1-5, "pattern_revelation": 1-5, "invitational_tone": 1-5, "actionability": "1-5或N/A", "naturalness": 1-5, "focus": 1-5}}"""

DIM_WEIGHTS = [
    ("acknowledgment",      0.20),
    ("insight_accuracy",    0.20),
    ("pattern_revelation",  0.10),
    ("invitational_tone",   0.10),
    ("actionability",       0.15),
    ("naturalness",         0.15),
    ("focus",               0.10),
]

DIM_LABELS = {
    "acknowledgment": "发心承认",
    "insight_accuracy": "洞察准确性",
    "pattern_revelation": "模式揭示",
    "invitational_tone": "邀请感",
    "actionability": "建议可操作性",
    "naturalness": "措辞自然度",
    "focus": "专一度",
}


def judge_popup(dialogue: str, popup: str, model: str = JUDGE_MODEL) -> Tuple[float, Dict]:
    """Score a popup, returning (weighted_score, per_dim_scores)."""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_KEY}",
        "Content-Type": "application/json",
    }
    prompt = SCORING_PROMPT.format(dialogue=dialogue, response=popup)
    payload = {
        "model": model,
        "max_tokens": 1024,  # v4 pro: reasoning + output share this budget
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": "你是一个严格的评估专家，只输出JSON，不输出其他内容。"},
            {"role": "user", "content": prompt},
        ],
    }

    for attempt in range(3):
        try:
            resp = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=90)
            resp.raise_for_status()
            data = resp.json()
            raw = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Parse JSON
            scores = _parse_json(raw)

            # Compute weighted average
            weighted_sum = 0.0
            total_weight = 0.0
            for dim_key, weight in DIM_WEIGHTS:
                val = scores.get(dim_key)
                if val == "N/A" or val is None:
                    if dim_key == "actionability":
                        continue  # skip, weight redistributed
                    continue
                if isinstance(val, (int, float)) and 1 <= val <= 5:
                    normalized = (val - 1) / 4
                    weighted_sum += normalized * weight
                    total_weight += weight

            if total_weight == 0:
                return 0.0, scores

            return weighted_sum / total_weight, scores
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                print(f"  [Judge] Failed after 3 attempts: {e}")
                return 0.0, {}


def _parse_json(raw: str) -> Dict:
    """Parse JSON from LLM response."""
    cleaned = raw.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Cannot parse JSON: {raw[:200]}")


# === Analysis: identify weaknesses ===
def analyze_results(results: List[Dict]) -> Dict:
    """Analyze per-dimension scores to find weaknesses and patterns."""
    dim_scores = {dim: [] for dim, _ in DIM_WEIGHTS}
    overall_scores = []

    for r in results:
        overall_scores.append(r["overall_score"])
        for dim, _ in DIM_WEIGHTS:
            val = r["per_dim"].get(dim)
            if isinstance(val, (int, float)):
                dim_scores[dim].append((val - 1) / 4)  # normalize to 0-1

    dim_means = {}
    for dim, scores in dim_scores.items():
        if scores:
            dim_means[dim] = {"mean": mean(scores), "std": stdev(scores) if len(scores) > 1 else 0}
        else:
            dim_means[dim] = {"mean": 0, "std": 0}

    # Rank dimensions from weakest to strongest
    ranked = sorted(dim_means.items(), key=lambda x: x[1]["mean"])

    return {
        "overall_mean": mean(overall_scores) if overall_scores else 0,
        "overall_std": stdev(overall_scores) if len(overall_scores) > 1 else 0,
        "dim_means": dim_means,
        "weakest_dims": [(dim, stats) for dim, stats in ranked[:3] if stats["mean"] < 0.9],
        "strongest_dims": [(dim, stats) for dim, stats in ranked[-3:]],
    }


# === Generate improved prompt variant ===
def generate_variant(current_prompt: str, analysis: Dict, round_num: int) -> str:
    """Use DeepSeek v4 pro to generate an improved prompt variant targeting weaknesses."""
    weakest_info = "\n".join(
        f"  - {DIM_LABELS[dim]} (平均 {stats['mean']:.3f}/1.0): 当前提示词在此维度表现最弱"
        for dim, stats in analysis["weakest_dims"]
    )

    strongest_info = "\n".join(
        f"  - {DIM_LABELS[dim]} (平均 {stats['mean']:.3f}/1.0): 表现较好，保持"
        for dim, stats in analysis["strongest_dims"]
    )

    strategies = [
        "精简冗余，聚焦核心方法论，让模型有更多'脑力'用于生成高质量弹窗而非理解复杂规则",
        "加强具体示例和反例，用before/after的对比方式让模型理解什么是好弹窗",
        "调整语气和角色设定，尝试不同的认知框架（如从'镜面'改为'翻译者'或'陪伴者'）",
        "增加具体的话术模板和句式引导，减少抽象原则",
        "重新平衡各维度的权重——对薄弱维度给予更明确的指导词",
    ]
    strategy = strategies[min(round_num - 1, len(strategies) - 1)]

    variant_prompt = f"""你是一位提示词工程专家。请基于以下分析改进提示词。

## 当前提示词
```
{current_prompt[:3000]}
```

## 评分分析（12条测试案例）
整体均分：{analysis['overall_mean']:.3f}/1.0

薄弱维度（需要改进）：
{weakest_info}

强项维度（保持）：
{strongest_info}

## 改进策略
{strategy}

## 要求
1. 对薄弱维度给出更明确、更具体的指导
2. 保持强项维度的质量不退化
3. 提示词整体长度控制在 500-1500 字
4. 保持中文输出
5. 直接输出改进后的提示词全文，不要解释改动

改进后的提示词："""

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "deepseek-v4-pro",
        "max_tokens": 4096,
        "temperature": 0.8,
        "messages": [{"role": "user", "content": variant_prompt}],
    }

    resp = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data.get("choices", [{}])[0].get("message", {}).get("content", current_prompt)


# === Head-to-head comparison ===
def head_to_head(prompt_a: str, prompt_b: str, test_set: List[Dict],
                 label_a: str, label_b: str) -> Dict:
    """Compare two prompts on the same test set."""
    results = {"a": [], "b": [], "winner": None, "margin": 0}

    for i, item in enumerate(test_set):
        dialogue = item["question"]
        d_short = dialogue[:60].replace("\n", " ")
        print(f"    [{i+1}/{len(test_set)}] {d_short}...")

        # Generate with both prompts
        popup_a = generate_popup(prompt_a, dialogue)
        time.sleep(0.3)
        popup_b = generate_popup(prompt_b, dialogue)
        time.sleep(0.3)

        # Judge both
        score_a, dims_a = judge_popup(dialogue, popup_a)
        time.sleep(0.3)
        score_b, dims_b = judge_popup(dialogue, popup_b)
        time.sleep(0.3)

        results["a"].append({"score": score_a, "dims": dims_a, "popup": popup_a[:200]})
        results["b"].append({"score": score_b, "dims": dims_b, "popup": popup_b[:200]})

        print(f"      {label_a}: {score_a:.3f}  |  {label_b}: {score_b:.3f}  |  Δ: {score_b - score_a:+.3f}")

    scores_a = [r["score"] for r in results["a"]]
    scores_b = [r["score"] for r in results["b"]]
    results["mean_a"] = mean(scores_a)
    results["mean_b"] = mean(scores_b)
    results["margin"] = results["mean_b"] - results["mean_a"]
    results["winner"] = label_b if results["margin"] > 0 else label_a

    return results


# === Main 5-round loop ===
def main():
    print("=" * 80)
    print("AUTO RESEARCH: 5-Round Prompt Optimization")
    print(f"Task model: {TASK_MODEL} | Judge model: {JUDGE_MODEL}")
    print(f"Starting from: v1.7 baseline ({len(PROMPT_V17)} chars)")
    print("=" * 80)

    # Load 12 test dialogues (fixed across rounds for comparability)
    test_set = load_test_set(12)
    print(f"Test set: {len(test_set)} dialogues")
    for i, item in enumerate(test_set):
        print(f"  [{i+1}] {item['question'][:80].replace(chr(10), ' ')}...")
    print()

    # Initialize
    current_prompt = PROMPT_V17
    current_label = "v1.7_baseline"
    all_rounds = []

    for round_num in range(1, 6):
        print(f"\n{'='*80}")
        print(f"ROUND {round_num}/5")
        print(f"Current best: {current_label} ({len(current_prompt)} chars)")
        print(f"{'='*80}")

        round_dir = os.path.join(RESULTS_DIR, f"r{round_num}")
        os.makedirs(round_dir, exist_ok=True)

        # Step 1: Generate & judge all test cases with current prompt
        print(f"\n[Step 1] Evaluating current prompt on {len(test_set)} dialogues...")
        current_results = []
        for i, item in enumerate(test_set):
            dialogue = item["question"]
            d_short = dialogue[:60].replace("\n", " ")
            print(f"  [{i+1}/{len(test_set)}] {d_short}...")

            popup = generate_popup(current_prompt, dialogue)
            score, dims = judge_popup(dialogue, popup)

            current_results.append({
                "dialogue": dialogue[:200],
                "popup": popup,
                "overall_score": score,
                "per_dim": dims,
            })
            print(f"      Score: {score:.3f}  |  Dims: { {k: dims.get(k) for k, _ in DIM_WEIGHTS} }")
            time.sleep(0.5)

        # Step 2: Analyze
        print(f"\n[Step 2] Analyzing weaknesses...")
        analysis = analyze_results(current_results)

        print(f"  Overall: {analysis['overall_mean']:.3f} ± {analysis['overall_std']:.3f}")
        print(f"  Weakest dimensions:")
        for dim, stats in analysis["weakest_dims"]:
            print(f"    - {DIM_LABELS[dim]}: {stats['mean']:.3f}")
        print(f"  Strongest dimensions:")
        for dim, stats in analysis["strongest_dims"]:
            print(f"    + {DIM_LABELS[dim]}: {stats['mean']:.3f}")

        # Step 3: Generate variant
        print(f"\n[Step 3] Generating improved variant...")
        variant_prompt = generate_variant(current_prompt, analysis, round_num)
        print(f"  Variant length: {len(variant_prompt)} chars")
        print(f"  Preview: {variant_prompt[:200]}...")

        # Step 4: Head-to-head on 6 test dialogues
        print(f"\n[Step 4] Head-to-head comparison on 6 dialogues...")
        h2h_test = test_set[:6]
        h2h = head_to_head(current_prompt, variant_prompt, h2h_test,
                          current_label, f"r{round_num}_variant")

        print(f"\n  Result: {h2h['mean_a']:.3f} ({current_label}) vs "
              f"{h2h['mean_b']:.3f} (variant)")
        print(f"  Margin: {h2h['margin']:+.3f} → Winner: {h2h['winner']} !")

        # Step 5: Decide
        if h2h["margin"] > 0.01:
            current_prompt = variant_prompt
            current_label = f"r{round_num}_variant"
            print(f"\n  ✅ Variant WINS! Upgrading to {current_label}")
        elif h2h["margin"] > -0.01:
            # Tie — keep the more compact one
            if len(variant_prompt) < len(current_prompt):
                current_prompt = variant_prompt
                current_label = f"r{round_num}_variant"
                print(f"\n  ⚖️  Tie → keeping more compact variant ({len(variant_prompt)} chars)")
            else:
                print(f"\n  ⚖️  Tie → keeping current ({len(current_prompt)} chars)")
        else:
            print(f"\n  ❌ Variant lost — keeping {current_label}")

        # Save round data
        round_data = {
            "round": round_num,
            "current_label": current_label,
            "current_prompt": current_prompt,
            "current_prompt_len": len(current_prompt),
            "analysis": {
                "overall_mean": analysis["overall_mean"],
                "overall_std": analysis["overall_std"],
                "weakest_dims": [(d, s["mean"]) for d, s in analysis["weakest_dims"]],
                "strongest_dims": [(d, s["mean"]) for d, s in analysis["strongest_dims"]],
            },
            "variant_prompt": variant_prompt,
            "variant_prompt_len": len(variant_prompt),
            "head_to_head": {
                "mean_current": h2h["mean_a"],
                "mean_variant": h2h["mean_b"],
                "margin": h2h["margin"],
                "winner": h2h["winner"],
            },
            "per_sample_current": current_results,
            "per_sample_h2h": {
                "current": h2h["a"],
                "variant": h2h["b"],
            },
        }

        with open(os.path.join(round_dir, "round_data.json"), "w", encoding="utf-8") as f:
            json.dump(round_data, f, ensure_ascii=False, indent=2)

        # Save current best prompt
        with open(os.path.join(round_dir, "best_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(current_prompt)

        all_rounds.append(round_data)
        print(f"\n  Round data saved to {round_dir}")

    # === Final Summary ===
    print(f"\n{'='*80}")
    print("5-ROUND AUTO RESEARCH COMPLETE")
    print(f"{'='*80}")
    print(f"Final best: {current_label}")
    print(f"Final prompt length: {len(current_prompt)} chars")
    print(f"\nRound-by-round scores:")
    for rd in all_rounds:
        print(f"  R{rd['round']}: {rd['analysis']['overall_mean']:.3f} "
              f"({rd['current_label']}, {rd['current_prompt_len']} chars) "
              f"| vs variant: {rd['head_to_head']['mean_variant']:.3f} "
              f"(Δ: {rd['head_to_head']['margin']:+.3f}, {rd['head_to_head']['winner']})")

    # Save final summary
    summary = {
        "config": {
            "task_model": TASK_MODEL,
            "judge_model": JUDGE_MODEL,
            "test_set_size": len(test_set),
            "starting_prompt": "v1.7_baseline",
            "starting_prompt_len": len(PROMPT_V17),
        },
        "final_prompt": current_prompt,
        "final_prompt_len": len(current_prompt),
        "final_label": current_label,
        "rounds": all_rounds,
    }

    with open(os.path.join(RESULTS_DIR, "final_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(RESULTS_DIR, "final_best_prompt.txt"), "w", encoding="utf-8") as f:
        f.write(current_prompt)

    print(f"\n✅ Final results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
