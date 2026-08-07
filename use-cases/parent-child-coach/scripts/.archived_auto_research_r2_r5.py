"""
Auto Research: Rounds 2-5 Continuation
=======================================
Continues from R1 winner. More robust error handling.
"""
import json, os, re, sys, time
from statistics import mean, stdev
from typing import Dict, List, Tuple

import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results", "auto_research")

DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY_PLACEHOLDER")
TASK_MODEL = "deepseek-v4-pro"
JUDGE_MODEL = "deepseek-v4-pro"

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
    ("acknowledgment", 0.20), ("insight_accuracy", 0.20),
    ("pattern_revelation", 0.10), ("invitational_tone", 0.10),
    ("actionability", 0.15), ("naturalness", 0.15), ("focus", 0.10),
]

DIM_LABELS = {
    "acknowledgment": "发心承认", "insight_accuracy": "洞察准确性",
    "pattern_revelation": "模式揭示", "invitational_tone": "邀请感",
    "actionability": "建议可操作性", "naturalness": "措辞自然度",
    "focus": "专一度",
}


def api_call(model: str, messages: List[Dict], max_tokens: int, temperature: float,
             timeout: int = 120) -> str:
    """Robust API call with connect/read timeout separation and retries."""
    headers = {"Authorization": f"Bearer {DEEPSEEK_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "max_tokens": max_tokens, "temperature": temperature, "messages": messages}

    for attempt in range(4):
        try:
            resp = requests.post(DEEPSEEK_URL, headers=headers, json=payload,
                                timeout=(30, timeout))
            resp.raise_for_status()
            data = resp.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        except requests.exceptions.Timeout:
            wait = 2 ** attempt * 5
            print(f"    ⚠ Timeout (attempt {attempt+1}/4), waiting {wait}s...", flush=True)
            time.sleep(wait)
        except requests.exceptions.ConnectionError as e:
            wait = 2 ** attempt * 10
            print(f"    ⚠ Connection error (attempt {attempt+1}/4): {e}, waiting {wait}s...", flush=True)
            time.sleep(wait)
        except Exception as e:
            wait = 2 ** attempt * 3
            print(f"    ⚠ API error (attempt {attempt+1}/4): {type(e).__name__}: {e}, waiting {wait}s...", flush=True)
            time.sleep(wait)

    raise RuntimeError(f"API call failed after 4 attempts")


def generate_popup(system_prompt: str, dialogue: str) -> str:
    return api_call(TASK_MODEL, [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"对话：\n{dialogue}\n\n请生成弹窗："},
    ], max_tokens=2048, temperature=0.7, timeout=120)


def judge_popup(dialogue: str, popup: str) -> Tuple[float, Dict]:
    prompt = SCORING_PROMPT.format(dialogue=dialogue, response=popup)

    for attempt in range(4):
        try:
            raw = api_call(JUDGE_MODEL, [
                {"role": "system", "content": "你是一个严格的评估专家，只输出JSON，不输出其他内容。"},
                {"role": "user", "content": prompt},
            ], max_tokens=1024, temperature=0.0, timeout=90)

            scores = _parse_json(raw)
            weighted_sum = 0.0
            total_weight = 0.0
            for dim_key, weight in DIM_WEIGHTS:
                val = scores.get(dim_key)
                if val == "N/A" or val is None:
                    continue
                if isinstance(val, (int, float)) and 1 <= val <= 5:
                    normalized = (val - 1) / 4
                    weighted_sum += normalized * weight
                    total_weight += weight
            if total_weight == 0:
                return 0.0, scores
            return weighted_sum / total_weight, scores
        except Exception as e:
            if attempt < 3:
                time.sleep(3)
            else:
                print(f"    ⚠ Judge failed after 4 attempts: {e}", flush=True)
                return 0.0, {}


def _parse_json(raw: str) -> Dict:
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


def analyze_results(results: List[Dict]) -> Dict:
    dim_scores = {dim: [] for dim, _ in DIM_WEIGHTS}
    overall_scores = []
    for r in results:
        overall_scores.append(r["overall_score"])
        for dim, _ in DIM_WEIGHTS:
            val = r["per_dim"].get(dim)
            if isinstance(val, (int, float)):
                dim_scores[dim].append((val - 1) / 4)
    dim_means = {}
    for dim, scores_ in dim_scores.items():
        dim_means[dim] = {"mean": mean(scores_) if scores_ else 0,
                          "std": stdev(scores_) if len(scores_) > 1 else 0}
    ranked = sorted(dim_means.items(), key=lambda x: x[1]["mean"])
    return {
        "overall_mean": mean(overall_scores) if overall_scores else 0,
        "overall_std": stdev(overall_scores) if len(overall_scores) > 1 else 0,
        "dim_means": dim_means,
        "weakest_dims": [(dim, stats) for dim, stats in ranked[:3] if stats["mean"] < 0.9],
        "strongest_dims": [(dim, stats) for dim, stats in ranked[-3:]],
    }


def generate_variant(current_prompt: str, analysis: Dict, round_num: int) -> str:
    weakest_info = "\n".join(
        f"  - {DIM_LABELS[dim]} (平均 {stats['mean']:.3f}/1.0): 当前提示词在此维度表现最弱"
        for dim, stats in analysis["weakest_dims"]
    )
    strongest_info = "\n".join(
        f"  - {DIM_LABELS[dim]} (平均 {stats['mean']:.3f}/1.0): 表现较好，保持"
        for dim, stats in analysis["strongest_dims"]
    )

    strategies = [
        "精简冗余，聚焦核心方法论",
        "加强具体示例和反例，用before/after对比",
        "调整语气和角色设定，尝试不同的认知框架",
        "增加具体的话术模板和句式引导",
        "重新平衡各维度权重，对薄弱维度给予更明确的指导词",
    ]
    strategy = strategies[min(round_num - 1, len(strategies) - 1)]

    variant_prompt_text = f"""你是一位提示词工程专家。请基于以下分析改进提示词。

## 当前提示词
```
{current_prompt[:4000]}
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
3. 提示词整体长度控制在 500-2000 字
4. 保持中文输出
5. 直接输出改进后的提示词全文，不要解释改动

改进后的提示词："""

    return api_call(TASK_MODEL, [{"role": "user", "content": variant_prompt_text}],
                    max_tokens=4096, temperature=0.8, timeout=180)


def head_to_head(prompt_a, prompt_b, test_set, label_a, label_b) -> Dict:
    results = {"a": [], "b": [], "winner": None, "margin": 0}
    for i, item in enumerate(test_set):
        dialogue = item["question"]
        d_short = dialogue[:60].replace("\n", " ")
        print(f"    [{i+1}/{len(test_set)}] {d_short}...", flush=True)

        popup_a = generate_popup(prompt_a, dialogue)
        time.sleep(0.3)
        popup_b = generate_popup(prompt_b, dialogue)
        time.sleep(0.3)

        score_a, dims_a = judge_popup(dialogue, popup_a)
        time.sleep(0.3)
        score_b, dims_b = judge_popup(dialogue, popup_b)
        time.sleep(0.3)

        results["a"].append({"score": score_a, "dims": dims_a, "popup": popup_a[:200]})
        results["b"].append({"score": score_b, "dims": dims_b, "popup": popup_b[:200]})
        print(f"      {label_a}: {score_a:.3f}  |  {label_b}: {score_b:.3f}  |  Δ: {score_b - score_a:+.3f}", flush=True)

    scores_a = [r["score"] for r in results["a"]]
    scores_b = [r["score"] for r in results["b"]]
    results["mean_a"] = mean(scores_a) if scores_a else 0
    results["mean_b"] = mean(scores_b) if scores_b else 0
    results["margin"] = results["mean_b"] - results["mean_a"]
    results["winner"] = label_b if results["margin"] > 0 else label_a
    return results


def load_test_set(n: int = 12) -> List[Dict]:
    merged_path = os.path.join(BASE_DIR, "data", "dataset_merged_train.json")
    with open(merged_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    step = max(1, len(data) // n)
    test_set = [data[i] for i in range(0, min(len(data), step * n), step)][:n]
    return [item for item in test_set if item.get("question")][:n]


def main():
    print("=" * 80, flush=True)
    print("AUTO RESEARCH: Rounds 2-5 Continuation", flush=True)
    print(f"Task: {TASK_MODEL} | Judge: {JUDGE_MODEL}", flush=True)
    print("=" * 80, flush=True)

    # Load R1 winner
    r1_data = json.load(open(os.path.join(RESULTS_DIR, "r1", "round_data.json"), "r", encoding="utf-8"))
    current_prompt = r1_data["current_prompt"]
    current_label = r1_data["current_label"]
    print(f"Starting from: {current_label} ({len(current_prompt)} chars)", flush=True)
    print(f"R1 baseline score: {r1_data['analysis']['overall_mean']:.3f}", flush=True)
    print(f"R1 H2H: variant won by {r1_data['head_to_head']['margin']:+.3f}", flush=True)

    # Load test set
    test_set = load_test_set(12)
    print(f"Test set: {len(test_set)} dialogues\n", flush=True)

    all_rounds = [r1_data]

    for round_num in range(2, 6):
        print(f"\n{'='*80}", flush=True)
        print(f"ROUND {round_num}/5", flush=True)
        print(f"Current best: {current_label} ({len(current_prompt)} chars)", flush=True)
        print(f"{'='*80}", flush=True)

        round_dir = os.path.join(RESULTS_DIR, f"r{round_num}")
        os.makedirs(round_dir, exist_ok=True)

        # Step 1: Evaluate current prompt
        print(f"\n[Step 1] Evaluating on {len(test_set)} dialogues...", flush=True)
        current_results = []
        for i, item in enumerate(test_set):
            dialogue = item["question"]
            d_short = dialogue[:60].replace("\n", " ")
            print(f"  [{i+1}/{len(test_set)}] {d_short}...", flush=True)

            try:
                popup = generate_popup(current_prompt, dialogue)
                score, dims = judge_popup(dialogue, popup)
            except Exception as e:
                print(f"    ❌ FAILED: {e}", flush=True)
                popup = ""
                score = 0.0
                dims = {}

            current_results.append({
                "dialogue": dialogue[:200],
                "popup": popup,
                "overall_score": score,
                "per_dim": dims,
            })
            print(f"      Score: {score:.3f}", flush=True)
            time.sleep(0.5)

        # Step 2: Analyze
        print(f"\n[Step 2] Analyzing weaknesses...", flush=True)
        analysis = analyze_results(current_results)
        print(f"  Overall: {analysis['overall_mean']:.3f} ± {analysis['overall_std']:.3f}", flush=True)
        print(f"  Weakest:", flush=True)
        for dim, stats in analysis["weakest_dims"]:
            print(f"    - {DIM_LABELS[dim]}: {stats['mean']:.3f}", flush=True)
        print(f"  Strongest:", flush=True)
        for dim, stats in analysis["strongest_dims"]:
            print(f"    + {DIM_LABELS[dim]}: {stats['mean']:.3f}", flush=True)

        # Step 3: Generate variant
        print(f"\n[Step 3] Generating improved variant...", flush=True)
        try:
            variant_prompt = generate_variant(current_prompt, analysis, round_num)
        except Exception as e:
            print(f"    ❌ Variant generation failed: {e}", flush=True)
            print(f"    Using current prompt as variant (will end up as tie)", flush=True)
            variant_prompt = current_prompt
        print(f"  Variant length: {len(variant_prompt)} chars", flush=True)
        print(f"  Preview: {variant_prompt[:200]}...", flush=True)

        # Step 4: Head-to-head
        print(f"\n[Step 4] Head-to-head on 6 dialogues...", flush=True)
        h2h_test = test_set[:6]
        try:
            h2h = head_to_head(current_prompt, variant_prompt, h2h_test,
                              current_label, f"r{round_num}_variant")
        except Exception as e:
            print(f"    ❌ H2H failed: {e}", flush=True)
            h2h = {"mean_a": 0, "mean_b": 0, "margin": 0, "winner": current_label,
                   "a": [], "b": []}

        print(f"\n  Result: {h2h['mean_a']:.3f} ({current_label}) vs "
              f"{h2h['mean_b']:.3f} (variant)", flush=True)
        print(f"  Margin: {h2h['margin']:+.3f} → Winner: {h2h['winner']} !", flush=True)

        # Step 5: Decide
        if h2h["margin"] > 0.01:
            current_prompt = variant_prompt
            current_label = f"r{round_num}_variant"
            print(f"  ✅ Variant WINS!", flush=True)
        elif h2h["margin"] > -0.01:
            if len(variant_prompt) < len(current_prompt):
                current_prompt = variant_prompt
                current_label = f"r{round_num}_variant"
                print(f"  ⚖️  Tie → keeping more compact variant", flush=True)
            else:
                print(f"  ⚖️  Tie → keeping current", flush=True)
        else:
            print(f"  ❌ Variant lost", flush=True)

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
                "mean_current": h2h["mean_a"], "mean_variant": h2h["mean_b"],
                "margin": h2h["margin"], "winner": h2h["winner"],
            },
            "per_sample_current": current_results,
            "per_sample_h2h": {"current": h2h["a"], "variant": h2h["b"]},
        }

        with open(os.path.join(round_dir, "round_data.json"), "w", encoding="utf-8") as f:
            json.dump(round_data, f, ensure_ascii=False, indent=2)
        with open(os.path.join(round_dir, "best_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(current_prompt)

        all_rounds.append(round_data)
        print(f"  ✅ Round {round_num} data saved", flush=True)

    # Final summary
    print(f"\n{'='*80}", flush=True)
    print("ALL 5 ROUNDS COMPLETE", flush=True)
    print(f"Final best: {current_label} ({len(current_prompt)} chars)", flush=True)
    for rd in all_rounds:
        print(f"  R{rd['round']}: {rd['analysis']['overall_mean']:.3f} "
              f"({rd['current_label']}, {rd['current_prompt_len']} chars) "
              f"| h2h margin: {rd['head_to_head']['margin']:+.3f}", flush=True)

    summary = {
        "config": {"task_model": TASK_MODEL, "judge_model": JUDGE_MODEL,
                   "test_set_size": len(test_set)},
        "final_prompt": current_prompt,
        "final_prompt_len": len(current_prompt),
        "final_label": current_label,
        "rounds": all_rounds,
    }
    with open(os.path.join(RESULTS_DIR, "final_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(RESULTS_DIR, "final_best_prompt.txt"), "w", encoding="utf-8") as f:
        f.write(current_prompt)

    print(f"\n✅ Final results saved to {RESULTS_DIR}/", flush=True)


if __name__ == "__main__":
    main()
