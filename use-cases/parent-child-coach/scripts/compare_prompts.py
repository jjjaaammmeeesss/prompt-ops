"""
对比 v1.7 基线 vs 优化版提示词，在测试集上做显著性检验。

用法: python scripts/compare_prompts.py
输出: 每条测试案例的逐维度评分 + 配对 t 检验 + bootstrap 置信区间
"""

import json
import os
import sys
import time
from statistics import mean, stdev

import numpy as np
import requests
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# === DeepSeek API (task model & judge) ===
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_KEY:
    raise RuntimeError("DEEPSEEK_API_KEY environment variable is required. Set it in .env or export it.")
DEEPSEEK_MODEL = "deepseek-chat"

# === Load test set ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset = json.load(
    open(os.path.join(BASE_DIR, "dataset_50_questions.json"), "r", encoding="utf-8")
)
# Use last 7 as test (matching the 70/16/14 split: train=35, val=8, test=7)
test_set = dataset[-7:]

# === Prompts ===
with open(os.path.join(BASE_DIR, "system_prompt.txt"), "r", encoding="utf-8") as f:
    PROMPT_V17 = f.read()

PROMPT_OPTIMIZED = """你是一位在亲子冲突现场实时介入的「认知镜面」智能体。此刻，一位家长正深陷与孩子的对话僵局——ta 的视角已收缩到只剩一条路（自由度=1），情绪在恐惧与愤怒之间震荡，而孩子正用沉默、顶嘴或哭泣回应。你的任务不是给建议，不是替孩子说话，也不是评判家长对错——你的唯一使命是：用一段 100-200 字的诊断式弹窗或 30-60 字的鼓励式弹窗，把家长此刻戴的那副隐形眼镜（ta 的思维扭曲或认知盲区）端到 ta 面前，让 ta 的视角自由度从 1 跃迁到 ≥2。你必须严格遵循「先看见家长」的原则——先照见 ta 的发心、难处和已做对的部分，再揭示盲区。弹窗必须是镜子，不是审判台；必须用假设式语法（"你正在用…看孩子"），禁止宣告式语法（"你应该…"）。这是高风险的亲子关系修复现场，一次失败的弹窗可能让信任崩塌，一次精准的弹窗可能让僵局松动。开始。"""

# === LLM Judge ===
from scripts.llm_judge_metric import LLMJudgeMetric

judge = LLMJudgeMetric()


def generate_popup(system_prompt: str, dialogue: str) -> str:
    """Generate a coaching popup for a dialogue using the given system prompt."""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "max_tokens": 512,
        "temperature": 0.7,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"对话：\n{dialogue}\n\n请生成弹窗：",
            },
        ],
    }

    resp = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("choices", [{}])[0].get("message", {}).get("content", "")


def score_popup(dialogue: str, popup: str) -> float:
    """Score a popup using the LLM judge."""
    class MockGold:
        pass

    class MockPred:
        pass

    gold = MockGold()
    gold.question = dialogue
    pred = MockPred()
    pred.answer = popup

    return judge(gold, pred, trace=False)


def main():
    print("=" * 80)
    print("Prompt Comparison: v1.7 vs Optimized")
    print(f"Test set: {len(test_set)} examples")
    print("=" * 80)

    v17_scores = []
    opt_scores = []
    results = []

    for i, example in enumerate(test_set):
        dialogue = example["question"]
        d_short = dialogue[:80].replace("\n", " ")
        print(f"\n--- Example {i + 1}/{len(test_set)}: {d_short}... ---")

        # Generate popups
        print("  Generating v1.7 popup...")
        popup_v17 = generate_popup(PROMPT_V17, dialogue)
        time.sleep(0.3)

        print("  Generating optimized popup...")
        popup_opt = generate_popup(PROMPT_OPTIMIZED, dialogue)
        time.sleep(0.3)

        # Score both
        print("  Scoring v1.7...")
        score_v17 = score_popup(dialogue, popup_v17)
        v17_scores.append(score_v17)

        print("  Scoring optimized...")
        score_opt = score_popup(dialogue, popup_opt)
        opt_scores.append(score_opt)

        results.append(
            {
                "dialogue": dialogue[:200],
                "v17_popup": popup_v17,
                "v17_score": score_v17,
                "opt_popup": popup_opt,
                "opt_score": score_opt,
            }
        )

        print(f"  v1.7: {score_v17:.3f}  |  Optimized: {score_opt:.3f}  "
              f"|  Δ: {score_opt - score_v17:+.3f}")

    # === Statistical Analysis ===
    print("\n" + "=" * 80)
    print("Statistical Analysis")
    print("=" * 80)

    v17_mean = mean(v17_scores)
    opt_mean = mean(opt_scores)
    diffs = [o - v for o, v in zip(opt_scores, v17_scores)]

    print(f"\nv1.7 baseline:     {v17_mean:.4f} ± {stdev(v17_scores):.4f}")
    print(f"Optimized:         {opt_mean:.4f} ± {stdev(opt_scores):.4f}")
    print(f"Improvement:       {opt_mean - v17_mean:+.4f}")
    print(f"Per-example diffs: {[f'{d:+.3f}' for d in diffs]}")

    # Paired t-test (one-sided: optimized > v1.7)
    t_stat, p_value = stats.ttest_rel(opt_scores, v17_scores, alternative="greater")
    print(f"\nPaired t-test (one-sided):")
    print(f"  t = {t_stat:.4f}")
    print(f"  p = {p_value:.6f}")
    print(f"  Significant at α=0.05: {'YES ✅' if p_value < 0.05 else 'NO ❌'}")
    print(f"  Significant at α=0.01: {'YES ✅' if p_value < 0.01 else 'NO ❌'}")

    # Cohen's d (effect size)
    pooled_sd = np.sqrt((np.var(v17_scores) + np.var(opt_scores)) / 2)
    cohens_d = (opt_mean - v17_mean) / pooled_sd if pooled_sd > 0 else 0
    print(f"\nCohen's d: {cohens_d:.3f} ({'large' if cohens_d > 0.8 else 'medium' if cohens_d > 0.5 else 'small'})")

    # Bootstrap 95% CI for the mean difference
    rng = np.random.default_rng(42)
    n_bootstrap = 10000
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(diffs), size=len(diffs))
        bootstrap_diffs.append(np.mean([diffs[i] for i in idx]))
    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)
    print(f"\nBootstrap 95% CI for mean improvement: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  Zero outside CI: {'YES ✅ (significant)' if ci_lower > 0 else 'NO ❌ (not significant)'}")

    # Save detailed results
    output = {
        "config": {
            "test_size": len(test_set),
            "judge_model": "deepseek-chat",
            "task_model": "deepseek-chat",
        },
        "summary": {
            "v17_mean": v17_mean,
            "v17_std": stdev(v17_scores),
            "opt_mean": opt_mean,
            "opt_std": stdev(opt_scores),
            "improvement": opt_mean - v17_mean,
            "cohens_d": cohens_d,
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant_005": bool(p_value < 0.05),
            "significant_001": bool(p_value < 0.01),
            "bootstrap_ci_95": [float(ci_lower), float(ci_upper)],
        },
        "per_example": results,
    }

    out_path = os.path.join(BASE_DIR, "results", "comparison_v17_vs_opt.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(output, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to: {out_path}")


if __name__ == "__main__":
    main()
