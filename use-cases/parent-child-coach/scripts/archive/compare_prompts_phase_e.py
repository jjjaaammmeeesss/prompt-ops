"""
Phase E: 专家验证 — 在专家打标 holdout 集上对比所有版本提示词。

对比:
  - v1.7 基线 (4815 chars)
  - R1 优化版 (374 chars, "亲子冲突现场")
  - R2 优化版 (359 chars, "亲子对话现场")

测试集:
  - 旧 holdout: 7 条 (保持与上轮可比)
  - 专家 holdout: 7 条 (含 5 条有专家评分)

用法: python scripts/compare_prompts_phase_e.py
输出: results/comparison_r2_expert.json
"""

import json
import os
import sys
import time
from statistics import mean, stdev

import numpy as np
import requests
from scipy import stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# === DeepSeek API ===
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY_PLACEHOLDER")
DEEPSEEK_MODEL = "deepseek-chat"

# === Load prompts ===
with open(os.path.join(BASE_DIR, "system_prompt_backup_v17.txt"), "r", encoding="utf-8") as f:
    PROMPT_V17 = f.read()

PROMPT_R1 = """你是一位在亲子冲突现场实时介入的「认知镜面」智能体。此刻，一位家长正深陷与孩子的对话僵局——ta 的视角已收缩到只剩一条路（自由度=1），情绪在恐惧与愤怒之间震荡，而孩子正用沉默、顶嘴或哭泣回应。你的任务不是给建议，不是替孩子说话，也不是评判家长对错——你的唯一使命是：用一段 100-200 字的诊断式弹窗或 30-60 字的鼓励式弹窗，把家长此刻戴的那副隐形眼镜（ta 的思维扭曲或认知盲区）端到 ta 面前，让 ta 的视角自由度从 1 跃迁到 ≥2。你必须严格遵循「先看见家长」的原则——先照见 ta 的发心、难处和已做对的部分，再揭示盲区。弹窗必须是镜子，不是审判台；必须用假设式语法（"你正在用…看孩子"），禁止宣告式语法（"你应该…"）。这是高风险的亲子关系修复现场，一次失败的弹窗可能让信任崩塌，一次精准的弹窗可能让僵局松动。开始。"""

PROMPT_R2 = open(os.path.join(BASE_DIR, "system_prompt.txt"), "r", encoding="utf-8").read()

# === LLM Judge ===
from llm_judge_metric import LLMJudgeMetric
judge = LLMJudgeMetric()


def generate_popup(system_prompt: str, dialogue: str) -> str:
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
            {"role": "user", "content": f"对话：\n{dialogue}\n\n请生成弹窗："},
        ],
    }
    resp = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data.get("choices", [{}])[0].get("message", {}).get("content", "")


def score_popup(dialogue: str, popup: str) -> float:
    class MockGold:
        pass
    class MockPred:
        pass
    gold = MockGold()
    gold.question = dialogue
    pred = MockPred()
    pred.answer = popup
    return judge(gold, pred, trace=False)


def run_comparison(test_set, test_label, prompts_dict):
    """Run comparison of all prompts on a given test set."""
    results = {name: {"scores": [], "popups": []} for name in prompts_dict}

    for i, item in enumerate(test_set):
        dialogue = item.get("dialogue") or item.get("question", "")
        d_short = dialogue[:80].replace("\n", " ")
        print(f"\n--- [{test_label}] Example {i+1}/{len(test_set)}: {d_short}... ---")

        for name, prompt in prompts_dict.items():
            print(f"  [{name}] Generating popup...")
            popup = generate_popup(prompt, dialogue)
            results[name]["popups"].append(popup)
            time.sleep(0.2)

            print(f"  [{name}] Scoring...")
            score = score_popup(dialogue, popup)
            results[name]["scores"].append(score)
            print(f"  [{name}] Score: {score:.3f}")

    return results


def compute_stats(name_a, scores_a, name_b, scores_b):
    """Compute paired statistical comparison."""
    diffs = [b - a for a, b in zip(scores_a, scores_b)]
    t_stat, p_value = stats.ttest_rel(scores_b, scores_a, alternative="greater")
    pooled_sd = np.sqrt((np.var(scores_a) + np.var(scores_b)) / 2)
    cohens_d = (mean(scores_b) - mean(scores_a)) / pooled_sd if pooled_sd > 0 else 0

    rng = np.random.default_rng(42)
    bootstrap_diffs = []
    for _ in range(10000):
        idx = rng.integers(0, len(diffs), size=len(diffs))
        bootstrap_diffs.append(np.mean([diffs[i] for i in idx]))
    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)

    return {
        "a_mean": mean(scores_a),
        "a_std": stdev(scores_a),
        "b_mean": mean(scores_b),
        "b_std": stdev(scores_b),
        "improvement": mean(diffs),
        "per_example_diffs": [round(d, 4) for d in diffs],
        "cohens_d": round(cohens_d, 4),
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(float(p_value), 6),
        "significant_005": bool(p_value < 0.05),
        "significant_001": bool(p_value < 0.01),
        "bootstrap_ci_95": [round(float(ci_lower), 4), round(float(ci_upper), 4)],
    }


def main():
    prompts = {
        "v1.7_baseline": PROMPT_V17,
        "R1_optimized": PROMPT_R1,
        "R2_optimized": PROMPT_R2,
    }

    print("=" * 80)
    print("Phase E: Expert Validation — Multi-prompt Comparison")
    print(f"Prompts: {list(prompts.keys())}")
    print(f"Prompt lengths: v1.7={len(PROMPT_V17)}, R1={len(PROMPT_R1)}, R2={len(PROMPT_R2)}")
    print("=" * 80)

    # === Test Set A: old 7-item holdout ===
    old_dataset = json.load(
        open(os.path.join(BASE_DIR, "dataset_50_questions.json"), "r", encoding="utf-8")
    )
    old_test = old_dataset[-7:]
    print(f"\n{'='*80}")
    print("TEST SET A: Old holdout (7 examples)")
    print(f"{'='*80}")
    results_a = run_comparison(old_test, "old", prompts)

    # === Test Set B: expert holdout ===
    expert_test = json.load(
        open(os.path.join(BASE_DIR, "data", "expert_test.json"), "r", encoding="utf-8")
    )
    print(f"\n{'='*80}")
    print(f"TEST SET B: Expert holdout ({len(expert_test)} examples, "
          f"{sum(1 for e in expert_test if e.get('expert_score') is not None)} with expert scores)")
    print(f"{'='*80}")
    results_b = run_comparison(expert_test, "expert", prompts)

    # === Statistical Analysis ===
    print(f"\n{'='*80}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*80}")

    output = {
        "config": {
            "test_set_a_size": len(old_test),
            "test_set_b_size": len(expert_test),
            "test_set_b_with_expert_scores": sum(1 for e in expert_test if e.get("expert_score") is not None),
            "judge_model": "deepseek-chat",
            "task_model": "deepseek-chat",
            "prompts": {name: len(p) for name, p in prompts.items()},
        },
        "test_set_a": {},
        "test_set_b": {},
        "expert_correlation": {},
    }

    # Compare R1 vs v1.7, R2 vs v1.7, R2 vs R1 on each test set
    comparisons = [
        ("v1.7_baseline", "R1_optimized"),
        ("v1.7_baseline", "R2_optimized"),
        ("R1_optimized", "R2_optimized"),
    ]

    for test_label, results, test_data in [
        ("test_set_a", results_a, old_test),
        ("test_set_b", results_b, expert_test),
    ]:
        print(f"\n--- {test_label} ---")
        for a_name, b_name in comparisons:
            stats_result = compute_stats(
                a_name, results[a_name]["scores"],
                b_name, results[b_name]["scores"],
            )
            key = f"{a_name}_vs_{b_name}"
            output[test_label][key] = stats_result

            sig = "✅" if stats_result["significant_005"] else "❌"
            print(f"\n  {a_name} vs {b_name}:")
            print(f"    {a_name}: {stats_result['a_mean']:.4f} ± {stats_result['a_std']:.4f}")
            print(f"    {b_name}: {stats_result['b_mean']:.4f} ± {stats_result['b_std']:.4f}")
            print(f"    Δ: {stats_result['improvement']:+.4f}, d={stats_result['cohens_d']:.3f}, "
                  f"p={stats_result['p_value']:.4f} {sig}")

        # Per-example details
        output[test_label]["per_example"] = []
        for i, item in enumerate(test_data):
            entry = {
                "dialogue": (item.get("dialogue") or item.get("question", ""))[:200],
                "expert_score": item.get("expert_score"),
                "expert_name": item.get("expert_name"),
            }
            for name in prompts:
                entry[f"{name}_score"] = round(results[name]["scores"][i], 4)
                entry[f"{name}_popup"] = results[name]["popups"][i][:300]
            output[test_label]["per_example"].append(entry)

    # === Expert correlation (Test Set B only) ===
    expert_scored = [(i, e) for i, e in enumerate(expert_test) if e.get("expert_score") is not None]
    if len(expert_scored) >= 3:
        print(f"\n--- Expert Score Correlation (Test Set B, n={len(expert_scored)}) ---")
        for name in prompts:
            judge_scores = [results_b[name]["scores"][i] for i, _ in expert_scored]
            expert_norm = [(e["expert_score"] - 1) / 9 for _, e in expert_scored]

            if len(judge_scores) >= 3:
                r, p = stats.pearsonr(judge_scores, expert_norm)
                rho, sp = stats.spearmanr(judge_scores, expert_norm)
                output["expert_correlation"][name] = {
                    "pearson_r": round(float(r), 4),
                    "pearson_p": round(float(p), 4),
                    "spearman_rho": round(float(rho), 4),
                    "spearman_p": round(float(sp), 4),
                    "judge_scores": [round(float(s), 4) for s in judge_scores],
                    "expert_scores_normalized": [round(float(s), 4) for s in expert_norm],
                }
                print(f"  {name}: r={r:.3f} (p={p:.3f}), ρ={rho:.3f} (p={sp:.3f})")

    # === Save ===
    out_path = os.path.join(BASE_DIR, "results", "comparison_r2_expert.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Results saved to: {out_path}")


if __name__ == "__main__":
    main()
