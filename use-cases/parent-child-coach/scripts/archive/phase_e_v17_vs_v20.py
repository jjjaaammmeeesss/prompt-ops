"""
Phase E Final: v1.7 vs R5 v2.0 on expert holdout
Task model: DeepSeek v4 pro | Judge: LLM Judge (DeepSeek v4 pro)
"""
import json, os, sys, time
from statistics import mean, stdev

import numpy as np
import requests
from scipy import stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "scripts"))

# API
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY_PLACEHOLDER")
TASK_MODEL = "deepseek-v4-pro"

# Load prompts
with open(os.path.join(BASE_DIR, "system_prompt_backup_v17.txt"), "r", encoding="utf-8") as f:
    PROMPT_V17 = f.read()
with open(os.path.join(BASE_DIR, "system_prompt_v2.0.txt"), "r", encoding="utf-8") as f:
    PROMPT_V20 = f.read()

# Load LLM Judge
from llm_judge_metric import LLMJudgeMetric
judge = LLMJudgeMetric()


def generate_popup(system_prompt: str, dialogue: str) -> str:
    headers = {"Authorization": f"Bearer {DEEPSEEK_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": TASK_MODEL, "max_tokens": 2048, "temperature": 0.7,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"对话：\n{dialogue}\n\n请生成弹窗："},
        ],
    }
    for attempt in range(4):
        try:
            resp = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=(30, 120))
            resp.raise_for_status()
            data = resp.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as e:
            if attempt < 3:
                time.sleep(2 ** attempt * 5)
            else:
                print(f"  ❌ Generation failed: {e}")
                return ""


def score_popup(dialogue: str, popup: str) -> float:
    class MockGold: pass
    class MockPred: pass
    gold = MockGold(); gold.question = dialogue
    pred = MockPred(); pred.answer = popup
    return judge(gold, pred, trace=False)


def main():
    prompts = {"v1.7_baseline": PROMPT_V17, "v2.0_R5": PROMPT_V20}
    print("=" * 80)
    print("Phase E: v1.7 vs v2.0 (R5) on Expert Holdout")
    print(f"Prompt lengths: v1.7={len(PROMPT_V17)}, v2.0={len(PROMPT_V20)}")
    print(f"Task model: {TASK_MODEL}")
    print("=" * 80)

    # Load expert test set
    expert_test = json.load(open(os.path.join(BASE_DIR, "data", "expert_test.json"), "r", encoding="utf-8"))
    print(f"Expert test set: {len(expert_test)} examples, "
          f"{sum(1 for e in expert_test if e.get('expert_score') is not None)} with expert scores")

    results = {name: {"scores": [], "popups": []} for name in prompts}

    for i, item in enumerate(expert_test):
        dialogue = item.get("dialogue") or item.get("question", "")
        d_short = dialogue[:80].replace("\n", " ")
        print(f"\n[{i+1}/{len(expert_test)}] {d_short}...")

        for name, prompt in prompts.items():
            print(f"  [{name}] Generating...")
            popup = generate_popup(prompt, dialogue)
            results[name]["popups"].append(popup)
            time.sleep(0.3)

            print(f"  [{name}] Scoring...")
            score = score_popup(dialogue, popup)
            results[name]["scores"].append(score)
            print(f"  [{name}] Score: {score:.3f}")

    # Stats
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")

    scores_v17 = results["v1.7_baseline"]["scores"]
    scores_v20 = results["v2.0_R5"]["scores"]

    for name, sc in [("v1.7", scores_v17), ("v2.0", scores_v20)]:
        print(f"  {name}: {mean(sc):.4f} ± {stdev(sc):.4f}")

    diffs = [b - a for a, b in zip(scores_v17, scores_v20)]
    t_stat, p_value = stats.ttest_rel(scores_v20, scores_v17, alternative="greater")
    pooled_sd = np.sqrt((np.var(scores_v17) + np.var(scores_v20)) / 2)
    cohens_d = (mean(scores_v20) - mean(scores_v17)) / pooled_sd if pooled_sd > 0 else 0

    rng = np.random.default_rng(42)
    bootstrap_diffs = []
    for _ in range(10000):
        idx = rng.integers(0, len(diffs), size=len(diffs))
        bootstrap_diffs.append(np.mean([diffs[i] for i in idx]))
    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)

    sig = "✅" if p_value < 0.05 else "❌"
    print(f"\n  v1.7 → v2.0: Δ = {mean(diffs):+.4f}")
    print(f"  Cohen's d = {cohens_d:.3f}")
    print(f"  Paired t-test: t={t_stat:.2f}, p={p_value:.4f} {sig}")
    print(f"  Bootstrap 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    # Per-example details
    print(f"\n  Per-example:")
    for i, item in enumerate(expert_test):
        exp = item.get("expert_score")
        exp_str = f" expert={exp}" if exp else ""
        print(f"    [{i+1}] v1.7={scores_v17[i]:.3f} | v2.0={scores_v20[i]:.3f} | Δ={diffs[i]:+.3f}{exp_str}")

    # Expert correlation
    expert_scored = [(i, e) for i, e in enumerate(expert_test) if e.get("expert_score") is not None]
    if len(expert_scored) >= 3:
        print(f"\n  Expert correlation (n={len(expert_scored)}):")
        for name, sc in [("v1.7", scores_v17), ("v2.0", scores_v20)]:
            js = [sc[i] for i, _ in expert_scored]
            es = [(expert_test[i]["expert_score"] - 1) / 9 for i, _ in expert_scored]
            r, p = stats.pearsonr(js, es)
            print(f"    {name}: Pearson r={r:.3f} (p={p:.3f})")

    # Save
    output = {
        "config": {"task_model": TASK_MODEL, "test_set": "expert_test",
                   "v1.7_len": len(PROMPT_V17), "v2.0_len": len(PROMPT_V20)},
        "v1.7_mean": mean(scores_v17), "v1.7_std": stdev(scores_v17),
        "v2.0_mean": mean(scores_v20), "v2.0_std": stdev(scores_v20),
        "improvement": mean(diffs), "cohens_d": round(cohens_d, 4),
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(float(p_value), 6),
        "significant_005": bool(p_value < 0.05),
        "bootstrap_ci_95": [round(float(ci_lower), 4), round(float(ci_upper), 4)],
        "per_example": [{"dialogue": (expert_test[i].get("dialogue") or expert_test[i].get("question", ""))[:200],
                         "expert_score": expert_test[i].get("expert_score"),
                         "v1.7_score": round(scores_v17[i], 4),
                         "v2.0_score": round(scores_v20[i], 4),
                         "v1.7_popup": results["v1.7_baseline"]["popups"][i][:300],
                         "v2.0_popup": results["v2.0_R5"]["popups"][i][:300]}
                        for i in range(len(expert_test))],
    }

    out_path = os.path.join(BASE_DIR, "results", "phase_e_final.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved: {out_path}")


if __name__ == "__main__":
    main()
