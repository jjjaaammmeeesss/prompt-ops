"""
LLM Judge 校准脚本 — 对比 judge 评分与专家评分，验证 judge 可靠性。

用法: python scripts/calibrate_judge.py
输出: results/calibration_report.json
"""

import json
import os
import sys
import time
from statistics import mean, stdev

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_judge_metric import LLMJudgeMetric, DIMS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "data", "expert_dataset.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "results", "calibration_report.json")


def normalize_expert(score_1_10: int) -> float:
    """Normalize expert score from 1-10 to 0-1."""
    return (score_1_10 - 1) / 9


def main():
    # Load dataset
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Filter calibratable records
    calibratable = [
        r for r in data
        if r.get("expert_score") is not None
        and r.get("system_popup")
        and r.get("dialogue")
    ]

    if len(calibratable) < 10:
        print(f"⚠ 可校准记录不足 ({len(calibratable)} < 10)，无法可靠计算相关性")
        print("  需要至少 10 条同时有 expert_score + system_popup + dialogue 的记录")
        return

    print(f"校准数据集: {len(calibratable)} 条记录")
    print(f"  标注人分布: 晓浩={sum(1 for r in calibratable if r['expert_name']=='晓浩')}, "
          f"廖老师={sum(1 for r in calibratable if r['expert_name']=='廖老师')}")

    # Initialize judge
    judge = LLMJudgeMetric()
    print(f"\nJudge model: {judge.model}")

    # Score each record
    results = []
    expert_scores = []
    judge_scores = []

    for i, record in enumerate(calibratable):
        dialogue = record["dialogue"]
        popup = record["system_popup"]
        expert_raw = record["expert_score"]

        # Create mock objects for LLMJudgeMetric
        class MockGold:
            pass

        class MockPred:
            pass

        gold = MockGold()
        gold.question = dialogue
        pred = MockPred()
        pred.answer = popup

        print(f"\n[{i+1}/{len(calibratable)}] {record['id'][:50]}...")
        print(f"  专家评分: {expert_raw}/10")

        # Get judge score
        try:
            j_score = judge(gold, pred, trace=False)
            judge_scores.append(j_score)
            expert_normalized = normalize_expert(expert_raw)
            expert_scores.append(expert_normalized)

            print(f"  Judge评分: {j_score:.3f}  |  专家归一化: {expert_normalized:.3f}  "
                  f"|  偏差: {j_score - expert_normalized:+.3f}")

            results.append({
                "id": record["id"],
                "expert_name": record["expert_name"],
                "expert_score_raw": expert_raw,
                "expert_score_normalized": round(expert_normalized, 4),
                "judge_score": round(j_score, 4),
                "deviation": round(j_score - expert_normalized, 4),
                "dialogue_preview": dialogue[:120],
                "popup_preview": popup[:120],
            })
        except Exception as e:
            print(f"  ❌ Judge 调用失败: {e}")
            continue

        time.sleep(0.3)

    if len(judge_scores) < 10:
        print(f"\n⚠ 成功评分的记录不足 ({len(judge_scores)} < 10)")
        return

    # === Statistical Analysis ===
    print("\n" + "=" * 60)
    print("校准统计分析")
    print("=" * 60)

    js = np.array(judge_scores)
    es = np.array(expert_scores)

    # Pearson r
    pearson_r, pearson_p = stats.pearsonr(js, es)
    print(f"\nPearson r: {pearson_r:.4f} (p={pearson_p:.4f})")

    # Spearman ρ
    spearman_rho, spearman_p = stats.spearmanr(js, es)
    print(f"Spearman ρ: {spearman_rho:.4f} (p={spearman_p:.4f})")

    # MAE
    mae = np.mean(np.abs(js - es))
    print(f"MAE: {mae:.4f}")

    # RMSE
    rmse = np.sqrt(np.mean((js - es) ** 2))
    print(f"RMSE: {rmse:.4f}")

    # Mean bias (positive = judge overrates vs expert)
    mean_bias = np.mean(js - es)
    print(f"Mean bias: {mean_bias:+.4f} ({'judge偏高' if mean_bias > 0.05 else 'judge偏低' if mean_bias < -0.05 else '基本一致'})")

    # Correlation strength assessment
    print(f"\n相关性判据:")
    if pearson_r >= 0.7:
        verdict = "强相关 ✅ — judge 可直接使用"
    elif pearson_r >= 0.4:
        verdict = "中等相关 ⚠️ — 建议检查维度偏差后使用"
    else:
        verdict = "弱相关 ❌ — 需要重修 judge rubric 或更换 judge model"
    print(f"  → {verdict}")

    # Score range comparison
    print(f"\n评分区间对比:")
    print(f"  专家评分: {min(expert_scores):.3f} - {max(expert_scores):.3f} "
          f"(均值 {mean(expert_scores):.3f}, 标准差 {stdev(expert_scores):.3f})")
    print(f"  Judge评分: {min(judge_scores):.3f} - {max(judge_scores):.3f} "
          f"(均值 {mean(judge_scores):.3f}, 标准差 {stdev(judge_scores):.3f})")

    # Save report
    report = {
        "config": {
            "judge_model": judge.model,
            "num_samples": len(judge_scores),
            "expert_distribution": {
                r["expert_name"]: sum(1 for x in results if x["expert_name"] == r["expert_name"])
                for r in results
            },
        },
        "correlation": {
            "pearson_r": round(float(pearson_r), 4),
            "pearson_p": round(float(pearson_p), 4),
            "spearman_rho": round(float(spearman_rho), 4),
            "spearman_p": round(float(spearman_p), 4),
            "mae": round(float(mae), 4),
            "rmse": round(float(rmse), 4),
            "mean_bias": round(float(mean_bias), 4),
        },
        "score_ranges": {
            "expert": {
                "min": round(float(min(expert_scores)), 3),
                "max": round(float(max(expert_scores)), 3),
                "mean": round(float(mean(expert_scores)), 3),
                "std": round(float(stdev(expert_scores)), 3),
            },
            "judge": {
                "min": round(float(min(judge_scores)), 3),
                "max": round(float(max(judge_scores)), 3),
                "mean": round(float(mean(judge_scores)), 3),
                "std": round(float(stdev(judge_scores)), 3),
            },
        },
        "verdict": verdict,
        "per_sample": results,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n校准报告已保存: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
