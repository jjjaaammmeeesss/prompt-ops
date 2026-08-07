"""[历史实验] 汇总 8 轮 auto-evolve 迭代结果（全部 discard）"""
import json
from pathlib import Path

R = Path("D:/prompt-ops/use-cases/parent-child-coach/results")
b = json.load(open(R / "auto_baseline_v25_full.json", encoding='utf-8'))
b_agg = b['aggregate']
print(f"基线 (n=3, 62 cases):")
print(f"  M1={b_agg['m1_trigger_accuracy']:.1%} M5={b_agg['m5_tone_match']:.1%} M6={b_agg['m6_insight_quality']:.2f} M7={b_agg['m7_safety_score']:.2f} overall={b_agg['overall_score']:.3f}")
print()
print(f"{'iter':<5}{'target':<14}{'version':<14}{'M1':<8}{'M5':<8}{'M6':<8}{'M7':<8}{'overall':<10}{'Δ':<10}{'reason'}")
print("─" * 110)

iters = [
    (1, "perception", "v2.6"),
    (2, "perception", "v2.7"),
    (3, "perception", "v2.8"),
    (4, "master", "v3.6"),
    (5, "master", "v3.7"),
    (6, "master", "v3.8"),
    (7, "production", "v3.3"),
]
for i, tgt, ver in iters:
    f = R / f"auto_iter_{i:02d}_{tgt}_{ver}_discard.json"
    if not f.exists():
        continue
    d = json.load(open(f, encoding='utf-8'))
    agg = d['aggregate']
    meta = d.get('meta', {})
    reason = meta.get('reason', '')[:60]
    delta = agg['overall_score'] - b_agg['overall_score']
    print(f"{i:<5}{tgt:<14}{ver:<14}{agg['m1_trigger_accuracy']:<8.1%}{agg['m5_tone_match']:<8.1%}{agg['m6_insight_quality']:<8.2f}{agg['m7_safety_score']:<8.2f}{agg['overall_score']:<10.3f}{delta:<+10.3f}{reason}")

print()
print("=== 关键发现 ===")
print(f"基线 overall = {b_agg['overall_score']:.3f}")
print(f"所有 7 次变异均被丢弃 — 最佳变体仍 Δ=-0.018")
