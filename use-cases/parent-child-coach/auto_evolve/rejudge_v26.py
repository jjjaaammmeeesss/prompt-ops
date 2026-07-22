"""用新的 should_keep 逻辑重新判断 v2.6"""
import json
import sys
sys.path.insert(0, "D:/prompt-ops/use-cases/parent-child-coach")

from auto_evolve.optimizer import should_keep
from auto_evolve.run_auto_evolve import load_full_report
from pathlib import Path

R = Path("D:/prompt-ops/use-cases/parent-child-coach/results")

baseline = load_full_report(R / "auto_baseline_v25_full.json")
v26 = load_full_report(R / "auto_iter_01_perception_v2.6_discard.json")

print(f"基线 (v3.1): overall={baseline.overall_score:.3f} M1={baseline.aggregate_m1:.1%} M5={baseline.aggregate_m5:.1%} M6={baseline.aggregate_m6:.2f} M7={baseline.aggregate_m7:.2f}")
print(f"v2.6:        overall={v26.overall_score:.3f} M1={v26.aggregate_m1:.1%} M5={v26.aggregate_m5:.1%} M6={v26.aggregate_m6:.2f} M7={v26.aggregate_m7:.2f}")
print()

keep, reason = should_keep(baseline, v26)
print(f"新 should_keep 判断: {'✅ KEEP' if keep else '❌ DISCARD'}")
print(f"原因: {reason}")

if keep:
    print("\n🎉 v2.6 应该被保留！但 prompt 文件已被删除，需要重新生成。")
    print("   DeepSeek 余额不足（402），无法重新跑评估。")
else:
    print(f"\nv2.6 仍然被 discard: {reason}")
