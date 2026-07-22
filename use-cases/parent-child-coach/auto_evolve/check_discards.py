"""检查 3 个 discard 变体"""
import json
for i in [1, 2, 3]:
    v = f'v2.{i+5}'
    fname = f'D:/prompt-ops/use-cases/parent-child-coach/results/auto_iter_{i:02d}_perception_{v}_discard.json'
    d = json.load(open(fname, encoding='utf-8'))
    agg = d['aggregate']
    meta = d.get('meta', {})
    print(f'=== iter {i}: perception {v} DISCARD ===')
    print(f'  reason: {meta.get("reason","")}')
    m1 = agg['m1_trigger_accuracy']
    m5 = agg['m5_tone_match']
    m6 = agg['m6_insight_quality']
    m7 = agg['m7_safety_score']
    ov = agg['overall_score']
    print(f'  M1={m1:.1%} M5={m5:.1%} M6={m6:.2f} M7={m7:.2f} overall={ov:.3f}')
    print()
