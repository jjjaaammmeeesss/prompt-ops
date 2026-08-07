"""检查 62-case baseline"""
import json
b = json.load(open('D:/prompt-ops/use-cases/parent-child-coach/results/auto_baseline_v25_full.json', encoding='utf-8'))
agg = b['aggregate']
print('=== 62-case baseline (n=3) ===')
m1 = agg['m1_trigger_accuracy']
m5 = agg['m5_tone_match']
m6 = agg['m6_insight_quality']
m7 = agg['m7_safety_score']
ov = agg['overall_score']
print(f'  M1={m1:.1%} M5={m5:.1%} M6={m6:.2f} M7={m7:.2f} overall={ov:.3f}')
print(f'  n_cases={len(b["per_case"])}')

# 失败统计
fails = []
for c in b['per_case']:
    issues = []
    if c.get('m5_tone_match') == 0 and c.get('gold_should_popup', True):
        issues.append(f"tone_mismatch(sys={c.get('sys_tone','')},gold={c.get('gold_tone','')})")
    if c.get('m6_insight_score') is not None and c['m6_insight_score'] < 3:
        issues.append(f"M6_low={c['m6_insight_score']:.1f}")
    if c.get('m7_safety_score') is not None and c['m7_safety_score'] < 4:
        issues.append(f"M7_low={c['m7_safety_score']:.1f}")
    if c.get('m1_trigger_match') == 0:
        issues.append(f"M1_mismatch(sys={c.get('sys_should_popup')},gold={c.get('gold_should_popup')})")
    if c.get('error'):
        issues.append('error')
    if issues:
        fails.append((c['case_id'], c.get('window_index',1), issues))

print(f'\n=== Stable failures (n=3 denoised): {len(fails)} ===')
for cid, wid, issues in fails:
    print(f'  {cid:10s} w{wid} | ' + ' | '.join(issues))
