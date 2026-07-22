"""检查 GB_001-GB_014 数据完整性"""
import json
ds = json.load(open('D:/prompt-ops/use-cases/parent-child-coach/data/golden_dataset.json', encoding='utf-8'))
for c in ds:
    if not c['case_id'].startswith('GB_'):
        continue
    diag = c.get('dialogue', '')
    win_text = c.get('windows', [{}])[0].get('window_text', '')
    ref = c.get('windows', [{}])[0].get('reference_popup', '')
    has_mojibake = any(ch in str(c) for ch in ['鍘', '瀛', '闂', '鎭'])
    print(f"{c['case_id']}: dialogue_len={len(diag)} win_text_len={len(win_text)} ref_len={len(ref)} mojibake={has_mojibake}")
