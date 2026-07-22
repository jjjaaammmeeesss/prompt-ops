"""Phase 2: Compare blind H2H results against golden_bank expert scores."""
import json, os, re
from statistics import mean, stdev

# Load Phase 1 blind results
blind_path = r'D:\prompt-ops\use-cases\parent-child-coach\results\auto_research_judge_v2\h2h_r5_v17_expert_blind.json'
blind = json.load(open(blind_path, encoding='utf-8'))

# Load all golden_bank scored windows
gb_dir = r'D:\星灵-soul-手搓\亲子沟通洞见\测试智能体\data\golden_bank'
gb_scores = []
for f in sorted(os.listdir(gb_dir)):
    if not f.endswith('.json'):
        continue
    d = json.load(open(os.path.join(gb_dir, f), encoding='utf-8'))
    full = d.get('full_dialogue', '')
    for w in d.get('windows', []):
        score = w.get('overall_score')
        if score is None:
            continue
        w_dialogue = w.get('window_dialogue', '') or full
        clean = re.sub(r'\d+[孩子妈妈爸爸儿子女儿老师同学]*[：:]\s*', '', w_dialogue)
        clean = re.sub(r'\s+', '', clean)[:120]
        gb_scores.append({
            'dialogue_clean': clean, 'score': score,
            'tone': w.get('expected_tone', ''),
            'golden_popup': w.get('golden_popup', ''),
            'window_label': w.get('window_label', ''),
            'case_title': d.get('case_title', ''), 'file': f,
        })

# Match
comparisons = []
for r in blind['results']:
    dialogue = r['dialogue']
    clean_d = re.sub(r'\d+[孩子妈妈爸爸儿子女儿老师同学]*[：:]\s*', '', dialogue)
    clean_d = re.sub(r'\s+', '', clean_d)[:120]
    best, best_overlap = None, 0
    for gb in gb_scores:
        overlap = 0
        for a, b in zip(clean_d, gb['dialogue_clean']):
            if a == b:
                overlap += 1
            else:
                break
        if overlap > best_overlap:
            best_overlap = overlap
            best = gb
    if best and best_overlap >= 30:
        exp_score = best['score']
        exp_norm = exp_score / 10.0
        v17_score = r['v17']['score']
        r5_score = r['r5']['score']
        v17_err = abs(v17_score - exp_norm)
        r5_err = abs(r5_score - exp_norm)
        closer = 'v1.7' if v17_err < r5_err else ('r5' if r5_err < v17_err else 'tie')
        comparisons.append({
            'case_title': r.get('case_title', ''),
            'source_file': r.get('source_file', ''),
            'window_label': r.get('window_label', ''),
            'expert_score_raw': exp_score,
            'expert_norm': exp_norm,
            'v17_score': v17_score,
            'r5_score': r5_score,
            'v17_error': round(v17_err, 6),
            'r5_error': round(r5_err, 6),
            'closer': closer,
            'expert_tone': best['tone'],
            'v17_veto': r['v17'].get('veto', ''),
            'r5_veto': r['r5'].get('veto', ''),
            'overlap_chars': best_overlap,
            'golden_file': best['file'],
        })

# Split valid vs parse-error
valid = [c for c in comparisons
         if 'parse_error' not in str(c['v17_veto'])
         and 'parse_error' not in str(c['r5_veto'])]
parse_err = [c for c in comparisons
             if 'parse_error' in str(c['v17_veto'])
             or 'parse_error' in str(c['r5_veto'])]

print(f'Total matched: {len(comparisons)}')
print(f'Valid (no parse errors): {len(valid)}')
print(f'With parse errors: {len(parse_err)}')
print()

if valid:
    v17_mae = mean(c['v17_error'] for c in valid)
    r5_mae = mean(c['r5_error'] for c in valid)
    v17_closer = sum(1 for c in valid if c['closer'] == 'v1.7')
    r5_closer = sum(1 for c in valid if c['closer'] == 'r5')
    tie = sum(1 for c in valid if c['closer'] == 'tie')

    def pearson(xs, ys):
        n = len(xs)
        mx, my = mean(xs), mean(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        dx = (sum((x - mx) ** 2 for x in xs) ** 0.5)
        dy = (sum((y - my) ** 2 for y in ys) ** 0.5)
        return num / (dx * dy) if dx * dy > 0 else 0

    v17_scores = [c['v17_score'] for c in valid]
    r5_scores = [c['r5_score'] for c in valid]
    exp_scores = [c['expert_norm'] for c in valid]

    r_v17 = pearson(v17_scores, exp_scores)
    r_r5 = pearson(r5_scores, exp_scores)

    def spearman(xs, ys):
        n = len(xs)
        def rank(vals):
            sorted_vals = sorted(enumerate(vals), key=lambda x: x[1])
            ranks = [0] * n
            for i, (orig_idx, _) in enumerate(sorted_vals):
                ranks[orig_idx] = i + 1
            return ranks
        rx = rank(xs)
        ry = rank(ys)
        return pearson(rx, ry)

    rho_v17 = spearman(v17_scores, exp_scores)
    rho_r5 = spearman(r5_scores, exp_scores)

    v17_var = stdev(v17_scores) if len(v17_scores) > 1 else 0
    r5_var = stdev(r5_scores) if len(r5_scores) > 1 else 0
    exp_var = stdev(exp_scores) if len(exp_scores) > 1 else 0

    print('=' * 70)
    print('Phase 2 - Expert Alignment (校标对标)')
    print(f'N={len(valid)} (valid, no parse errors)')
    print()
    print(f'  {"":20} {"v1.7":<14} {"r5":<14} {"Expert":<14}')
    print(f'  {"MAE":20} {v17_mae:.4f}        {r5_mae:.4f}')
    print(f'  {"Closer Count":20} {v17_closer:<14} {r5_closer:<14}')
    print(f'  {"Pearson r":20} {r_v17:+.4f}        {r_r5:+.4f}')
    print(f'  {"Spearman rho":20} {rho_v17:+.4f}        {rho_r5:+.4f}')
    print(f'  {"Score stdev":20} {v17_var:.4f}        {r5_var:.4f}        {exp_var:.4f}')
    print()

    print(f'  Score distribution:')
    print(f'  {"":20} {"v1.7":<14} {"r5":<14} {"Expert":<14}')
    for label, lo, hi in [
        ('0.0-0.2', 0, 0.2), ('0.2-0.4', 0.2, 0.4), ('0.4-0.6', 0.4, 0.6),
        ('0.6-0.8', 0.6, 0.8), ('0.8-1.0', 0.8, 1.0)
    ]:
        v17_n = sum(1 for s in v17_scores if lo <= s < hi)
        r5_n = sum(1 for s in r5_scores if lo <= s < hi)
        exp_n = sum(1 for s in exp_scores if lo <= s < hi)
        print(f'  {label:<20} {v17_n:<14} {r5_n:<14} {exp_n:<14}')

    print()
    print(f'  {"Case":<32} {"Exp":>4} {"v1.7":>6} {"r5":>6} {"dv17":>8} {"dr5":>8} {"Win":>5}')
    print(f'  {"-"*32} {"-"*4} {"-"*6} {"-"*6} {"-"*8} {"-"*8} {"-"*5}')
    for c in valid:
        title = c['case_title'][:30]
        print(f'  {title:<32} {c["expert_score_raw"]:>4} {c["v17_score"]:.4f} {c["r5_score"]:.4f} '
              f'{c["v17_error"]:+8.4f} {c["r5_error"]:+8.4f} {c["closer"]:>5}')

    # Save
    out = {
        'summary': {
            'n_total_matched': len(comparisons),
            'n_valid': len(valid),
            'n_parse_errors': len(parse_err),
            'v17_mae': v17_mae,
            'r5_mae': r5_mae,
            'v17_closer_count': v17_closer,
            'r5_closer_count': r5_closer,
            'tie_count': tie,
            'pearson_r': {'v1.7': r_v17, 'r5': r_r5},
            'spearman_rho': {'v1.7': rho_v17, 'r5': rho_r5},
            'v17_score_std': v17_var,
            'r5_score_std': r5_var,
            'expert_score_std': exp_var,
        },
        'comparisons': valid,
        'parse_error_comparisons': parse_err,
    }
    out_path = r'D:\prompt-ops\use-cases\parent-child-coach\results\auto_research_judge_v2\h2h_r5_v17_expert_compare_v2.json'
    json.dump(out, open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'\nSaved to {out_path}')
