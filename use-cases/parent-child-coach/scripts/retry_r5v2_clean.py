"""Clean retry: regenerate + re-judge failed r5 v2 cases."""
import json, time, re, os
from pathlib import Path
from statistics import mean
import requests

PROJECT = Path(r'D:\prompt-ops\use-cases\parent-child-coach')
RESULTS_DIR = PROJECT / 'results' / 'auto_research_judge_v2'
GB_DIR = Path(r'D:\星灵-soul-手搓\亲子沟通洞见\测试智能体\data\golden_bank')

R5V2_PROMPT = (RESULTS_DIR / 'final_best_prompt_v2.txt').read_text(encoding='utf-8')

DS_KEY = None
for line in open(PROJECT / '.env', encoding='utf-8'):
    if line.startswith('DEEPSEEK_API_KEY='):
        DS_KEY = line.split('=', 1)[1].strip(); break
if not DS_KEY:
    DS_KEY = os.getenv('DEEPSEEK_API_KEY', '')
DS_URL = 'https://api.deepseek.com/v1/chat/completions'
CL_URL = 'https://s.lconai.com/v1/messages'
CL_KEY = 'CLAUDE_API_KEY_PLACEHOLDER'

# ── Load data ──
blind = json.load(open(RESULTS_DIR / 'h2h_r5_v17_expert_blind.json', encoding='utf-8'))
prev = json.load(open(RESULTS_DIR / 'r5v2_rubric_validation.json', encoding='utf-8'))

failed_files = set()
for r in prev['results']:
    if r['version'] != 'r5v2':
        continue
    popup = r.get('popup', '')
    plen = len(popup)
    # Empty, very short, or truncated (ends without sentence-ending punctuation)
    if plen < 50 or (plen < 120 and not popup.rstrip()[-1] in ('。', '！', '？', '"', '」', '”')):
        failed_files.add(r['file'])
print('Failed cases:', failed_files)

# Build cases from golden_bank + blind results
cases = []
for f in sorted(GB_DIR.glob('GB_*.json')):
    if f.name not in failed_files:
        continue
    gb = json.load(open(f, encoding='utf-8'))
    for w in gb.get('windows', []):
        wd = w.get('window_dialogue', '') or gb.get('full_dialogue', '')
        if not wd.strip():
            continue
        wd_clean = re.sub(r'\d+[：:]\s*', '', wd).strip()
        for br in blind['results']:
            r_clean = re.sub(r'\s+', '', br['dialogue'])[:100]
            g_clean = re.sub(r'\s+', '', wd_clean)[:100]
            overlap = sum(1 for a, b in zip(g_clean, r_clean) if a == b)
            if overlap >= 40:
                case = {
                    'file': f.name,
                    'window': w.get('window_label', ''),
                    'expected_tone': w.get('expected_tone', ''),
                    'dialogue': br['dialogue'],
                    'golden_popup': (w.get('golden_popup') or '').strip(),
                    'hit_list': w.get('hit_list', []) + w.get('golden_sentences', []),
                    'forbidden_list': w.get('forbidden_list', []) + w.get('problem_sentences', []),
                }
                cases.append(case)
                break
        break  # first window only

print('Loaded {} cases for retry'.format(len(cases)))

# ── Judge prompt template ──
RJ_TEMPLATE = (
    '你是弹窗质量评审员。对比AI弹窗和专家基准，四个维度打分。\n'
    '\n'
    '## 1. 类型匹配 (type_match) 0/1\n'
    '专家期望: {expected_tone}。AI弹窗类型是否一致？1=一致 0=不一致。\n'
    '\n'
    '## 2. 洞察点覆盖 (insight_coverage) 1-5\n'
    '专家指定洞察点: {hit_list}。AI覆盖了多少？5=全覆盖 1=未命中。\n'
    '\n'
    '## 3. 主要矛盾一致性 (core_conflict_alignment) 1-5\n'
    '专家参考弹窗: {golden_popup}。AI的核心矛盾是否与专家一致？5=高度一致 1=完全不一致。\n'
    '\n'
    '## 4. 避雷 (forbidden_check) 0/1\n'
    '专家禁止项: {forbidden_list}。AI是否触碰红线？1=完全避开 0=触碰。\n'
    '\n'
    '输出JSON: {{"type_match":1,"insight_coverage":4,"core_conflict_alignment":4,"forbidden_check":1,"brief_reason":"..."}}\n'
    '\n'
    '对话: {dialogue}\n'
    'AI弹窗: {popup}'
)

# ── Retry ──
updated = []
for i, c in enumerate(cases):
    label = c['file'] + ':' + c['window']
    print('\n[{}/{}] {} tone={}'.format(i+1, len(cases), label, c['expected_tone']), end=' ', flush=True)

    # Generate
    ds_headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + DS_KEY}
    user_msg = R5V2_PROMPT + '\n\n对话：\n' + c['dialogue']
    popup = None
    for attempt in range(4):
        try:
            resp = requests.post(DS_URL, json={
                'model': 'deepseek-v4-pro',
                'messages': [{'role': 'user', 'content': user_msg}],
                'temperature': 0.7, 'max_tokens': 600,
            }, headers=ds_headers, timeout=(30, 90))
            resp.raise_for_status()
            text = resp.json()['choices'][0]['message']['content'].strip()
            if len(text) >= 30:
                popup = text
                break
            print('(short:{})'.format(len(text)), end='')
        except Exception as e:
            print('(e:{})'.format(type(e).__name__), end='')
        time.sleep(2)

    if not popup:
        print(' GEN_FAILED')
        continue
    print(' gen({}c)'.format(len(popup)), end=' ', flush=True)

    # Judge r5 v2
    hit_text = '\n'.join('- ' + str(h) for h in c['hit_list']) if c['hit_list'] else '(无)'
    forbid_text = '\n'.join('- ' + str(f) for f in c['forbidden_list']) if c['forbidden_list'] else '(无)'
    jprompt = RJ_TEMPLATE.format(
        expected_tone=c['expected_tone'],
        hit_list=hit_text,
        golden_popup=c['golden_popup'] or '(无)',
        forbidden_list=forbid_text,
        dialogue=c['dialogue'],
        popup=popup,
    )

    cl_headers = {
        'x-api-key': CL_KEY,
        'anthropic-version': '2023-06-01',
        'Content-Type': 'application/json',
    }
    judge_result = None
    for attempt in range(3):
        try:
            resp = requests.post(CL_URL, json={
                'model': 'claude-opus-4-8', 'max_tokens': 512, 'temperature': 0.0,
                'thinking': {'type': 'disabled'},
                'system': 'Only output JSON.',
                'messages': [{'role': 'user', 'content': jprompt}],
            }, headers=cl_headers, timeout=(30, 90))
            resp.raise_for_status()
            for block in resp.json().get('content', []):
                if isinstance(block, dict) and block.get('type') == 'text':
                    m = re.search(r'\{[^{}]*"type_match"[^{}]*\}', block['text'], re.DOTALL)
                    if m:
                        judge_result = json.loads(m.group(0))
                        break
                    m = re.search(r'\{.*\}', block['text'], re.DOTALL)
                    if m:
                        try:
                            judge_result = json.loads(m.group(0))
                            break
                        except:
                            pass
            if judge_result:
                break
        except:
            pass
        time.sleep(2)

    if not judge_result:
        print(' JUDGE_FAILED')
        continue

    entry = {
        'file': c['file'], 'window': c['window'], 'version': 'r5v2',
        'expected_tone': c['expected_tone'], 'popup': popup,
    }
    entry.update(judge_result)
    print('type={} ins={} con={}'.format(
        judge_result.get('type_match'),
        judge_result.get('insight_coverage'),
        judge_result.get('core_conflict_alignment')))
    updated.append(entry)

# ── Merge back ──
new_results = [r for r in prev['results']
               if not (r['version'] == 'r5v2' and r['file'] in {u['file'] for u in updated})]
new_results += updated
prev['results'] = new_results

# ── Summary ──
r5v2 = [r for r in new_results if r['version'] == 'r5v2' and r.get('type_match', -1) >= 0]
v17 = [r for r in new_results if r['version'] == 'v17' and r.get('type_match', -1) >= 0]

print('\n' + '=' * 60)
print('FINAL: r5 v2 vs v1.7')
print('=' * 60)
for label, res in [('r5 v2', r5v2), ('v1.7', v17)]:
    type_ok = sum(1 for r in res if r.get('type_match') == 1)
    ins = mean(r.get('insight_coverage', 1) for r in res)
    con = mean(r.get('core_conflict_alignment', 1) for r in res)
    forbid_ok = sum(1 for r in res if r.get('forbidden_check') == 1)
    print('\n  {} (N={}):'.format(label, len(res)))
    print('    类型匹配: {}/{} ({:.0%})'.format(type_ok, len(res), type_ok/len(res)))
    print('    洞察点覆盖: {:.1f}/5  主要矛盾: {:.1f}/5  避雷: {}/{}'.format(ins, con, forbid_ok, len(res)))

# ── H2H ──
from collections import defaultdict
cases_h2h = defaultdict(dict)
for r in new_results:
    if r.get('type_match', -1) < 0:
        continue
    cases_h2h[(r['file'], r['window'])][r['version']] = r

r5w, v17w, tie = 0, 0, 0
print('\n  {:<25} {:>10} {:>6} {:>6} {:>6}'.format('Case', 'Tone', 'v1.7', 'r5v2', 'Win'))
print('  ' + '-'*55)
for (file, win), vers in sorted(cases_h2h.items()):
    if 'r5v2' not in vers or 'v17' not in vers:
        continue
    def sc(r):
        return r.get('type_match', 0)*3 + r.get('insight_coverage', 0) + r.get('core_conflict_alignment', 0) + r.get('forbidden_check', 0)*2
    s5, s17 = sc(vers['r5v2']), sc(vers['v17'])
    if s5 > s17:
        r5w += 1; w = 'r5v2'
    elif s17 > s5:
        v17w += 1; w = 'v1.7'
    else:
        tie += 1; w = 'tie'
    label = (file + ':' + win)[:24]
    tone = vers['r5v2'].get('expected_tone', '')
    print('  {:<25} {:>10} {:>6} {:>6} {:>6}'.format(label, tone, s17, s5, w))
print('\n  H2H: r5v2={}  v1.7={}  tie={}'.format(r5w, v17w, tie))

out_path = RESULTS_DIR / 'r5v2_rubric_validation.json'
json.dump(prev, open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print('\nSaved to ' + str(out_path))
