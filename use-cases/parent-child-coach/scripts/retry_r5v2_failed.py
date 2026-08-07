"""Retry failed r5 v2 generations and re-judge."""
import json, time, re, os
from pathlib import Path
import requests

PROJECT = Path(r'D:\prompt-ops\use-cases\parent-child-coach')
RESULTS_DIR = PROJECT / 'results' / 'auto_research_judge_v2'
R5V2_PROMPT = (RESULTS_DIR / 'final_best_prompt_v2.txt').read_text(encoding='utf-8')

# Load previous results and identify failed cases (short output < 50 chars)
prev = json.load(open(RESULTS_DIR / 'r5v2_rubric_validation.json', encoding='utf-8'))
failed = [r for r in prev['results'] if r['version'] == 'r5v2' and len(r.get('popup', '')) < 50]
print(f'Failed cases to retry: {len(failed)}')
for f in failed:
    print(f'  {f["file"]}:{f["window"]} ({len(f.get("popup",""))} chars)')

# API config
DS_URL = "https://api.deepseek.com/v1/chat/completions"
DS_KEY = None
env_path = PROJECT / '.env'
if env_path.exists():
    for line in open(env_path, encoding='utf-8'):
        if line.startswith("DEEPSEEK_API_KEY="):
            DS_KEY = line.split("=", 1)[1].strip(); break
if not DS_KEY:
    DS_KEY = os.getenv("DEEPSEEK_API_KEY")

CL_URL = "https://s.lconai.com/v1/messages"
CL_KEY = "CLAUDE_API_KEY_PLACEHOLDER"

# Load rubric prompt from phase2_rubric_align.py
RUBRIC_PROMPT = """你是弹窗质量评审员。你的任务是对比「AI 生成的弹窗」和「人类专家标注的基准」，从四个维度打分。

# 评分维度

## 1. 类型匹配 (type_match) — 0 或 1
专家标注了该场景期望的弹窗类型：
- **诊断式** (diagnostic)：家长陷入了思维惯性或盲区，弹窗应该帮他"看见没注意到的东西"
- **鼓励式** (empowering)：家长已经做得很好，弹窗应该"肯定他已经做到的"，不需要诊断或建议

**判断标准**：AI 弹窗的整体语气和意图，是否与专家期望的类型一致？
- 1 = 一致
- 0 = 不一致

## 2. 洞察点覆盖 (insight_coverage) — 1-5 分
专家标注了弹窗"必须命中的洞察点"（hit_list / golden_sentences）。
- 5 = 覆盖了所有关键洞察点
- 4 = 覆盖了大部分
- 3 = 覆盖了一半
- 2 = 覆盖了少数
- 1 = 完全没命中

## 3. 主要矛盾一致性 (core_conflict_alignment) — 1-5 分
专家手写了 golden_popup，其中指出了该场景的"主要矛盾"。
AI 弹窗识别的核心矛盾，与专家是否一致？
- 5 = 高度一致，说的是同一件事
- 1 = 完全不一致

## 4. 避雷检查 (forbidden_check) — 0 或 1
AI 弹窗是否触碰了专家标注的红线？
- 1 = 完全避开
- 0 = 触碰了红线

# 产出格式
```json
{{
  "type_match": 1,
  "insight_coverage": 4,
  "core_conflict_alignment": 4,
  "forbidden_check": 1,
  "brief_reason": "一句话"
}}
```

---

**对话**：
{dialogue}

**专家期望类型**：{expected_tone}
**专家洞察点**：
{hit_list}
**专家参考弹窗**：
{golden_popup}
**专家禁止项**：
{forbidden_list}

**AI 生成的弹窗**：
{popup}
"""

# Load original matches data for retry
GB_DIR = Path(r'D:\星灵-soul-手搓\亲子沟通洞见\测试智能体\data\golden_bank')
blind = json.load(open(RESULTS_DIR / 'h2h_r5_v17_expert_blind.json', encoding='utf-8'))

def gen(dialogue):
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {DS_KEY}'}
    for attempt in range(4):
        try:
            resp = requests.post(DS_URL, json={
                'model': 'deepseek-v4-pro',
                'messages': [{'role': 'user', 'content': f"{R5V2_PROMPT}\n\n对话：\n{dialogue}"}],
                'temperature': 0.7, 'max_tokens': 600,
            }, headers=headers, timeout=(30, 90))
            resp.raise_for_status()
            text = resp.json()['choices'][0]['message']['content'].strip()
            if len(text) >= 30:
                return text
            print(f'    attempt {attempt+1}: only {len(text)} chars, retrying...')
        except Exception as e:
            print(f'    attempt {attempt+1}: {e}')
            time.sleep(3)
    return '[GEN_FAILED]'

def judge(match, popup):
    hit_list = '\n'.join(f'- {h}' for h in (match.get('hit_list',[]) + match.get('golden_sentences',[]))) or '(无)'
    forbidden_list = '\n'.join(f'- {f}' for f in (match.get('forbidden_list',[]) + match.get('problem_sentences',[]))) or '(无)'
    prompt = RUBRIC_PROMPT.format(
        dialogue=match['dialogue'], expected_tone=match['expected_tone'],
        hit_list=hit_list, golden_popup=match.get('golden_popup','') or '(无)',
        forbidden_list=forbidden_list, popup=popup,
    )
    headers = {'x-api-key': CL_KEY, 'anthropic-version': '2023-06-01', 'Content-Type': 'application/json'}
    for attempt in range(3):
        try:
            resp = requests.post(CL_URL, json={
                'model': 'claude-opus-4-8', 'max_tokens': 512, 'temperature': 0.0,
                'thinking': {'type': 'disabled'},
                'system': '你是弹窗质量评审员。只输出JSON。',
                'messages': [{'role': 'user', 'content': prompt}],
            }, headers=headers, timeout=(30, 90))
            resp.raise_for_status()
            for block in resp.json().get('content', []):
                if isinstance(block, dict) and block.get('type') == 'text':
                    m = re.search(r'\{[^{}]*"type_match"[^{}]*\}', block['text'], re.DOTALL)
                    if m: return json.loads(m.group(0))
                    m = re.search(r'\{.*\}', block['text'], re.DOTALL)
                    if m:
                        try: return json.loads(m.group(0))
                        except: pass
            return {'error': 'parse'}
        except Exception as e:
            if attempt == 2: return {'error': str(e)[:80]}
            time.sleep(2)
    return {'error': 'retries'}

# Retry
updated = []
for i, f in enumerate(failed):
    print(f'\n[{i+1}/{len(failed)}] {f["file"]}:{f["window"]} tone={f["expected_tone"]}')

    # Find full match data from the original results
    match = None
    for r in prev['results']:
        if r['file'] == f['file'] and r['window'] == f['window'] and r['version'] == 'v17':
            match = {'file': f['file'], 'window': f['window'],
                     'expected_tone': f['expected_tone'], 'dialogue': '', 'v17_popup': r['popup']}
            break

    # Need dialogue — find from blind results
    for m in prev['results']:
        if m['file'] == f['file'] and m['window'] == f['window'] and 'dialogue' in m:
            match['dialogue'] = m.get('dialogue', '')
    if not match or not match.get('dialogue'):
        print('  SKIP: no dialogue found')
        continue

    # Also need golden_bank data
    gb_path = GB_DIR / f['file']
    if gb_path.exists():
        gb = json.load(open(gb_path, encoding='utf-8'))
        for w in gb.get('windows', []):
            if w.get('window_label', '') == f['window']:
                match['hit_list'] = w.get('hit_list', [])
                match['golden_sentences'] = w.get('golden_sentences', [])
                match['forbidden_list'] = w.get('forbidden_list', [])
                match['problem_sentences'] = w.get('problem_sentences', [])
                match['golden_popup'] = (w.get('golden_popup') or '').strip()
                break

    # Generate
    print('  gen...', end=' ', flush=True)
    popup = gen(match['dialogue'])
    print(f'({len(popup)} chars)', end=' ', flush=True)

    if popup == '[GEN_FAILED]':
        print('FAILED')
        updated.append({**f, 'popup': '[GEN_FAILED]'})
        continue

    # Judge
    print('judge...', end=' ', flush=True)
    j = judge(match, popup)
    entry = {**f, 'version': 'r5v2', 'popup': popup, **j}
    if 'error' in j:
        print(f'ERROR: {j["error"]}')
    else:
        print(f'type={j.get("type_match")} ins={j.get("insight_coverage")} con={j.get("core_conflict_alignment")}')
    updated.append(entry)

# Merge back
new_results = [r for r in prev['results'] if not (r['version']=='r5v2' and r['file'] in {u['file'] for u in updated})]
new_results += updated
prev['results'] = new_results

# Recompute summary
r5v2 = [r for r in new_results if r['version']=='r5v2' and r.get('type_match',-1)>=0]
v17 = [r for r in new_results if r['version']=='v17' and r.get('type_match',-1)>=0]

from statistics import mean
print('\n' + '='*60)
print('UPDATED RESULTS')
print('='*60)
for label, res in [('r5 v2', r5v2), ('v1.7', v17)]:
    type_ok = sum(1 for r in res if r.get('type_match')==1)
    ins = mean(r.get('insight_coverage',1) for r in res)
    con = mean(r.get('core_conflict_alignment',1) for r in res)
    print(f'\n  {label} (N={len(res)}):')
    print(f'    类型匹配: {type_ok}/{len(res)} ({type_ok/len(res):.0%})')
    print(f'    洞察点覆盖: {ins:.1f}/5')
    print(f'    主要矛盾一致: {con:.1f}/5')

# Save
out_path = RESULTS_DIR / 'r5v2_rubric_validation.json'
json.dump(prev, open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print(f'\nSaved to {out_path}')
