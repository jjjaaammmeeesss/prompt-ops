"""Validate r5 v2 prompt against rubric: generate new popups, judge vs v1.7."""
import json, os, re, time
from pathlib import Path
from statistics import mean
import requests

PROJECT = Path(r'D:\prompt-ops\use-cases\parent-child-coach')
GB_DIR = Path(r'D:\星灵-soul-手搓\亲子沟通洞见\测试智能体\data\golden_bank')
RESULTS_DIR = PROJECT / 'results' / 'auto_research_judge_v2'
BLIND_PATH = RESULTS_DIR / 'h2h_r5_v17_expert_blind.json'
R5V2_PROMPT = (RESULTS_DIR / 'final_best_prompt_v2.txt').read_text(encoding='utf-8')

DS_URL = "https://api.deepseek.com/v1/chat/completions"
DS_KEY = None
env_path = PROJECT / '.env'
if env_path.exists():
    for line in open(env_path, encoding='utf-8'):
        if line.startswith("DEEPSEEK_API_KEY="):
            DS_KEY = line.split("=", 1)[1].strip(); break
if not DS_KEY:
    DS_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DS_KEY:
    raise RuntimeError("DEEPSEEK_API_KEY required")

CL_URL = "https://s.lconai.com/v1/messages"
CL_KEY = "CLAUDE_API_KEY_PLACEHOLDER"
CL_MODEL = "claude-opus-4-8"

# Same rubric judge prompt as phase2_rubric_align.py
RUBRIC_JUDGE_PROMPT = """你是弹窗质量评审员。你的任务是对比「AI 生成的弹窗」和「人类专家标注的基准」，从四个维度打分。

# 评分维度

## 1. 类型匹配 (type_match) — 0 或 1
专家标注了该场景期望的弹窗类型：
- **诊断式** (diagnostic)：家长陷入了思维惯性或盲区，弹窗应该帮他"看见没注意到的东西"
- **鼓励式** (empowering)：家长已经做得很好，弹窗应该"肯定他已经做到的"，不需要诊断或建议

**判断标准**：AI 弹窗的整体语气和意图，是否与专家期望的类型一致？
- 1 = 一致（专家要鼓励，AI 就在鼓励；专家要诊断，AI 就在诊断）
- 0 = 不一致（专家要鼓励，AI 却在诊断/给建议；或专家要诊断，AI 却在单纯夸奖）

## 2. 洞察点覆盖 (insight_coverage) — 1-5 分
专家标注了弹窗"必须命中的洞察点"（hit_list / golden_sentences）。

**判断标准**：AI 弹窗覆盖了多少个专家指定的洞察点？
- 5 = 覆盖了所有关键洞察点，且表达自然
- 4 = 覆盖了大部分（≥75%）
- 3 = 覆盖了一半左右
- 2 = 只覆盖了少数（≤25%）
- 1 = 完全没有命中任何专家洞察点

## 3. 主要矛盾一致性 (core_conflict_alignment) — 1-5 分
专家手写了 golden_popup（参考弹窗），其中指出了该场景的"主要矛盾"——即家长和孩子之间最核心的那个张力是什么。

**判断标准**：AI 弹窗识别的核心矛盾，与专家 golden_popup 中的核心矛盾，是否一致？
- 5 = 高度一致，AI 弹窗和专家弹窗在说"同一件事"
- 4 = 大部分一致，AI 弹窗的核心判断与专家对齐，只是表达不同
- 3 = 部分一致，方向对但不够精准
- 2 = 偏差较大，AI 弹窗关注了次要问题而忽略了主要矛盾
- 1 = 完全不一致，AI 弹窗说的跟专家关注的不是同一件事

## 4. 避雷检查 (forbidden_check) — 0 或 1
专家标注了弹窗"不应该出现"的内容（forbidden_list / problem_sentences）。

**判断标准**：AI 弹窗是否触碰了这些红线？
- 1 = 完全避开了所有红线
- 0 = 触碰了至少一个红线

# 产出格式
```json
{{
  "type_match": 1,
  "insight_coverage": 4,
  "core_conflict_alignment": 4,
  "forbidden_check": 1,
  "brief_reason": "一句话总结对比结论"
}}
```

---

现在请评审以下弹窗：

**对话**：
{dialogue}

**专家期望类型**：{expected_tone}
**专家洞察点 (hit_list / golden_sentences)**：
{hit_list}
**专家参考弹窗 (golden_popup)**：
{golden_popup}
**专家禁止项 (forbidden_list / problem_sentences)**：
{forbidden_list}

**AI 生成的弹窗**：
{popup}
"""


def load_matches():
    """Load matched blind results with rich golden_bank annotations."""
    blind = json.load(open(BLIND_PATH, encoding='utf-8'))
    matches = []
    seen = set()
    for f in sorted(GB_DIR.glob('GB_*.json')):
        d = json.load(open(f, encoding='utf-8'))
        for w in d.get('windows', []):
            wd = w.get('window_dialogue', '') or d.get('full_dialogue', '')
            if not wd.strip(): continue
            gb_clean = re.sub(r'\d+[：:]\s*', '', wd)
            gb_clean = re.sub(r'\s+', '', gb_clean)[:100]
            for r in blind['results']:
                r_clean = re.sub(r'\s+', '', r['dialogue'])[:100]
                overlap = sum(1 for a, b in zip(gb_clean, r_clean) if a == b)
                if overlap < 40: continue
                gp = (w.get('golden_popup') or '').strip()
                if not gp and not w.get('hit_list'): continue
                # Dedup
                key = f.name + w.get('window_label', '')
                if key in seen: continue
                seen.add(key)
                matches.append({
                    'file': f.name, 'window': w.get('window_label', ''),
                    'expected_tone': w.get('expected_tone', ''),
                    'golden_popup': gp,
                    'hit_list': w.get('hit_list', []),
                    'golden_sentences': w.get('golden_sentences', []),
                    'forbidden_list': w.get('forbidden_list', []),
                    'problem_sentences': w.get('problem_sentences', []),
                    'dialogue': r['dialogue'],
                    'v17_popup': r['v17']['popup'],
                })
                break
    return matches


def deepseek_gen(dialogue):
    """Generate popup using DeepSeek with r5 v2 prompt."""
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {DS_KEY}',
    }
    full_prompt = f"{R5V2_PROMPT}\n\n对话：\n{dialogue}"
    for attempt in range(3):
        try:
            resp = requests.post(DS_URL, json={
                'model': 'deepseek-v4-pro',
                'messages': [{'role': 'user', 'content': full_prompt}],
                'temperature': 0.7, 'max_tokens': 512,
            }, headers=headers, timeout=(30, 60))
            resp.raise_for_status()
            return resp.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            if attempt == 2:
                return f'[ERROR: {e}]'
            time.sleep(2)
    return '[ERROR]'


def rubrics_judge(match, popup):
    """Judge a popup against expert rubric using Claude."""
    hit_list = match['hit_list'] + match['golden_sentences']
    forbidden_list = match['forbidden_list'] + match['problem_sentences']

    prompt = RUBRIC_JUDGE_PROMPT.format(
        dialogue=match['dialogue'],
        expected_tone=match['expected_tone'],
        hit_list='\n'.join(f'- {h}' for h in hit_list) if hit_list else '(无)',
        golden_popup=match['golden_popup'] or '(专家未提供参考弹窗)',
        forbidden_list='\n'.join(f'- {f}' for f in forbidden_list) if forbidden_list else '(无)',
        popup=popup,
    )

    headers = {
        'x-api-key': CL_KEY,
        'anthropic-version': '2023-06-01',
        'Content-Type': 'application/json',
    }

    for attempt in range(3):
        try:
            resp = requests.post(CL_URL, json={
                'model': CL_MODEL, 'max_tokens': 512, 'temperature': 0.0,
                'thinking': {'type': 'disabled'},
                'system': '你是弹窗质量评审员。只输出JSON，不要markdown包裹。',
                'messages': [{'role': 'user', 'content': prompt}],
            }, headers=headers, timeout=(30, 90))
            resp.raise_for_status()
            data = resp.json()
            for block in data.get('content', []):
                if isinstance(block, dict) and block.get('type') == 'text':
                    text = block['text']
                    m = re.search(r'\{[^{}]*"type_match"[^{}]*\}', text, re.DOTALL)
                    if m: return json.loads(m.group(0))
                    m = re.search(r'\{.*\}', text, re.DOTALL)
                    if m:
                        try: return json.loads(m.group(0))
                        except: pass
                    return {'error': 'parse_failed', 'raw': text[:200]}
            return {'error': 'no_text_block', 'raw': str(data)[:200]}
        except requests.exceptions.HTTPError as e:
            return {'error': f'HTTP {e.response.status_code}'}
        except Exception as e:
            if attempt == 2: return {'error': str(e)[:100]}
            time.sleep(2 ** attempt * 3)
    return {'error': 'max_retries'}


def main():
    print('=' * 60)
    print('r5 v2 Validation: Rubric-based Expert Alignment')
    print('=' * 60)

    matches = load_matches()
    print(f'\nUsable matches: {len(matches)}')

    results = []
    for i, m in enumerate(matches):
        # Generate r5 v2 popup
        print(f'  [{i+1}/{len(matches)}] {m["file"]}:{m["window"]} gen...', end=' ', flush=True)
        r5v2_popup = deepseek_gen(m['dialogue'])
        if r5v2_popup.startswith('[ERROR'):
            print(f'GEN ERROR: {r5v2_popup[:60]}')
            continue
        print(f'({len(r5v2_popup)} chars)', end=' ', flush=True)

        # Judge r5 v2
        print('judge r5v2...', end=' ', flush=True)
        r5v2_j = rubrics_judge(m, r5v2_popup)

        # Judge v1.7
        print('v17...', end=' ', flush=True)
        v17_j = rubrics_judge(m, m['v17_popup'])

        for ver, judge, popup in [('r5v2', r5v2_j, r5v2_popup), ('v17', v17_j, m['v17_popup'])]:
            entry = {
                'file': m['file'], 'window': m['window'],
                'version': ver, 'expected_tone': m['expected_tone'],
                **judge, 'popup': popup,
            }
            if 'error' in judge:
                print(f'{ver}=ERROR ', end='')
                entry['type_match'] = -1
            else:
                print(f'{ver}:type={judge.get("type_match")} ins={judge.get("insight_coverage")} con={judge.get("core_conflict_alignment")} ', end='')
            results.append(entry)
        print()

    # Summary
    r5v2 = [r for r in results if r['version'] == 'r5v2' and r.get('type_match', -1) >= 0]
    v17 = [r for r in results if r['version'] == 'v17' and r.get('type_match', -1) >= 0]

    print('\n' + '=' * 60)
    print('RESULTS: r5 v2 vs v1.7')
    print('=' * 60)

    for label, res in [('r5 v2', r5v2), ('v1.7', v17)]:
        if not res: continue
        type_ok = sum(1 for r in res if r.get('type_match') == 1)
        forbid_ok = sum(1 for r in res if r.get('forbidden_check') == 1)
        ins = mean(r.get('insight_coverage', 1) for r in res)
        con = mean(r.get('core_conflict_alignment', 1) for r in res)
        print(f'\n  {label} (N={len(res)}):')
        print(f'    类型匹配: {type_ok}/{len(res)} ({type_ok/len(res):.0%})')
        print(f'    洞察点覆盖: {ins:.1f}/5')
        print(f'    主要矛盾一致: {con:.1f}/5')
        print(f'    避雷通过: {forbid_ok}/{len(res)}')

    # H2H
    print(f'\n  {"Case":<25} {"Tone":>10} {"v1.7":>8} {"r5v2":>8} {"Winner":>8}')
    print(f'  {"-"*25} {"-"*10} {"-"*8} {"-"*8} {"-"*8}')
    r5_w, v17_w, tie = 0, 0, 0
    for m in matches:
        r5j = next((r for r in r5v2 if r['file']==m['file'] and r['window']==m['window']), None)
        vj = next((r for r in v17 if r['file']==m['file'] and r['window']==m['window']), None)
        if not r5j or not vj: continue
        def sc(r):
            return r.get('type_match',0)*2 + r.get('insight_coverage',0) + r.get('core_conflict_alignment',0) + r.get('forbidden_check',0)*2
        sr, sv = sc(r5j), sc(vj)
        if sr > sv: r5_w += 1; w = 'r5v2'
        elif sv > sr: v17_w += 1; w = 'v1.7'
        else: tie += 1; w = 'tie'
        label = f'{m["file"]}:{m["window"]}'[:24]
        print(f'  {label:<25} {m["expected_tone"]:>10} {sv:>8} {sr:>8} {w:>8}')
    print(f'\n  H2H: r5v2={r5_w}  v1.7={v17_w}  tie={tie}')

    # Save
    out_path = RESULTS_DIR / 'r5v2_rubric_validation.json'
    json.dump({'matches': len(matches), 'results': results}, open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'\nSaved to {out_path}')


if __name__ == '__main__':
    main()
