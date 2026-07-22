"""
Phase 2 v2: Rubric-based expert alignment.
Evaluates r5 and v1.7 popups against golden_bank qualitative criteria:
  1. Type match (tone: diagnostic vs empowering)
  2. Insight coverage (hit_list, golden_sentences)
  3. Core conflict alignment (main contradiction match with golden_popup)
  4. Forbidden check (forbidden_list, problem_sentences)
"""
import json, os, re, time
from pathlib import Path
from statistics import mean
import requests

PROJECT = Path(r'D:\prompt-ops\use-cases\parent-child-coach')
GB_DIR = Path(r'D:\星灵-soul-手搓\亲子沟通洞见\测试智能体\data\golden_bank')
RESULTS_DIR = PROJECT / 'results' / 'auto_research_judge_v2'
BLIND_PATH = RESULTS_DIR / 'h2h_r5_v17_expert_blind.json'

CLAUDE_URL = 'https://s.lconai.com/v1/messages'
CLAUDE_KEY = 'CLAUDE_API_KEY_PLACEHOLDER'
CLAUDE_MODEL = 'claude-opus-4-8'

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
    for f in sorted(GB_DIR.glob('GB_*.json')):
        d = json.load(open(f, encoding='utf-8'))
        for w in d.get('windows', []):
            wd = w.get('window_dialogue', '') or d.get('full_dialogue', '')
            if not wd.strip():
                continue
            gb_clean = re.sub(r'\d+[：:]\s*', '', wd)
            gb_clean = re.sub(r'\s+', '', gb_clean)[:100]
            for r in blind['results']:
                r_clean = re.sub(r'\s+', '', r['dialogue'])[:100]
                overlap = sum(1 for a, b in zip(gb_clean, r_clean) if a == b)
                if overlap < 40:
                    continue
                gp = (w.get('golden_popup') or '').strip()
                fb = (w.get('overall_feedback') or '').strip()
                # Only keep cases with at least golden_popup or hit_list
                if not gp and not w.get('hit_list'):
                    continue
                matches.append({
                    'file': f.name,
                    'window': w.get('window_label', ''),
                    'expected_tone': w.get('expected_tone', ''),
                    'should_popup': w.get('should_popup', True),
                    'golden_popup': gp,
                    'hit_list': w.get('hit_list', []),
                    'golden_sentences': w.get('golden_sentences', []),
                    'forbidden_list': w.get('forbidden_list', []),
                    'problem_sentences': w.get('problem_sentences', []),
                    'overall_feedback': fb,
                    'dialogue': r['dialogue'],
                    'r5_popup': r['r5']['popup'],
                    'v17_popup': r['v17']['popup'],
                })
                break
    return matches


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
        'x-api-key': CLAUDE_KEY,
        'anthropic-version': '2023-06-01',
        'Content-Type': 'application/json',
    }

    for attempt in range(3):
        try:
            resp = requests.post(CLAUDE_URL, json={
                'model': CLAUDE_MODEL,
                'max_tokens': 512,
                'temperature': 0.0,
                'thinking': {'type': 'disabled'},
                'system': '你是弹窗质量评审员。只输出JSON，不要markdown包裹。',
                'messages': [{'role': 'user', 'content': prompt}],
            }, headers=headers, timeout=(30, 90))
            resp.raise_for_status()
            data = resp.json()
            for block in data.get('content', []):
                if isinstance(block, dict) and block.get('type') == 'text':
                    text = block['text']
                    # Parse JSON
                    m = re.search(r'\{[^{}]*"type_match"[^{}]*\}', text, re.DOTALL)
                    if m:
                        return json.loads(m.group(0))
                    m = re.search(r'\{.*\}', text, re.DOTALL)
                    if m:
                        try:
                            return json.loads(m.group(0))
                        except:
                            pass
                    return {'error': 'parse_failed', 'raw': text[:200]}
            return {'error': 'no_text_block', 'raw': str(data)[:200]}
        except requests.exceptions.HTTPError as e:
            return {'error': f'HTTP {e.response.status_code}: {e.response.text[:100]}'}
        except Exception as e:
            if attempt == 2:
                return {'error': str(e)[:100]}
            time.sleep(2 ** attempt * 3)
    return {'error': 'max_retries'}


def main():
    print('=' * 60)
    print('Phase 2 v2: Rubric-based Expert Alignment')
    print('=' * 60)

    matches = load_matches()
    print(f'\nUsable matches (with golden_popup or hit_list): {len(matches)}')

    results = []
    for i, m in enumerate(matches):
        for version, popup in [('r5', m['r5_popup']), ('v17', m['v17_popup'])]:
            print(f'  [{i+1}/{len(matches)}] {m["file"]}:{m["window"]} {version}...', end=' ', flush=True)
            judgement = rubrics_judge(m, popup)
            results.append({
                'file': m['file'], 'window': m['window'],
                'version': version, 'expected_tone': m['expected_tone'],
                **judgement,
            })
            if 'error' in judgement:
                print(f'ERROR: {judgement["error"][:60]}')
            else:
                score = (judgement.get('type_match', 0) * 1.0 +
                         judgement.get('insight_coverage', 1) / 5.0 +
                         judgement.get('core_conflict_alignment', 1) / 5.0 +
                         judgement.get('forbidden_check', 0) * 0.5) / 3.0
                print(f'type={judgement.get("type_match")} '
                      f'insight={judgement.get("insight_coverage")} '
                      f'conflict={judgement.get("core_conflict_alignment")} '
                      f'forbid={judgement.get("forbidden_check")} '
                      f'| composite={score:.3f}')

    # Summary
    r5_res = [r for r in results if r['version'] == 'r5' and 'error' not in r]
    v17_res = [r for r in results if r['version'] == 'v17' and 'error' not in r]

    print('\n' + '=' * 60)
    print('RUBRIC ALIGNMENT RESULTS')
    print('=' * 60)

    for label, res in [('r5', r5_res), ('v1.7', v17_res)]:
        if not res:
            continue
        print(f'\n  {label} (N={len(res)}):')

        def composite(r):
            return ((r.get('type_match', 0) * 1.0 +
                     r.get('insight_coverage', 1) / 5.0 +
                     r.get('core_conflict_alignment', 1) / 5.0 +
                     r.get('forbidden_check', 0) * 0.5) / 3.0)

        scores = [composite(r) for r in res]
        type_matches = sum(1 for r in res if r.get('type_match') == 1)
        forbids_ok = sum(1 for r in res if r.get('forbidden_check') == 1)
        avg_insight = mean(r.get('insight_coverage', 1) for r in res)
        avg_conflict = mean(r.get('core_conflict_alignment', 1) for r in res)

        print(f'    Composite score:     {mean(scores):.3f}')
        print(f'    Type match rate:     {type_matches}/{len(res)} ({type_matches/len(res):.0%})')
        print(f'    Avg insight cover:   {avg_insight:.1f}/5')
        print(f'    Avg conflict align:  {avg_conflict:.1f}/5')
        print(f'    Forbidden clear:     {forbids_ok}/{len(res)} ({forbids_ok/len(res):.0%})')

    # Per-case detail
    print('\n  Per-case:')
    print(f'  {"Case":<24} {"Tone":>8} {"v:type":>6} {"v:ins":>6} {"v:con":>6} {"v:for":>6} {"r:type":>6} {"r:ins":>6} {"r:con":>6} {"r:for":>6}')
    print(f'  {"-"*24} {"-"*8} {"-"*6} {"-"*6} {"-"*6} {"-"*6} {"-"*6} {"-"*6} {"-"*6} {"-"*6}')
    for m in matches:
        r5 = next((r for r in results if r['file'] == m['file'] and r['window'] == m['window'] and r['version'] == 'r5'), {})
        v17 = next((r for r in results if r['file'] == m['file'] and r['window'] == m['window'] and r['version'] == 'v17'), {})
        label = f'{m["file"]}:{m["window"]}'
        print(f'  {label:<24} {m["expected_tone"]:>8} '
              f'{v17.get("type_match","?"):>6} {v17.get("insight_coverage","?"):>6} '
              f'{v17.get("core_conflict_alignment","?"):>6} {v17.get("forbidden_check","?"):>6} '
              f'{r5.get("type_match","?"):>6} {r5.get("insight_coverage","?"):>6} '
              f'{r5.get("core_conflict_alignment","?"):>6} {r5.get("forbidden_check","?"):>6}')

    # Save
    out = {
        'method': 'rubric-based expert alignment (type, insight, conflict, forbidden)',
        'n_matches': len(matches),
        'summary': {
            'r5': {
                'n': len(r5_res),
                'composite_mean': mean([((r.get('type_match',0)*1.0 + r.get('insight_coverage',1)/5.0 + r.get('core_conflict_alignment',1)/5.0 + r.get('forbidden_check',0)*0.5)/3.0) for r in r5_res]) if r5_res else 0,
                'type_match_rate': sum(1 for r in r5_res if r.get('type_match')==1)/len(r5_res) if r5_res else 0,
                'avg_insight_coverage': mean(r.get('insight_coverage',1) for r in r5_res) if r5_res else 0,
                'avg_conflict_alignment': mean(r.get('core_conflict_alignment',1) for r in r5_res) if r5_res else 0,
                'forbidden_clear_rate': sum(1 for r in r5_res if r.get('forbidden_check')==1)/len(r5_res) if r5_res else 0,
            },
            'v17': {
                'n': len(v17_res),
                'composite_mean': mean([((r.get('type_match',0)*1.0 + r.get('insight_coverage',1)/5.0 + r.get('core_conflict_alignment',1)/5.0 + r.get('forbidden_check',0)*0.5)/3.0) for r in v17_res]) if v17_res else 0,
                'type_match_rate': sum(1 for r in v17_res if r.get('type_match')==1)/len(v17_res) if v17_res else 0,
                'avg_insight_coverage': mean(r.get('insight_coverage',1) for r in v17_res) if v17_res else 0,
                'avg_conflict_alignment': mean(r.get('core_conflict_alignment',1) for r in v17_res) if v17_res else 0,
                'forbidden_clear_rate': sum(1 for r in v17_res if r.get('forbidden_check')==1)/len(v17_res) if v17_res else 0,
            },
        },
        'results': results,
    }
    out_path = RESULTS_DIR / 'h2h_r5_v17_rubric_alignment.json'
    json.dump(out, open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'\nSaved to {out_path}')


if __name__ == '__main__':
    main()
