"""v2.3 活跃校标 · 基于 golden_bank 专家标注的端到端评估。

评估维度（与 v2.3 prompt 结构对齐）：
  1. 结构完整性 — 是否使用了 `——` 分隔符？前后段落是否完整？
  2. 类型匹配   — 诊断式 vs 鼓励式，是否与专家期望一致？
  3. 洞察覆盖   — 是否命中了专家指定的洞察点？
  4. 矛盾一致性 — 识别的主要矛盾是否与专家一致？
  5. 建议质量   — `——` 之后的建议句是否具体、可执行？
  6. 避雷检查   — 是否避开了专家标注的红线？

用法:
  python scripts/evaluate_v23.py -n 13              # 用 13 个校标案例评估
  python scripts/evaluate_v23.py --all               # 评估所有可匹配案例
  python scripts/evaluate_v23.py -n 13 --compare v1.7 # 与 v1.7 对比
"""
import json, os, re, sys, time, argparse
from pathlib import Path
from statistics import mean
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

PROJECT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT / 'results' / 'auto_research_judge_v2'
GB_DIR = Path(r'D:\星灵-soul-手搓\亲子沟通洞见\测试智能体\data\golden_bank')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── API ──
DS_URL = "https://api.deepseek.com/v1/chat/completions"
DS_KEY = None
env_path = PROJECT / '.env'
if env_path.exists():
    for line in open(env_path, encoding='utf-8'):
        if line.startswith("DEEPSEEK_API_KEY="):
            DS_KEY = line.split("=", 1)[1].strip(); break
if not DS_KEY:
    DS_KEY = os.getenv("DEEPSEEK_API_KEY", "")
if not DS_KEY:
    raise RuntimeError("DEEPSEEK_API_KEY required")

CL_URL = "https://s.lconai.com/v1/messages"
CL_KEY = "CLAUDE_API_KEY_PLACEHOLDER"
CL_MODEL = "claude-opus-4-8"

# ── Judge Prompt · v2.3 专用 ──
JUDGE_V23 = """你是亲子沟通弹窗评审员。你的任务是用人类专家的标准，从六个维度给 v2.3 弹窗打分。

## 评分维度

### 1. 结构完整性 (structure_score) — 0 或 1
v2.3 的弹窗格式要求：洞察段落 + `——`（单独一行）+ 建议段落。
- 1 = `——` 分隔符存在，前后段落完整、有意义
- 0 = 缺少 `——`，或前后某一段缺失/无意义

### 2. 类型匹配 (type_match) — 0 或 1
专家标注了该场景期望的弹窗类型：{expected_tone}
- 诊断式: 弹窗帮家长看见"没注意到的东西"，指出盲区或惯性思维
- 鼓励式: 弹窗肯定家长已经做对的事，让他看见自己的好
- 1 = AI 弹窗的类型与专家一致
- 0 = 不一致

### 3. 洞察覆盖 (insight_coverage) — 1-5
专家指定了必须命中的洞察点：
{hit_list}
AI 弹窗的洞察段落（`——` 之前的文字）覆盖了多少？无需逐字匹配，关键是核心意思到位。
- 5 = 全部覆盖  4 = 大部分  3 = 一半  2 = 少数  1 = 完全没命中

### 4. 矛盾一致性 (conflict_alignment) — 1-5
专家手写了参考弹窗，其中指出了场景的"主要矛盾"：
{golden_popup}
AI 弹窗识别的核心矛盾，与专家说的是不是同一件事？
- 5 = 高度一致  3 = 方向对但不够精准  1 = 完全不一致

### 5. 建议质量 (suggestion_quality) — 1-5
v2.3 要求在 `——` 之后给出具体建议。评估建议段落的：
- 具体性：不是空泛的"你可以试试"，而是给出了可操作的做法或话术
- 贴切性：建议与前面的洞察一致，不是突然跳到另一个话题
- 语气：像懂他的人轻声提醒，不是居高临下的指导
- 5 = 极好  3 = 中规中矩  1 = 空洞或不当

### 6. 避雷 (forbidden_check) — 0 或 1
专家标注了禁止出现的内容：
{forbidden_list}
AI 弹窗是否触碰了任何红线？
- 1 = 完全避开  0 = 触碰了红线

## 产出格式
```json
{{
  "structure_score": 1,
  "type_match": 1,
  "insight_coverage": 4,
  "conflict_alignment": 4,
  "suggestion_quality": 4,
  "forbidden_check": 1,
  "brief_reason": "一句话总结"
}}
```

对话：
{dialogue}

AI 弹窗：
{popup}
"""


def load_cases():
    """Load golden_bank cases with dialogue matched from expert_dataset or blind results."""
    blind_path = RESULTS_DIR / 'h2h_r5_v17_expert_blind.json'
    blind = None
    if blind_path.exists():
        blind = json.load(open(blind_path, encoding='utf-8'))

    cases = []
    seen = set()
    for f in sorted(GB_DIR.glob('GB_*.json')):
        gb = json.load(open(f, encoding='utf-8'))
        for w in gb.get('windows', []):
            wd = w.get('window_dialogue', '') or gb.get('full_dialogue', '')
            if not wd.strip():
                continue
            gp = (w.get('golden_popup') or '').strip()
            hl = w.get('hit_list', [])
            # Need at minimum: tone + (golden_popup or hit_list)
            if not w.get('expected_tone') or (not gp and not hl):
                continue

            wd_clean = re.sub(r'\d+[：:]\s*', '', wd).strip()
            dialogue = wd_clean  # fallback: golden_bank's own dialogue

            # Try to find better-formatted dialogue from blind results
            if blind:
                for br in blind['results']:
                    r_clean = re.sub(r'\s+', '', br['dialogue'])[:100]
                    g_clean = re.sub(r'\s+', '', wd_clean)[:100]
                    if sum(1 for a, b in zip(g_clean, r_clean) if a == b) >= 40:
                        dialogue = br['dialogue']
                        break

            key = f.name + ':' + w.get('window_label', '')
            if key in seen:
                continue
            seen.add(key)

            cases.append({
                'id': key,
                'source_file': f.name,
                'window': w.get('window_label', ''),
                'expected_tone': w.get('expected_tone', ''),
                'should_popup': w.get('should_popup', True),
                'dialogue': dialogue,
                'golden_popup': gp,
                'hit_list': hl,
                'golden_sentences': w.get('golden_sentences', []),
                'forbidden_list': w.get('forbidden_list', []),
                'problem_sentences': w.get('problem_sentences', []),
                'overall_feedback': (w.get('overall_feedback') or '').strip(),
            })
    return cases


def deepseek(prompt, dialogue):
    """Generate popup with v2.3 prompt."""
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {DS_KEY}'}
    for attempt in range(3):
        try:
            resp = requests.post(DS_URL, json={
                'model': 'deepseek-v4-pro',
                'messages': [{'role': 'user', 'content': f'{prompt}\n\n对话：\n{dialogue}'}],
                'temperature': 0.7, 'max_tokens': 600,
            }, headers=headers, timeout=(30, 90))
            resp.raise_for_status()
            text = resp.json()['choices'][0]['message']['content'].strip()
            if len(text) >= 30:
                return text
        except:
            pass
        time.sleep(2)
    return None


def claude_judge(judge_prompt):
    """Call Claude to judge a popup."""
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
                'system': '只输出JSON。',
                'messages': [{'role': 'user', 'content': judge_prompt}],
            }, headers=headers, timeout=(30, 90))
            resp.raise_for_status()
            for block in resp.json().get('content', []):
                if isinstance(block, dict) and block.get('type') == 'text':
                    # Parse JSON
                    m = re.search(r'\{[^{}]*"structure_score"[^{}]*\}', block['text'], re.DOTALL)
                    if m:
                        return json.loads(m.group(0))
                    m = re.search(r'\{.*\}', block['text'], re.DOTALL)
                    if m:
                        try:
                            return json.loads(m.group(0))
                        except:
                            pass
        except:
            pass
        time.sleep(2 ** attempt * 3)
    return None


def evaluate_one(case, prompt):
    """Generate popup + judge it."""
    popup = deepseek(prompt, case['dialogue'])
    if not popup:
        return {'case_id': case['id'], 'error': 'generation_failed'}

    hit_list = case['hit_list'] + case['golden_sentences']
    forbidden_list = case['forbidden_list'] + case['problem_sentences']

    jp = JUDGE_V23.format(
        expected_tone=case['expected_tone'],
        hit_list='\n'.join('- ' + str(h) for h in hit_list) if hit_list else '(无)',
        golden_popup=case['golden_popup'] or '(无)',
        forbidden_list='\n'.join('- ' + str(f) for f in forbidden_list) if forbidden_list else '(无)',
        dialogue=case['dialogue'],
        popup=popup,
    )

    judgement = claude_judge(jp)
    if not judgement:
        return {'case_id': case['id'], 'error': 'judge_failed', 'popup': popup}

    return {
        'case_id': case['id'],
        'source_file': case['source_file'],
        'window': case['window'],
        'expected_tone': case['expected_tone'],
        'dialogue': case['dialogue'],
        'popup': popup,
        **judgement,
    }


def compute_composite(r):
    """Composite score: weights tuned for v2.3 priorities."""
    return (
        r.get('structure_score', 0) * 1.5 +
        r.get('type_match', 0) * 2.0 +
        r.get('insight_coverage', 1) / 5.0 * 1.5 +
        r.get('conflict_alignment', 1) / 5.0 * 1.5 +
        r.get('suggestion_quality', 1) / 5.0 * 1.5 +
        r.get('forbidden_check', 0) * 2.0
    )


def print_summary(results, label):
    """Print evaluation summary."""
    valid = [r for r in results if 'error' not in r]
    if not valid:
        print(f'  {label}: 0 valid results')
        return

    n = len(valid)
    structure_ok = sum(1 for r in valid if r.get('structure_score') == 1)
    type_ok = sum(1 for r in valid if r.get('type_match') == 1)
    forbid_ok = sum(1 for r in valid if r.get('forbidden_check') == 1)
    ins = mean(r.get('insight_coverage', 1) for r in valid)
    con = mean(r.get('conflict_alignment', 1) for r in valid)
    sug = mean(r.get('suggestion_quality', 1) for r in valid)
    comp = mean(compute_composite(r) for r in valid)

    print(f'  {label} (N={n}):')
    print(f'    结构完整: {structure_ok}/{n} ({structure_ok/n:.0%})  类型匹配: {type_ok}/{n} ({type_ok/n:.0%})')
    print(f'    洞察覆盖: {ins:.1f}/5  矛盾一致: {con:.1f}/5  建议质量: {sug:.1f}/5')
    print(f'    避雷通过: {forbid_ok}/{n}  综合分: {comp:.2f}')
    return valid


def main():
    parser = argparse.ArgumentParser(description='v2.3 活跃校标评估')
    parser.add_argument('-n', type=int, default=13, help='评估案例数量')
    parser.add_argument('--all', action='store_true', help='评估所有可匹配案例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--compare', type=str, default=None,
                        help='与指定版本的 prompt 对比 (如 v1.7 → system_prompt.txt)')
    args = parser.parse_args()

    # Load
    prompt_v23 = (PROJECT / 'system_prompt_v2.3.txt').read_text(encoding='utf-8')
    cases = load_cases()

    import random
    random.seed(args.seed)
    if not args.all:
        cases = random.sample(cases, min(args.n, len(cases)))

    print('=' * 60)
    print(f'v2.3 活跃校标 | {len(cases)} cases | seed={args.seed}')
    print(f'Prompt: system_prompt_v2.3.txt ({len(prompt_v23)} chars)')
    print('=' * 60)

    # Evaluate v2.3
    t0 = time.time()
    print('\n[v2.3] Generating + Judging...')
    results_v23 = []
    for i, c in enumerate(cases):
        label = c['id']
        print(f'  [{i+1}/{len(cases)}] {label} ...', end=' ', flush=True)
        r = evaluate_one(c, prompt_v23)
        results_v23.append(r)
        if 'error' in r:
            print(f'ERROR: {r["error"]}')
        else:
            comp = compute_composite(r)
            print(f'struct={r.get("structure_score")} type={r.get("type_match")} '
                  f'ins={r.get("insight_coverage")} con={r.get("conflict_alignment")} '
                  f'sug={r.get("suggestion_quality")} forbid={r.get("forbidden_check")} '
                  f'| {comp:.2f}')
    elapsed = time.time() - t0
    print(f'\n  Done in {elapsed:.0f}s ({elapsed/len(cases):.0f}s/case)')

    # Summary
    print('\n' + '=' * 60)
    print('RESULTS')
    print('=' * 60)
    valid_v23 = print_summary(results_v23, 'v2.3')

    # Compare with another version if requested
    if args.compare:
        compare_path = PROJECT / args.compare if args.compare.endswith('.txt') else PROJECT / f'system_prompt_{args.compare}.txt'
        if not compare_path.exists():
            compare_path = PROJECT / 'system_prompt.txt'  # fallback to v1.7
        if compare_path.exists():
            prompt_other = compare_path.read_text(encoding='utf-8')
            print(f'\n[{args.compare}] Generating + Judging...')
            results_other = []
            for i, c in enumerate(cases):
                label = c['id']
                print(f'  [{i+1}/{len(cases)}] {label} ...', end=' ', flush=True)
                r = evaluate_one(c, prompt_other)
                results_other.append(r)
                if 'error' in r:
                    print(f'ERROR: {r["error"]}')
                else:
                    comp = compute_composite(r)
                    print(f'struct={r.get("structure_score")} type={r.get("type_match")} '
                          f'ins={r.get("insight_coverage")} con={r.get("conflict_alignment")} '
                          f'sug={r.get("suggestion_quality")} forbid={r.get("forbidden_check")} '
                          f'| {comp:.2f}')
            print()
            valid_other = print_summary(results_other, args.compare)

            # H2H
            if valid_v23 and valid_other:
                v23_by_id = {r['case_id']: r for r in valid_v23}
                other_by_id = {r['case_id']: r for r in valid_other}
                v23_w = other_w = tie = 0
                for cid in v23_by_id:
                    if cid not in other_by_id:
                        continue
                    sv = compute_composite(v23_by_id[cid])
                    so = compute_composite(other_by_id[cid])
                    if sv > so: v23_w += 1
                    elif so > sv: other_w += 1
                    else: tie += 1
                print(f'\n  H2H: v2.3={v23_w}  {args.compare}={other_w}  tie={tie}')

                results_v23 += results_other

    # Save
    out = {
        'config': {
            'prompt': 'system_prompt_v2.3.txt',
            'prompt_chars': len(prompt_v23),
            'n_cases': len(cases),
            'seed': args.seed,
            'judge_dimensions': ['structure_score', 'type_match', 'insight_coverage',
                                 'conflict_alignment', 'suggestion_quality', 'forbidden_check'],
        },
        'results': results_v23,
    }
    out_path = RESULTS_DIR / 'evaluate_v23_results.json'
    json.dump(out, open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'\nSaved to {out_path}')


if __name__ == '__main__':
    main()
