"""
Calibrate Judge v2.0 against golden_bank expert scores.

Key insight from Case_5: the judge needs a "type gate" — first determine
whether the situation calls for 诊断式 or 鼓励式, then score quality.
The current judge only assesses writing quality, missing the type mismatch.

Approach:
  1. Load 77 golden_bank scored windows + match to blind H2H results
  2. Build few-shot calibration examples from golden_bank
  3. Add a "tone-type pre-check" to the judge prompt
  4. Run calibrated judge on the 30-expert test set
  5. Compute correlation improvement vs original judge
"""
import json, os, re, time, sys
from pathlib import Path
from datetime import datetime
from statistics import mean, stdev

# Paths
PROJECT = Path(r'D:\prompt-ops\use-cases\parent-child-coach')
GB_DIR = Path(r'D:\星灵-soul-手搓\亲子沟通洞见\测试智能体\data\golden_bank')
RESULTS_DIR = PROJECT / 'results' / 'auto_research_judge_v2'
BLIND_PATH = RESULTS_DIR / 'h2h_r5_v17_expert_blind.json'
R5_PROMPT_PATH = RESULTS_DIR / 'final_best_prompt.txt'
V17_PROMPT_PATH = PROJECT / 'system_prompt.txt'

# API config
DEEPSEEK_URL = os.environ.get('DEEPSEEK_URL', 'http://9.134.115.203:8080/v1/chat/completions')
DEEPSEEK_KEY = os.environ.get('DEEPSEEK_KEY', 'sk-default')
CLAUDE_URL = os.environ.get('CLAUDE_URL', 'http://9.134.115.203:8081/v1/messages')
CLAUDE_KEY = os.environ.get('CLAUDE_KEY', 'sk-default')

import requests


# ── Step 1: Load golden_bank scored windows ──────────────────────────
def load_golden_bank():
    """Load all scored windows from golden_bank, with dialogue text."""
    windows = []
    for f in sorted(GB_DIR.glob('GB_*.json')):
        d = json.load(open(f, encoding='utf-8'))
        full_dialogue = d.get('full_dialogue', '')
        case_title = d.get('case_title', '')
        annotator = d.get('annotator', '?')
        for w in d.get('windows', []):
            score = w.get('overall_score')
            if score is None:
                continue
            w_dialogue = w.get('window_dialogue', '') or full_dialogue
            # Normalize: strip numeric speaker labels
            dialogue_clean = re.sub(r'\d+[：:]\s*', '', w_dialogue).strip()
            windows.append({
                'source_file': f.name,
                'case_title': case_title,
                'annotator': annotator,
                'window_label': w.get('window_label', ''),
                'dialogue': dialogue_clean,
                'expert_score': score,  # 1-10
                'expected_tone': w.get('expected_tone', ''),
                'should_popup': w.get('should_popup', True),
                'golden_popup': w.get('golden_popup', ''),
                'golden_sentences': w.get('golden_sentences', []),
                'problem_sentences': w.get('problem_sentences', []),
                'hit_list': w.get('hit_list', []),
                'forbidden_list': w.get('forbidden_list', []),
                'overall_feedback': w.get('overall_feedback', ''),
                'blind_spot': w.get('blind_spot', ''),
            })
    return windows


# ── Step 2: Match golden_bank to blind H2H results ───────────────────
def match_to_blind(gb_windows, blind_results):
    """Match golden_bank windows to blind H2H results by dialogue prefix."""
    matches = []
    for gb in gb_windows:
        gb_clean = re.sub(r'\s+', '', gb['dialogue'])[:100]
        best, best_overlap = None, 0
        for r in blind_results:
            r_clean = re.sub(r'\s+', '', r['dialogue'])[:100]
            overlap = 0
            for a, b in zip(gb_clean, r_clean):
                if a == b:
                    overlap += 1
                else:
                    break
            if overlap > best_overlap:
                best_overlap = overlap
                best = r
        if best and best_overlap >= 40:
            matches.append({**gb, 'blind_result': best, 'overlap': best_overlap})
    return matches


# ── Step 3: API helpers ──────────────────────────────────────────────
def deepseek(system_prompt, dialogue, max_retries=3):
    """Generate popup using DeepSeek."""
    headers = {'Content-Type': 'application/json'}
    if DEEPSEEK_KEY != 'sk-default':
        headers['Authorization'] = f'Bearer {DEEPSEEK_KEY}'

    full_prompt = f"{system_prompt}\n\n对话：\n{dialogue}"

    for attempt in range(max_retries):
        try:
            resp = requests.post(DEEPSEEK_URL, json={
                'model': 'deepseek-v4-pro',
                'messages': [{'role': 'user', 'content': full_prompt}],
                'temperature': 0.7,
                'max_tokens': 512,
            }, headers=headers, timeout=60)
            data = resp.json()
            content = data['choices'][0]['message']['content']
            return content.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                return f'[ERROR: {e}]'
            time.sleep(2)
    return '[ERROR]'


def claude_judge(judge_prompt, dialogue, popup, max_retries=3):
    """Judge a popup using Claude."""
    headers = {'Content-Type': 'application/json'}
    if CLAUDE_KEY != 'sk-default':
        headers['x-api-key'] = CLAUDE_KEY

    full_prompt = judge_prompt.replace('{dialogue}', dialogue).replace('{popup}', popup)

    for attempt in range(max_retries):
        try:
            resp = requests.post(CLAUDE_URL, json={
                'model': 'claude-opus-4-8',
                'max_tokens': 1024,
                'messages': [{'role': 'user', 'content': full_prompt}],
                'thinking': {'type': 'disabled'},
            }, headers=headers, timeout=90)
            data = resp.json()
            # Handle Anthropic Messages format
            for block in data.get('content', []):
                if block.get('type') == 'text':
                    return block['text']
            return f'[ERROR: No text block. Content: {str(data)[:200]}]'
        except Exception as e:
            if attempt == max_retries - 1:
                return f'[ERROR: {e}]'
            time.sleep(2)
    return '[ERROR]'


def parse_json(text):
    """Extract JSON from judge response."""
    # Try code block
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except:
            pass
    # Try raw JSON
    m = re.search(r'\{[^{}]*"veto"[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except:
            pass
    # Try multi-line brace match
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except:
            pass
    return {'veto': f'parse_error: {text[:100]}', 'score': 0}


def compute_score(dims):
    """Compute weighted score from dimensions."""
    weights = {'看见感': 0.25, '对话忠实度': 0.20, '命中核心': 0.20,
               '人话感': 0.20, '温度': 0.15}
    score = 0
    for dim, w in weights.items():
        val = dims.get(dim, 0)
        if isinstance(val, (int, float)):
            score += (val / 5.0) * w
    return score


# ── Step 4: Build calibrated judge prompt ────────────────────────────
def build_calibrated_prompt(few_shot_examples):
    """Build a judge prompt with type-gate and few-shot calibration."""

    few_shot_text = ''
    for i, ex in enumerate(few_shot_examples):
        few_shot_text += f"""
### 校准范例 {i+1}
**对话**：{ex['dialogue'][:300]}
**弹窗**：{ex['popup'][:300]}
**专家评语**：{ex.get('expert_feedback', '')}
**专家期望语气**：{ex.get('expected_tone', '')}
**专家打分**：{ex['expert_score']}/10
"""

    prompt = f"""你是「弹窗质量评审员」。你的任务是给亲子沟通弹窗打分。

# 第一步：类型匹配检查（否决项）
在评分之前，先判断这个场景需要什么类型的弹窗：
- **鼓励式**：家长已经做得很好，弹窗应该肯定他已经做到的，让他看见自己的好。不需要诊断、不需要建议。
- **诊断式**：家长陷入了某种思维惯性或盲区，弹窗应该帮他看见没注意到的东西。

**如果场景需要鼓励式但弹窗用了诊断式 → 弹窗类型不匹配，最高不超过 3/5 在"看见感"和"命中核心"维度。**
**如果弹窗里加了"要不要试试…""你可以…"之类的建议句 → 额外扣分，因为弹窗不是建议书。**

# 第二步：五维评分（每个维度 1-5 分）

| 维度 | 权重 | 1分 | 5分 |
|------|------|-----|-----|
| **看见感** | 0.25 | 没看出来家长在做什么 | 精准点出家长自己都没意识到的关键动作 |
| **对话忠实度** | 0.20 | 编造了对话里不存在的事 | 每句话都有原文依据 |
| **命中核心** | 0.20 | 抓了皮毛，没触及真正问题 | 一击命中场景最关键的那个点 |
| **人话感** | 0.20 | 术语满天飞，像教科书 | 就像一个人在旁边轻声说话 |
| **温度** | 0.15 | 冷冰冰或居高临下 | 温暖、尊重、和当事人站在一起 |

# 第三步：否决项
- **事实性错误**：弹窗说的事在对话里根本没发生 → veto，总分 0
- **语气严重误判**：家长在发火，弹窗却夸他"很温柔" → veto，总分 0

# 产出格式
```json
{{
  "tone_check": {{
    "scene_needs": "鼓励式/诊断式",
    "popup_type": "鼓励式/诊断式/混合",
    "type_match": true/false,
    "type_penalty_note": "如果不匹配，简述原因"
  }},
  "veto": null,
  "看见感": 3,
  "对话忠实度": 3,
  "命中核心": 3,
  "人话感": 3,
  "温度": 3,
  "brief_reason": "一句话理由"
}}
```

{few_shot_text}

现在请按以上标准评审以下弹窗：

**对话**：
{{dialogue}}

**弹窗**：
{{popup}}
"""
    return prompt


# ── Step 5: Main calibration flow ────────────────────────────────────
def main():
    print('=' * 60)
    print('Judge v2.0 Calibration')
    print(f'Started: {datetime.now().isoformat()}')
    print('=' * 60)

    # Load data
    print('\n[1/5] Loading golden_bank...')
    gb_windows = load_golden_bank()
    print(f'  Scored windows: {len(gb_windows)}')

    # Score distribution
    scores = [w['expert_score'] for w in gb_windows]
    print(f'  Score range: {min(scores)}-{max(scores)}, mean={mean(scores):.1f}, σ={stdev(scores):.1f}')

    # Load blind results
    print('\n[2/5] Loading blind H2H results...')
    blind = json.load(open(BLIND_PATH, encoding='utf-8'))
    print(f'  Blind results: {len(blind["results"])}')

    # Match
    print('\n[3/5] Matching golden_bank ↔ blind results...')
    matched = match_to_blind(gb_windows, blind['results'])
    print(f'  Matched: {len(matched)} windows')

    # Build few-shot examples from matched data
    # Take 5 diverse examples across the score range
    matched_sorted = sorted(matched, key=lambda x: x['expert_score'])
    few_shot_indices = []
    # Sample across score distribution
    for target_score in [1, 3, 5, 7, 9]:
        candidates = [m for m in matched_sorted if abs(m['expert_score'] - target_score) <= 1]
        if candidates:
            few_shot_indices.append(candidates[len(candidates)//2])  # middle of range

    few_shot_examples = []
    for m in few_shot_indices:
        # Get the better popup (r5 or v17) for this example
        br = m['blind_result']
        # Use r5 popup as it's generally better written
        popup = br['r5']['popup']
        few_shot_examples.append({
            'dialogue': m['dialogue'],
            'popup': popup,
            'expert_score': m['expert_score'],
            'expected_tone': m['expected_tone'],
            'expert_feedback': m.get('overall_feedback', ''),
        })

    print(f'  Built {len(few_shot_examples)} few-shot calibration examples')
    for ex in few_shot_examples:
        print(f'    - score={ex["expert_score"]}, tone={ex["expected_tone"]}')

    # Build calibrated prompt
    print('\n[4/5] Building calibrated judge prompt...')
    calibrated_prompt = build_calibrated_prompt(few_shot_examples)

    # Save the calibrated prompt
    cal_prompt_path = RESULTS_DIR / 'judge_v2_calibrated_prompt.txt'
    cal_prompt_path.write_text(calibrated_prompt, encoding='utf-8')
    print(f'  Saved to {cal_prompt_path}')

    # Run calibration on matched examples
    print(f'\n[5/5] Testing calibrated judge on {len(matched)} matched cases...')
    print(f'  This will take ~{len(matched) * 10}s...')

    results = []
    for i, m in enumerate(matched):
        br = m['blind_result']
        dialogue = br['dialogue']

        for version in ['r5', 'v17']:
            popup = br[version]['popup']

            # Run with calibrated judge
            raw = claude_judge(calibrated_prompt, dialogue, popup)
            parsed = parse_json(raw)
            score = compute_score(parsed)

            # Apply type penalty
            type_match = parsed.get('tone_check', {}).get('type_match', True)
            if not type_match:
                # Penalty: cap 看见感 and 命中核心 at 3
                for dim in ['看见感', '命中核心']:
                    if parsed.get(dim, 0) > 3:
                        parsed[dim] = 3
                score = compute_score(parsed)

            results.append({
                'source_file': m['source_file'],
                'window_label': m['window_label'],
                'version': version,
                'expert_score': m['expert_score'],
                'expert_tone': m['expected_tone'],
                'judge_score': score,
                'type_match': type_match,
                'tone_check': parsed.get('tone_check', {}),
                'dims': {k: v for k, v in parsed.items() if k in ['看见感', '对话忠实度', '命中核心', '人话感', '温度']},
                'veto': parsed.get('veto'),
                'reason': parsed.get('brief_reason', ''),
            })

            print(f'  [{i+1}/{len(matched)}] {m["source_file"]}:{m["window_label"]} {version} '
                  f'expert={m["expert_score"]} judge={score:.4f} type_match={type_match}')

    # Compute calibration metrics
    r5_results = [r for r in results if r['version'] == 'r5']
    v17_results = [r for r in results if r['version'] == 'v17']

    def pearson(xs, ys):
        n = len(xs)
        mx, my = mean(xs), mean(ys)
        num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
        dx = (sum((x-mx)**2 for x in xs)**0.5)
        dy = (sum((y-my)**2 for y in ys)**0.5)
        return num/(dx*dy) if dx*dy > 0 else 0

    print('\n' + '=' * 60)
    print('CALIBRATION RESULTS')
    print('=' * 60)

    for label, res in [('r5', r5_results), ('v1.7', v17_results)]:
        js = [r['judge_score'] for r in res]
        es = [r['expert_score']/10.0 for r in res]
        mae = mean(abs(j-e) for j,e in zip(js, es))
        r_val = pearson(js, es)
        type_match_rate = sum(1 for r in res if r['type_match']) / len(res)
        print(f'\n  {label}:')
        print(f'    MAE: {mae:.4f}')
        print(f'    Pearson r: {r_val:+.4f}')
        print(f'    Type match rate: {type_match_rate:.1%}')
        print(f'    Judge score mean: {mean(js):.4f}, σ: {stdev(js):.4f}')
        print(f'    Expert score mean: {mean(es):.4f}, σ: {stdev(es):.4f}')

    # Save
    out = {
        'config': {
            'few_shot_count': len(few_shot_examples),
            'total_matched': len(matched),
            'calibrated_prompt_path': str(cal_prompt_path),
        },
        'few_shot_examples': few_shot_examples,
        'results': results,
    }
    out_path = RESULTS_DIR / 'judge_v2_calibration_results.json'
    json.dump(out, open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'\nSaved to {out_path}')

    return results


if __name__ == '__main__':
    main()
