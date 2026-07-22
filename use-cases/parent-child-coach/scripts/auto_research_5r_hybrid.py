"""
Auto Research: 5-Round Iteration with Judge v2.0
=================================================
Task Model:  DeepSeek v4 pro (via api.deepseek.com, OpenAI format)
Judge Model: Claude Opus 4.8 (via s.lconai.com, Anthropic Messages format)

Judge v2.0 — 基于52个专家打标用例提炼的5维度模型:
  看见感        0.25  家长读完会不会觉得"你懂我"？
  对话忠实度     0.20  每个判断都能在对话中找到原文依据？
  命中核心       0.20  抓住了这个窗口最该被看见的那个点？
  人话感         0.20  像一个真实的人在说话？（无术语/模板/框架标签）
  温度           0.15  读完觉得"在帮我"还是"在教我"？
  + 一级否决: 事实性错误/语气严重误判 → 0分

Algorithm per round:
  1. Evaluate current prompt on 12 test dialogues
  2. Analyze per-dimension scores, identify weakest/strongest dims
  3. Generate improved variant via DeepSeek
  4. Head-to-head on 6 dialogues: current vs variant
  5. Select winner → becomes next round's current

Starting from v1.7 baseline. Saves to results/auto_research_judge_v2/
"""

import json
import os
import re
import sys
import time
from statistics import mean, stdev
from typing import Dict, List, Tuple

import requests

# === Paths ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results", "auto_research_judge_v2")
os.makedirs(RESULTS_DIR, exist_ok=True)

# === DeepSeek API (Task Model — OpenAI format) ===
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_KEY:
    # Fallback: read from .env file
    env_path = os.path.join(BASE_DIR, ".env")
    if os.path.exists(env_path):
        for line in open(env_path, encoding="utf-8"):
            if line.startswith("DEEPSEEK_API_KEY="):
                DEEPSEEK_KEY = line.split("=", 1)[1].strip()
                break
if not DEEPSEEK_KEY:
    raise RuntimeError("DEEPSEEK_API_KEY env var required. Set it or add to .env file.")
TASK_MODEL = "deepseek-v4-pro"

# === Claude API via 智创聚合 (Judge Model — Anthropic Messages format) ===
CLAUDE_URL = "https://s.lconai.com/v1/messages"
CLAUDE_KEY = "CLAUDE_API_KEY_PLACEHOLDER"
CLAUDE_MODEL = "claude-opus-4-8"

# === v1.7 Baseline (loaded from file) ===
BASELINE_PATH = os.path.join(BASE_DIR, "system_prompt_backup_v17.txt")
with open(BASELINE_PATH, "r", encoding="utf-8") as f:
    PROMPT_V17 = f.read().strip()

# === Judge v2.0 Scoring Prompt (5-dimension + veto + few-shot) ===
SCORING_PROMPT = """你是一名亲子沟通教练弹窗的评估专家。你的任务是给AI生成的弹窗从五个维度打分。

## 评分规则

### 一级否决（先检查，触发任一则总分=0，不进入维度评分）

**事实性错误**：弹窗编造了对话中不存在的内容——认错主语（把妈妈说成爸爸/反之）、引用其他窗口的句子、或描述了对话中根本没发生的事。
**语气严重误判**：家长明明做得很好的场景给了诊断式弹窗；或者严重问题被轻描淡写为鼓励。

---

### 维度评分（仅在通过否决后）

#### 1. 看见感（1-5分）
家长读完会不会觉得"你懂我"？弹窗是否看见并精准说出了家长此刻的情绪、困境、发心或脆弱？

- 5分：精准命中家长自己都说不清的那个情绪/困境/发心。不仅看见表层行为，更穿透到底层的恐惧、无力、或未被自己察觉的善意。
- 3分：看见了表层情绪或动机，但停在家长自己已经意识到的层面，没有挖到更深。
- 1分：完全没看见家长这个人——直接跳到评判、纠正或建议。

#### 2. 对话忠实度（1-5分）
弹窗的每一个判断都能在对话原文中找到具体依据？不编造、不夸大、不替任何一方脑补未言明的心理活动？

- 5分：每个判断都能在对话中找到具体的句子作为依据。不编造、不夸大。
- 3分：方向大致正确，但有2-3处推断过度或轻微偏离。
- 1分：严重脱离对话——编造场景、对不存在的事件进行诊断。

#### 3. 命中核心（1-5分）
弹窗是否抓住了这个窗口最该被看见的那个点？读完让人觉得"对，就是这个"。

- 5分：一击即中。具体到"这个时刻、这个反应里，有一个你自己没看到的东西"。
- 3分：方向对但不够精准。点到了领域但没说出具体方式、时刻、影响。
- 1分：完全抓错重点，或停留在"换个角度""多理解孩子"这种泛泛之谈。

#### 4. 人话感（1-5分）
弹窗像一个真实的人在说话吗？没有任何术语、框架标签、模板句式？如果弹窗里出现了让普通家长停下来想"这个词什么意思"的内容，就是失败。

- 5分：就是朋友在耳边说话。短句、口语、朴素、精准。没有任何术语、框架标签、模板句式。
- 3分：有2-3处不够自然——轻微模板痕迹、个别句子偏书面。
- 1分：术语直接暴露（如"多极""代偿""关系根""认知扭曲""CBT"等）、明显使用填充句式（如"你正戴着X的眼镜""你缺了一个X的框架"）、或文艺到让人看不懂。

#### 5. 温度（1-5分）
弹窗的整体姿态是盟友还是教师？

- 5分：从头到尾是盟友。即使是指出盲区，也让人感到被尊重、被信任。邀请的语气，不是宣告。
- 3分：姿态在盟友和教师之间摇摆——前半段像朋友，后半段开始"你应该……"。
- 1分：有明显说教感或审判感。使用"你是错的""你应该"等宣告句式。

---

## Few-shot 校准样本

以下是一个真实对话的三版弹窗，用于校准你的评分标准：

### 对话
孩子最近不怎么跟妈妈说话，妈妈担心，翻了孩子的日记和抽屉。孩子发现后说："你翻我东西？"妈妈说："我只是想看看你最近怎么了。"

### 弹窗 A ("信任的门不是撞开的，是敲开的")
你很担心——他不跟你说话，你就想从抽屉里找答案。但你从担心滑到了翻东西，用"我只是想看看你最近怎么了"来合理化越界。他想让你知道的，不是他的东西放在哪里，是他的感受被尊重。信任的门不是撞开的，是敲开的。——你可以这么说："妈妈刚才不该翻你东西。我很担心你，但我应该先问你一句——最近有什么事想跟我聊聊吗？"

校准：看见感=5, 对话忠实度=5, 命中核心=5, 人话感=5, 温度=5。无术语、无模板、句句基于对话、盟友姿态。

### 弹窗 B (含"多极"术语和"你正戴着X的眼镜"模板)
你是因为真的担心、又不知道该怎么办才翻日记的。我猜你心里清楚这样做不对，但那个瞬间你的恐惧比尊重跑得更快。你正戴着一副"只有我看到的才算数"的眼镜——你觉得关心就能越过边界，可孩子感受到的是"妈妈的眼神穿透了门板"。你缺了一个"多极"的框架：你眼里的"了解"，是他的"暴露"。

校准：看见感=3, 对话忠实度=3, 命中核心=4, 人话感=1, 温度=3。"多极"术语泄漏、"你正戴着X的眼镜""你缺了一个X的框架"是模板填充句式。

### 弹窗 C (有模板感但比B好)
你是因为孩子什么都不跟你说，心里又急又怕，才翻的日记——你想知道怎么才能走近她。但你正在用"我是你妈"这副眼镜，把越界读成了关心。孩子看到的事实是：我最私密的东西，被我最需要信任的人打开了，连问都没问。

校准：看见感=4, 对话忠实度=4, 命中核心=4, 人话感=2, 温度=4。"我是你妈这副眼镜"仍是模板句式，但比B少了很多术语。

**核心原则：好的弹窗让家长感觉被理解，而不是被分析。术语和模板句式是减分项，不是加分项。**

---

## 待评估内容

对话：
{dialogue}

AI教练的回应：
{response}

请输出JSON（只输出JSON，不要其他文字）：
{{"veto": null或"事实性错误"或"语气严重误判", "being_seen": 1-5, "dialogue_fidelity": 1-5, "core_insight": 1-5, "natural_language": 1-5, "warmth": 1-5, "brief_reason": "一句话简述核心判断"}}"""

# === Dimension weights (v2.0) ===
DIM_WEIGHTS = [
    ("being_seen", 0.25), ("dialogue_fidelity", 0.20),
    ("core_insight", 0.20), ("natural_language", 0.20),
    ("warmth", 0.15),
]

DIM_LABELS = {
    "being_seen": "看见感", "dialogue_fidelity": "对话忠实度",
    "core_insight": "命中核心", "natural_language": "人话感",
    "warmth": "温度",
}


# ============================================================================
# DeepSeek API (OpenAI format) — for task model (generation + variant)
# ============================================================================

def deepseek_call(messages: List[Dict], max_tokens: int, temperature: float,
                  timeout: int = 120) -> str:
    """Call DeepSeek API (OpenAI format) with retries."""
    headers = {"Authorization": f"Bearer {DEEPSEEK_KEY}", "Content-Type": "application/json"}
    payload = {"model": TASK_MODEL, "max_tokens": max_tokens,
               "temperature": temperature, "messages": messages}

    last_error = None
    for attempt in range(4):
        try:
            resp = requests.post(DEEPSEEK_URL, headers=headers, json=payload,
                                timeout=(30, timeout))
            resp.raise_for_status()
            data = resp.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        except requests.exceptions.Timeout:
            wait = 2 ** attempt * 5
            print(f"    ⚠ DeepSeek timeout (attempt {attempt+1}/4), waiting {wait}s...", flush=True)
            time.sleep(wait)
        except requests.exceptions.ConnectionError as e:
            wait = 2 ** attempt * 10
            print(f"    ⚠ DeepSeek connection error (attempt {attempt+1}/4): {e}, waiting {wait}s...", flush=True)
            time.sleep(wait)
        except Exception as e:
            last_error = e
            wait = 2 ** attempt * 3
            print(f"    ⚠ DeepSeek error (attempt {attempt+1}/4): {type(e).__name__}: {e}, waiting {wait}s...", flush=True)
            time.sleep(wait)

    raise RuntimeError(f"DeepSeek API call failed after 4 attempts. Last error: {last_error}")


def generate_popup(system_prompt: str, dialogue: str) -> str:
    """Generate coaching popup using DeepSeek task model."""
    return deepseek_call([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"对话：\n{dialogue}\n\n请生成弹窗："},
    ], max_tokens=2048, temperature=0.7, timeout=120)


# ============================================================================
# Claude API (Anthropic Messages format) — for judge model (scoring)
# ============================================================================

def claude_judge_call(prompt_text: str) -> str:
    """Call Claude via 智创聚合 (Anthropic Messages format) with retries."""
    headers = {
        "x-api-key": CLAUDE_KEY,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": 1024,
        "temperature": 0.0,
        "system": "你是严格的亲子沟通弹窗评估专家。只输出JSON，不输出其他内容。核心原则：好的弹窗让家长感觉被理解，而不是被分析。术语和模板句式是减分项，不是加分项。评分标准已通过few-shot样本校准——请内化那些样本中的评分逻辑。",
        "messages": [{"role": "user", "content": prompt_text}],
    }

    last_error = None
    for attempt in range(4):
        try:
            resp = requests.post(CLAUDE_URL, headers=headers, json=payload,
                                timeout=(30, 90))
            resp.raise_for_status()
            data = resp.json()
            content = data.get("content", [])
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    return block["text"]
            if content and isinstance(content[0], dict) and "text" in content[0]:
                return content[0]["text"]
            raise KeyError(f"No text block in Claude response. Content: {json.dumps(content, ensure_ascii=False)[:200]}")
        except requests.exceptions.Timeout:
            wait = 2 ** attempt * 5
            print(f"    ⚠ Claude timeout (attempt {attempt+1}/4), waiting {wait}s...", flush=True)
            time.sleep(wait)
        except requests.exceptions.ConnectionError as e:
            wait = 2 ** attempt * 10
            print(f"    ⚠ Claude connection error (attempt {attempt+1}/4): {e}, waiting {wait}s...", flush=True)
            time.sleep(wait)
        except Exception as e:
            last_error = e
            wait = 2 ** attempt * 3
            print(f"    ⚠ Claude error (attempt {attempt+1}/4): {type(e).__name__}: {e}, waiting {wait}s...", flush=True)
            time.sleep(wait)

    raise RuntimeError(f"Claude judge call failed after 4 attempts. Last error: {last_error}")


# ============================================================================
# Judge logic (v2.0 — 5 dims + veto)
# ============================================================================

def _parse_json(raw: str) -> Dict:
    """Extract JSON scores from raw text."""
    cleaned = raw.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Cannot parse JSON from: {raw[:200]}")


def judge_popup(dialogue: str, popup: str) -> Tuple[float, Dict]:
    """Score a popup using Claude judge v2.0 (5-dim + veto).

    Returns (weighted_score, per_dim_scores).
    Veto triggered → returns (0.0, scores) with veto field set.
    """
    prompt = SCORING_PROMPT.format(dialogue=dialogue, response=popup)

    for attempt in range(4):
        try:
            raw = claude_judge_call(prompt)
            scores = _parse_json(raw)

            # Check veto gate
            veto = scores.get("veto")
            if veto and str(veto).strip() not in ("null", "none", ""):
                return 0.0, scores

            # Compute weighted average
            weighted_sum = 0.0
            total_weight = 0.0
            for dim_key, weight in DIM_WEIGHTS:
                val = scores.get(dim_key)
                if val is None:
                    continue
                if isinstance(val, (int, float)) and 1 <= val <= 5:
                    normalized = (val - 1) / 4  # 1→0.0, 5→1.0
                    weighted_sum += normalized * weight
                    total_weight += weight

            if total_weight == 0:
                return 0.0, scores
            return weighted_sum / total_weight, scores

        except Exception as e:
            if attempt < 3:
                time.sleep(3)
            else:
                print(f"    ⚠ Claude judge failed after 4 attempts: {e}", flush=True)
                return 0.0, {}


# ============================================================================
# Analysis & variant generation
# ============================================================================

def analyze_results(results: List[Dict]) -> Dict:
    """Analyze per-dimension scores across all samples."""
    dim_scores = {dim: [] for dim, _ in DIM_WEIGHTS}
    overall_scores = []
    veto_count = 0
    for r in results:
        overall_scores.append(r["overall_score"])
        per_dim = r.get("per_dim", {})
        if per_dim.get("veto") and str(per_dim["veto"]).strip() not in ("null", "none", ""):
            veto_count += 1
        for dim, _ in DIM_WEIGHTS:
            val = per_dim.get(dim)
            if isinstance(val, (int, float)):
                dim_scores[dim].append((val - 1) / 4)

    dim_means = {}
    for dim, scores_ in dim_scores.items():
        dim_means[dim] = {
            "mean": mean(scores_) if scores_ else 0,
            "std": stdev(scores_) if len(scores_) > 1 else 0,
        }

    ranked = sorted(dim_means.items(), key=lambda x: x[1]["mean"])
    return {
        "overall_mean": mean(overall_scores) if overall_scores else 0,
        "overall_std": stdev(overall_scores) if len(overall_scores) > 1 else 0,
        "dim_means": dim_means,
        "veto_count": veto_count,
        "weakest_dims": [(dim, stats) for dim, stats in ranked[:2] if stats["mean"] < 0.95],
        "strongest_dims": [(dim, stats) for dim, stats in ranked[-2:]],
    }


def generate_variant(current_prompt: str, analysis: Dict, round_num: int) -> str:
    """Use DeepSeek to generate an improved prompt variant targeting weaknesses."""
    weakest_info = "\n".join(
        f"  - {DIM_LABELS[dim]} (平均 {stats['mean']:.3f}/1.0): 当前提示词在此维度表现最弱"
        for dim, stats in analysis["weakest_dims"]
    ) if analysis["weakest_dims"] else "  (所有维度均已接近天花板)"

    strongest_info = "\n".join(
        f"  - {DIM_LABELS[dim]} (平均 {stats['mean']:.3f}/1.0): 表现较好，保持"
        for dim, stats in analysis["strongest_dims"]
    )

    veto_info = ""
    if analysis.get("veto_count", 0) > 0:
        veto_info = f"\n⚠️ 一级否决触发 {analysis['veto_count']} 次（事实性错误或语气误判），必须消除。"

    strategies = [
        "精简冗余，聚焦核心方法论。删掉不必要的分类、表格、框架术语",
        "加强具体示例和反例，用before/after对比。但要确保示例本身不说术语",
        "调整语气和角色设定——更像朋友而非教练/分析师。减少'你必须XX'类的硬性指令",
        "关键修复：所有给模型用的内部诊断框架必须标注'仅供内部参考，禁止在输出中暴露以下词汇：...'",
        "重新平衡：在保持洞察准确性的前提下，最大限度地简化指令，让模型有空间用自然语言表达",
    ]
    strategy = strategies[min(round_num - 1, len(strategies) - 1)]

    variant_prompt_text = f"""你是一位提示词工程专家。请基于以下分析改进提示词。

## 当前提示词
```
{current_prompt[:4000]}
```

## 评分分析（Claude Opus 4.8 独立评分，Judge v2.0 五维度模型）
整体均分：{analysis['overall_mean']:.3f}/1.0
{veto_info}

薄弱维度（需要改进）：
{weakest_info}

强项维度（保持）：
{strongest_info}

## 改进策略
{strategy}

## 关键约束（必须遵守）
1. 提示词中使用的任何内部框架术语（如"多极""关系根""CBT歪曲""七层结构"等），必须明确标注"仅供内部诊断使用，禁止在弹窗输出中直接出现"
2. 禁止使用模板句式指令，如"必须用'你正戴着X的眼镜'句式"——这会污染模型输出
3. 用"让家长感觉你懂他"替代"命名思维歪曲类型"作为诊断质量的标准
4. 提示词整体长度控制在 500-2500 字
5. 保持中文输出
6. 直接输出改进后的提示词全文，不要解释改动

改进后的提示词："""

    return deepseek_call(
        [{"role": "user", "content": variant_prompt_text}],
        max_tokens=4096, temperature=0.8, timeout=180,
    )


# ============================================================================
# Head-to-head comparison
# ============================================================================

def head_to_head(prompt_a, prompt_b, test_set, label_a, label_b) -> Dict:
    """Compare two prompts on the same test set using judge v2.0."""
    results = {"a": [], "b": [], "winner": None, "margin": 0}
    for i, item in enumerate(test_set):
        dialogue = item["question"]
        d_short = dialogue[:60].replace("\n", " ")
        print(f"    [{i+1}/{len(test_set)}] {d_short}...", flush=True)

        popup_a = generate_popup(prompt_a, dialogue)
        time.sleep(0.3)
        popup_b = generate_popup(prompt_b, dialogue)
        time.sleep(0.3)

        score_a, dims_a = judge_popup(dialogue, popup_a)
        time.sleep(0.3)
        score_b, dims_b = judge_popup(dialogue, popup_b)
        time.sleep(0.3)

        veto_a = dims_a.get("veto", "") if dims_a else ""
        veto_b = dims_b.get("veto", "") if dims_b else ""
        reason_a = dims_a.get("brief_reason", "") if dims_a else ""
        reason_b = dims_b.get("brief_reason", "") if dims_b else ""

        results["a"].append({"score": score_a, "dims": dims_a, "popup": popup_a[:300]})
        results["b"].append({"score": score_b, "dims": dims_b, "popup": popup_b[:300]})
        veto_str = f" [VETO:{veto_a}]" if veto_a and str(veto_a).strip() not in ("null",) else ""
        veto_str_b = f" [VETO:{veto_b}]" if veto_b and str(veto_b).strip() not in ("null",) else ""
        print(f"      {label_a}: {score_a:.4f}{veto_str} | {label_b}: {score_b:.4f}{veto_str_b} | Δ: {score_b - score_a:+.4f}", flush=True)

    scores_a = [r["score"] for r in results["a"]]
    scores_b = [r["score"] for r in results["b"]]
    results["mean_a"] = mean(scores_a) if scores_a else 0
    results["mean_b"] = mean(scores_b) if scores_b else 0
    results["margin"] = results["mean_b"] - results["mean_a"]
    results["winner"] = label_b if results["margin"] > 0 else label_a
    return results


# ============================================================================
# Test set loading
# ============================================================================

def load_test_set(n: int = 12) -> List[Dict]:
    """Load n evenly-spaced samples from merged train dataset."""
    merged_path = os.path.join(BASE_DIR, "data", "dataset_merged_train.json")
    with open(merged_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    step = max(1, len(data) // n)
    test_set = [data[i] for i in range(0, min(len(data), step * n), step)][:n]
    return [item for item in test_set if item.get("question")][:n]


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80, flush=True)
    print("AUTO RESEARCH: 5-Round Hybrid — Judge v2.0 (5-dim + veto)", flush=True)
    print(f"Task Model:  {TASK_MODEL} (api.deepseek.com)", flush=True)
    print(f"Judge Model: {CLAUDE_MODEL} (s.lconai.com) — 5-dim + veto + few-shot", flush=True)
    print(f"Baseline:    v1.7 ({len(PROMPT_V17)} chars)", flush=True)
    print(f"Results:     {RESULTS_DIR}", flush=True)
    print("=" * 80, flush=True)

    # Load test set
    test_set = load_test_set(12)
    print(f"\nTest set: {len(test_set)} dialogues", flush=True)
    for i, item in enumerate(test_set):
        d_short = item["question"][:80].replace("\n", " ")
        print(f"  [{i+1}] {d_short}...", flush=True)

    current_prompt = PROMPT_V17
    current_label = "v1.7_baseline"
    all_rounds = []

    for round_num in range(1, 6):
        print(f"\n{'='*80}", flush=True)
        print(f"ROUND {round_num}/5", flush=True)
        print(f"Current best: {current_label} ({len(current_prompt)} chars)", flush=True)
        print(f"{'='*80}", flush=True)

        round_dir = os.path.join(RESULTS_DIR, f"r{round_num}")
        os.makedirs(round_dir, exist_ok=True)

        # Step 1: Evaluate current prompt
        print(f"\n[Step 1] Evaluating on {len(test_set)} dialogues...", flush=True)
        current_results = []
        for i, item in enumerate(test_set):
            dialogue = item["question"]
            d_short = dialogue[:60].replace("\n", " ")
            print(f"  [{i+1}/{len(test_set)}] {d_short}...", flush=True)

            try:
                popup = generate_popup(current_prompt, dialogue)
                score, dims = judge_popup(dialogue, popup)
            except Exception as e:
                print(f"    ❌ FAILED: {e}", flush=True)
                popup = ""
                score = 0.0
                dims = {}

            veto = dims.get("veto", "") if dims else ""
            reason = dims.get("brief_reason", "") if dims else ""
            current_results.append({
                "dialogue": dialogue[:200],
                "popup": popup,
                "overall_score": score,
                "per_dim": dims,
            })
            veto_str = f" [VETO:{veto}]" if veto and str(veto).strip() not in ("null", "none", "") else ""
            dim_summary = {k: dims.get(k) for k in DIM_LABELS}
            print(f"      Score: {score:.4f}{veto_str}  dims: {json.dumps(dim_summary, ensure_ascii=False)} | {reason}", flush=True)
            time.sleep(0.5)

        # Step 2: Analyze
        print(f"\n[Step 2] Analyzing weaknesses...", flush=True)
        analysis = analyze_results(current_results)
        print(f"  Overall: {analysis['overall_mean']:.4f} ± {analysis['overall_std']:.4f}", flush=True)
        if analysis["veto_count"] > 0:
            print(f"  ⚠️  Veto count: {analysis['veto_count']}", flush=True)
        if analysis["weakest_dims"]:
            print(f"  Weakest:", flush=True)
            for dim, stats in analysis["weakest_dims"]:
                print(f"    - {DIM_LABELS[dim]}: {stats['mean']:.3f}", flush=True)
        else:
            print(f"  All dims near ceiling!", flush=True)
        print(f"  Strongest:", flush=True)
        for dim, stats in analysis["strongest_dims"]:
            print(f"    + {DIM_LABELS[dim]}: {stats['mean']:.3f}", flush=True)

        # Step 3: Generate variant
        print(f"\n[Step 3] Generating improved variant...", flush=True)
        try:
            variant_prompt = generate_variant(current_prompt, analysis, round_num)
            variant_prompt = variant_prompt.strip()
            if variant_prompt.startswith("```"):
                variant_prompt = re.sub(r"^```\w*\n?", "", variant_prompt)
                variant_prompt = re.sub(r"\n?```$", "", variant_prompt)
        except Exception as e:
            print(f"    ❌ Variant generation failed: {e}", flush=True)
            variant_prompt = current_prompt
        print(f"  Variant length: {len(variant_prompt)} chars", flush=True)
        print(f"  Preview: {variant_prompt[:200]}...", flush=True)

        # Step 4: Head-to-head
        print(f"\n[Step 4] Head-to-head on 6 dialogues...", flush=True)
        h2h_test = test_set[:6]
        try:
            h2h = head_to_head(current_prompt, variant_prompt, h2h_test,
                              current_label, f"r{round_num}_variant")
        except Exception as e:
            print(f"    ❌ H2H failed: {e}", flush=True)
            h2h = {"mean_a": 0, "mean_b": 0, "margin": 0, "winner": current_label,
                   "a": [], "b": []}

        print(f"\n  {h2h['mean_a']:.4f} ({current_label}) vs "
              f"{h2h['mean_b']:.4f} (variant)", flush=True)
        print(f"  Margin: {h2h['margin']:+.4f} → Winner: {h2h['winner']}", flush=True)

        # Step 5: Decide
        if h2h["margin"] > 0.01:
            current_prompt = variant_prompt
            current_label = f"r{round_num}_variant"
            print(f"  ✅ Variant WINS!", flush=True)
        elif h2h["margin"] > -0.01:
            if len(variant_prompt) < len(current_prompt):
                current_prompt = variant_prompt
                current_label = f"r{round_num}_variant"
                print(f"  ⚖️  Tie → keeping more compact variant", flush=True)
            else:
                print(f"  ⚖️  Tie → keeping current", flush=True)
        else:
            print(f"  ❌ Variant lost, keeping current", flush=True)

        # Save round data
        round_data = {
            "round": round_num,
            "current_label": current_label,
            "current_prompt": current_prompt,
            "current_prompt_len": len(current_prompt),
            "analysis": {
                "overall_mean": analysis["overall_mean"],
                "overall_std": analysis["overall_std"],
                "veto_count": analysis.get("veto_count", 0),
                "weakest_dims": [(d, s["mean"]) for d, s in analysis["weakest_dims"]],
                "strongest_dims": [(d, s["mean"]) for d, s in analysis["strongest_dims"]],
            },
            "variant_prompt": variant_prompt,
            "variant_prompt_len": len(variant_prompt),
            "head_to_head": {
                "mean_current": h2h["mean_a"],
                "mean_variant": h2h["mean_b"],
                "margin": h2h["margin"],
                "winner": h2h["winner"],
            },
            "per_sample_current": current_results,
            "per_sample_h2h": {"current": h2h["a"], "variant": h2h["b"]},
        }

        with open(os.path.join(round_dir, "round_data.json"), "w", encoding="utf-8") as f:
            json.dump(round_data, f, ensure_ascii=False, indent=2)
        with open(os.path.join(round_dir, "best_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(current_prompt)

        all_rounds.append(round_data)
        print(f"  ✅ Round {round_num} data saved", flush=True)

    # Final summary
    print(f"\n{'='*80}", flush=True)
    print("ALL 5 ROUNDS COMPLETE", flush=True)
    print(f"Final best: {current_label} ({len(current_prompt)} chars)", flush=True)
    for rd in all_rounds:
        print(f"  R{rd['round']}: {rd['analysis']['overall_mean']:.4f} "
              f"({rd['current_label']}, {rd['current_prompt_len']} chars) "
              f"| h2h margin: {rd['head_to_head']['margin']:+.4f}", flush=True)

    summary = {
        "config": {
            "task_model": TASK_MODEL,
            "task_api": "api.deepseek.com (OpenAI format)",
            "judge_model": CLAUDE_MODEL,
            "judge_api": "s.lconai.com (Anthropic Messages format)",
            "judge_version": "v2.0 (5-dim + veto + few-shot)",
            "test_set_size": len(test_set),
            "baseline": "v1.7",
        },
        "final_prompt": current_prompt,
        "final_prompt_len": len(current_prompt),
        "final_label": current_label,
        "rounds": all_rounds,
    }
    with open(os.path.join(RESULTS_DIR, "final_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(RESULTS_DIR, "final_best_prompt.txt"), "w", encoding="utf-8") as f:
        f.write(current_prompt)

    print(f"\n✅ Final results saved to {RESULTS_DIR}/", flush=True)
    print(f"   final_summary.json", flush=True)
    print(f"   final_best_prompt.txt", flush=True)


if __name__ == "__main__":
    main()
