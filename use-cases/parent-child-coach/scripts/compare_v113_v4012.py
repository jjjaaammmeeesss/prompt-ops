"""v1.13 vs v4.0.12 横向对比脚本 · 五道门框架（门1 + 门4）。

两个版本用相同数据集、相同 judge、相同轮数，在同一脚本内跑，保证可比。

用法:
  python scripts/compare_v113_v4012.py --gate 1 --rounds 3     # 门1：校标集统计检验
  python scripts/compare_v113_v4012.py --gate 4 --rounds 3     # 门4：独立测试盲评
  python scripts/compare_v113_v4012.py --gate all --rounds 3   # 全部门
"""

import argparse
import json
import math
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# ── Paths ────────────────────────────────────────────────────
_realtime_parent = Path(__file__).resolve().parent.parent
_project_root = Path(__file__).resolve().parents[2]

if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_realtime_parent) not in sys.path:
    sys.path.insert(0, str(_realtime_parent))

from dotenv import load_dotenv
load_dotenv("D:/星灵-soul-手搓/亲子沟通洞见/测试智能体/.env")

sys.path.insert(0, str(_realtime_parent / "scripts"))
from llm_judge_metric import LLMJudgeMetric

# ── Config ───────────────────────────────────────────────────
V113_PROMPT_PATH = Path(
    r"D:\星灵-soul-手搓\亲子沟通洞见\测试智能体\prompt-ops"
    r"\use-cases\parent-child-coach\prompt_A轨_v1.13_候选版.md"
)
V4012_PROMPT_PATH = _realtime_parent / "system_prompt_v4.0.12.txt"
EXPERT_DATASET = _realtime_parent / "data" / "expert_dataset_full_71.json"
INDEPENDENT_DATASET = _realtime_parent / "data" / "new_12_independent.json"

QIANFAN_URL = "https://qianfan.baidubce.com/v2/chat/completions"
QIANFAN_KEY = os.getenv("BAIDU_QIANFAN_KEY", "")
SUT_MODEL = "deepseek-v4-pro"
JUDGE_MODEL = "glm-5.2"

OUTPUT_DIR = _realtime_parent / "results" / "compare_v113_v4012"


# ═══════════════════════════════════════════════════════════════
# Qianfan API Client
# ═══════════════════════════════════════════════════════════════

class QianfanClient:
    """Thin wrapper around Baidu Qianfan ModelBuilder API (OpenAI-compatible)."""

    def __init__(self, model: str = SUT_MODEL, timeout: int = 120):
        self.model = model
        self.timeout = timeout
        if not QIANFAN_KEY:
            raise RuntimeError("BAIDU_QIANFAN_KEY env var required")

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 640,
        temperature: float = 0.3,
    ) -> Tuple[str, Dict[str, Any]]:
        """Return (content_text, usage_dict)."""
        headers = {
            "Authorization": f"Bearer {QIANFAN_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        resp = requests.post(
            QIANFAN_URL, headers=headers, json=payload, timeout=self.timeout
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return content, usage


# ═══════════════════════════════════════════════════════════════
# v1.13 Adapter
# ═══════════════════════════════════════════════════════════════

def extract_v113_prompts(md_path: Path) -> Tuple[str, str]:
    """Parse v1.13 markdown: System Prompt (``` fence) + User Prompt (```` fence)."""
    lines = md_path.read_text(encoding="utf-8").splitlines(keepends=True)

    # System prompt: first ``` block
    sys_start = sys_end = None
    for i, line in enumerate(lines):
        s = line.rstrip()
        if s == "```" and sys_start is None:
            sys_start = i
        elif s == "```" and sys_start is not None and i > sys_start:
            sys_end = i
            break
    system_prompt = "".join(lines[sys_start + 1 : sys_end])

    # User prompt: first ```` block
    usr_start = usr_end = None
    for i, line in enumerate(lines):
        s = line.rstrip()
        if s == "````" and usr_start is None:
            usr_start = i
        elif s == "````" and usr_start is not None and i > usr_start:
            usr_end = i
            break
    user_prompt = "".join(lines[usr_start + 1 : usr_end])
    return system_prompt, user_prompt


def parse_v113_output(raw: str) -> dict:
    """Fuzzy JSON parse v1.13 output (same pattern as v1.7)."""
    raw = raw.strip()
    # Strip ``` fences
    if raw.startswith("```"):
        first_nl = raw.find("\n")
        if first_nl > 0:
            raw = raw[first_nl + 1 :]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Extract first { ... } block
    first_brace = raw.find("{")
    last_brace = raw.rfind("}")
    if first_brace >= 0 and last_brace > first_brace:
        try:
            return json.loads(raw[first_brace : last_brace + 1])
        except json.JSONDecodeError:
            pass
    return {"_parse_error": "JSON 解析失败", "_raw": raw[:500]}


def generate_v113(
    client: QianfanClient,
    system_prompt: str,
    user_template: str,
    dialogue: str,
) -> Tuple[str, dict]:
    """Generate popup with v1.13 prompt. Returns (popup_text, metadata)."""
    filled = user_template.replace("{user_input}", dialogue)
    filled = filled.replace("{context_block}", "（首次对话，无前序上下文）")
    filled = filled.replace("{profile_context}", "（无用户画像）")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": filled},
    ]
    raw, usage = client.chat(messages=messages, max_tokens=2048, temperature=0.3)

    obj = parse_v113_output(raw)
    if not obj.get("should_popup", True):
        return "", {
            "tone": "skip",
            "should_popup": False,
            "skip_reason": obj.get("skip_reason"),
            "usage": usage,
        }

    tone = obj.get("tone", "diagnostic")
    insight = obj.get("popup_insight") or ""
    suggestion = obj.get("popup_suggestion") or ""

    if tone == "empowering":
        popup = insight.strip()
    else:
        parts = [p for p in [insight.strip(), suggestion.strip()] if p]
        popup = "\n\n".join(parts)

    return popup, {
        "tone": tone,
        "should_popup": True,
        "skip_reason": None,
        "usage": usage,
    }


# ═══════════════════════════════════════════════════════════════
# v4.0.12 Adapter
# ═══════════════════════════════════════════════════════════════

def generate_v4012(
    client: QianfanClient,
    system_prompt: str,
    dialogue: str,
    tone: str,
) -> str:
    """Generate popup with v4.0.12 prompt. Returns popup_text."""
    if tone == "诊断式":
        type_instruction = (
            "请生成**诊断式弹窗**（80-200字）。"
            "必须：先承认发心 → 揭示具体模式 → 给出一个微小可做的尝试。"
        )
    else:
        type_instruction = (
            "请生成**鼓励式弹窗**（30-80字）。"
            "必须：具体点出家长刚展现的积极模式 → 简短有力。"
        )
    user_content = f"""当前对话：
{dialogue}

{type_instruction}

请直接输出弹窗全文（不附加解释、不输出JSON、不输出"弹窗："等前缀）："""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    raw, usage = client.chat(messages=messages, max_tokens=640, temperature=0.3)
    return raw.strip()


# ═══════════════════════════════════════════════════════════════
# Statistical Functions (from test-agent SystemComparator)
# ═══════════════════════════════════════════════════════════════

def bootstrap_ci(
    deltas: List[float],
    n_bootstrap: int = 2000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Bootstrap confidence interval for mean of paired differences."""
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(n_bootstrap):
        sample = [deltas[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    alpha = (1 - ci_level) / 2
    lo_idx = int(alpha * n_bootstrap)
    hi_idx = int((1 - alpha) * n_bootstrap)
    return means[lo_idx], means[hi_idx]


def wilcoxon_signed_rank(deltas: List[float]) -> Tuple[float, float]:
    """Wilcoxon signed-rank test. Returns (W_statistic, approximate_p_value)."""
    # Remove zero differences
    non_zero = [d for d in deltas if abs(d) > 1e-10]
    n = len(non_zero)
    if n == 0:
        return 0.0, 1.0

    # Rank absolute differences
    abs_d = [abs(d) for d in non_zero]
    ranked = sorted(enumerate(abs_d), key=lambda x: x[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and ranked[j][1] == ranked[i][1]:
            j += 1
        avg_rank = (i + j + 1) / 2  # 1-indexed average
        for k in range(i, j):
            ranks[ranked[k][0]] = avg_rank
        i = j

    # W = sum of ranks for positive differences
    W = sum(ranks[k] for k in range(n) if non_zero[k] > 0)

    # Normal approximation
    mean_W = n * (n + 1) / 4
    # Handle ties: compute tie correction
    unique_vals = {}
    for d in abs_d:
        unique_vals[d] = unique_vals.get(d, 0) + 1
    tie_corr = sum(c * (c - 1) * (c + 1) for c in unique_vals.values())
    std_W = math.sqrt(n * (n + 1) * (2 * n + 1) / 24 - tie_corr / 48)

    if std_W < 1e-10:
        return W, 1.0

    z = (W - mean_W) / std_W
    # Two-sided p-value from normal approximation
    p_value = 2 * (1 - _normal_cdf(abs(z)))
    return W, p_value


def _normal_cdf(x: float) -> float:
    """Standard normal CDF (Abramowitz and Stegun approximation)."""
    if x < 0:
        return 1 - _normal_cdf(-x)
    b = [0.31938153, -0.356563782, 1.781477937, -1.821255978, 1.330274429]
    p = 0.2316419
    t = 1 / (1 + p * x)
    poly = b[0] * t + b[1] * t**2 + b[2] * t**3 + b[3] * t**4 + b[4] * t**5
    return 1 - (1 / math.sqrt(2 * math.pi)) * math.exp(-(x**2) / 2) * poly


# ═══════════════════════════════════════════════════════════════
# Judge Runner
# ═══════════════════════════════════════════════════════════════

def score_with_judge(
    judge: LLMJudgeMetric,
    dialogue: str,
    golden: str,
    popup: str,
) -> float:
    """Score a popup against the golden answer using LLM judge."""
    from dspy import Example

    if not popup:
        return 0.0
    gold_ex = Example(question=dialogue, answer=golden)
    pred_ex = Example(answer=popup)
    try:
        return judge(gold_ex, pred_ex, trace=False)
    except Exception as e:
        print(f"    ⚠️ Judge exception: {e}")
        return 0.0


# ═══════════════════════════════════════════════════════════════
# Gate 1: Statistical Testing (Gold-Anchored)
# ═══════════════════════════════════════════════════════════════

def run_gate1(
    cases: List[dict],
    sut_client: QianfanClient,
    judge: LLMJudgeMetric,
    v113_sys: str,
    v113_usr: str,
    v4012_prompt: str,
    rounds: int = 3,
) -> dict:
    """Gate 1: 70 cases × 2 versions × N rounds, gold-anchored judge."""
    n = len(cases)
    print(f"\n{'='*60}")
    print(f"  Gate 1 · 统计检验 — {n} 条校标 × {rounds} 轮")
    print(f"{'='*60}")

    all_rounds: List[dict] = []

    for r in range(rounds):
        print(f"\n--- Round {r+1}/{rounds} ---")
        round_data = {"round": r + 1, "v113": [], "v4012": [], "deltas": []}

        # Randomize order for fairness (same seed per round for reproducibility)
        rng = random.Random(42 + r)
        indices = list(range(n))
        rng.shuffle(indices)

        for idx_idx, idx in enumerate(indices):
            case = cases[idx]
            dialogue = case["question"]
            golden = case["answer"]
            expected_tone = case.get("tone", "诊断式")
            case_id = case.get("id", f"case_{idx}")

            print(f"  [{idx_idx+1}/{n}] {case_id} tone={expected_tone}")

            # --- v1.13 ---
            try:
                popup_113, meta_113 = generate_v113(
                    sut_client, v113_sys, v113_usr, dialogue
                )
                score_113 = score_with_judge(judge, dialogue, golden, popup_113)
            except Exception as e:
                print(f"    ❌ v1.13 error: {e}")
                popup_113, meta_113, score_113 = "", {"tone": "error"}, 0.0

            # --- v4.0.12 ---
            try:
                popup_4012 = generate_v4012(
                    sut_client, v4012_prompt, dialogue, expected_tone
                )
                score_4012 = score_with_judge(judge, dialogue, golden, popup_4012)
            except Exception as e:
                print(f"    ❌ v4.0.12 error: {e}")
                popup_4012, score_4012 = "", 0.0

            delta = score_113 - score_4012
            print(
                f"    v1.13: {score_113:.3f} | v4.0.12: {score_4012:.3f} "
                f"| Δ: {delta:+.3f} | tone_113={meta_113.get('tone')}"
            )

            round_data["v113"].append(
                {
                    "case_id": case_id,
                    "expected_tone": expected_tone,
                    "actual_tone": meta_113.get("tone"),
                    "should_popup": meta_113.get("should_popup"),
                    "score": score_113,
                    "popup": popup_113,
                }
            )
            round_data["v4012"].append(
                {
                    "case_id": case_id,
                    "expected_tone": expected_tone,
                    "score": score_4012,
                    "popup": popup_4012,
                }
            )
            round_data["deltas"].append(delta)

        all_rounds.append(round_data)

    # ── Aggregate across rounds ──
    v113_scores = [
        s for rd in all_rounds for s in [e["score"] for e in rd["v113"]]
    ]
    v4012_scores = [
        s for rd in all_rounds for s in [e["score"] for e in rd["v4012"]]
    ]
    deltas = [d for rd in all_rounds for d in rd["deltas"]]

    v113_mean = sum(v113_scores) / len(v113_scores) if v113_scores else 0
    v4012_mean = sum(v4012_scores) / len(v4012_scores) if v4012_scores else 0
    delta_mean = sum(deltas) / len(deltas) if deltas else 0

    v113_pass = sum(1 for s in v113_scores if s >= 0.70)
    v4012_pass = sum(1 for s in v4012_scores if s >= 0.70)

    ci_lo, ci_hi = bootstrap_ci(deltas)
    W_stat, p_value = wilcoxon_signed_rank(deltas)

    # v1.13 skip rate
    skip_count = sum(
        1
        for rd in all_rounds
        for e in rd["v113"]
        if e.get("actual_tone") == "skip"
    )
    skip_rate = skip_count / len(v113_scores) if v113_scores else 0

    # ── Print summary ──
    print(f"\n{'='*60}")
    print(f"  Gate 1 · 统计结果")
    print(f"{'='*60}")
    print(f"  v1.13:    μ={v113_mean:.3f}  pass={v113_pass}/{len(v113_scores)}")
    print(f"  v4.0.12:  μ={v4012_mean:.3f}  pass={v4012_pass}/{len(v4012_scores)}")
    print(f"  Δ (v1.13 - v4.0.12): {delta_mean:+.4f}")
    print(f"  Bootstrap 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  Wilcoxon: W={W_stat:.1f}, p={p_value:.4f}")
    print(f"  v1.13 不弹率: {skip_rate:.1%}")

    # ── Verdict ──
    ci_significant = ci_lo > 0 or ci_hi < 0
    p_significant = p_value < 0.05

    if ci_significant and p_significant:
        if delta_mean > 0:
            verdict = "v1.13 显著更优 ✅"
        else:
            verdict = "v4.0.12 显著更优 ✅"
    else:
        verdict = "TIE — 差异不显著"

    if skip_rate > 0.30:
        verdict += " ⚠️ v1.13 不弹率 >30%（前置分流可能过度抑制）"

    print(f"  → {verdict}")

    return {
        "gate": 1,
        "n_cases": n,
        "rounds": rounds,
        "v113_mean": v113_mean,
        "v4012_mean": v4012_mean,
        "v113_pass_rate": v113_pass / len(v113_scores) if v113_scores else 0,
        "v4012_pass_rate": v4012_pass / len(v4012_scores) if v4012_scores else 0,
        "delta_mean": delta_mean,
        "bootstrap_ci": [ci_lo, ci_hi],
        "wilcoxon_W": W_stat,
        "wilcoxon_p": p_value,
        "v113_skip_rate": skip_rate,
        "ci_significant": ci_significant,
        "p_significant": p_significant,
        "verdict": verdict,
        "rounds_data": all_rounds,
    }


# ═══════════════════════════════════════════════════════════════
# Gate 4: Independent Testing (Gold-Anchored + Blind A/B)
# ═══════════════════════════════════════════════════════════════

BLIND_AB_PROMPT = """你是一名亲子沟通教练弹窗的评估专家。下面是一段亲子对话，以及**两份**不同版本的系统生成的弹窗。

请独立评估每份弹窗，然后选出你认为更好的一份。如果两份质量相当，选 "TIE"。

## 对话
{dialogue}

## 弹窗 A
{popup_a}

## 弹窗 B
{popup_b}

## 评估标准
1. 是否抓住了对话中最关键的洞察？
2. 语气温度是否合适？
3. 语言是否自然、像人说的？
4. 是否有事实错误或编造内容？

## 输出格式
只输出 JSON（不要其他文字）：
{{"winner": "A"|"B"|"TIE", "reason_A": "弹窗A一句话评价", "reason_B": "弹窗B一句话评价", "brief_reason": "选择理由"}}"""


def blind_ab_judge(
    client: QianfanClient,
    dialogue: str,
    popup_a: str,
    popup_b: str,
) -> dict:
    """Blind A/B comparison: judge sees two unlabeled popups, picks winner."""
    prompt = BLIND_AB_PROMPT.format(
        dialogue=dialogue,
        popup_a=popup_a or "（未弹窗）",
        popup_b=popup_b or "（未弹窗）",
    )
    messages = [
        {
            "role": "system",
            "content": "你是严格的亲子沟通弹窗评估专家。只输出JSON，不输出其他内容。",
        },
        {"role": "user", "content": prompt},
    ]
    raw, usage = client.chat(
        messages=messages, max_tokens=512, temperature=0.1
    )

    # Parse JSON
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {"winner": "TIE", "brief_reason": "parse_error", "_raw": raw[:300]}


def run_gate4(
    cases: List[dict],
    sut_client: QianfanClient,
    judge: LLMJudgeMetric,
    v113_sys: str,
    v113_usr: str,
    v4012_prompt: str,
    rounds: int = 3,
) -> dict:
    """Gate 4: 12 independent cases × 2 versions × N rounds, gold-anchored + blind A/B."""
    n = len(cases)
    print(f"\n{'='*60}")
    print(f"  Gate 4 · 独立测试 — {n} 条用例 × {rounds} 轮")
    print(f"{'='*60}")

    # Use separate client for blind judge (or same model is fine since
    # blind judge is a different task from scoring judge)
    blind_client = QianfanClient(model=JUDGE_MODEL, timeout=90)

    all_results = []
    gold_v113_scores = []
    gold_v4012_scores = []
    ab_winners = []  # "v113" | "v4012" | "TIE"

    for r in range(rounds):
        print(f"\n--- Round {r+1}/{rounds} ---")

        for i, case in enumerate(cases):
            dialogue = case["question"]
            golden = case["answer"]
            expected_tone = case.get("tone", "诊断式")
            case_id = case.get("case_id", f"case_{i}")

            print(f"  [{i+1}/{n}] {case_id} tone={expected_tone}")

            # --- Generate ---
            try:
                popup_113, meta_113 = generate_v113(
                    sut_client, v113_sys, v113_usr, dialogue
                )
                score_113 = score_with_judge(judge, dialogue, golden, popup_113)
            except Exception as e:
                print(f"    ❌ v1.13 error: {e}")
                popup_113, meta_113, score_113 = "", {"tone": "error"}, 0.0

            try:
                popup_4012 = generate_v4012(
                    sut_client, v4012_prompt, dialogue, expected_tone
                )
                score_4012 = score_with_judge(judge, dialogue, golden, popup_4012)
            except Exception as e:
                print(f"    ❌ v4.0.12 error: {e}")
                popup_4012, score_4012 = "", 0.0

            gold_v113_scores.append(score_113)
            gold_v4012_scores.append(score_4012)

            # --- Blind A/B (randomize order to prevent position bias) ---
            swap = random.choice([True, False])
            if swap:
                ab_result = blind_ab_judge(
                    blind_client, dialogue, popup_4012, popup_113
                )
                raw_winner = ab_result.get("winner", "TIE")
                if raw_winner == "A":
                    ab_winner = "v4012"
                elif raw_winner == "B":
                    ab_winner = "v113"
                else:
                    ab_winner = "TIE"
            else:
                ab_result = blind_ab_judge(
                    blind_client, dialogue, popup_113, popup_4012
                )
                raw_winner = ab_result.get("winner", "TIE")
                if raw_winner == "A":
                    ab_winner = "v113"
                elif raw_winner == "B":
                    ab_winner = "v4012"
                else:
                    ab_winner = "TIE"

            ab_winners.append(ab_winner)

            print(
                f"    v1.13: {score_113:.3f} | v4.0.12: {score_4012:.3f} "
                f"| Blind AB: {ab_winner} (swap={swap})"
            )

            all_results.append(
                {
                    "case_id": case_id,
                    "round": r + 1,
                    "expected_tone": expected_tone,
                    "v113_score": score_113,
                    "v4012_score": score_4012,
                    "v113_popup": popup_113,
                    "v4012_popup": popup_4012,
                    "v113_tone": meta_113.get("tone"),
                    "blind_ab_winner": ab_winner,
                    "swap": swap,
                }
            )

    # ── Aggregate ──
    v113_mean = sum(gold_v113_scores) / len(gold_v113_scores)
    v4012_mean = sum(gold_v4012_scores) / len(gold_v4012_scores)
    gold_deltas = [
        a - b for a, b in zip(gold_v113_scores, gold_v4012_scores)
    ]

    ci_lo, ci_hi = bootstrap_ci(gold_deltas)
    W_stat, p_value = wilcoxon_signed_rank(gold_deltas)

    # Blind A/B tally
    v113_wins = sum(1 for w in ab_winners if w == "v113")
    v4012_wins = sum(1 for w in ab_winners if w == "v4012")
    ties = sum(1 for w in ab_winners if w == "TIE")
    total = len(ab_winners)
    v113_win_rate = v113_wins / total if total else 0
    v4012_win_rate = v4012_wins / total if total else 0

    # Bootstrap CI on win rate
    blind_deltas = []
    for w in ab_winners:
        if w == "v113":
            blind_deltas.append(1)
        elif w == "v4012":
            blind_deltas.append(0)
        else:
            blind_deltas.append(0.5)  # TIE = half point each
    blind_ci_lo, blind_ci_hi = bootstrap_ci(blind_deltas)

    # v1.13 skip rate
    skip_count = sum(
        1 for r in all_results if r.get("v113_tone") == "skip"
    )
    skip_rate = skip_count / len(all_results) if all_results else 0

    # ── Print summary ──
    print(f"\n{'='*60}")
    print(f"  Gate 4 · 统计结果")
    print(f"{'='*60}")
    print(f"  Gold-anchored:")
    print(f"    v1.13:    μ={v113_mean:.3f}")
    print(f"    v4.0.12:  μ={v4012_mean:.3f}")
    print(f"    Δ: {v113_mean - v4012_mean:+.4f}")
    print(f"    Bootstrap 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"    Wilcoxon: W={W_stat:.1f}, p={p_value:.4f}")
    print(f"  Blind A/B:")
    print(f"    v1.13 wins: {v113_wins}/{total} ({v113_win_rate:.1%})")
    print(f"    v4.0.12 wins: {v4012_wins}/{total} ({v4012_win_rate:.1%})")
    print(f"    TIE: {ties}/{total}")
    print(f"    v1.13 win rate 95% CI: [{blind_ci_lo:.3f}, {blind_ci_hi:.3f}]")
    print(f"  v1.13 不弹率: {skip_rate:.1%}")

    # ── Verdict ──
    gold_sig = (ci_lo > 0 or ci_hi < 0) and p_value < 0.05
    blind_sig = blind_ci_lo > 0.5 or blind_ci_hi < 0.5

    if gold_sig and blind_sig:
        if v113_mean > v4012_mean and blind_ci_lo > 0.5:
            verdict = "v1.13 显著更优 ✅（高置信度：两指标一致）"
        elif v4012_mean > v113_mean and blind_ci_hi < 0.5:
            verdict = "v4.0.12 显著更优 ✅（高置信度：两指标一致）"
        else:
            verdict = "⚠️ 两指标矛盾，需人工分析"
    elif gold_sig:
        verdict = (
            f"{'v1.13' if v113_mean > v4012_mean else 'v4.0.12'} "
            f"Gold-anchored 显著更优，Blind A/B 不显著"
        )
    elif blind_sig:
        winner = "v1.13" if blind_ci_lo > 0.5 else "v4.0.12"
        verdict = f"{winner} Blind A/B 显著更优，Gold-anchored 不显著"
    else:
        verdict = "TIE — Gold-anchored 和 Blind A/B 均不显著"

    print(f"  → {verdict}")

    return {
        "gate": 4,
        "n_cases": n,
        "rounds": rounds,
        "gold_v113_mean": v113_mean,
        "gold_v4012_mean": v4012_mean,
        "gold_delta_mean": v113_mean - v4012_mean,
        "gold_bootstrap_ci": [ci_lo, ci_hi],
        "gold_wilcoxon_W": W_stat,
        "gold_wilcoxon_p": p_value,
        "blind_v113_wins": v113_wins,
        "blind_v4012_wins": v4012_wins,
        "blind_ties": ties,
        "blind_v113_win_rate": v113_win_rate,
        "blind_win_rate_ci": [blind_ci_lo, blind_ci_hi],
        "v113_skip_rate": skip_rate,
        "verdict": verdict,
        "results": all_results,
    }


# ═══════════════════════════════════════════════════════════════
# Gate 3: Candidate Validation (Structural Check)
# ═══════════════════════════════════════════════════════════════

def run_gate3(
    v113_path: Path, v4012_path: Path
) -> dict:
    """Gate 3: structural integrity check of both prompts."""
    print(f"\n{'='*60}")
    print(f"  Gate 3 · 候选校验")
    print(f"{'='*60}")

    v113_text = v113_path.read_text(encoding="utf-8")
    v4012_text = v4012_path.read_text(encoding="utf-8")

    checks = {}

    # ── v4.0.12 checks ──
    v4012_checks = []
    sections = ["一、", "二、", "三、", "四、", "五、", "六、", "七、", "八、"]
    for s in sections:
        v4012_checks.append(
            {"check": f"Section {s} 存在", "pass": s in v4012_text}
        )
    v4012_checks.append(
        {
            "check": "周易卦象速查表完整",
            "pass": "卦象" in v4012_text and "弹窗策略" in v4012_text,
        }
    )
    v4012_checks.append(
        {
            "check": "tone 灵活覆盖规则（≥2 积极行为）",
            "pass": "≥2" in v4012_text or "≥ 2" in v4012_text,
        }
    )
    v4012_checks.append(
        {
            "check": "全面禁脑补规则",
            "pass": "脑补" in v4012_text or "虚构" in v4012_text,
        }
    )
    all_v4012 = all(c["pass"] for c in v4012_checks)
    print(f"  v4.0.12: {'✅' if all_v4012 else '❌'}")
    for c in v4012_checks:
        print(f"    [{'✅' if c['pass'] else '❌'}] {c['check']}")
    checks["v4012"] = {"passed": all_v4012, "checks": v4012_checks}

    # ── v1.13 checks ──
    v113_checks = []
    v113_checks.append(
        {"check": "System Prompt（``` 围栏）存在", "pass": "```" in v113_text}
    )
    v113_checks.append(
        {
            "check": "User Prompt（```` 围栏）存在",
            "pass": "````" in v113_text,
        }
    )
    v113_checks.append(
        {
            "check": "JSON schema 完整（should_popup / tone / popup_insight / popup_suggestion）",
            "pass": all(
                k in v113_text
                for k in [
                    "should_popup",
                    "tone",
                    "popup_insight",
                    "popup_suggestion",
                ]
            ),
        }
    )
    # v1.13 uses Chinese labels like "第 0.5 步" or "# Step 0"
    step_flow_ok = (
        ("Step 0" in v113_text or "第 0 步" in v113_text)
        and ("0.5" in v113_text or "前置分流" in v113_text)
        and ("Step 1" in v113_text or "第 1 步" in v113_text)
        and ("Step 2" in v113_text or "第 2 步" in v113_text)
        and ("Step 3" in v113_text or "第 3 步" in v113_text)
    )
    v113_checks.append(
        {
            "check": "Step 0→0.5→1→2A/2B→3 流程完整",
            "pass": step_flow_ok,
        }
    )
    v113_checks.append(
        {
            "check": "占位符 {user_input}/{profile_context}/{context_block} 存在",
            "pass": all(
                p in v113_text
                for p in ["{user_input}", "{profile_context}", "{context_block}"]
            ),
        }
    )
    v113_checks.append(
        {
            "check": "F1 逐句原文锚定",
            "pass": "原文引用" in v113_text or "逐句" in v113_text,
        }
    )
    v113_checks.append(
        {
            "check": "F2 反话强制响应",
            "pass": "反话" in v113_text,
        }
    )
    v113_checks.append(
        {
            "check": "F3 深度自检 (L1/L2/L3)",
            "pass": "L1" in v113_text and "L2" in v113_text,
        }
    )
    v113_checks.append(
        {
            "check": "F4 前置分流 (Step 0.5)",
            "pass": "0.5" in v113_text or "前置分流" in v113_text,
        }
    )
    all_v113 = all(c["pass"] for c in v113_checks)
    print(f"  v1.13: {'✅' if all_v113 else '❌'}")
    for c in v113_checks:
        print(f"    [{'✅' if c['pass'] else '❌'}] {c['check']}")
    checks["v113"] = {"passed": all_v113, "checks": v113_checks}

    all_pass = all_v4012 and all_v113
    print(f"\n  → Gate 3: {'PASS ✅' if all_pass else 'FAIL ❌'}")

    return {"gate": 3, "passed": all_pass, "checks": checks}


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="v1.13 vs v4.0.12 横向对比（五道门框架）"
    )
    parser.add_argument(
        "--gate",
        choices=["1", "3", "4", "all"],
        default="all",
        help="Which gate to run (default: all)",
    )
    parser.add_argument(
        "--rounds", type=int, default=3, help="Number of rounds per case (default: 3)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit cases to first N (0 = all, for testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config and exit without calling APIs",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("╔══════════════════════════════════════════════════╗")
    print("║  v1.13 vs v4.0.12 · 五道门横向对比            ║")
    print("╚══════════════════════════════════════════════════╝")
    print(f"  SUT:  {SUT_MODEL} @ Baidu Qianfan")
    print(f"  Judge: {JUDGE_MODEL} @ Baidu Qianfan")
    print(f"  Rounds: {args.rounds}")
    print(f"  Output: {OUTPUT_DIR}")

    if args.dry_run:
        print("\n[Dry run — exiting]")
        return

    # ── Init clients ──
    sut_client = QianfanClient(model=SUT_MODEL, timeout=120)

    # ── Gate 3: Candidate Validation ──
    if args.gate in ("3", "all"):
        gate3_result = run_gate3(V113_PROMPT_PATH, V4012_PROMPT_PATH)
        gate3_path = OUTPUT_DIR / f"gate3_report_{timestamp}.json"
        gate3_path.write_text(
            json.dumps(gate3_result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n  Gate 3 报告: {gate3_path}")

        if not gate3_result["passed"]:
            print("\n  ❌ Gate 3 未通过，终止后续门。请修复 prompt 结构问题。")
            return

    # ── Load prompts ──
    v113_sys, v113_usr = extract_v113_prompts(V113_PROMPT_PATH)
    v4012_prompt = V4012_PROMPT_PATH.read_text(encoding="utf-8")
    print(f"\n  v1.13: system={len(v113_sys)}字, user={len(v113_usr)}字")
    print(f"  v4.0.12: {len(v4012_prompt)}字")

    # ── Gate 1: Statistical Testing ──
    if args.gate in ("1", "all"):
        with open(EXPERT_DATASET, "r", encoding="utf-8") as f:
            expert_cases = json.load(f)
        if args.limit > 0:
            expert_cases = expert_cases[: args.limit]
            print(f"\n  ⚠️ 限制为前 {args.limit} 条（测试模式）")

        judge = LLMJudgeMetric(judge_backend="qianfan", timeout=90)
        gate1_result = run_gate1(
            cases=expert_cases,
            sut_client=sut_client,
            judge=judge,
            v113_sys=v113_sys,
            v113_usr=v113_usr,
            v4012_prompt=v4012_prompt,
            rounds=args.rounds,
        )

        gate1_path = OUTPUT_DIR / f"gate1_report_{timestamp}.json"
        gate1_path.write_text(
            json.dumps(gate1_result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n  Gate 1 报告: {gate1_path}")

    # ── Gate 4: Independent Testing ──
    if args.gate in ("4", "all"):
        with open(INDEPENDENT_DATASET, "r", encoding="utf-8") as f:
            independent_cases = json.load(f)
        if args.limit > 0:
            independent_cases = independent_cases[: args.limit]
            print(f"\n  ⚠️ 限制为前 {args.limit} 条（测试模式）")

        judge4 = LLMJudgeMetric(judge_backend="qianfan", timeout=90)
        gate4_result = run_gate4(
            cases=independent_cases,
            sut_client=sut_client,
            judge=judge4,
            v113_sys=v113_sys,
            v113_usr=v113_usr,
            v4012_prompt=v4012_prompt,
            rounds=args.rounds,
        )

        gate4_path = OUTPUT_DIR / f"gate4_report_{timestamp}.json"
        gate4_path.write_text(
            json.dumps(gate4_result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n  Gate 4 报告: {gate4_path}")

    print(f"\n{'='*60}")
    print(f"  全部测试完成。结果保存在: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
