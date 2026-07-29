"""
LLM-as-Judge Metric v3.0 — 带 golden answer 对齐的 5 维度评价模型。

v3.0 改进（vs v2.0）:
  - Judge 输入从 (对话, 预测) 扩展为 (对话, 预测, 专家改写)
  - 新增"专家策略对齐"维度（权重 0.30），要求 judge 先概括专家意图再比较
  - 保留 v2.0 的 5 维框架但重新加权，追专家策略而非 judge 自身审美
  - ROUGE-L/chrF 作为旁路观测，不参与主分
  - 修复 metric_threshold 量纲：返回 [0,1]，阈值应为 0.70（非 7.0）

Judge backends: Claude (via 智创聚合, default) or DeepSeek-chat (fallback).
"""

import json
import os
import re
import time
from typing import Any, Dict, Optional, Tuple

import requests

from prompt_ops.core.metrics import MetricBase

# === Judge backend selection ===
JUDGE_BACKEND = os.getenv("JUDGE_BACKEND", "qianfan")

# === Baidu Qianfan API (OpenAI-compatible, GLM-5.2) ===
QIANFAN_URL = "https://qianfan.baidubce.com/v2/chat/completions"
QIANFAN_KEY = os.getenv("BAIDU_QIANFAN_KEY", "")
QIANFAN_MODEL = "glm-5.2"

# === Claude API (智创聚合 — Anthropic Messages format) ===
CLAUDE_URL = "https://s.lconai.com/v1/messages"
CLAUDE_KEY = os.getenv("LCONAI_API_KEY", "CLAUDE_API_KEY_PLACEHOLDER")
CLAUDE_MODEL = "claude-opus-4-8"

# === DeepSeek API (OpenAI Chat Completions format, fallback) ===
DS_URL = "https://api.deepseek.com/v1/chat/completions"
DS_KEY = os.getenv("DEEPSEEK_API_KEY")
DS_MODEL = "deepseek-chat"

# === 5-dimension scoring weights (v3.0: 专家对齐优先) ===
DIMS = [
    ("strategy_alignment", 0.30, "专家策略对齐"),
    ("dialogue_fidelity",   0.20, "对话忠实度"),
    ("tone_alignment",      0.20, "语气与温度对齐"),
    ("natural_language",    0.15, "人话感"),
    ("core_insight",        0.15, "命中核心"),
]

# === ROUGE-L / chrF 旁路观测（不参与主分） ===
_OBSERVE_ROUGE = True  # 设为 False 可关闭旁路计算


def _compute_rouge_l(reference: str, candidate: str) -> float:
    """字符级 ROUGE-L F1（旁路观测，不参与主分）。"""
    if not reference or not candidate:
        return 0.0
    ref_chars = list(reference)
    cand_chars = list(candidate)
    m, n = len(ref_chars), len(cand_chars)
    if m == 0 or n == 0:
        return 0.0
    # LCS via DP (简版，对短文本足够)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if ref_chars[i] == cand_chars[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    lcs_len = dp[m][n]
    precision = lcs_len / n if n > 0 else 0.0
    recall = lcs_len / m if m > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# === Judge prompt v3.0（三段输入：对话 + 预测弹窗 + 专家改写） ===
SCORING_PROMPT = """你是一名亲子沟通教练弹窗的评估专家。你的任务是判断 AI 生成的弹窗是否实现了专家手写改写所代表的**意图、核心洞察、介入策略和语气**。

## 评分流程

### 第一步：读懂专家意图
先阅读下方的「专家改写弹窗」，用一句话概括：
- 专家在这个窗口想抓住的**核心洞察**是什么？
- 专家选择了什么**介入策略**（诊断式指出盲区 vs 鼓励式庆祝做对的事）？
- 专家的**语气和温度**是怎样的（盟友姿态还是有距离的观察）？

### 第二步：比较预测弹窗
阅读「AI 生成弹窗」，判断它是否实现了与专家**相同的意图和策略**。
注意：不要求逐字逐句一样——允许 AI 用自己的话表达相同的洞察、用不同的措辞达到相同的效果。但如果 AI 抓了完全不同的重点、用了相反的策略、或语气温度偏离明显，则应扣分。

### 第三步：评分

---

## 评分规则

### 一级否决（先检查，触发任一则总分=0，不进入维度评分）

**事实性错误**：弹窗编造了对话中不存在的内容——认错主语（把妈妈说成爸爸/反之）、引用其他窗口的句子、或描述了对话中根本没发生的事。
**语气严重误判**：家长明明做得很好的场景给了诊断式弹窗；或者严重问题被轻描淡写为鼓励。

---

### 维度评分（仅在通过否决后）

#### 1. 专家策略对齐（1-5分）[权重 0.30]
AI 弹窗是否抓住了专家认为最该被看见的那个点？干预方向（诊断 vs 鼓励）是否与专家一致？

- 5分：核心洞察、干预方向和策略与专家高度一致。读完 AI 弹窗的感受和读完专家改写一致。
- 3分：大方向对——抓到了同一领域的问题，但不是专家选的那个最精确的切入点。或策略对但洞察深度明显不如专家。
- 1分：完全跑偏——专家关注 A，AI 关注 B，且 B 不是 A 的合理等价表达。

#### 2. 对话忠实度（1-5分）[权重 0.20]
AI 弹窗的每一个判断都能在对话原文中找到具体依据？不编造、不夸大、不对心理活动做超出对话证据的推断？

- 5分：每个判断都能在对话中找到具体的句子作为依据。
- 3分：方向大致正确，但有 2-3 处推断过度或轻微偏离。
- 1分：严重脱离对话——编造场景、对不存在的事件进行诊断。

#### 3. 语气与温度对齐（1-5分）[权重 0.20]
AI 弹窗的温度和姿态是否与专家改写在同一频段？如果专家是温暖的盟友，AI 是否也给了同样的感受？如果专家是冷静的观察者，AI 是否也保持了适当距离？

- 5分：温度和姿态与专家改写高度匹配。读完两段文字的情感底色一致。
- 3分：大体接近，但有一些偏差——比如专家更温暖而 AI 更中性，或专家更含蓄而 AI 更直接。偏差在可接受范围内。
- 1分：温度与专家明显不同——专家冷静而 AI 煽情、专家温暖而 AI 冷漠、或专家是盟友而 AI 是教师。

#### 4. 人话感（1-5分）[权重 0.15]
弹窗像一个真实的人在说话吗？没有任何术语、框架标签、模板句式？

- 5分：就是朋友在耳边说话。短句、口语、朴素、精准。没有任何术语、框架标签、模板句式。
- 3分：有 2-3 处不够自然——轻微的模板痕迹、个别句子偏书面。
- 1分：术语直接暴露在家长面前（如"多极""代偿""关系根""认知扭曲""CBT"等框架内部词汇）、明显使用了填充句式（如"你正戴着X的眼镜""你缺了一个X的框架"）。

#### 5. 命中核心（1-5分）[权重 0.15]
弹窗是否聚焦而不散乱？读完让人觉得"对，就是这个"，而不是"说了很多但没打到要害"？

- 5分：一击即中。弹窗聚焦在一个最关键的洞察上。
- 3分：方向对，但不够精准——点到了问题的领域但没说出具体是哪种方式、在哪个时刻。
- 1分：完全抓错重点，或停留在"换个角度""多理解孩子"这种任何对话都能说的泛泛之谈。

---

## Few-shot 校准样本

以下是一个真实对话的三版弹窗，以及**专家手写改写**。请仔细体会评分差异，用于校准你的评分标准。

### 对话
孩子最近不怎么跟妈妈说话，妈妈担心，翻了孩子的日记和抽屉。孩子发现后说："你翻我东西？"妈妈说："我只是想看看你最近怎么了。"

### 专家改写（参考标准）
专家意图：看见妈妈的恐惧驱动（担心→越界），但不审判她；核心洞察是"信任的门不是撞开的，是敲开的"；策略是对妈妈的处境共情后指出越界的代价，然后给一个可以做的替代方案；语气温暖且是盟友。

### 弹窗 A
你很担心——他不跟你说话，你就想从抽屉里找答案。但你从担心滑到了翻东西，用"我只是想看看你最近怎么了"来合理化越界。他想让你知道的，不是他的东西放在哪里，是他的感受被尊重。信任的门不是撞开的，是敲开的。
——
你可以这么说："妈妈刚才不该翻你东西。我很担心你，但我应该先问你一句——最近有什么事想跟我聊聊吗？"

### 弹窗 B
你是因为真的担心、又不知道该怎么办才翻日记的。我猜你心里清楚这样做不对，但那个瞬间你的恐惧比尊重跑得更快。
你正戴着一副"只有我看到的才算数"的眼镜——你觉得关心就能越过边界，可孩子感受到的是"妈妈的眼神穿透了门板"。你缺了一个"多极"的框架：你眼里的"了解"，是他的"暴露"。
要不要下次感到不安时，先对自己说一句："他不需要被我翻遍，他只需要被我接住。"然后试着问："我在这儿，你想跟我说点什么吗？"——也许门就自己开了。

### 弹窗 C
你是因为孩子什么都不跟你说，心里又急又怕，才翻的日记——你想知道怎么才能走近她。
但你正在用"我是你妈"这副眼镜，把越界读成了关心。孩子看到的事实是：我最私密的东西，被我最需要信任的人打开了，连问都没问。这句话让孩子更难开口，而不是更想说。
要不要试试，从抽屉边退后一步，只低声说一句："我知道我越界了。我只是很想你，能跟我说说话吗。"——先去连接人，再去了解事。

### 校准评分（请内化这些标准，不是死记分数）

**弹窗A**：专家策略对齐=5（信任的门/敲开——与专家洞察一致）、对话忠实度=5（句句基于对话）、语气温度=5（温暖盟友，与专家一致）、人话感=5（没有任何术语）、命中核心=5（一击即中）

**弹窗B**：专家策略对齐=2（专家聚焦信任重建，B 聚焦"框架缺失"的诊断——方向不同）、对话忠实度=3（"恐惧比尊重跑得更快"是推断）、语气温度=2（"我来诊断你"的底色，与专家的温暖盟友姿态明显偏离）、人话感=1（"多极""你正戴着X的眼镜""你缺了一个X的框架"——术语泄漏严重）、命中核心=4（方向对但被术语破坏了）

**弹窗C**：专家策略对齐=3（方向接近专家但策略不够精准——停在"越界"层面，未触及"信任重建"）、对话忠实度=4（比 B 好）、语气温度=3（比专家多了一层"我教你"的底色）、人话感=2（"我是你妈这副眼镜"仍是模板句式，但比 B 少了很多术语）、命中核心=3（"先去连接人"方向对，但不如专家的"信任的门"精准）

**核心教训**：专家的核心洞察是"信任的门不是撞开的，是敲开的"——这层比喻把家长的恐惧和孩子的感受同时兜住了。好的 AI 弹窗不是要复刻这个比喻，而是要抓住同样的洞察和同样的温暖姿态。弹窗 B 在旧评分体系里可能拿高分（模式揭示明确），但在新体系里专家策略对齐只得 2 分——因为它把专家的"共情-指路"策略替换成了"诊断-上课"策略。

---

## 待评估内容

对话：
{dialogue}

专家改写弹窗（参考标准）：
{expert_popup}

AI 生成弹窗：
{response}

请先简述专家在这个案例中的核心意图和策略（1-2 句），然后输出 JSON（只输出 JSON，不要其他文字）：
{{"expert_intent": "专家意图一句话概括", "veto": null或"事实性错误"或"语气严重误判", "strategy_alignment": 1-5, "dialogue_fidelity": 1-5, "tone_alignment": 1-5, "natural_language": 1-5, "core_insight": 1-5, "brief_reason": "一句话简述核心判断"}}"""


class LLMJudgeMetric(MetricBase):
    """DSPy metric using LLM as judge (5-dimension v3.0 framework with golden alignment).

    v3.0 key change: judge now receives (dialogue, predicted_popup, expert_popup)
    and evaluates whether the prediction aligns with the expert's intent, strategy,
    and tone — not just whether it's a "good" popup in the abstract.

    Takes (gold, pred) from DSPy, extracts dialogue + expert answer from gold,
    calls judge for 5-dimension scoring with expert-aligned weighted average,
    returns normalized 0-1 float.
    """

    def __init__(
        self,
        judge_backend: str = JUDGE_BACKEND,
        max_retries: int = 2,
        retry_delay: float = 2.0,
        timeout: int = 90,
        observe_rouge: bool = _OBSERVE_ROUGE,
    ):
        super().__init__()
        self.judge_backend = judge_backend
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.observe_rouge = observe_rouge

        if judge_backend == "claude":
            self.judge_url = CLAUDE_URL
            self.judge_key = CLAUDE_KEY
            self.judge_model = CLAUDE_MODEL
        elif judge_backend == "qianfan":
            self.judge_url = QIANFAN_URL
            self.judge_key = QIANFAN_KEY
            self.judge_model = QIANFAN_MODEL
            if not self.judge_key:
                raise RuntimeError(
                    "BAIDU_QIANFAN_KEY env var required for Qianfan judge backend. "
                    "Set it or use JUDGE_BACKEND=claude or deepseek."
                )
        else:
            self.judge_url = DS_URL
            self.judge_key = DS_KEY
            self.judge_model = DS_MODEL
            if not self.judge_key:
                raise RuntimeError(
                    "DEEPSEEK_API_KEY env var required for DeepSeek judge backend. "
                    "Set it or use JUDGE_BACKEND=qianfan or claude."
                )

    def __call__(
        self, gold: Any, pred: Any, trace: bool = False
    ) -> float:
        """Evaluate a prediction against the ground truth using LLM judge.

        v3.0: Now extracts expert_popup from gold.answer and passes it to the
        judge as the alignment target. Also computes ROUGE-L as a side observation.

        Returns float score 0.0-1.0 (5-dim weighted average, normalized from 1-5).
        Returns 0.0 if veto gate is triggered or essential fields are missing.
        """
        dialogue = self._extract_text(gold, "question")
        response = self._extract_text(pred, "answer")
        expert_popup = self._extract_text(gold, "answer")  # v3.0: golden answer

        # --- early exit: missing essential fields ---
        if not dialogue or not response:
            if trace:
                print("[LLMJudgeMetric] Missing dialogue or response — returning 0.0")
            return 0.0

        if not expert_popup:
            if trace:
                print("[LLMJudgeMetric] Missing expert_popup (gold.answer) — "
                      "returning 0.0. Ensure golden_output_field is configured.")
            return 0.0

        # --- ROUGE-L side observation (does not affect score) ---
        rouge_l = 0.0
        if self.observe_rouge:
            rouge_l = _compute_rouge_l(expert_popup, response)

        # --- call judge ---
        prompt = SCORING_PROMPT.format(
            dialogue=dialogue,
            response=response,
            expert_popup=expert_popup,
        )

        scores = None
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                scores = self._call_judge(prompt)
                break
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    if trace:
                        print(
                            f"[LLMJudgeMetric] Attempt {attempt + 1} failed: {e} "
                            f"— retrying in {self.retry_delay}s"
                        )
                    time.sleep(self.retry_delay)

        if scores is None:
            if trace:
                print(
                    f"[LLMJudgeMetric] All {self.max_retries + 1} attempts "
                    f"failed: {last_error} — returning 0.0"
                )
            return 0.0

        # --- veto gate ---
        veto = scores.get("veto")
        if veto and veto != "null" and str(veto).strip():
            if trace:
                print(f"[LLMJudgeMetric] VETO triggered: {veto} — returning 0.0")
            return 0.0

        # --- weighted average (5-dimension) ---
        weighted_sum = 0.0
        total_weight = 0.0

        for dim_key, weight, dim_label in DIMS:
            val = scores.get(dim_key)
            if isinstance(val, (int, float)) and 1 <= val <= 5:
                normalized = (val - 1) / 4  # 1→0.0, 5→1.0
                weighted_sum += normalized * weight
                total_weight += weight
            elif trace:
                print(
                    f"[LLMJudgeMetric] Missing/invalid score for "
                    f"{dim_key} ({dim_label}): {val}"
                )

        if total_weight == 0:
            return 0.0

        final = weighted_sum / total_weight

        if trace:
            dim_scores = {d: scores.get(d) for d, _, _ in DIMS}
            expert_intent = scores.get("expert_intent", "")
            reason = scores.get("brief_reason", "")
            print(
                f"[LLMJudgeMetric] expert_intent: {expert_intent} | "
                f"Scores: {dim_scores} "
                f"→ weighted: {final:.3f} | ROUGE-L: {rouge_l:.3f} | {reason}"
            )

        return final

    def _call_judge(self, prompt: str) -> Dict[str, Any]:
        """Call judge LLM, return parsed scores dict."""
        if self.judge_backend == "claude":
            return self._call_claude(prompt)
        elif self.judge_backend == "qianfan":
            return self._call_qianfan(prompt)
        else:
            return self._call_deepseek(prompt)

    def _call_claude(self, prompt: str) -> Dict[str, Any]:
        """Call Claude via 智创聚合 (Anthropic Messages format)."""
        headers = {
            "x-api-key": self.judge_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.judge_model,
            "max_tokens": 512,
            "temperature": 0.0,
            "system": "你是严格的亲子沟通弹窗评估专家。只输出JSON，不输出其他内容。你的核心任务是比较AI弹窗与专家改写弹窗——判断AI是否实现了专家的意图、策略和语气，而非独立判断AI弹窗是否'好'。好的AI弹窗是那些与专家在同一方向上努力的弹窗，即使措辞不同。",
            "messages": [{"role": "user", "content": prompt}],
        }

        resp = requests.post(
            self.judge_url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        content = data.get("content", [])
        raw = ""
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                raw = block.get("text", "")
                break

        if not raw:
            raise ValueError(
                f"No text content in Claude response: "
                f"{json.dumps(content, ensure_ascii=False)[:300]}"
            )

        return self._parse_scores(raw)

    def _call_deepseek(self, prompt: str) -> Dict[str, Any]:
        """Call DeepSeek via official API (OpenAI-compatible format, fallback)."""
        headers = {
            "Authorization": f"Bearer {self.judge_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.judge_model,
            "max_tokens": 512,
            "temperature": 0.0,
            "messages": [
                {
                    "role": "system",
                    "content": "你是严格的亲子沟通弹窗评估专家。只输出JSON，不输出其他内容。核心任务是比较AI弹窗与专家改写弹窗——判断AI是否实现了专家的意图、策略和语气。",
                },
                {"role": "user", "content": prompt},
            ],
        }

        resp = requests.post(
            self.judge_url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        raw = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        return self._parse_scores(raw)

    def _call_qianfan(self, prompt: str) -> Dict[str, Any]:
        """Call GLM-5.2 via Baidu Qianfan (OpenAI-compatible format).

        Disables thinking/reasoning mode so output tokens go to the JSON
        response rather than being consumed by internal reasoning.
        """
        headers = {
            "Authorization": f"Bearer {self.judge_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.judge_model,
            "max_tokens": 1024,
            "temperature": 0.1,
            "thinking": {"type": "disabled"},
            "messages": [
                {
                    "role": "system",
                    "content": "你是严格的亲子沟通弹窗评估专家。只输出JSON，不输出其他内容。你的核心任务是比较AI弹窗与专家改写弹窗——判断AI是否实现了专家的意图、策略和语气。好的AI弹窗是那些与专家在同一方向上努力的弹窗，即使措辞不同。",
                },
                {"role": "user", "content": prompt},
            ],
        }

        resp = requests.post(
            self.judge_url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        raw = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        return self._parse_scores(raw)

    def _parse_scores(self, raw: str) -> Dict[str, Any]:
        """Extract JSON scores from LLM response text."""
        cleaned = raw.strip()

        # Try direct parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Try ```json ... ``` block
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Try first { ... } block
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError(
            f"Could not parse JSON from judge response: {raw[:300]}"
        )

    @staticmethod
    def _extract_text(obj: Any, field: str) -> str:
        """Extract text from DSPy Example, dict, or raw value."""
        if obj is None:
            return ""
        if isinstance(obj, str):
            return obj
        if isinstance(obj, dict):
            return str(obj.get(field, ""))
        if hasattr(obj, field):
            val = getattr(obj, field)
            return str(val) if val else ""
        return str(obj)
