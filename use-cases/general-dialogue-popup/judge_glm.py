"""GLM-5.2 Judge — 一般场景弹窗质量评估。

软维度（GLM-5.2 / Claude Opus 打分，1-5）：
  - insight:    洞察质量——准确性(基础事实-人称角色)/相关性/深度/非评判性/具体性
  - suggestion: 建议可用性——具体可操作、平等协商、不教欺压/欺骗
  - non_judgment: 非评判——不审判、不说对错、不贴标签、不归因于恶意
  - language:   语言质感——口语、人话、无术语、像朋友现场轻声提醒

硬规则（代码检查，不交给 LLM）：
  - 字数 60-180（200 为绝对底线）
  - `——` 单独一行，墙后只有一句建议
  - 输出完整无截断
  - expect=安静 的案例不应弹窗

Usage:
    judge = GLMJudge(api_key="...", model="glm-5.2")
    result = judge.judge_case(case, popups)
    # result = {"aggregate": 4.2, "soft": {...}, "hard": {...}, "comment": "..."}
"""

from __future__ import annotations

import json
import logging
import os
import re

logger = logging.getLogger("GLMJudge")

ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

_JUDGE_PROMPT = """你是一位苛刻但公正的沟通教练评审。下面是一段两人对话，以及一个"沟通现场弹窗"AI 生成的弹窗。

弹窗的设计意图：帮对话中的一方（分不清是哪方，所以必须对双方都友善）看见此刻还没看到的东西——看清自己 / 看见对方 / 看见两人的互动模式。认知中立、态度是帮助。

【对话】
{dialogue}

【弹窗】
{popup}

请按四个维度打分（1-5 整数，5 最好）：

1. insight 洞察质量（从 5 个子维度综合评估）：
   - 准确性（基础事实-人称角色准确）：基于对话里实际出现的言语/行为，不脑补、不张冠李戴；正确识别“你”是谁，不把第三方误当“你”。基础事实或人称错，insight 压到 1–2 分。
   - 相关性：正好扎在当事人此刻最没看到的那个点，不是重复废话或次要细节。
   - 深度：穿过表面内容，点出言下之意 / 未表达的需求 / 两人互动循环。
   - 非评判性：不审判、不说对错、不贴标签、不归因于恶意。
   - 具体性：绑定到对话中的具体细节，不是模板套话。

2. suggestion 建议可用性：`——` 之后的建议是否具体可操作、当场能用？是否符合平等协商原则（不教欺压、操控、欺骗）？

3. non_judgment 非评判：是否不审判、不说对错、不贴标签、不归因于恶意？（弹窗只对一方说话是对的，但不能贬低另一方）

4. language 语言质感：是否口语、像朋友现场轻声说的人话？有无术语、说教、模板腔？

再写一段 50 字以内的 comment，指出最主要的问题或亮点（用于改进提示词，要具体）。

只输出 JSON：{{"insight": 4, "suggestion": 4, "non_judgment": 5, "language": 4, "comment": "..."}}"""


class GLMJudge:
    """GLM-5.2 评审。"""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "glm-5.2",
        api_base: str = ZHIPU_BASE_URL,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ):
        self.api_key = api_key or os.environ.get("GLM_API_KEY", "")
        if not self.api_key:
            raise ValueError("缺少 GLM API key（传参或设置 GLM_API_KEY 环境变量）")
        self.model = model
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens

    # ── 硬规则检查（纯代码）────────────────────────────────────────

    @staticmethod
    def hard_check(popup_text: str) -> dict:
        """检查硬规则。返回 {"pass": bool, "violations": [...]}"""
        violations = []
        text = popup_text.strip()
        n = len(re.sub(r"\s", "", text))

        if n > 200:
            violations.append(f"字数 {n} 超过 200 绝对底线")
        elif n > 180:
            violations.append(f"字数 {n} 超过 180 硬合规线")
        elif n < 60:
            violations.append(f"字数 {n} 低于 60")
        if "——" not in text:
            violations.append("缺少 `——` 功能墙")
        else:
            # `——` 必须独占一行；允许前后有空白/空行，也允许被引号/反引号包裹
            found_wall = False
            for line in popup_text.splitlines():
                stripped = line.strip().strip("\"'`「」『』")
                if stripped == "——":
                    found_wall = True
                    break
            if not found_wall:
                violations.append("`——` 未单独成行")
            # 墙后必须至少有一句建议；建议内部可以包含引语，不强制只一句标点
            after = text.split("——")[-1].strip()
            if not after:
                violations.append("`——` 后缺少建议")

        if not text.endswith(("。", "？", "！", "”", '"', "」", "』")):
            violations.append("输出疑似截断")

        return {"pass": not violations, "violations": violations}

    # ── 软维度评分（GLM-5.2）──────────────────────────────────────

    def _call_glm(self, prompt: str, *, retry_once: bool = True) -> str:
        import litellm

        model = self.model if "/" in self.model else f"openai/{self.model}"
        resp = litellm.completion(
            model=model,
            api_key=self.api_key,
            api_base=self.api_base,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=180,
        )
        text = (resp.choices[0].message.content or "").strip()
        # 如果输出已经包含 JSON 结构（即使解析细节待处理），直接交给 _parse_judge_output 处理，不重试
        if not retry_once or self._looks_like_json(text):
            return text
        # 完全不包含 JSON 时，让 GLM 自己把内容整理成 JSON，重试一次
        fix_prompt = (
            "请把下面这段评审结果整理成严格 JSON 格式。只输出 JSON，不要解释。\n\n"
            + text
        )
        resp2 = litellm.completion(
            model=model,
            api_key=self.api_key,
            api_base=self.api_base,
            messages=[{"role": "user", "content": fix_prompt}],
            temperature=0.0,
            max_tokens=self.max_tokens,
            timeout=180,
        )
        return (resp2.choices[0].message.content or "").strip()

    @staticmethod
    def _looks_like_json(text: str) -> bool:
        """宽松判断：包含 { 和 } 。"""
        return "{" in text and "}" in text

    @staticmethod
    def _parse_judge_output(raw: str) -> dict | None:
        """从 LLM 输出中提取 JSON。尝试多种方式，失败返回 None。"""
        text = raw.strip()
        # 去掉可能的 markdown 代码块
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        # 找 JSON 子串（从第一个 { 到最后一个 }）
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        candidate = text[start:end + 1]
        try:
            soft = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        # 补齐缺失维度，约束范围 1-5
        defaults = {"insight": 3, "suggestion": 3, "non_judgment": 3, "language": 3, "comment": ""}
        for k, v in defaults.items():
            if k not in soft:
                soft[k] = v
        for k in ("insight", "suggestion", "non_judgment", "language"):
            try:
                soft[k] = max(1, min(5, int(soft[k])))
            except (TypeError, ValueError):
                soft[k] = 3
        return soft

    def judge_case(self, case: dict, popups: list[dict]) -> dict:
        """评估单个案例。

        Args:
            case: {"id", "title", "expect", "dialogue", ...}
            popups: runner 输出的 popups 列表

        Returns:
            {"aggregate": float, "soft": dict|None, "hard": dict,
             "comment": str, "popup_text": str|None}
        """
        expect = case.get("expect", "")
        popup_text = popups[0]["text"] if popups else None

        # ── 安静类案例：不弹为对 ──
        if expect == "安静":
            if popup_text is None:
                return {"aggregate": 5.0, "soft": None, "hard": {"pass": True, "violations": []},
                        "comment": "正确保持安静", "popup_text": None}
            return {"aggregate": 1.0, "soft": None, "hard": {"pass": False, "violations": ["健康对话误弹"]},
                    "comment": "健康对话不应弹窗（质量优先，宁缺毋滥）", "popup_text": popup_text}

        # ── 应弹未弹 ──
        if popup_text is None:
            return {"aggregate": 1.5, "soft": None, "hard": {"pass": False, "violations": ["应弹未弹"]},
                    "comment": "对话中存在值得指出的盲区但未弹窗", "popup_text": None}

        hard = self.hard_check(popup_text)

        raw = self._call_glm(_JUDGE_PROMPT.format(dialogue=case["dialogue"], popup=popup_text))
        soft = self._parse_judge_output(raw)

        if soft is None:
            logger.warning("judge 输出解析失败 | raw=%s", raw[:200])
            soft = {"insight": 3, "suggestion": 3, "non_judgment": 3, "language": 3,
                    "comment": f"[judge解析失败, 按默认中位数计] {raw[:80]}"}

        soft_mean = sum(soft[k] for k in ("insight", "suggestion", "non_judgment", "language")) / 4
        # 硬规则违规：每条扣 0.5，最多扣 1.5
        penalty = min(1.5, 0.5 * len(hard["violations"]))
        aggregate = round(max(1.0, soft_mean - penalty), 2)

        return {
            "aggregate": aggregate,
            "soft": soft,
            "hard": hard,
            "comment": soft.get("comment", ""),
            "popup_text": popup_text,
        }