"""星鸾 Claude Opus 评审 — 作为 GLM-5.2 的替代/备用裁判。

复用 GLMJudge 的硬规则与评分维度，仅把软评分调用从 GLM 换成 Anthropic 原生 API。
"""

from __future__ import annotations

import json
import logging
import os

from judge_glm import GLMJudge, _JUDGE_PROMPT

logger = logging.getLogger("ClaudeJudge")


class ClaudeJudge:
    """星鸾 Claude Opus 评审器（Anthropic API 兼容）。"""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-opus-4-7",
        api_base: str = "https://luanapi.xingluan.cn",
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
        if not self.api_key:
            raise ValueError("缺少 Anthropic/Xingluan API token（传参或设置 ANTHROPIC_AUTH_TOKEN）")
        self.model = model
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens

    # 直接复用 GLMJudge 的硬规则
    hard_check = staticmethod(GLMJudge.hard_check)

    def _call_claude(self, prompt: str, *, retry_once: bool = True) -> str:
        import anthropic

        client = anthropic.Anthropic(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=180,
        )
        resp = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text = ""
        for block in resp.content:
            if block.type == "text":
                text = block.text
                break

        if not retry_once or GLMJudge._looks_like_json(text):  # type: ignore
            return text

        fix_resp = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0.0,
            messages=[{
                "role": "user",
                "content": "请把下面这段评审结果整理成严格 JSON 格式。只输出 JSON，不要解释。\n\n" + text,
            }],
        )
        for block in fix_resp.content:
            if block.type == "text":
                return block.text
        return ""

    def judge_case(self, case: dict, popups: list[dict]) -> dict:
        """评估单个案例。接口与 GLMJudge.judge_case 保持一致。"""
        expect = case.get("expect", "")
        popup_text = popups[0]["text"] if popups else None

        # 安静类案例：不弹为对
        if expect == "安静":
            if popup_text is None:
                return {
                    "aggregate": 5.0,
                    "soft": None,
                    "hard": {"pass": True, "violations": []},
                    "comment": "正确保持安静",
                    "popup_text": None,
                }
            return {
                "aggregate": 1.0,
                "soft": None,
                "hard": {"pass": False, "violations": ["健康对话误弹"]},
                "comment": "健康对话不应弹窗（质量优先，宁缺毋滥）",
                "popup_text": popup_text,
            }

        # 应弹未弹
        if popup_text is None:
            return {
                "aggregate": 1.5,
                "soft": None,
                "hard": {"pass": False, "violations": ["应弹未弹"]},
                "comment": "对话中存在值得指出的盲区但未弹窗",
                "popup_text": None,
            }

        hard = self.hard_check(popup_text)

        raw = self._call_claude(_JUDGE_PROMPT.format(dialogue=case["dialogue"], popup=popup_text))
        soft = GLMJudge._parse_judge_output(raw)  # type: ignore

        if soft is None:
            logger.warning("judge 输出解析失败 | raw=%s", raw[:200])
            soft = {
                "insight": 3,
                "suggestion": 3,
                "non_judgment": 3,
                "language": 3,
                "comment": f"[judge解析失败, 按默认中位数计] {raw[:80]}",
            }

        soft_mean = sum(soft[k] for k in ("insight", "suggestion", "non_judgment", "language")) / 4
        penalty = min(1.5, 0.5 * len(hard["violations"]))
        aggregate = round(max(1.0, soft_mean - penalty), 2)

        return {
            "aggregate": aggregate,
            "soft": soft,
            "hard": hard,
            "comment": soft.get("comment", ""),
            "popup_text": popup_text,
        }
