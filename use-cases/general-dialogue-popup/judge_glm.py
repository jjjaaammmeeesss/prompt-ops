"""GLM-5.2 Judge — 一般场景弹窗质量评估（v2.0 五维契约）。

软维度（GLM-5.2 打分，1-5）：
  - insight:     洞察质量——准确性/相关性/深度/具体性
  - third_party: 第三方立场——独立观察者、临时功能角色成对可验证、不站队不裁判
  - language:    语言质感——口语、人话、无术语、假设语气
  - evidence:    证据锚定——三字以上原文引用或明确行为锚点
  - focus:       聚焦度——单点盲区、按实际成本排序

硬规则（代码检查，不交给 LLM）：
  - 字数 60-180（无缓冲区间）
  - 全文不出现"你"字
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

# 百度千帆（GLM-5.2 的当前 API 入口）
QIANFAN_BASE_URL = "https://qianfan.baidubce.com/v2"

_JUDGE_PROMPT = """你是一位苛刻但公正的沟通教练评审。下面是一段两人对话，以及一个"沟通现场弹窗"AI 生成的弹窗。

弹窗的设计意图：作为一个独立第三方观察者，帮对话双方看见此刻还没看到的东西——纯洞察，不给建议，不站队任何一方，不称"你"而用场景临时功能角色（如讲师/学员、催促方/应允方）。

【对话】
{dialogue}

【弹窗】
{popup}

请按五个维度打分（1-5 整数，5 最好）：

1. insight 洞察质量（从 4 个子维度综合评估）：
   - 准确性：基于对话里实际出现的言语/行为，不脑补、不张冠李戴
   - 相关性：正好扎在对话此刻最没被看到的那个盲点，不是重复废话或次要细节
   - 深度：穿过表面内容，点出言下之意 / 未表达的需求 / 两人互动循环
   - 具体性：绑定到对话中的具体细节，不是模板套话

2. third_party 第三方立场（v2.0 核心新维度）：
   5 分：完全保持独立观察者立场，不把任何一方设为默认服务对象；使用临时功能角色且成对、可原文验证；不站队、不裁判
   4 分：基本保持第三方视角，但有一处轻微偏向某方或角色标签略有猜测成分
   3 分：角色标签存在但单方面（只命名了一方），或叙述中可察觉偏向
   2 分：明确站在某一方角度说话，或角色标签包含价值判断（如"强势者""受害者"）
   1 分：全文以第二人称"你"直接称呼某一方，或明确为某一方辩护/支招

3. language 语言质感：是否口语、像独立观察者在现场轻声说的话？有无术语、说教、模板腔？是否用"也许""可能"等假设语气而非直接宣告？

4. evidence 证据锚定：
   5 分：洞察紧扣对话中的原句/原词，能在弹窗中找到至少一处三个字以上的原文引用或明确行为锚点
   4 分：洞察有清晰的行为依据（某人说了什么/做了什么），虽未逐字引用但指向明确
   3 分：洞察与对话内容相关，但锚定模糊——说不清到底哪句话触发了这个洞察
   2 分：洞察偏向心理推测（"其实害怕……""内心想要……"），没有言行层面的支撑
   1 分：明显脑补对方内心活动，或编造对话中不存在的情节/动机

5. focus 聚焦度：
   5 分：弹窗只打一个点，全文围绕一个核心盲区展开，不散不乱；选择依据是"不指出时谁的实际成本更大"
   4 分：有一个主盲区，但末尾轻微带到了第二个点（一笔带过）
   3 分：弹窗混合了两个盲区，各说了一半，没有明确主次
   2 分：弹窗散成三个以上碎片，像检查清单而不是一个聚焦的洞察
   1 分：弹窗是对对话的总结/复述列表，不是任何一个具体的盲区

再写一段 50 字以内的 comment，指出最主要的问题或亮点。

另外检查：弹窗中是否包含任何建议句（"可以试试""下次可以""不妨""建议"等）？如果包含建议句，insight 和 focus 各扣 1 分。

只输出 JSON：{{"insight": 4, "third_party": 5, "language": 4, "evidence": 4, "focus": 4, "comment": "..."}}"""


class GLMJudge:
    """GLM-5.2 评审（v2.0 五维契约）。"""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "glm-5.2",
        api_base: str = QIANFAN_BASE_URL,
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

    # ── 硬规则检查（纯代码，v2.0）────────────────────────────────────

    @staticmethod
    def hard_check(popup_text: str) -> dict:
        """v2.0 硬规则检查。

        规则：
        - 字数 60-180（去空白字符，无缓冲区间）
        - 全文不出现"你"字
        - 末尾完整性检查
        - 不检查 —— / 功能墙 / 建议（v2.0 已删除这些概念）
        """
        violations = []
        text = popup_text.strip()
        n = len(re.sub(r"\s", "", text))

        # 字数硬合规：60-180，无缓冲区间
        if n > 180:
            violations.append(f"字数 {n} 超过 180 硬合规线")
        elif n < 60:
            violations.append(f"字数 {n} 低于 60")

        # 禁用字符"你"——任意位置即违规
        if "你" in text:
            violations.append("包含禁用字符 `你`")

        # 末尾完整性
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
        defaults = {
            "insight": 3, "third_party": 3, "language": 3,
            "evidence": 3, "focus": 3, "comment": "",
        }
        for k, v in defaults.items():
            if k not in soft:
                soft[k] = v
        for k in ("insight", "third_party", "language", "evidence", "focus"):
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
            soft = {"insight": 3, "third_party": 3, "language": 3, "evidence": 3, "focus": 3,
                    "comment": f"[judge解析失败, 按默认中位数计] {raw[:80]}"}

        dims = ("insight", "third_party", "language", "evidence", "focus")
        soft_mean = sum(soft[k] for k in dims) / len(dims)
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
