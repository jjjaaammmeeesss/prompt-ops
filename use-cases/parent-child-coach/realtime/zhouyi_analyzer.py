"""Stage 1 — 周易三爻八卦状态分类器。

使用 LLM 分析亲子对话中家长的三爻状态（下/中/上），
识别八卦卦象，评估风险等级，输出结构化的 ZhouYiState。
"""

import json
import logging
import re
import time
from typing import Any, Dict, Optional

from .output_schemas import (
    YaoState,
    Trigram,
    PopupTone,
    ZhouYiState,
)
from .zhouyi_prompts import (
    ZHOUYI_ANALYZER_SYSTEM,
    build_analyzer_user_prompt,
)

logger = logging.getLogger("prompt_ops.realtime.zhouyi_analyzer")


class ZhouYiAnalyzer:
    """Stage 1：周易八卦状态分类器。

    将对话窗口文本送入 LLM，输出三爻状态和八卦分类。
    针对快速/便宜模型优化，输出控制在 ~200 tokens。

    Usage:
        analyzer = ZhouYiAnalyzer(model_adapter)
        state = analyzer.analyze(dialogue_window)
        print(state.trigram.symbol, state.trigram.chinese_name)
    """

    # 默认配置
    DEFAULT_TEMPERATURE = 0.0
    DEFAULT_MAX_TOKENS = 256
    DEFAULT_TIMEOUT = 15.0  # 秒

    def __init__(
        self,
        model_adapter,  # LiteLLMModelAdapter or similar
        temperature: float = None,
        max_tokens: int = None,
        timeout: float = None,
    ):
        """初始化分析器。

        Args:
            model_adapter: LLM 适配器（需实现 generate_with_chat_format 方法）
            temperature: LLM 温度（默认 0.0，分类任务不需要创造性）
            max_tokens: 最大输出 token
            timeout: API 调用超时秒数
        """
        self.model = model_adapter
        self.temperature = temperature or self.DEFAULT_TEMPERATURE
        self.max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS
        self.timeout = timeout or self.DEFAULT_TIMEOUT

        # 确保模型有 generate_with_chat_format 方法
        if not hasattr(self.model, "generate_with_chat_format"):
            logger.warning(
                "Model adapter does not have generate_with_chat_format; "
                "falling back to generate()"
            )

    def analyze(self, dialogue_window: str) -> ZhouYiState:
        """分析对话窗口，返回周易状态分类。

        Args:
            dialogue_window: 待分析的对话文本（最近 N 轮）

        Returns:
            ZhouYiState: 包含卦象、爻状态、风险等信息的结构化结果
        """
        if not dialogue_window or len(dialogue_window.strip()) < 20:
            return self._default_state(
                reason="对话文本过短，无法进行有意义的分析"
            )

        try:
            raw_json = self._call_llm(dialogue_window)
            state = self._parse_response(raw_json)
            state = self._validate_state(state)
            logger.info(
                f"ZhouYi analysis: {state.trigram.symbol} {state.trigram.chinese_name} "
                f"({state.trigram.yao_pattern}) | {state.container_status} | "
                f"risk={state.risk_level} | conf={state.confidence:.2f}"
            )
            return state

        except Exception as e:
            logger.error(f"ZhouYi analysis failed: {e}", exc_info=True)
            return self._default_state(reason=f"分析失败: {str(e)[:80]}")

    def _call_llm(self, dialogue: str) -> str:
        """调用 LLM 进行分析，返回原始响应文本。"""
        user_prompt = build_analyzer_user_prompt(dialogue)
        messages = [
            {"role": "system", "content": ZHOUYI_ANALYZER_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]

        start = time.time()

        if hasattr(self.model, "generate_with_chat_format"):
            raw = self.model.generate_with_chat_format(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        else:
            # 回退：拼接 system + user 为单个 prompt
            combined = f"{ZHOUYI_ANALYZER_SYSTEM}\n\n{user_prompt}"
            raw = self.model.generate(
                prompt=combined,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

        elapsed = time.time() - start
        logger.debug(f"LLM analysis call took {elapsed:.2f}s")
        return raw

    def _parse_response(self, raw: str) -> ZhouYiState:
        """解析 LLM 的 JSON 响应为 ZhouYiState。

        处理多种可能的格式：纯JSON、```json 代码块、含杂文。
        """
        cleaned = raw.strip()

        # 尝试 1: 直接解析
        try:
            data = json.loads(cleaned)
            return self._dict_to_state(data)
        except json.JSONDecodeError:
            pass

        # 尝试 2: ```json ... ``` 代码块
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1).strip())
                return self._dict_to_state(data)
            except json.JSONDecodeError:
                pass

        # 尝试 3: 提取第一个 {...} 块
        match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                return self._dict_to_state(data)
            except json.JSONDecodeError:
                pass

        # 尝试 4: 查找更深层嵌套的 JSON
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                return self._dict_to_state(data)
            except json.JSONDecodeError:
                pass

        raise ValueError(f"无法从LLM响应中解析JSON: {raw[:200]}")

    def _dict_to_state(self, data: Dict[str, Any]) -> ZhouYiState:
        """将解析后的字典转换为 ZhouYiState。"""
        # 解析爻状态
        lower = self._parse_yao(data.get("下爻", ""))
        middle = self._parse_yao(data.get("中爻", ""))
        upper = self._parse_yao(data.get("上爻", ""))

        # 查卦象
        try:
            trigram = Trigram.from_yao_states(lower, middle, upper)
        except ValueError:
            # 回退：基于多数爻状态猜测
            trigram = self._fallback_trigram(lower, middle, upper)

        # 解析建议类型
        tone_str = data.get("建议类型", "诊断式")
        if "鼓励" in tone_str:
            suggested_tone = PopupTone.ENCOURAGING
        else:
            suggested_tone = PopupTone.DIAGNOSTIC

        return ZhouYiState(
            trigram=trigram,
            lower_yao=lower,
            middle_yao=middle,
            upper_yao=upper,
            container_status=str(data.get("容器判定", "不适用")),
            risk_level=str(data.get("风险", "低")),
            suggested_tone=suggested_tone,
            confidence=float(data.get("置信度", 0.5)),
            brief_reason=str(data.get("一句话", "")),
        )

    @staticmethod
    def _parse_yao(value: str) -> YaoState:
        """将字符串解析为 YaoState。"""
        v = str(value).strip()
        if v in ("控", "掌控", "阴"):
            return YaoState.RONG_QI
        if v in ("失", "失控", "阳"):
            return YaoState.SHI_KONG
        # 回退：包含"失"→失控，否则默认掌控
        if "失" in v:
            return YaoState.SHI_KONG
        logger.warning(f"Unknown yao value '{v}', defaulting to 控")
        return YaoState.RONG_QI

    @staticmethod
    def _fallback_trigram(lower: YaoState, middle: YaoState,
                          upper: YaoState) -> Trigram:
        """当 LLM 输出无法精确匹配时的回退逻辑。"""
        try:
            return Trigram.from_yao_states(lower, middle, upper)
        except ValueError:
            # 基于失控数量做最佳猜测
            loss_count = sum(
                1 for y in (lower, middle, upper) if y == YaoState.SHI_KONG
            )
            if loss_count == 0:
                return Trigram.KUN
            elif loss_count == 1:
                # 需要判断具体位置
                if upper == YaoState.SHI_KONG:
                    return Trigram.ZHEN
                elif middle == YaoState.SHI_KONG:
                    return Trigram.KAN
                else:
                    return Trigram.GEN
            elif loss_count == 2:
                if lower == YaoState.RONG_QI:
                    return Trigram.DUI
                elif middle == YaoState.RONG_QI:
                    return Trigram.LI
                else:
                    return Trigram.XUN
            else:
                return Trigram.QIAN

    def _validate_state(self, state: ZhouYiState) -> ZhouYiState:
        """验证和修正 LLM 输出。

        检查项目：
        1. 卦象与爻状态一致
        2. confidence 在有效范围
        3. 纯稳态坤卦的容器判定应为"不适用"
        4. 乾卦必须有容器判定（安全型 vs 危险型）
        """
        # 验证爻-卦一致性
        try:
            expected = Trigram.from_yao_states(
                state.lower_yao, state.middle_yao, state.upper_yao
            )
            if expected != state.trigram:
                logger.warning(
                    f"Trigram mismatch: got {state.trigram.chinese_name}, "
                    f"expected {expected.name} from yao states. Correcting."
                )
                state.trigram = expected
        except ValueError:
            state.trigram = self._fallback_trigram(
                state.lower_yao, state.middle_yao, state.upper_yao
            )

        # 钳制 confidence
        state.confidence = max(0.0, min(1.0, state.confidence))

        # 统一风险等级为中文（v4.0.14：消除中英文混用导致的门控失效）
        _risk_normalize = {"low": "低", "medium": "中", "high": "高"}
        state.risk_level = _risk_normalize.get(
            str(state.risk_level).strip().lower(), state.risk_level
        )

        # 纯稳态坤卦时修正容器判定
        if state.is_stable:
            if state.container_status in ("有容器", "无容器"):
                state.container_status = "不适用"

        # 乾卦确保有容器判定
        if state.is_full_release and state.container_status == "不适用":
            state.container_status = "无容器"  # 默认保守判定
            state.risk_level = "高"

        return state

    def _default_state(self, reason: str = "") -> ZhouYiState:
        """返回默认/安全的分析状态（坤卦 - 稳定）。"""
        return ZhouYiState(
            trigram=Trigram.KUN,
            lower_yao=YaoState.RONG_QI,
            middle_yao=YaoState.RONG_QI,
            upper_yao=YaoState.RONG_QI,
            container_status="不适用",
            risk_level="low",
            suggested_tone=PopupTone.ENCOURAGING,
            confidence=0.1,
            brief_reason=reason or "默认安全状态",
        )
