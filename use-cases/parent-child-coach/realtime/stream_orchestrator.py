"""流式编排器 — 实时对话处理的主引擎。

组件：
  - TextBuffer: 累积文本 + 滑动窗口管理
  - TriggerEngine: 字数触发 + 关键词触发
  - DebounceGate: 弹窗去抖 + 状态跟踪
  - StreamOrchestrator: 串联所有组件的主循环

数据流：
  流式文本 → TextBuffer → TriggerEngine 判断是否触发
  → ZhouYiAnalyzer (Stage 1) → DebounceGate 判断是否弹窗
  → PopupGenerator (Stage 2) → 输出弹窗
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .output_schemas import (
    YaoState,
    Trigram,
    PopupTone,
    ZhouYiState,
    Popup,
    TriggerEvent,
    risk_rank,
)
from .zhouyi_analyzer import ZhouYiAnalyzer
from .popup_generator import PopupGenerator

logger = logging.getLogger("prompt_ops.realtime.orchestrator")


# ============================================================
# TextBuffer — 滑动窗口文本缓冲
# ============================================================

class TextBuffer:
    """累积流式文本，维护滑动分析窗口。

    设计要点：
    - full_text: 完整对话历史（用于关键词匹配）
    - window: 最近 N 字的滑动窗口（用于 LLM 分析）
    - 窗口之间有 lookback 重叠，保证上下文连续性
    """

    def __init__(self, window_size: int = 3000, lookback: int = 500):
        """初始化文本缓冲。

        Args:
            window_size: 分析窗口最大字符数
            lookback: 与上一窗口的重叠字符数
        """
        self.full_text: str = ""
        self.window_size = window_size
        self.lookback = lookback
        self._last_window_end: int = 0  # 上次窗口结束位置

    def append(self, text: str) -> str:
        """追加新文本，返回当前分析窗口。

        Args:
            text: 新到达的文本片段

        Returns:
            str: 当前用于分析的滑动窗口文本
        """
        self.full_text += text
        return self.current_window

    @property
    def total_chars(self) -> int:
        """完整对话总字符数。"""
        return len(self.full_text)

    @property
    def current_window(self) -> str:
        """当前滑动分析窗口。

        从上次窗口结束位置-回看向前，取 window_size 字符。
        """
        start = max(0, self._last_window_end - self.lookback)
        end = min(len(self.full_text), start + self.window_size)
        window = self.full_text[start:end]
        return window

    def mark_window_analyzed(self):
        """标记当前窗口已被分析，下次窗口从此处开始。"""
        self._last_window_end = len(self.full_text)

    def get_recent(self, chars: int = 500) -> str:
        """获取最近 N 个字符。"""
        return self.full_text[-chars:]


# ============================================================
# TriggerEngine — 触发管理
# ============================================================

class TriggerEngine:
    """判断是否触发分析。支持字数触发和关键词触发。

    触发规则（优先级从高到低）：
    1. critical 关键词 → 立即触发
    2. 累积字数 ≥ char_trigger → 触发
    3. warning/opportunity 关键词 + 累积 ≥ 30 字 → 触发
    """

    # 默认配置
    DEFAULT_CHAR_TRIGGER = 120
    DEFAULT_MIN_CHARS = 60
    DEFAULT_MIN_INTERVAL_MS = 3000
    DEFAULT_KEYWORD_MIN_CHARS = 30

    def __init__(
        self,
        char_trigger: int = None,
        min_chars_for_analysis: int = None,
        min_interval_ms: int = None,
        keyword_file: str = None,
        keyword_min_chars: int = None,
    ):
        """初始化触发引擎。

        Args:
            char_trigger: 字数触发阈值
            min_chars_for_analysis: 最少需要累积多少字才触发分析
            min_interval_ms: 两次触发之间的最小间隔（毫秒）
            keyword_file: 关键词配置文件路径（JSON）
            keyword_min_chars: 关键词触发所需的最少累积字数
        """
        self.char_trigger = char_trigger or self.DEFAULT_CHAR_TRIGGER
        self.min_chars_for_analysis = (
            min_chars_for_analysis or self.DEFAULT_MIN_CHARS
        )
        self.min_interval_ms = min_interval_ms or self.DEFAULT_MIN_INTERVAL_MS
        self.keyword_min_chars = (
            keyword_min_chars or self.DEFAULT_KEYWORD_MIN_CHARS
        )

        # 累积计数
        self._chars_since_last_trigger: int = 0
        self._last_trigger_time: float = 0.0

        # 加载关键词
        self.keywords: Dict[str, list] = {"critical": [], "warning": [],
                                           "opportunity": []}
        if keyword_file:
            self._load_keywords(keyword_file)

    def _load_keywords(self, path: str):
        """从 JSON 文件加载关键词配置。"""
        resolved = Path(path)
        if not resolved.exists():
            # 尝试相对于当前模块路径
            resolved = Path(__file__).parent / path

        if resolved.exists():
            data = json.loads(resolved.read_text(encoding="utf-8"))
            for group in ("critical", "warning", "opportunity"):
                if group in data:
                    self.keywords[group] = data[group]
            logger.info(
                f"Loaded keywords: {sum(len(v) for v in self.keywords.values())} "
                f"patterns across {len(self.keywords)} groups"
            )
        else:
            logger.warning(f"Keyword file not found: {path}")

    def feed(self, new_chars: int, text: str) -> Optional[TriggerEvent]:
        """送入新字符，检查是否触发。

        Args:
            new_chars: 自上次调用以来新增的字符数
            text: 当前完整对话文本

        Returns:
            TriggerEvent 如果触发，否则 None
        """
        self._chars_since_last_trigger += new_chars

        # 检查最小间隔
        now = time.time()
        if self._last_trigger_time > 0:
            elapsed_ms = (now - self._last_trigger_time) * 1000
            if elapsed_ms < self.min_interval_ms:
                return None  # 间隔太短，不触发

        # 优先级1: critical 关键词（不管字数，立即触发）
        keyword = self._match_keywords(text, "critical")
        if keyword:
            return self._fire("keyword", text, keyword)

        # 最少字数检查（非 critical 关键词需要）
        if self._chars_since_last_trigger < self.min_chars_for_analysis:
            return None

        # 优先级2: warning/opportunity 关键词 + 字数门控
        if self._chars_since_last_trigger >= self.keyword_min_chars:
            for group in ("warning", "opportunity"):
                keyword = self._match_keywords(text, group)
                if keyword:
                    return self._fire("keyword", text, keyword)

        # 优先级3: 字数触发
        if self._chars_since_last_trigger >= self.char_trigger:
            return self._fire("char_count", text)

        return None

    def _match_keywords(self, text: str, group: str) -> Optional[str]:
        """在文本中查找指定组的关键词。"""
        for kw in self.keywords.get(group, []):
            if kw in text:
                return kw
        return None

    def _fire(self, source: str, text: str,
              keyword: str = None) -> TriggerEvent:
        """触发分析，重置计数器。"""
        accumulated = self._chars_since_last_trigger
        self._chars_since_last_trigger = 0
        self._last_trigger_time = time.time()

        return TriggerEvent(
            source=source,
            accumulated_chars=accumulated,
            window_text=text,
            keyword_matched=keyword,
        )

    def reset(self):
        """重置触发引擎状态。"""
        self._chars_since_last_trigger = 0
        self._last_trigger_time = 0.0


# ============================================================
# DebounceGate — 弹窗去抖
# ============================================================

class DebounceGate:
    """防止弹窗过频。基于卦象变化、时间间隔和历史状态做决策。

    规则：
    0. context-drift 放行（v4.0.14 新增，优先级最高）：
       risk_level 较上次弹窗升高 ≥1 级，或卦象转向兑/乾/巽
       → 强制放行并采用 diagnostic tone（修复 C-03：冲突升级被 debounce 吞掉）
    1. 卦象变化 → 弹（有新信息）
    2. 卦象不变但超过冷却时间 + 重复次数未达上限 → 弹
    3. 风险等级变为 high → 强制弹（越过冷却）
    4. 其他情况 → 抑制
    """

    DEFAULT_COOLDOWN_SECONDS = 15.0
    DEFAULT_SAME_STATE_MAX_REPEATS = 2
    DEFAULT_ABSOLUTE_MIN_INTERVAL = 5.0

    # context-drift 放行的目标卦象（冲突/能量升级信号）
    DRIFT_RELEASE_TRIGRAMS = (Trigram.DUI, Trigram.QIAN, Trigram.XUN)

    def __init__(
        self,
        cooldown_seconds: float = None,
        same_state_max_repeats: int = None,
        absolute_min_interval: float = None,
    ):
        """初始化去抖门控。

        Args:
            cooldown_seconds: 相同卦象弹窗的最小冷却秒数
            same_state_max_repeats: 相同卦象最多连续弹窗次数
            absolute_min_interval: 任意两个弹窗之间的绝对最小间隔
        """
        self.cooldown_seconds = (
            cooldown_seconds or self.DEFAULT_COOLDOWN_SECONDS
        )
        self.same_state_max_repeats = (
            same_state_max_repeats or self.DEFAULT_SAME_STATE_MAX_REPEATS
        )
        self.absolute_min_interval = (
            absolute_min_interval or self.DEFAULT_ABSOLUTE_MIN_INTERVAL
        )

        # 状态追踪
        self._last_popup_time: float = 0.0
        self._last_shown_trigram: Optional[Trigram] = None
        self._last_shown_tone: Optional[PopupTone] = None
        self._last_shown_risk_level: Optional[str] = None
        self._trigram_repeat_count: int = 0
        self._popup_history: List[Popup] = []

    def should_show(self, state: ZhouYiState,
                    now: float = None) -> tuple[bool, str]:
        """判断是否应该弹窗。

        Args:
            state: 当前的周易分析状态
            now: 当前时间戳（默认 time.time()）

        Returns:
            (should_show, reason): 是否弹窗及原因
        """
        if now is None:
            now = time.time()

        # 规则0: context-drift 放行（v4.0.14，越过绝对最小间隔与冷却）
        # 冲突升级信号必须弹——debounce 的本职是防重复，不是压制升级
        if self._last_shown_trigram is not None:
            risk_escalated = (
                risk_rank(state.risk_level)
                >= risk_rank(self._last_shown_risk_level or "低") + 1
            )
            drift_to_conflict = (
                state.trigram in self.DRIFT_RELEASE_TRIGRAMS
                and state.trigram != self._last_shown_trigram
            )
            if risk_escalated or drift_to_conflict:
                # 升级场景强制 diagnostic tone（修复 C-03 FC_TONE_OFF）
                old_risk = self._last_shown_risk_level
                state.suggested_tone = PopupTone.DIAGNOSTIC
                self._update_state(state, now)
                if risk_escalated:
                    return True, (
                        f"context drift 放行: 风险升级 "
                        f"{old_risk} → {state.risk_level}"
                    )
                return True, (
                    f"context drift 放行: 卦象转向 "
                    f"{state.trigram.chinese_name}（兑/乾/巽）"
                )

        # 绝对最小间隔检查
        if self._last_popup_time > 0:
            elapsed = now - self._last_popup_time
            if elapsed < self.absolute_min_interval:
                return False, f"绝对最小间隔未到 ({elapsed:.1f}s < {self.absolute_min_interval}s)"

        # 风险升级 → 强制弹
        if risk_rank(state.risk_level) >= 2 and self._last_shown_trigram is not None:
            if elapsed >= self.absolute_min_interval:
                self._update_state(state, now)
                return True, "高风险升级，强制弹窗"

        # 卦象变化 → 弹
        if state.trigram != self._last_shown_trigram:
            old_name = self._last_shown_trigram.chinese_name if self._last_shown_trigram else "初始"
            self._update_state(state, now)
            return True, f"卦象变化: {old_name} → {state.trigram.chinese_name}"

        # 同卦象，检查冷却和重复次数
        elapsed = now - self._last_popup_time if self._last_popup_time > 0 else 999

        if elapsed < self.cooldown_seconds:
            return False, f"冷却中 ({elapsed:.1f}s < {self.cooldown_seconds}s)"

        if self._trigram_repeat_count >= self.same_state_max_repeats:
            return False, (
                f"相同卦象已达上限 "
                f"({self._trigram_repeat_count}/{self.same_state_max_repeats})"
            )

        # 弹窗类型没变且卦象没变 → 抑制
        if state.suggested_tone == self._last_shown_tone:
            return False, "卦象和弹窗类型均未变化"

        self._update_state(state, now)
        return True, "冷却已过，弹窗类型改变"

    def _update_state(self, state: ZhouYiState, now: float):
        """更新内部状态追踪。"""
        if state.trigram == self._last_shown_trigram:
            self._trigram_repeat_count += 1
        else:
            self._trigram_repeat_count = 1

        self._last_popup_time = now
        self._last_shown_trigram = state.trigram
        self._last_shown_tone = state.suggested_tone
        self._last_shown_risk_level = state.risk_level

    def record_popup(self, popup: Popup):
        """记录已展示的弹窗。"""
        self._popup_history.append(popup)
        # 只保留最近 20 个
        if len(self._popup_history) > 20:
            self._popup_history = self._popup_history[-20:]

    @property
    def popup_history(self) -> List[Popup]:
        return self._popup_history

    def reset(self):
        """重置去抖门控状态。"""
        self._last_popup_time = 0.0
        self._last_shown_trigram = None
        self._last_shown_tone = None
        self._last_shown_risk_level = None
        self._trigram_repeat_count = 0
        self._popup_history = []


# ============================================================
# StreamOrchestrator — 主编排器
# ============================================================

class StreamOrchestrator:
    """流式弹窗系统主编排器。

    串联 TextBuffer → TriggerEngine → ZhouYiAnalyzer → DebounceGate
    → PopupGenerator → 输出回调。

    Usage:
        orchestrator = StreamOrchestrator(
            analyzer=stage1,
            generator=stage2,
            output_callback=lambda popup: print(popup.full_text),
        )

        # 模拟流式输入
        for chunk in text_chunks:
            popup = await orchestrator.process_chunk(chunk)
            if popup:
                print(f"[弹窗] {popup.full_text}")
    """

    def __init__(
        self,
        analyzer: ZhouYiAnalyzer,
        generator: PopupGenerator,
        output_callback: Callable[[Popup], Any] = None,
        char_trigger: int = 120,
        min_chars_for_analysis: int = 60,
        min_interval_ms: int = 3000,
        keyword_file: str = None,
        cooldown_seconds: float = 15.0,
        same_state_max_repeats: int = 2,
        absolute_min_interval: float = 5.0,
        window_size: int = 3000,
        lookback: int = 500,
        stable_block_enabled: bool = True,
    ):
        """初始化编排器。

        Args:
            analyzer: Stage 1 周易分析器
            generator: Stage 2 弹窗生成器
            output_callback: 弹窗输出回调函数
            char_trigger: 字数触发阈值
            min_chars_for_analysis: 最少分析字符数
            min_interval_ms: 最小触发间隔（毫秒）
            keyword_file: 关键词配置文件路径
            cooldown_seconds: 弹窗冷却秒数
            same_state_max_repeats: 同卦象最多连续弹窗数
            absolute_min_interval: 任意两个弹窗之间的绝对最小间隔
            window_size: 分析窗口大小
            lookback: 窗口回看字符数
            stable_block_enabled: P0 硬拦截开关（v4.0.14）——
                ZhouYi 判定 risk_level=低 + 坤卦 + container_status=不适用
                时一律不弹窗（修复 B-01/B-02/A-01 日常/优秀对话误触发）
        """
        self.analyzer = analyzer
        self.generator = generator
        self.output_callback = output_callback or self._default_output

        # 组件
        self.buffer = TextBuffer(window_size=window_size, lookback=lookback)
        self.trigger = TriggerEngine(
            char_trigger=char_trigger,
            min_chars_for_analysis=min_chars_for_analysis,
            min_interval_ms=min_interval_ms,
            keyword_file=keyword_file,
        )
        self.debounce = DebounceGate(
            cooldown_seconds=cooldown_seconds,
            same_state_max_repeats=same_state_max_repeats,
            absolute_min_interval=absolute_min_interval,
        )

        # P0 硬拦截开关（v4.0.14）
        self.stable_block_enabled = stable_block_enabled

        # 运行时状态
        self._analysis_count: int = 0
        self._popup_count: int = 0
        self._suppressed_count: int = 0
        self._zhouyi_states: List[ZhouYiState] = []

    async def process_chunk(self, chunk: str) -> Optional[Popup]:
        """处理一个文本块。

        完整流程：追加→触发检查→分析→去抖→生成→输出。
        支持同步和异步回调。

        Args:
            chunk: 新到达的文本片段

        Returns:
            生成的 Popup（如果弹窗），否则 None
        """
        if not chunk:
            return None

        # 1. 追加到缓冲
        chars_before = self.buffer.total_chars
        window = self.buffer.append(chunk)
        new_chars = self.buffer.total_chars - chars_before

        # 2. 检查触发
        trigger_event = self.trigger.feed(new_chars, self.buffer.full_text)
        if trigger_event is None:
            return None

        logger.info(
            f"Triggered by {trigger_event.source}"
            + (f" ({trigger_event.keyword_matched})"
               if trigger_event.keyword_matched else "")
            + f" | {new_chars} new chars | "
            f"total {self.buffer.total_chars} chars"
        )

        # 3. Stage 1: 周易分析
        self._analysis_count += 1
        zhouyi_state = self.analyzer.analyze(window)
        self._zhouyi_states.append(zhouyi_state)
        self.buffer.mark_window_analyzed()

        logger.info(
            f"Analysis #{self._analysis_count}: "
            f"{zhouyi_state.trigram.symbol} {zhouyi_state.trigram.chinese_name} "
            f"({zhouyi_state.trigram.yao_pattern}) | "
            f"risk={zhouyi_state.risk_level} | "
            f"tone={zhouyi_state.suggested_tone.value}"
        )

        # 4. P0 硬拦截（v4.0.14，v4.0.20 收紧）：低风险 + 坤卦（纯稳态）+ 容器不适用
        #    **且 Stage1 建议类型也为「不弹窗」** 才拦截。
        #    v4.0.20 修复（缺陷B · P0 过度保守）：原 P0 只看三要素，把"和谐但有教育契机"
        #    （Stage1 判建议类型=诊断式/鼓励式）也误当纯日常拦了。现在 P0 与 Stage1
        #    「建议类型」联动——仅当 Stage1 也判「不弹窗」（真·纯日常无契机）才拦；
        #    Stage1 判了诊断式/鼓励式/看见孩子（识别到情绪信号/教育契机）则放行进 Stage2。
        #    依据：REN-42 裁判诊断 — B-01/B-02/A-01 的 ZhouYi 终态均为
        #    「坤·稳定承载型·risk_level=低」，系统已正确识别无冲突，
        #    但触发门仍放行弹窗（FC_UNSUPPORTED）。
        if (
            self.stable_block_enabled
            and zhouyi_state.risk_level == "低"
            and zhouyi_state.trigram == Trigram.KUN
            and zhouyi_state.container_status == "不适用"
            and "不弹窗" in (zhouyi_state.suggestion or "")
        ):
            self._suppressed_count += 1
            logger.info(
                "Popup suppressed: P0 硬拦截 "
                "(risk=低 + 坤 + 容器不适用 + Stage1建议类型=不弹窗，稳态日常对话)"
            )
            return None

        # 5. 去抖检查
        should_show, reason = self.debounce.should_show(zhouyi_state)

        if not should_show:
            self._suppressed_count += 1
            logger.info(f"Popup suppressed: {reason}")
            return None

        # 6. Stage 2: 生成弹窗
        self._popup_count += 1
        popup = self.generator.generate(
            dialogue_window=window,
            zhouyi_state=zhouyi_state,
            previous_popups=self.debounce.popup_history,
        )

        # 6b. 生成自检未通过（如 P2 鼓励式缺少 repair phrase）→ 不弹
        if not popup.should_popup:
            self._popup_count -= 1
            self._suppressed_count += 1
            logger.info("Popup suppressed: 生成自检未通过（should_popup=False）")
            return None

        # 7. 记录和输出
        self.debounce.record_popup(popup)

        logger.info(
            f"Popup #{self._popup_count} shown: "
            f"{popup.tone.value} | {popup.char_count} chars | "
            f"reason: {reason}"
        )

        # 调用回调（支持同步和异步）
        result = self.output_callback(popup)
        # 如果是协程，等待
        if hasattr(result, "__await__"):
            await result

        return popup

    async def process_full_text(self, text: str,
                                 chunk_size: int = 60) -> List[Popup]:
        """处理完整对话文本（模拟流式输入）。

        将完整文本按 chunk_size 切片，逐块送入 process_chunk。

        Args:
            text: 完整对话文本
            chunk_size: 每块字符数

        Returns:
            List[Popup]: 所有生成的弹窗
        """
        popups = []

        # 按换行分割成句子，再按 chunk_size 合并
        lines = text.split("\n")
        current_chunk = ""

        for line in lines:
            current_chunk += line + "\n"

            if len(current_chunk) >= chunk_size:
                popup = await self.process_chunk(current_chunk)
                if popup:
                    popups.append(popup)
                current_chunk = ""
                # 小延迟模拟实时
                await self._async_sleep(0.05)

        # 处理剩余文本
        if current_chunk.strip():
            popup = await self.process_chunk(current_chunk)
            if popup:
                popups.append(popup)

        return popups

    @staticmethod
    async def _async_sleep(seconds: float):
        """异步睡眠。"""
        import asyncio
        await asyncio.sleep(seconds)

    @staticmethod
    def _default_output(popup: Popup):
        """默认输出回调：打印弹窗到 stdout。"""
        symbol = popup.zhouyi_context.trigram.symbol if popup.zhouyi_context else ""
        print(f"\n{'='*50}")
        print(f"  {symbol} {popup.tone.value.upper()} 弹窗 ({popup.char_count}字)")
        print(f"{'='*50}")
        print(popup.full_text)
        print(f"{'='*50}\n")

    @property
    def stats(self) -> dict:
        """运行统计。"""
        return {
            "total_chars": self.buffer.total_chars,
            "analysis_count": self._analysis_count,
            "popup_count": self._popup_count,
            "suppressed_count": self._suppressed_count,
            "suppression_rate": (
                self._suppressed_count / max(1, self._analysis_count)
            ),
            "last_state": (
                self._zhouyi_states[-1].to_dict()
                if self._zhouyi_states else None
            ),
        }

    def reset(self):
        """重置编排器状态（用于新一轮对话）。"""
        self.buffer = TextBuffer(
            window_size=self.buffer.window_size,
            lookback=self.buffer.lookback,
        )
        self.trigger.reset()
        self.debounce.reset()
        self._analysis_count = 0
        self._popup_count = 0
        self._suppressed_count = 0
        self._zhouyi_states = []
