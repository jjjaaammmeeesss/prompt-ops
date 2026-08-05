"""Stage 2 — 弹窗内容生成器。

加载现有系统提示词，融入周易八卦分析上下文，
生成诊断式（100-200字）或鼓励式（30-60字）弹窗内容。

v4.0.14 变更（P2）:
- 诊断式弹窗强制「至少 1 句 parent-quotable repair phrase」，
  首次生成缺失时自动重试一次，仍缺失则该弹窗不通过（should_popup=False）。
  v4.0.18 修正：P2 原误配为鼓励式，现修正为诊断式——诊断式指出问题后需教家长怎么说话。

v4.0.16 变更（FC_TONE_OFF + FC_STALE 代码层闭环）:
- FC_TONE_OFF: generate() 调 LLM 前扫描 dialogue_window，若命中家长行为
  override 关键词（催促/打断、评判贴标签、命令单向权力、轻度贬低），
  强制 tone=DIAGNOSTIC，冻结 tone 灵活覆盖。
- FC_STALE: generate() 返回前与 previous_popups 做语义相似度比对，
  超过阈值（默认 0.70）则拒绝弹窗（should_popup=False），避免跨窗口复读。
- PopupGenerator.__init__ 新增 dedup_config 参数，由 cli_demo 从
  config.yaml 的 dedup 段读入。
"""

import difflib
import logging
import os
import re
import time
from pathlib import Path
from typing import List, Optional

from .output_schemas import (
    PopupTone,
    ZhouYiState,
    Popup,
    risk_rank,
)
logger = logging.getLogger("prompt_ops.realtime.popup_generator")

# === 弹窗生成的系统提示词增强 ===
# v4.0: 静态卦象策略已移入 system_prompt_v4.0.txt 第八节。
# 此模板只注入动态的每调用上下文（具体卦象/风险/洞察），
# LLM 自行参考提示词中的策略表来调整弹窗。

ZHOUYI_CONTEXT_TEMPLATE = """
## 周易卦象上下文（本轮实时分析结果）

当前沟通状态属于 **{symbol} {trigram_name}（{yao_pattern}）**——{description}。

容器状态：{container_status}
风险等级：{risk_level}
分析洞察：{brief_reason}

请参考系统提示词第八节「八卦弹窗策略速查」中对应卦象的指导来生成弹窗。

---
"""

# === 弹窗字数限制 ===
DIAGNOSTIC_MIN_CHARS = 80
DIAGNOSTIC_MAX_CHARS = 200
ENCOURAGING_MIN_CHARS = 20
ENCOURAGING_MAX_CHARS = 80

# === P2: parent-quotable repair phrase 检测（v4.0.14 新增） ===
# 引号内 ≥4 字的完整话术视为"家长可直接引用的话"，
# 兼容中文引号「」『』“”与英文引号 ""。
_QUOTABLE_PHRASE_RE = re.compile(r'[「『“"]([^」』”"]{4,})[」』”"]')

# 重试时的强化指令
_REPAIR_PHRASE_RETRY_INSTRUCTION = (
    "⚠️ 上一次输出不合格：缺少家长可直接引用的话术。"
    "必须重新生成，并在弹窗末尾以「你可以这样说：\"……\"」的形式"
    "给出至少一句家长能脱口说出的完整话术（引号内为实际措辞）。"
)


def has_quotable_phrase(text: str) -> bool:
    """检测文本中是否含至少一句引号内的可直接引用话术（≥4字）。"""
    return bool(_QUOTABLE_PHRASE_RE.search(text or ""))


# === v4.0.16: 家长行为 tone override（FC_TONE_OFF 代码层闭环） ===
# 与 system_prompt_v4.0.16+ §「家长行为 tone override」对齐。
# 命中任一类别即强制 DIAGNOSTIC，冻结 tone 灵活覆盖。
# 优先级：安全路由（DebounceGate context-drift）> 本规则 > 卦象 tone > tone 灵活覆盖。
PARENT_OVERRIDE_KEYWORDS = {
    "催促/打断": [
        "快点", "快一点", "别说了", "行了行了", "行了我知道",
        "别废话", "闭嘴", "你能不能快点", "动作快", "抓紧时间",
        "你快点", "少啰嗦", "有完没完",
    ],
    "评判贴标签": [
        "你就是磨蹭", "你太敏感", "你这个人就是", "你就是个",
        "你太矫情", "你就是太", "矫情", "你就是故意", "你总是",
        "你每次都", "你就是不上心",
    ],
    "命令单向权力": [
        "我让你做你就做", "少废话", "按我说的", "我让你",
        "没有为什么", "我说了算", "听我的", "不许顶嘴",
        "你少跟我", "我是你妈", "我是你爸", "照我说的做",
    ],
    "轻度贬低/否定情绪": [
        "这有什么好哭", "至于吗", "想太多", "无理取闹",
        "小题大做", "娇气", "这有什么", "有什么好哭",
        "别那么娇", "你至于", "哭什么哭", "有什么好闹",
    ],
}


def detect_parent_override(dialogue: str) -> Optional[str]:
    """扫描对话文本，若命中家长行为 tone override 关键词，返回命中的类别名。

    与 prompt 层「家长行为 tone override」规则对齐——代码层作为硬约束兜底，
    防止 LLM 在 tone 灵活覆盖判定中误将含单向权力行为的对话切为鼓励式。

    Returns:
        命中的类别名（如 "催促/打断"），或 None。
    """
    if not dialogue:
        return None
    for category, keywords in PARENT_OVERRIDE_KEYWORDS.items():
        for kw in keywords:
            if kw in dialogue:
                return category
    return None


# === v4.0.16: 跨窗口语义去重（FC_STALE 代码层闭环） ===
def semantic_similarity(a: str, b: str) -> float:
    """计算两段文本的相似度（difflib SequenceMatcher ratio）。

    与 prompt 层「跨窗口语义去重自检」对齐——代码层作为硬约束兜底，
    在 LLM 仍输出高相似度弹窗时拒绝展示。
    """
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


class PopupGenerator:
    """Stage 2：弹窗内容生成器。

    加载现有系统提示词，注入周易分析上下文，生成弹窗。

    Usage:
        generator = PopupGenerator(
            model_adapter=stage2_model,
            system_prompt_path="../system_prompt.txt",
        )
        popup = generator.generate(dialogue_window, zhouyi_state)
    """

    DEFAULT_TEMPERATURE = 0.3
    DEFAULT_MAX_TOKENS = 640

    def __init__(
        self,
        model_adapter,
        system_prompt_path: str = None,
        temperature: float = None,
        max_tokens: int = None,
        dedup_config: dict = None,
    ):
        """初始化弹窗生成器。

        Args:
            model_adapter: LLM 适配器（Stage 2 — 更高能力的模型）
            system_prompt_path: 现有系统提示词文件路径
            temperature: LLM 温度
            max_tokens: 最大输出 token
            dedup_config: 跨窗口语义去重配置（v4.0.16），含
                enabled / semantic_similarity_threshold / history_window
        """
        self.model = model_adapter
        self.temperature = temperature or self.DEFAULT_TEMPERATURE
        self.max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS

        # v4.0.16: 跨窗口语义去重配置（FC_STALE 代码层闭环）
        _dedup = dedup_config or {}
        self.dedup_enabled = _dedup.get("enabled", True)
        self.dedup_threshold = _dedup.get("semantic_similarity_threshold", 0.70)
        self.dedup_history_window = _dedup.get("history_window", 5)

        # 加载系统提示词
        self.system_prompt = self._load_system_prompt(system_prompt_path)
        logger.info(
            f"PopupGenerator initialized with system prompt "
            f"({len(self.system_prompt)} chars)"
        )

    def _load_system_prompt(self, path: str = None) -> str:
        """加载系统提示词文件。

        查找顺序：
        1. 传入的路径
        2. v4.0: system_prompt_v4.0.txt（周易八卦集成版）
        3. v2.x: system_prompt_v2.3.txt / v2.2.txt / v2.1.txt
        4. 原始: system_prompt.txt（生产版回退）
        """
        realtime_dir = Path(__file__).parent

        candidates = []
        if path:
            # 先尝试从 realtime/ 目录解析（config.yaml 中的相对路径以此为基础）
            candidates.append((realtime_dir / path).resolve())
            candidates.append(Path(path).resolve())
        candidates.extend([
            realtime_dir / ".." / "system_prompt_v4.0.txt",
            realtime_dir / ".." / "system_prompt_v2.3.txt",
            realtime_dir / ".." / "system_prompt_v2.2.txt",
            realtime_dir / ".." / "system_prompt_v2.1.txt",
            realtime_dir / ".." / "system_prompt.txt",
        ])

        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved.exists():
                logger.info(f"Loading system prompt from: {resolved}")
                return resolved.read_text(encoding="utf-8")

        logger.warning(
            "No system prompt file found. Using built-in fallback. "
            f"Searched: {[str(c.resolve()) for c in candidates]}"
        )
        return self._fallback_system_prompt()

    @staticmethod
    def _fallback_system_prompt() -> str:
        """内置的最小化回退提示词。"""
        return """你是「亲子沟通教练 · A轨现场弹窗」智能体。
你的工作是：在家长与孩子对话的当下，弹出一段文字。

弹窗分两种：
- 诊断式弹窗（100-200字）：当家长需要照见盲区时使用
- 鼓励式弹窗（30-60字）：当家长做到值得肯定的瞬间时使用

核心原则：
1. 先看见家长的发心和难处，再揭示模式
2. 像朋友在耳边说话——口语化、短句、不绕弯
3. 永远用邀请的语气，不是命令或宣告
4. 每个弹窗含一个当下可做的微小尝试

当前对话：
{dialogue}

请直接输出弹窗全文（不附加解释）。"""

    def generate(
        self,
        dialogue_window: str,
        zhouyi_state: ZhouYiState,
        previous_popups: Optional[List[Popup]] = None,
    ) -> Popup:
        """生成弹窗内容。

        Args:
            dialogue_window: 当前对话窗口文本
            zhouyi_state: Stage 1 分析的周易状态
            previous_popups: 之前的弹窗历史（可选，用于避免重复）

        Returns:
            Popup: 包含弹窗类型、正文和建议的完整弹窗
        """
        tone = zhouyi_state.suggested_tone

        # v4.0.16: 家长行为 tone override（FC_TONE_OFF 代码层闭环）
        # 在调 LLM 前固定 tone——prompt 层的 override 规则因 tone 已在
        # _build_messages 中固化进 type_instruction，LLM 无权改 tone，
        # 故必须在代码层先做覆盖。
        override_reason = detect_parent_override(dialogue_window)
        if override_reason and tone == PopupTone.ENCOURAGING:
            logger.info(
                f"FC_TONE_OFF override: 命中「{override_reason}」，"
                f"强制 diagnostic（原 suggested_tone=encouraging）"
            )
            tone = PopupTone.DIAGNOSTIC

        try:
            raw_text = self._call_llm(dialogue_window, zhouyi_state, tone)
            popup = self._parse_popup_output(raw_text, tone, zhouyi_state)

            # P2（v4.0.14）：诊断式弹窗强制含 ≥1 句 parent-quotable
            # repair phrase，缺失则重试一次，仍缺失则该弹窗不通过。
            if (
                popup.tone == PopupTone.DIAGNOSTIC
                and not has_quotable_phrase(popup.full_text)
            ):
                logger.warning(
                    "Diagnostic popup missing quotable repair phrase; retrying once"
                )
                raw_text = self._call_llm(
                    dialogue_window, zhouyi_state, tone,
                    extra_instruction=_REPAIR_PHRASE_RETRY_INSTRUCTION,
                )
                popup = self._parse_popup_output(raw_text, tone, zhouyi_state)
                if not has_quotable_phrase(popup.full_text):
                    logger.warning(
                        "Diagnostic popup still missing quotable repair phrase "
                        "after retry; rejecting popup (P2)"
                    )
                    popup.should_popup = False
                    return popup

            # v4.0.16: 跨窗口语义去重（FC_STALE 代码层闭环）
            # 与本次会话最近 N 条弹窗比对相似度，超过阈值则拒绝展示。
            if (
                self.dedup_enabled
                and previous_popups
                and popup.should_popup
            ):
                recent = previous_popups[-self.dedup_history_window:]
                for idx, prev in enumerate(recent):
                    sim = semantic_similarity(popup.full_text, prev.full_text)
                    if sim >= self.dedup_threshold:
                        logger.warning(
                            f"FC_STALE dedup: 与最近第 {len(recent) - idx} 条弹窗"
                            f"相似度 {sim:.2f} ≥ {self.dedup_threshold}，"
                            f"拒绝弹窗（避免复读）"
                        )
                        popup.should_popup = False
                        return popup

            logger.info(
                f"Generated {popup.tone.value} popup ({popup.char_count} chars): "
                f"{popup.popup_insight[:60]}..."
            )
            return popup

        except Exception as e:
            logger.error(f"Popup generation failed: {e}", exc_info=True)
            return self._fallback_popup(zhouyi_state, str(e))

    def _build_messages(
        self,
        dialogue: str,
        zhouyi_state: ZhouYiState,
        tone: PopupTone,
        extra_instruction: str = None,
    ) -> list:
        """构建 LLM 消息列表。

        v4.0: 静态卦象策略已移入 system_prompt_v4.0.txt 第八节，
        此处只注入动态的每调用上下文（具体卦象/风险/洞察）。
        """
        # 构建周易上下文（动态数据，不含静态策略）
        zhouyi_context = ZHOUYI_CONTEXT_TEMPLATE.format(
            symbol=zhouyi_state.trigram.symbol,
            trigram_name=zhouyi_state.trigram.chinese_name,
            yao_pattern=zhouyi_state.trigram.yao_pattern,
            description=zhouyi_state.trigram.description,
            container_status=zhouyi_state.container_status,
            risk_level=zhouyi_state.risk_level,
            brief_reason=zhouyi_state.brief_reason,
        )

        # 组合系统提示词
        system_content = zhouyi_context + "\n" + self.system_prompt

        # 弹窗类型指令
        if tone == PopupTone.DIAGNOSTIC:
            type_instruction = (
                f"请生成**诊断式弹窗**（{DIAGNOSTIC_MIN_CHARS}-{DIAGNOSTIC_MAX_CHARS}字）。"
                "必须：先承认发心 → 揭示具体模式 → 给出一个微小可做的尝试。"
                "必须包含至少一句家长可直接引用的话术"
                "（以「你可以这样说：\"……\"」形式给出，引号内为实际措辞）。"
            )
        else:
            type_instruction = (
                f"请生成**鼓励式弹窗**（{ENCOURAGING_MIN_CHARS}-{ENCOURAGING_MAX_CHARS}字）。"
                "必须：具体点出家长刚展现的积极模式 → 简短有力。"
            )

        user_content = f"""当前对话：
{dialogue}

{type_instruction}
{extra_instruction or ""}

请直接输出弹窗全文（不附加解释、不输出JSON、不输出"弹窗："等前缀）："""

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    def _call_llm(
        self,
        dialogue: str,
        zhouyi_state: ZhouYiState,
        tone: PopupTone,
        extra_instruction: str = None,
    ) -> str:
        """调用 LLM 生成弹窗文本。"""
        messages = self._build_messages(
            dialogue, zhouyi_state, tone, extra_instruction=extra_instruction
        )

        start = time.time()

        if hasattr(self.model, "generate_with_chat_format"):
            raw = self.model.generate_with_chat_format(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        else:
            combined = "\n\n".join(
                f"{m['role']}: {m['content']}" for m in messages
            )
            raw = self.model.generate(
                prompt=combined,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

        elapsed = time.time() - start
        logger.debug(f"LLM generation call took {elapsed:.2f}s")
        return raw

    def _parse_popup_output(
        self,
        raw: str,
        tone: PopupTone,
        zhouyi_state: ZhouYiState,
    ) -> Popup:
        """解析 LLM 输出为 Popup 对象。

        处理 "——" 分隔符、去除前缀标签、字数检查。
        """
        text = raw.strip()

        # 去除常见前缀
        for prefix in ["弹窗：", "弹窗:", "诊断式弹窗：", "鼓励式弹窗：",
                        "诊断：", "鼓励：", "【弹窗】", "【诊断】", "【鼓励】"]:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()

        # 分离 insight 和 suggestion（用 —— 或 -- 分隔）
        insight = text
        suggestion = ""

        for sep in ["\n——\n", "\n——", "——\n", "——",
                     "\n--\n", "\n--", "--\n", "--"]:
            if sep in text:
                parts = text.split(sep, 1)
                insight = parts[0].strip()
                suggestion = parts[1].strip() if len(parts) > 1 else ""
                break

        # 字数检查
        char_count = len(insight) + (len(suggestion) + 2 if suggestion else 0)

        if tone == PopupTone.DIAGNOSTIC:
            if char_count > DIAGNOSTIC_MAX_CHARS + 20:
                logger.warning(
                    f"Diagnostic popup too long: {char_count} chars "
                    f"(max {DIAGNOSTIC_MAX_CHARS})"
                )
        else:
            if char_count > ENCOURAGING_MAX_CHARS + 20:
                logger.warning(
                    f"Encouraging popup too long: {char_count} chars "
                    f"(max {ENCOURAGING_MAX_CHARS})"
                )

        return Popup(
            should_popup=True,
            tone=tone,
            popup_insight=insight,
            popup_suggestion=suggestion,
            zhouyi_context=zhouyi_state,
            timestamp=time.time(),
        )

    def _fallback_popup(self, zhouyi_state: ZhouYiState,
                        error: str) -> Popup:
        """当生成失败时返回回退弹窗。"""
        trigram_name = zhouyi_state.trigram.chinese_name
        pattern = zhouyi_state.trigram.yao_pattern

        if risk_rank(zhouyi_state.risk_level) >= 2:
            insight = (
                f"这一刻，对话的能量很高。"
                f"你不一定做错了什么——但继续下去，可能两败俱伤。"
                f"能不能先停三秒？就三秒。"
            )
            tone = PopupTone.DIAGNOSTIC
        elif zhouyi_state.suggested_tone == PopupTone.ENCOURAGING:
            insight = (
                f"你刚刚的回应里有一种力量——"
                f"在{pattern}的沟通过程中，你选择了承载而不是反击。"
                f"孩子会记住这一刻。"
            )
            tone = PopupTone.ENCOURAGING
        else:
            insight = (
                f"在你和孩子的这段对话里，沟通呈现了{trigram_name}卦的模式。"
                f"看看是不是这样——你有一个自己还没注意到的习惯反应。"
                f"愿不愿意下次停一秒，看看另一种可能？"
            )
            tone = PopupTone.DIAGNOSTIC

        return Popup(
            should_popup=True,
            tone=tone,
            popup_insight=insight,
            popup_suggestion="",
            zhouyi_context=zhouyi_state,
            timestamp=time.time(),
        )
