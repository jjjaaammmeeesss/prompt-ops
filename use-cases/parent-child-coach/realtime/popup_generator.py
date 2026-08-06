"""Stage 2 — 弹窗内容生成器。

加载现有系统提示词，融入周易八卦分析上下文，
生成诊断式（100-200字）或鼓励式（60-120字）或看见孩子（50-100字）弹窗内容。

v4.0.14 变更（P2）:
- 仅诊断式弹窗强制「至少 1 句 parent-quotable repair phrase」，
  首次生成缺失时自动重试一次，仍缺失则该弹窗不通过（should_popup=False）。
  v4.0.18 修正：P2 原误配为鼓励式，现修正为诊断式——诊断式指出问题后需教家长怎么说话。
  v4.0.19 收窄：P2 仅覆盖诊断式（不再覆盖看见孩子），鼓励式/看见孩子不强制话术。

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

# === 执行器版本号 + 适配 prompt（双向同步元规则，见 CLAUDE.md）===
# __version__ 自增记录本执行器迭代；PROMPT_VERSION 声明它适配的 prompt 版本，
# 两者必须与生产 realtime/config.yaml 一致（由 scripts/check_prompt_executor_sync.py 校验）。
__version__ = "1.0"           # 生产执行器版本号
PROMPT_VERSION = "v4.0.19"    # 适配的 prompt 版本（= system_prompt_v4.0.19.txt）

# === 弹窗生成的系统提示词增强 ===
# v4.0+: 静态卦象策略已移入 system_prompt_v4.0.19.txt 第八节。
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
ENCOURAGING_MIN_CHARS = 60
ENCOURAGING_MAX_CHARS = 100
CHILD_INSIGHT_MIN_CHARS = 60
CHILD_INSIGHT_MAX_CHARS = 100

# === P2: parent-quotable repair phrase 检测（v4.0.14 新增） ===
# 引号内 ≥4 字的完整话术视为"家长可直接引用的话"，
# 兼容中文引号「」『』“”与英文引号 ""。
_QUOTABLE_PHRASE_RE = re.compile(r'[「『“"]([^」』”"]{4,})[」』”"]')

# === v4.0.19: LLM 声明的弹窗类型解析（方案①，三类型平等闭环） ===
# LLM 在预分析块第 0 项输出"类型：诊断式/鼓励式/看见孩子"，
# parse 以此定 Popup.tone 与字数检查，不再被卦象 suggested_tone 锁死。
_DECLARED_TYPE_RE = re.compile(r"类型[:：]\s*(诊断式|鼓励式|看见孩子)")
_DECLARED_TYPE_MAP = {
    "诊断式": PopupTone.DIAGNOSTIC,
    "鼓励式": PopupTone.ENCOURAGING,
    "看见孩子": PopupTone.CHILD_INSIGHT,
}

# 预分析元信息行（0.类型 / 1.元信息 / 2.关键句归属 / 3.错别字）。逐行匹配，
# 用于即使 LLM 漏掉 `==========` 分隔符也兜底剥离，防止元信息泄露进弹窗正文。
_PRE_ANALYSIS_LINE_RE = re.compile(
    r"^\s*\d\.\s*(类型|元信息|关键句归属|错别字)\s*[:：]"
)
_META_SEPARATOR_LINES = {"==========", "---", "--"}


def _strip_meta_lines(text: str) -> str:
    """按行删除预分析元信息行与分隔符行，返回纯弹窗正文。

    与 `==========` split 配合构成双重保险：即使 LLM 漏输出分隔符，
    元信息也不会残留进弹窗正文（修复 v4.0.19 元信息泄露）。
    正文中的全角破折号「——」不在此列，不受影响。
    """
    if not text:
        return ""
    kept = []
    for line in text.splitlines():
        if _PRE_ANALYSIS_LINE_RE.match(line):
            continue
        if line.strip() in _META_SEPARATOR_LINES:
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def _strip_label(text: str, label: str) -> str:
    """去掉正文行首的「洞察句：/建议句：」等标签，保留内容。"""
    t = text.strip()
    for sep in (f"{label}：", f"{label}:"):
        if t.startswith(sep):
            return t[len(sep):].strip()
    return t

# 重试时的强化指令
_REPAIR_PHRASE_RETRY_INSTRUCTION = (
    "⚠️ 上一次输出不合格：缺少家长可直接引用的话术。"
    "必须重新生成，并在弹窗末尾以「你可以这样说：\"……\"」的形式"
    "给出至少一句家长能脱口说出的完整话术（引号内为实际措辞）。"
)


def _length_retry_instruction(max_chars: int, char_count: int) -> str:
    """字数门重试指令（v4.0.19）：压缩超长弹窗到目标区间（kimi 实证有效结构）。"""
    return (
        f"⚠️ 上一次输出过长：当前 {char_count} 字，超过上限 {max_chars} 字。"
        f"必须压缩到 {max_chars} 字以内，超 {max_chars + 10} 字即为失败："
        "只保留核心内容"
        "（诊断式=洞察核心+建议句；鼓励式/看见孩子=具体肯定+对孩子的好影响+完整弧线），"
        "删除泛化展开、重复铺垫、套话短句。不得新增对话中不存在的内容，"
        "只输出弹窗正文。"
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


# === v4.0.19: child_insight 检测（FC_CHILD_INSIGHT 代码层） ===
# 当家长无明显负面行为，但孩子展现出值得被看见的特征时触发。
# 检测信号：孩子话轮占比高 + 孩子表达了独特视角/情感/创意。
CHILD_EXPRESSION_SIGNALS = [
    "我觉得", "我想", "我喜欢", "我不喜欢", "我怕", "我担心",
    "我发现了", "我知道了", "我自己", "我来", "我能", "我会",
    "因为", "所以", "但是我不", "可是我",
]


def detect_child_insight_opportunity(dialogue: str) -> bool:
    """检测对话窗口是否适合使用 child_insight 弹窗。

    条件：
    1. 未命中 FC_TONE_OFF（调用方负责保证）
    2. 孩子话轮占比 > 30%（以孩子发言行数 / 总对话行数估算）
    3. 孩子表达了至少 1 个特征信号（感受/想法/观点）

    Returns:
        True 如果建议使用 child_insight。
    """
    if not dialogue:
        return False

    lines = [l.strip() for l in dialogue.split("\n") if l.strip()]
    if len(lines) < 3:
        return False

    # 估算孩子话轮占比（以数字编号开头的行为孩子或家长发言）
    child_lines = 0
    parent_lines = 0
    for line in lines:
        # 匹配 "1.", "2." 等编号格式的对话行
        if line[0].isdigit():
            # 简单启发式：孩子行通常包含特定的孩子表达信号
            if any(sig in line for sig in CHILD_EXPRESSION_SIGNALS):
                child_lines += 1
            elif any(kw in line for kw in ["快点", "不许", "必须", "给我", "你应该", "你怎麼", "你怎么"]):
                parent_lines += 1
            else:
                # 无法判断的行，默认算孩子（保守估计）
                child_lines += 1

    total = child_lines + parent_lines
    if total == 0:
        return False

    child_ratio = child_lines / total
    return child_ratio >= 0.30


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
        1. 传入的路径（生产由 realtime/config.yaml 指定）
        2. 当前生产：system_prompt_v4.0.19.txt（与 PROMPT_VERSION 一致）
        3. v2.x 历史回退：system_prompt_v2.3.txt / v2.2.txt / v2.1.txt
        4. 原始: system_prompt.txt（生产版回退）
        """
        realtime_dir = Path(__file__).parent

        candidates = []
        if path:
            # 先尝试从 realtime/ 目录解析（config.yaml 中的相对路径以此为基础）
            candidates.append((realtime_dir / path).resolve())
            candidates.append(Path(path).resolve())
        candidates.extend([
            # 当前生产 prompt（与 PROMPT_VERSION / realtime/config.yaml 保持一致）
            realtime_dir / ".." / "system_prompt_v4.0.19.txt",
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
        # 卦象 suggested_tone 作为软起点（soft bias），不锁死任何类型。
        # v4.0.19: 三种弹窗类型（诊断/鼓励/看见孩子）地位平等、无优先级，
        # 由 LLM 结合对话全文与下方 soft-bias 信号综合裁决，代码不再强制覆盖。
        tone = zhouyi_state.suggested_tone

        # 收集 soft-bias 信号：只作为"倾向提示"注入 LLM，不强制 tone。
        # 取代旧 FC_TONE_OFF（无条件强制 diagnostic）与 child_insight 限 encouraging。
        bias_notes: list = []

        override_reason = detect_parent_override(dialogue_window)
        if override_reason:
            bias_notes.append(
                f"⚠️ 倾向信号（FC_TONE_OFF）：对话含家长单向权力/不接住行为"
                f"（{override_reason}类，如催促/打断/贴标签），可优先考虑诊断式。"
                "仅作倾向参考，最终请以对话全文为准。"
            )
            logger.info(
                f"soft-bias FC_TONE_OFF: 命中「{override_reason}」，"
                f"注入诊断倾向（原 suggested_tone={tone}）"
            )

        if detect_child_insight_opportunity(dialogue_window):
            bias_notes.append(
                "💡 倾向信号（FC_CHILD_INSIGHT）：对话以孩子特征表达为主，"
                "可优先考虑『看见孩子』弹窗。仅作倾向参考，最终请以对话全文为准。"
            )
            logger.info(
                "soft-bias FC_CHILD_INSIGHT: 检测到孩子特征表达信号，注入看见孩子倾向"
            )

        try:
            raw_text = self._call_llm(
                dialogue_window, zhouyi_state, tone, bias_notes=bias_notes
            )
            popup = self._parse_popup_output(raw_text, tone, zhouyi_state)

            # P2（v4.0.14）：仅诊断式弹窗强制含 ≥1 句 parent-quotable repair phrase，
            # 缺失则重试一次，仍缺失则该弹窗不通过。鼓励式/看见孩子不强制话术。
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
                    bias_notes=bias_notes,
                )
                popup = self._parse_popup_output(raw_text, tone, zhouyi_state)
                if not has_quotable_phrase(popup.full_text):
                    logger.warning(
                        "Diagnostic popup still missing quotable repair phrase "
                        "after retry; rejecting popup (P2)"
                    )
                    popup.should_popup = False
                    return popup

            # 字数门重试（v4.0.19）：超上限则带压缩指令重试一次，仍超则保留原样。
            # 这是让 60-100 / 100-200 标准真正生效的核心机制（kimi 实证压缩指令有效）。
            _max = {
                PopupTone.DIAGNOSTIC: DIAGNOSTIC_MAX_CHARS,
                PopupTone.ENCOURAGING: ENCOURAGING_MAX_CHARS,
                PopupTone.CHILD_INSIGHT: CHILD_INSIGHT_MAX_CHARS,
            }.get(popup.tone)
            if _max and popup.char_count > _max:
                logger.warning(
                    f"Popup too long ({popup.char_count}>{_max}); "
                    "retrying once to compress"
                )
                raw_text = self._call_llm(
                    dialogue_window, zhouyi_state, tone,
                    extra_instruction=_length_retry_instruction(
                        _max, popup.char_count
                    ),
                    bias_notes=bias_notes,
                )
                popup = self._parse_popup_output(raw_text, tone, zhouyi_state)
                if popup.char_count > _max:
                    logger.warning(
                        f"Popup still over {_max} chars ({popup.char_count}) "
                        "after retry; keeping as-is"
                    )

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
        bias_notes: Optional[list] = None,
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

        # 弹窗类型指令（v4.0.19: 三类型平等，LLM 综合裁决，无优先级）
        # 不再按 tone 分支强制单一类型——三种类型地位平等，由 LLM 结合
        # 对话全文与下方 soft-bias 倾向信号自行判断。tone 仅作初始参考起点。
        type_instruction = (
            "请根据对话内容，在以下三种弹窗类型中自行判断并生成——"
            "三种类型地位平等、无优先级，最终以对话全文为准：\n"
            f" - **诊断式**（{DIAGNOSTIC_MIN_CHARS}-{DIAGNOSTIC_MAX_CHARS}字）："
            "照见家长盲区，先承认发心→揭示具体模式→给出一个微小可做的尝试，"
            "必须包含至少一句家长可直接引用的话术"
            "（以「你可以这样说：\"……\"」形式给出，引号内为实际措辞）。\n"
            f" - **鼓励式**（{ENCOURAGING_MIN_CHARS}-{ENCOURAGING_MAX_CHARS}字）："
            "一段完整的纯肯定，具体点出家长刚展现的积极模式（摘实际言行+好影响），"
            "覆盖完整弧线，不含话术/建议。\n"
            f" - **看见孩子**（{CHILD_INSIGHT_MIN_CHARS}-{CHILD_INSIGHT_MAX_CHARS}字）："
            "洞察孩子的性格特征，结构「你的孩子可能是[具体特征描述]」+"
            "「ta可能更适合用[教育方式]来引导」，特征从对话实际言行提炼、禁止空洞形容词。\n"
            f"系统初始倾向为「{tone.value}」，仅作参考起点，不锁死类型。"
        )
        if bias_notes:
            type_instruction += "\n\n倾向信号（仅作参考，非强制）：\n" + "\n".join(bias_notes)

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
        bias_notes: Optional[list] = None,
    ) -> str:
        """调用 LLM 生成弹窗文本。"""
        messages = self._build_messages(
            dialogue, zhouyi_state, tone, extra_instruction=extra_instruction,
            bias_notes=bias_notes,
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
        v4.0.19（方案①）：预分析块（`==========` 之前）第 0 项
        "类型：诊断式/鼓励式/看见孩子"决定 Popup.tone 与字数检查；
        正文为 `==========` 之后的弹窗内容。不再被卦象 suggested_tone 锁死。
        """
        text = raw.strip()

        # 去除常见前缀
        for prefix in ["弹窗：", "弹窗:", "诊断式弹窗：", "鼓励式弹窗：",
                        "看见孩子弹窗：", "看见孩子：",
                        "诊断：", "鼓励：", "【弹窗】", "【诊断】", "【鼓励】",
                        "【看见孩子】"]:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()

        # v4.0.19: 预分析块与正文以 ========== 为界。
        # 预分析里提取 LLM 声明的类型；无分隔符则整体视为正文、回退卦象 tone。
        resolved_tone = tone
        if "==========" in text:
            preambule, _, body = text.partition("==========")
            m = _DECLARED_TYPE_RE.search(preambule)
            if m:
                resolved_tone = _DECLARED_TYPE_MAP[m.group(1)]
                logger.debug(
                    f"LLM 声明类型 {m.group(1)}（卦象建议 {tone.value}）"
                )
        else:
            body = text
        # 兜底：按行剥离残留的元信息行（防御 LLM 漏分隔符/格式变形），
        # 防止 0.类型/1.元信息/2.关键句归属/3.错别字 泄露进弹窗正文。
        body = _strip_meta_lines(body)

        # 分离 insight 和 suggestion（用 —— 或 -- 分隔）
        insight = body
        suggestion = ""

        for sep in ["\n——\n", "\n——", "——\n", "——",
                     "\n--\n", "\n--", "--\n", "--"]:
            if sep in body:
                parts = body.split(sep, 1)
                insight = _strip_label(parts[0], "洞察句")
                suggestion = (
                    _strip_label(parts[1], "建议句")
                    if len(parts) > 1 else ""
                )
                break
        else:
            # 无分隔符：可能是整段洞察，或带标签但缺建议句
            insight = _strip_label(insight, "洞察句")
            suggestion = _strip_label(suggestion, "建议句")

        # 字数检查（基于 LLM 声明的类型）
        char_count = len(insight) + (len(suggestion) + 2 if suggestion else 0)

        if resolved_tone == PopupTone.DIAGNOSTIC:
            if char_count > DIAGNOSTIC_MAX_CHARS + 20:
                logger.warning(
                    f"Diagnostic popup too long: {char_count} chars "
                    f"(max {DIAGNOSTIC_MAX_CHARS})"
                )
        elif resolved_tone == PopupTone.CHILD_INSIGHT:
            if char_count > CHILD_INSIGHT_MAX_CHARS + 20:
                logger.warning(
                    f"Child insight popup too long: {char_count} chars "
                    f"(max {CHILD_INSIGHT_MAX_CHARS})"
                )
        else:
            if char_count > ENCOURAGING_MAX_CHARS + 20:
                logger.warning(
                    f"Encouraging popup too long: {char_count} chars "
                    f"(max {ENCOURAGING_MAX_CHARS})"
                )

        return Popup(
            should_popup=True,
            tone=resolved_tone,
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
        elif zhouyi_state.suggested_tone == PopupTone.CHILD_INSIGHT:
            insight = (
                f"在这段对话里，孩子展现了一些值得被看见的特质。"
                f"花一点时间，看看孩子是怎样的人——"
                f"这是你了解ta最好的窗口。"
            )
            tone = PopupTone.CHILD_INSIGHT
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
