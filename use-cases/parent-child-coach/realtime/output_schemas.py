"""数据模型定义 — 周易八卦实时弹窗系统。

定义系统中所有核心数据类型：爻状态、卦象、弹窗类型、分析结果、弹窗内容、触发事件。
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class YaoState(str, Enum):
    """爻的状态：掌控（阴/容器）或 失控（阳/生命力）。"""

    RONG_QI = "控"  # 容器 / yin / 稳定承载
    SHI_KONG = "失"  # 生命力 / yang / 能量外溢


class Trigram(Enum):
    """八卦定义 — 三爻组合形成八种亲子沟通能量路径。

    每个卦象包含：符号、名称、下爻、中爻、上爻、含义简述。
    """

    # (符号, 名称, 下爻, 中爻, 上爻, 含义)
    KUN = ("☷", "坤", YaoState.RONG_QI, YaoState.RONG_QI, YaoState.RONG_QI, "稳定承载型")
    ZHEN = ("☳", "震", YaoState.RONG_QI, YaoState.RONG_QI, YaoState.SHI_KONG, "安全爆发型")
    KAN = ("☵", "坎", YaoState.RONG_QI, YaoState.SHI_KONG, YaoState.RONG_QI, "波动修复型")
    DUI = ("☱", "兑", YaoState.RONG_QI, YaoState.SHI_KONG, YaoState.SHI_KONG, "释放扩散型")
    GEN = ("☶", "艮", YaoState.SHI_KONG, YaoState.RONG_QI, YaoState.RONG_QI, "及时收束型")
    LI = ("☲", "离", YaoState.SHI_KONG, YaoState.RONG_QI, YaoState.SHI_KONG, "照见反复型")
    XUN = ("☴", "巽", YaoState.SHI_KONG, YaoState.SHI_KONG, YaoState.RONG_QI, "穿透落地型")
    QIAN = ("☰", "乾", YaoState.SHI_KONG, YaoState.SHI_KONG, YaoState.SHI_KONG, "生命外放型")

    def __init__(self, symbol: str, chinese_name: str, lower: YaoState,
                 middle: YaoState, upper: YaoState, description: str):
        self.symbol = symbol
        self.chinese_name = chinese_name
        self.lower = lower
        self.middle = middle
        self.upper = upper
        self.description = description

    @property
    def yao_pattern(self) -> str:
        """返回三爻模式字符串，如 '控控控'、'失控失'。"""
        return f"{self.lower.value}{self.middle.value}{self.upper.value}"

    @classmethod
    def from_yao_states(cls, lower: YaoState, middle: YaoState,
                        upper: YaoState) -> "Trigram":
        """根据三爻状态查找对应卦象。"""
        for trigram in cls:
            if (trigram.lower == lower and trigram.middle == middle
                    and trigram.upper == upper):
                return trigram
        raise ValueError(f"无法匹配卦象: {lower}{middle}{upper}")


class PopupTone(str, Enum):
    """弹窗类型：诊断式 或 鼓励式。"""

    DIAGNOSTIC = "diagnostic"  # 诊断式（100-200字）：照见盲区
    ENCOURAGING = "encouraging"  # 鼓励式（30-60字）：肯定正向时刻


# === 风险等级排序（v4.0.14 新增，供 P0 硬拦截 / P1 context-drift 放行使用） ===
# LLM 输出为中文（低/中/高），历史代码路径可能产出英文（low/medium/high），统一兼容。
RISK_ORDER = {
    "低": 0, "low": 0,
    "中": 1, "medium": 1,
    "高": 2, "high": 2,
}


def risk_rank(level: str) -> int:
    """将风险等级字符串映射为可比较的整数（低=0 / 中=1 / 高=2）。

    未识别的等级保守按 0 处理。
    """
    return RISK_ORDER.get(str(level).strip(), 0)


@dataclass
class ZhouYiState:
    """Stage 1 输出：周易八卦状态分类结果。

    Attributes:
        trigram: 识别的八卦卦象
        lower_yao: 下爻（开始/起点状态）
        middle_yao: 中爻（过程/互动发展）
        upper_yao: 上爻（结束/当前走向）
        container_status: 容器判定 — "有容器的失控" | "无容器的失控" | "纯稳态" | "不适用"
        risk_level: 风险等级 — "low" | "medium" | "high"
        suggested_tone: 建议弹窗类型
        confidence: LLM 分类置信度 (0.0-1.0)
        brief_reason: 一句话解释分类依据
    """

    trigram: Trigram
    lower_yao: YaoState
    middle_yao: YaoState
    upper_yao: YaoState
    container_status: str = "不适用"
    risk_level: str = "low"
    suggested_tone: PopupTone = PopupTone.DIAGNOSTIC
    confidence: float = 1.0
    brief_reason: str = ""

    @property
    def is_full_release(self) -> bool:
        """是否乾卦（全失控）。"""
        return self.trigram == Trigram.QIAN

    @property
    def is_stable(self) -> bool:
        """是否坤卦（全掌控/稳态）。"""
        return self.trigram == Trigram.KUN

    @property
    def has_transition(self) -> bool:
        """是否存在明显的状态转换（中间爻与两端不同）。"""
        return (self.lower_yao != self.middle_yao
                or self.middle_yao != self.upper_yao)

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return {
            "trigram_symbol": self.trigram.symbol,
            "trigram_name": self.trigram.chinese_name,
            "trigram_description": self.trigram.description,
            "yao_pattern": self.trigram.yao_pattern,
            "lower_yao": self.lower_yao.value,
            "middle_yao": self.middle_yao.value,
            "upper_yao": self.upper_yao.value,
            "container_status": self.container_status,
            "risk_level": self.risk_level,
            "suggested_tone": self.suggested_tone.value,
            "confidence": self.confidence,
            "brief_reason": self.brief_reason,
        }


@dataclass
class Popup:
    """Stage 2 输出：弹窗内容。

    Attributes:
        should_popup: 是否应该弹窗
        tone: 弹窗类型
        popup_insight: 弹窗正文（核心洞察）
        popup_suggestion: 可选建议（用 —— 分隔）
        zhouyi_context: 关联的周易分析状态
        timestamp: 弹窗生成时间戳
    """

    should_popup: bool
    tone: PopupTone
    popup_insight: str
    popup_suggestion: str = ""
    zhouyi_context: Optional[ZhouYiState] = None
    timestamp: float = 0.0

    @property
    def full_text(self) -> str:
        """返回完整的弹窗文本（洞察 + 建议）。"""
        if self.popup_suggestion:
            return f"{self.popup_insight}\n——\n{self.popup_suggestion}"
        return self.popup_insight

    @property
    def char_count(self) -> int:
        """弹窗字符数。"""
        return len(self.full_text)

    def to_dict(self) -> dict:
        """序列化为字典，与现有 dataset answer 格式兼容。"""
        result = {
            "should_popup": self.should_popup,
            "tone": self.tone.value,
            "popup_insight": self.popup_insight,
            "popup_suggestion": self.popup_suggestion,
            "timestamp": self.timestamp,
        }
        if self.zhouyi_context:
            result["zhouyi_state"] = self.zhouyi_context.to_dict()
        return result


@dataclass
class TriggerEvent:
    """触发事件 — 描述什么触发了本轮分析。

    Attributes:
        source: 触发来源 — "char_count" | "keyword" | "manual"
        keyword_matched: 匹配到的关键词（如有）
        accumulated_chars: 自上次分析累积的字符数
        window_text: 当前分析窗口的文本
    """

    source: str
    accumulated_chars: int
    window_text: str
    keyword_matched: Optional[str] = None
