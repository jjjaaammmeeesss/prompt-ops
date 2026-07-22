"""
Golden Dataset Parser — 解析专家手动标注文件 → 结构化 JSON。

输入：专家手动打标_全部用例汇整_20260716_141130.md (~19,000 行)
输出：golden_dataset.json

支持三种标注格式：
  1. 晓浩/子阳完整⑩项格式（最丰富）
  2. 廖老师快速通道格式
  3. GB_ 早期简单格式

用法：
    python golden_dataset.py \
      --input "C:/Users/h/Desktop/专家手动打标_全部用例汇整_20260716_141130.md" \
      --output "data/golden_dataset.json"
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class WindowAnnotation:
    """单窗口的专家标注。"""
    window_index: int = 0
    window_text: str = ""
    system_popup_text: str = ""
    system_tone: str = ""
    should_popup: bool | None = None  # 专家判断：该弹/不该弹
    expected_tone: str = ""  # 诊断式 / 鼓励式 / child_insight / mixed
    trigger_sentence: str = ""  # 应该在哪个句子上弹窗
    hit_checklist: list[str] = field(default_factory=list)  # ⑧ 命中清单
    forbid_checklist: list[str] = field(default_factory=list)  # ⑨ 禁止清单
    reference_popup: str = ""  # ⑩ 参考弹窗全文
    overall_score: float | None = None  # ⑤ 整体打分 1-10
    overall_feedback: str = ""  # ⑥ 整体反馈
    core_blind_spot: str = ""  # ⑦ 核心痛点
    good_sentences: list[dict] = field(default_factory=list)  # ★ 好句
    bad_sentences: list[dict] = field(default_factory=list)  # ⚠ 问题句


@dataclass
class GoldenCase:
    """单条黄金案例。"""
    case_id: str = ""
    annotator: str = ""  # 晓浩 / 子阳 / 廖老师 / 早期
    dialogue: str = ""  # 完整对话
    windows: list[WindowAnnotation] = field(default_factory=list)
    sequence_note: str = ""  # 序列标注
    content_note: str = ""  # 内容标注（额外说明）


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class GoldenDatasetParser:
    """解析专家手动标注 markdown 文件。"""

    def __init__(self, input_path: str):
        self.input_path = Path(input_path)
        self.text = self.input_path.read_text(encoding="utf-8")
        self.lines = self.text.split("\n")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self) -> list[GoldenCase]:
        """解析全部案例，返回 GoldenCase 列表。"""
        case_boundaries = self._find_case_boundaries()
        cases: list[GoldenCase] = []

        for i, (start, title) in enumerate(case_boundaries):
            end = case_boundaries[i + 1][0] if i + 1 < len(case_boundaries) else len(self.lines)
            section = "\n".join(self.lines[start:end])

            case = self._parse_case_section(section, title)
            if case and case.windows:  # 至少有一个窗口
                cases.append(case)

        return cases

    def save(self, cases: list[GoldenCase], output_path: str) -> None:
        """将解析结果保存为 JSON。"""
        records = []
        for case in cases:
            records.append({
                "case_id": case.case_id,
                "annotator": case.annotator,
                "dialogue": case.dialogue.strip(),
                "windows": [
                    {
                        "window_index": w.window_index,
                        "window_text": w.window_text.strip(),
                        "system_popup_text": w.system_popup_text.strip(),
                        "system_tone": w.system_tone,
                        "should_popup": w.should_popup,
                        "expected_tone": w.expected_tone,
                        "trigger_sentence": w.trigger_sentence,
                        "hit_checklist": w.hit_checklist,
                        "forbid_checklist": w.forbid_checklist,
                        "reference_popup": w.reference_popup.strip(),
                        "overall_score": w.overall_score,
                        "overall_feedback": w.overall_feedback.strip(),
                        "core_blind_spot": w.core_blind_spot.strip(),
                        "good_sentences": w.good_sentences,
                        "bad_sentences": w.bad_sentences,
                    }
                    for w in case.windows
                ],
                "sequence_note": case.sequence_note.strip(),
                "content_note": case.content_note.strip(),
            })

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"✅ 保存 {len(records)} 条案例 → {output}")

    # ------------------------------------------------------------------
    # Case boundary detection
    # ------------------------------------------------------------------

    def _find_case_boundaries(self) -> list[tuple[int, str]]:
        """找到所有案例的起始行号和标题。"""
        boundaries: list[tuple[int, str]] = []
        for i, line in enumerate(self.lines):
            # Match: ## Case X/Y: ... or ## GB_XXX：...
            if re.match(r"^## (Case \d+/\d+|GB_\d+)", line):
                boundaries.append((i, line.strip()))
        return boundaries

    # ------------------------------------------------------------------
    # Case-level parsing
    # ------------------------------------------------------------------

    def _parse_case_section(self, section: str, title: str) -> GoldenCase | None:
        """解析单个案例 section。"""
        case = GoldenCase()
        case.case_id = self._extract_case_id(title)
        case.annotator = self._detect_annotator(section, title)

        # 提取完整对话（GB_ 格式可能没有完整对话）
        case.dialogue = self._extract_dialogue(section)

        # 提取 windows
        case.windows = self._extract_windows(section)

        # 对于 GB_ 早期格式，整个 section 就是一个窗口，使用校标分类
        if title.startswith("GB_") and not case.windows:
            win = self._parse_gb_window(section)
            case.windows.append(win)

        # 提取序列标注和内容标注
        case.sequence_note = self._extract_field(
            section, r"序列标注.*?\n>\s*(.+?)(?:\n\n|\n---|\Z)", default=""
        )
        case.content_note = self._extract_field(
            section, r"内容标注.*?说明\).*?\n>\s*(.+?)(?:\n\n|\n> --|\Z)", default=""
        )

        return case

    @staticmethod
    def _extract_case_id(title: str) -> str:
        """从标题提取 case_id。"""
        # "## Case 1/10: ..." → "C1-001"
        m = re.match(r"## Case (\d+)/(\d+):", title)
        if m:
            return f"C{m.group(2)}-{int(m.group(1)):03d}"
        # "## GB_001：..." → "GB_001"
        m = re.match(r"## (GB_\d+)", title)
        if m:
            return m.group(1)
        # Fallback
        return re.sub(r"[^\w\-]", "_", title[:60])

    @staticmethod
    def _detect_annotator(section: str, title: str) -> str:
        """检测标注者。

        优先级：GB_ 前缀 > section 内显式署名 > case_id 推断 > section 特征匹配
        """
        # 1. GB_ → 早期 (must come first to avoid section-header contamination)
        if "GB_" in title:
            return "早期"

        # 2. Infer from title pattern (more reliable than section text)
        # "## Case X/10:" → 晓浩
        if re.match(r"## Case \d+/10:", title):
            return "晓浩"
        # "## Case X/13:" → 子阳 (第一批)
        if re.match(r"## Case \d+/13:", title):
            return "子阳"
        # "## Case X/11:" → 廖老师
        if re.match(r"## Case \d+/11:", title):
            return "廖老师"
        # "## Case X/4:" or "## Case X/5:" or "## Case X/3:" → 子阳
        if re.match(r"## Case \d+/(4|5|3):", title):
            return "子阳"

        # 3. Explicit annotator names in section (fallback)
        head = section[:2000]
        if "晓浩" in head or "晓浩已手写" in head:
            return "晓浩"
        if "廖老师" in head:
            return "廖老师"
        if "子阳" in head:
            return "子阳"

        # 4. Section features
        if "快速通道" in head:
            return "廖老师"
        if "⑧ 命中清单" in head:
            return "子阳"

        return "未知"

    # ------------------------------------------------------------------
    # Dialogue extraction
    # ------------------------------------------------------------------

    def _extract_dialogue(self, section: str) -> str:
        """提取完整对话文本。"""
        # 找到 ### 💬 完整对话 后的代码块
        m = re.search(
            r"### 💬 完整对话\s*\n\s*```\s*\n(.+?)```",
            section, re.DOTALL
        )
        if m:
            dialogue = m.group(1).strip()
            # 去掉行号前缀 "1：" → ""
            dialogue = re.sub(r"^\d+\s*[：:]\s*", "", dialogue, flags=re.MULTILINE)
            return dialogue
        return ""

    # ------------------------------------------------------------------
    # Window extraction
    # ------------------------------------------------------------------

    def _extract_windows(self, section: str) -> list[WindowAnnotation]:
        """提取所有窗口的标注。"""
        windows: list[WindowAnnotation] = []

        # 先找 甲/乙/丙 标记
        markers = list(re.finditer(r"### ([甲乙丙丁戊]) ·", section))

        if markers:
            # 有显式窗口标记
            for i, m in enumerate(markers):
                start = m.start()
                end = markers[i + 1].start() if i + 1 < len(markers) else len(section)
                win = self._parse_single_window(section[start:end], i + 1)
                windows.append(win)
            return windows

        # 无显式标记：尝试按 🤖 系统弹窗 切分
        popup_markers = list(re.finditer(r"#### 🤖 系统弹窗", section))
        if popup_markers:
            for i, m in enumerate(popup_markers):
                start = m.start()
                end = popup_markers[i + 1].start() if i + 1 < len(popup_markers) else len(section)
                win = self._parse_single_window(section[start:end], i + 1)
                windows.append(win)
            return windows

        # 无任何标记：整段当作一个窗口
        win = self._parse_single_window(section, 1)
        if win.should_popup is not None or win.system_popup_text.strip():
            windows.append(win)
        return windows

    def _parse_single_window(self, win_section: str, window_index: int) -> WindowAnnotation:
        """解析单个窗口 section。"""
        win = WindowAnnotation(window_index=window_index)

        # 提取窗口对话文本
        win.window_text = self._extract_window_dialogue(win_section)

        # 提取系统弹窗
        win.system_popup_text = self._extract_system_popup(win_section)
        win.system_tone = self._extract_system_tone(win_section)

        # 提取专家标注
        win.should_popup = self._extract_should_popup(win_section)
        win.expected_tone = self._extract_expected_tone(win_section)
        win.trigger_sentence = self._extract_trigger_sentence(win_section)
        win.hit_checklist = self._extract_hit_checklist(win_section)
        win.forbid_checklist = self._extract_forbid_checklist(win_section)
        win.reference_popup = self._extract_reference_popup(win_section)
        win.overall_score = self._extract_overall_score(win_section)
        win.overall_feedback = self._extract_overall_feedback(win_section)
        win.core_blind_spot = self._extract_core_blind_spot(win_section)
        win.good_sentences, win.bad_sentences = self._extract_sentence_feedback(win_section)

        return win

    def _parse_gb_window(self, section: str) -> WindowAnnotation:
        """解析 GB_ 早期格式的窗口。"""
        win = WindowAnnotation(window_index=1)

        # 弹窗全文 = system_popup_text + reference_popup
        m = re.search(r"### 弹窗全文\s*\n\s*```\s*\n(.+?)```", section, re.DOTALL)
        if m:
            win.system_popup_text = m.group(1).strip()
            win.reference_popup = m.group(1).strip()  # GB_ 格式中弹窗即参考

        # 校标分类：弹·鼓励式 / 不弹
        if re.search(r"校标分类：弹", section):
            win.should_popup = True
        elif re.search(r"校标分类：不弹", section):
            win.should_popup = False

        m = re.search(r"校标分类：弹·(诊断式|鼓励式)", section)
        if m:
            win.expected_tone = _normalize_tone(m.group(1))

        # 逐句标注
        win.good_sentences, win.bad_sentences = self._extract_sentence_feedback(section)

        return win

    # ------------------------------------------------------------------
    # Individual field extractors
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_window_dialogue(section: str) -> str:
        """提取当前窗口的对话文本。"""
        m = re.search(
            r"#### 💬 当前窗口原文\s*\n\s*```\s*\n(.+?)```",
            section, re.DOTALL
        )
        if m:
            text = m.group(1).strip()
            text = re.sub(r"^\d+\s*[：:]\s*", "", text, flags=re.MULTILINE)
            return text
        return ""

    @staticmethod
    def _extract_system_popup(section: str) -> str:
        """提取系统弹窗文本。"""
        # 格式1: #### 🤖 系统弹窗\n\n> text
        m = re.search(
            r"#### 🤖 系统弹窗.*?\n\n>\s*(.+?)(?:\n\n|####|$)",
            section, re.DOTALL
        )
        if m:
            text = m.group(1).strip()
            # 去掉引用符号 >
            text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)
            return text
        # 格式2: #### 🤖 系统弹窗（句 X 触发）\n\n**语气**: ...\n\n> text
        m = re.search(
            r"#### 🤖 系统弹窗.*?\n\n\*\*语气\*\*:.*?\n\n>\s*(.+?)(?:\n\n####|$)",
            section, re.DOTALL
        )
        if m:
            text = m.group(1).strip()
            text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)
            return text
        return ""

    @staticmethod
    def _extract_system_tone(section: str) -> str:
        """提取系统语气。"""
        m = re.search(r"\*\*语气\*\*:\s*(\S+)", section)
        if m:
            return m.group(1).strip()
        return ""

    @staticmethod
    def _extract_should_popup(section: str) -> bool | None:
        """提取专家判断：是否该弹窗。"""
        # 格式1: ① 是否该弹... \n\n> [·] 该弹
        if re.search(r"①\s*是否该弹.+?>\s*\[[·x✓✔\s]*\]\s*该弹", section, re.DOTALL):
            return True
        if re.search(r"①\s*是否该弹.+?>\s*\[[·x✓✔\s]*\]\s*(?:不弹|不该弹)", section, re.DOTALL):
            return False
        # 格式2: 校标分类
        if re.search(r"校标分类：弹", section):
            return True
        if re.search(r"校标分类：不弹", section):
            return False
        # 格式3: 表格格式
        if re.search(r"\|\s*\*\*该弹窗\?\*\*\s*\|\s*\[[·x✓✔\s]*\]\s*是", section):
            return True
        if re.search(r"\|\s*\*\*该弹窗\?\*\*\s*\|\s*\[[·x✓✔\s]*\]\s*否", section):
            return False
        # 格式4: 快速通道标记
        if re.search(r"该弹[窗\?]?\s*[：:]\s*是|弹窗\s*[：:]\s*是|是否弹窗\?\s*是", section):
            return True
        if re.search(r"该弹[窗\?]?\s*[：:]\s*否|弹窗\s*[：:]\s*否|是否弹窗\?\s*否", section):
            return False

        return None

    @staticmethod
    def _extract_expected_tone(section: str) -> str:
        """提取专家期望的口吻。"""
        # 格式1: ③ 应该弹什么口吻：\n\n> [ ·] 诊断式
        m = re.search(
            r"③\s*应该弹什么口吻.+?>\s*\[[·x✓✔\s]*\]\s*(诊断式|鼓励式|child.insight|mixed|不弹窗)",
            section, re.DOTALL
        )
        if m:
            return _normalize_tone(m.group(1))
        # 格式2: 弹窗口吻 | [ ] 诊断式
        m = re.search(
            r"\*\*弹窗口吻\*\*\s*\|\s*\[[·x✓✔\s]*\]\s*(诊断式|鼓励式|child.insight|mixed)",
            section
        )
        if m:
            return _normalize_tone(m.group(1))
        # 格式3: 校标分类：弹·诊断式
        m = re.search(r"校标分类：弹·(诊断式|鼓励式)", section)
        if m:
            return _normalize_tone(m.group(1))
        # 格式4: 快速通道
        m = re.search(r"(?:tone|口吻|语气)\s*[：:]\s*(diagnostic|empowering|child_insight|mixed|诊断式|鼓励式)", section)
        if m:
            return _normalize_tone(m.group(1))
        return ""

    @staticmethod
    def _extract_trigger_sentence(section: str) -> str:
        """提取弹窗触发句。"""
        m = re.search(
            r"②\s*应该在哪个对话句子上弹窗.*?\n>\s*(.+?)(?:\n\n|\n>\s*\n|\n\*\*)",
            section, re.DOTALL
        )
        if m:
            return m.group(1).strip()
        return ""

    @staticmethod
    def _extract_hit_checklist(section: str) -> list[str]:
        """提取 ⑧ 命中清单。"""
        items: list[str] = []
        # 格式1: ⑧ 命中清单... > [ ] 1. xxx > [ ] 2. yyy
        m = re.search(
            r"⑧\s*命中清单.*?\n((?:>\s*\[[·x✓✔\s]*\].*?\n)+)",
            section
        )
        if m:
            for line in m.group(1).split("\n"):
                item = re.sub(r"^>\s*\[[·x✓✔\s]*\]\s*\d*\.?\s*", "", line).strip()
                if item and item != "___":
                    items.append(item)
            return items
        # 格式2: hit_checklist in 快速通道
        m = re.search(
            r"(?:命中清单|hit_checklist|required).*?[：:]\s*(.+?)(?:\n|$)",
            section
        )
        if m:
            text = m.group(1).strip()
            # Split by numbered items or semicolons
            parts = re.split(r"[；;]|\d+\.\s*", text)
            items = [p.strip() for p in parts if p.strip()]
        return items

    @staticmethod
    def _extract_forbid_checklist(section: str) -> list[str]:
        """提取 ⑨ 禁止清单。"""
        items: list[str] = []
        m = re.search(
            r"⑨\s*禁止清单.*?\n((?:>\s*\[[·x✓✔\s]*\].*?\n)+)",
            section, re.DOTALL
        )
        if m:
            for line in m.group(1).split("\n"):
                item = re.sub(r"^>\s*\[[·x✓✔\s]*\]\s*(?:禁止[：:]\s*)?", "", line).strip()
                if item and item != "___":
                    items.append(item)
            return items
        # 格式2: forbid_checklist in 快速通道
        m = re.search(
            r"(?:禁止清单|forbid_checklist|forbidden).*?[：:]\s*(.+?)(?:\n|$)",
            section
        )
        if m:
            text = m.group(1).strip()
            parts = re.split(r"[；;]|\d+\.\s*", text)
            items = [p.strip() for p in parts if p.strip()]
        return items

    @staticmethod
    def _extract_reference_popup(section: str) -> str:
        """提取 ⑩ 参考弹窗全文。"""
        m = re.search(
            r"参考弹窗全文.*?\n\n>\s*(.+?)(?:\n\n(?:---|\*\*|#)|$)",
            section, re.DOTALL
        )
        if m:
            text = m.group(1).strip()
            text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)
            # 去掉分隔线
            text = re.sub(r"\n>\s*-+\s*\n", "\n", text)
            return text
        return ""

    @staticmethod
    def _extract_overall_score(section: str) -> float | None:
        """提取 ⑤ 整体打分。"""
        # 格式1: ⑤ 整体打分（1~10）：\n\n> [ ] 9/10
        m = re.search(
            r"⑤\s*整体打分.+?>\s*\[[·x✓✔\s]*\]\s*(\d+)\s*/\s*10",
            section, re.DOTALL
        )
        if m:
            return float(m.group(1))
        # 格式2: 快速通道 - ⑤ 整体打分 8/10
        m = re.search(
            r"⑤\s*整体打分[^0-9]*(\d+)\s*/\s*10",
            section
        )
        if m:
            return float(m.group(1))
        # 格式3: overall_score: 8
        m = re.search(
            r"(?:overall_score|score)\s*[：:]\s*(\d+)(?:\s*/\s*10)?",
            section
        )
        if m:
            return float(m.group(1))
        return None

    @staticmethod
    def _extract_overall_feedback(section: str) -> str:
        """提取 ⑥ 整体反馈。"""
        m = re.search(
            r"⑥\s*整体反馈.*?\n>\s*(.+?)(?:\n\n|\n>\s*\n|\n\*\*)",
            section, re.DOTALL
        )
        if m:
            text = m.group(1).strip()
            text = re.sub(r"^>\s*\[[·x✓✔\s]*\]\s*", "", text, flags=re.MULTILINE)
            return text
        return ""

    @staticmethod
    def _extract_core_blind_spot(section: str) -> str:
        """提取 ⑦ 核心痛点/盲区。"""
        m = re.search(
            r"⑦\s*(?:核心痛点标注|主要矛盾标注).*?\n>\s*(?:盲区[：:]?\s*)?(.+?)(?:\n\n|\n\*\*)",
            section, re.DOTALL
        )
        if m:
            return m.group(1).strip()
        return ""

    @staticmethod
    def _extract_sentence_feedback(section: str) -> tuple[list[dict], list[dict]]:
        """提取 ④ 句级反馈（★/⚠）。"""
        good: list[dict] = []
        bad: list[dict] = []

        # 好句 ★
        for m in re.finditer(
            r"\[[·x✓✔\s]*\]\s*★\s*[>_]?\s*(.+?)\s*—\s*理由[：:]?\s*(.+?)(?:\n|$)",
            section
        ):
            text = m.group(1).strip().rstrip("_")
            reason = m.group(2).strip().rstrip("_")
            if text and text not in ("___", "____", "__________"):
                good.append({"text": text, "reason": reason})

        # 问题句 ⚠
        for m in re.finditer(
            r"\[[·x✓✔\s]*\]\s*⚠\s*[>_]?\s*(.+?)\s*—\s*理由[：:]?\s*(.+?)(?:\n|$)",
            section
        ):
            text = m.group(1).strip().rstrip("_")
            reason = m.group(2).strip().rstrip("_")
            if text and text not in ("___", "____", "__________"):
                bad.append({"text": text, "reason": reason})

        return good, bad

    @staticmethod
    def _extract_field(section: str, pattern: str, default: str = "") -> str:
        """通用字段提取。"""
        m = re.search(pattern, section, re.DOTALL)
        if m:
            return m.group(1).strip()
        return default


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_tone(tone: str) -> str:
    """统一口吻名称。"""
    t = tone.strip()
    mapping = {
        "诊断式": "diagnostic",
        "鼓励式": "empowering",
        "child_insight": "child_insight",
        "child-insight": "child_insight",
        "mixed": "mixed",
        "不弹窗": "no_popup",
        "no_popup": "no_popup",
        "diagnostic": "diagnostic",
        "empowering": "empowering",
    }
    return mapping.get(t, t)


# ---------------------------------------------------------------------------
# Summary & validation
# ---------------------------------------------------------------------------


def summarize(cases: list[GoldenCase]) -> None:
    """打印解析摘要。"""
    total = len(cases)
    total_windows = sum(len(c.windows) for c in cases)
    with_popup = sum(
        1 for c in cases for w in c.windows if w.should_popup is True
    )
    with_tone = sum(
        1 for c in cases for w in c.windows if w.expected_tone
    )
    with_hit = sum(
        1 for c in cases for w in c.windows if w.hit_checklist
    )
    with_forbid = sum(
        1 for c in cases for w in c.windows if w.forbid_checklist
    )
    with_ref = sum(
        1 for c in cases for w in c.windows if w.reference_popup.strip()
    )
    with_score = sum(
        1 for c in cases for w in c.windows if w.overall_score is not None
    )

    annotators: dict[str, int] = {}
    for c in cases:
        annotators[c.annotator] = annotators.get(c.annotator, 0) + 1

    print(f"📊 解析摘要")
    print(f"  案例总数: {total}")
    print(f"  窗口总数: {total_windows}")
    print(f"  should_popup 标注: {with_popup}")
    print(f"  expected_tone 标注: {with_tone}")
    print(f"  命中清单: {with_hit}")
    print(f"  禁止清单: {with_forbid}")
    print(f"  参考弹窗: {with_ref}")
    print(f"  整体打分: {with_score}")
    print(f"  标注者分布: {annotators}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(description="解析专家手动标注为黄金数据集")
    parser.add_argument(
        "--input", "-i",
        default="C:/Users/h/Desktop/专家手动打标_全部用例汇整_20260716_141130.md",
        help="专家标注 markdown 文件路径",
    )
    parser.add_argument(
        "--output", "-o",
        default="D:/prompt-ops/use-cases/parent-child-coach/data/golden_dataset.json",
        help="输出 JSON 路径",
    )
    args = parser.parse_args()

    gdp = GoldenDatasetParser(args.input)
    cases = gdp.parse()
    summarize(cases)
    gdp.save(cases, args.output)


if __name__ == "__main__":
    main()
