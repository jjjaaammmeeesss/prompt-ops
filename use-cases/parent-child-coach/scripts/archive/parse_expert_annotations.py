"""
解析 5 个专家打标 markdown 文件 → 统一 JSON 数据集。

支持三种格式：
  1. worksheet_multi_window  — 13cases, 快速通道, 快速通道_v2
  2. worksheet_single_window — 10cases (晓浩)
  3. manual_review           — manual_review_20260626

用法: python scripts/parse_expert_annotations.py
输出: data/expert_dataset.json
"""

import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

# === Paths ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "人工打标用例_52个")
OUTPUT_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "expert_dataset.json")

# === Regex patterns ===
# Checkbox states: [x] (checked), [·] or [-] (dot/dash checked), [ ] (empty)
CHECKED_RE = re.compile(r"\[[x·\-]\]")
EMPTY_CHECK_RE = re.compile(r"\[\s*\]")

# Dialogue line number stripping
DIALOGUE_NUM_RE = re.compile(r"^\d+[：:\.．]\s*")

# Window label pattern — matches full header line including range
WINDOW_HEADER_RE = re.compile(
    r"###\s*(甲|乙|丙|丁|戊|己|庚|辛|壬|癸)\s*·\s*300字窗口\s*·\s*句\s*(\d+)\s*[-–]\s*(\d+)"
)

# Case boundary patterns
CASE_N_M_RE = re.compile(r"^##\s*Case\s+(\d+)/(\d+)\s*[:：]")
GB_CASE_RE = re.compile(r"^##\s*(GB_\d+)")

# Field extraction patterns
FIELD_SHOULD_POPUP_RE = re.compile(r"[①②]?\s*是否该弹[窗]?\s*[：:]")
FIELD_TRIGGER_RE = re.compile(r"[①②]?\s*应该在哪[个些]?对话句子上弹窗|触发句")
FIELD_TONE_RE = re.compile(r"[①②③]?\s*应该弹什么口吻|弹窗口吻")
FIELD_SENTENCE_FB_RE = re.compile(r"[④]\s*弹窗内容句级反馈")
FIELD_OVERALL_SCORE_RE = re.compile(r"[⑤]\s*整体打分")
FIELD_OVERALL_FB_RE = re.compile(r"[⑥]\s*整体反馈")
FIELD_CORE_ISSUE_RE = re.compile(r"[⑦]\s*(核心痛点标注|主要矛盾标注)")
FIELD_HIT_LIST_RE = re.compile(r"[⑧]\s*命中清单")
FIELD_FORBIDDEN_RE = re.compile(r"[⑨]\s*禁止清单")
FIELD_STANDARD_RE = re.compile(r"[⑩]\s*改写为校标")
FIELD_REF_POPUP_RE = re.compile(r"参考弹窗全文")

# Score extraction
SCORE_RE = re.compile(r"(\d+)\s*/\s*10")
TRIGGER_SENTENCE_RE = re.compile(r"句\s*(\d+)")
GOOD_SENTENCE_RE = re.compile(r"\[[x·\-\s]\]\s*★\s*(.+?)(?:\s*[—\-]\s*理由[：:]\s*(.*?))?\s*$", re.MULTILINE)
PROBLEM_SENTENCE_RE = re.compile(r"\[[x·\-\s]\]\s*⚠\s*(.+?)(?:\s*[—\-]\s*理由[：:]\s*(.*?))?\s*$", re.MULTILINE)

# System popup metadata
SYSTEM_TONE_RE = re.compile(r"\*\*语气\*\*\s*[:：]\s*(\S+)")
SYSTEM_CHANNEL_RE = re.compile(r"\*\*通道\*\*\s*[:：]\s*(\S+)")


@dataclass
class ExpertRecord:
    """Single expert-annotated window record."""
    id: str
    source_file: str
    case_title: str = ""
    window_label: str = ""
    sentence_range: list[int] = field(default_factory=lambda: [0, 0])
    dialogue: str = ""
    system_popup: str = ""
    system_tone: str = ""
    system_channel: str = ""
    expert_score: Optional[int] = None
    expert_tone: str = ""
    should_popup: Optional[bool] = None
    trigger_sentences: list[int] = field(default_factory=list)
    core_blind_spot: str = ""
    good_sentences: list[dict] = field(default_factory=list)
    problem_sentences: list[dict] = field(default_factory=list)
    hit_checklist: list[str] = field(default_factory=list)
    forbidden_list: list[str] = field(default_factory=list)
    reference_popup: str = ""
    overall_feedback: str = ""
    expert_name: str = ""


def is_checked(line: str) -> bool:
    """Check if a line has a checked checkbox ([x], [·], or [-])."""
    return bool(CHECKED_RE.search(line))


def strip_dialogue_numbers(text: str) -> str:
    """Remove leading line numbers from dialogue text."""
    lines = []
    for line in text.strip().split("\n"):
        line = line.strip()
        line = DIALOGUE_NUM_RE.sub("", line)
        lines.append(line)
    return "\n".join(lines)


def extract_score(text: str) -> Optional[int]:
    """Extract 1-10 score from text like '7 / 10' or '9/10'."""
    m = SCORE_RE.search(text)
    if m:
        score = int(m.group(1))
        if 1 <= score <= 10:
            return score
    return None


def extract_trigger_sentences(text: str) -> list[int]:
    """Extract trigger sentence numbers from annotation text."""
    return [int(m) for m in TRIGGER_SENTENCE_RE.findall(text)]


def extract_good_problem_sentences(text: str) -> tuple[list[dict], list[dict]]:
    """Extract ★ good and ⚠ problem sentences with reasons."""
    good = []
    problem = []

    # Match ★ good sentences (checked ones)
    for m in GOOD_SENTENCE_RE.finditer(text):
        content = m.group(1).strip()
        reason = (m.group(2) or "").strip()
        if content and content not in ("_______", "_______________"):
            good.append({"text": content, "reason": reason})

    # Match ⚠ problem sentences (checked ones)
    for m in PROBLEM_SENTENCE_RE.finditer(text):
        content = m.group(1).strip()
        reason = (m.group(2) or "").strip()
        if content and content not in ("_______", "_______________"):
            problem.append({"text": content, "reason": reason})

    return good, problem


def extract_list_items(text: str, prefix_pattern: str = r"\d+\.") -> list[str]:
    """Extract numbered list items from text."""
    items = []
    lines = text.strip().split("\n")
    for line in lines:
        line = line.strip()
        # Match "[x] N. text" or "N. text" patterns
        m = re.match(rf"(?:\[[x·\-\s]\]\s*)?{prefix_pattern}\s*(.+)", line)
        if m:
            item = m.group(1).strip()
            if item and "_______" not in item:
                items.append(item)
    return items


def extract_section(text: str, start_pattern: str, end_patterns: list[str] | None = None) -> str:
    """Extract text between a start marker and the next section marker."""
    lines = text.split("\n")
    result = []
    in_section = False

    for line in lines:
        if not in_section:
            if re.search(start_pattern, line):
                in_section = True
                # Include the rest of this line after the marker
                remaining = re.sub(start_pattern, "", line).strip()
                if remaining:
                    result.append(remaining)
                continue
        else:
            # Check if this line starts a new section
            if end_patterns:
                is_end = any(re.search(p, line) for p in end_patterns)
            else:
                is_end = bool(re.match(r"^#{2,4}\s", line))

            if is_end:
                break
            result.append(line)

    return "\n".join(result).strip()


# ============================================================
# Format 1 & 2: Worksheet parser (multi-window & single-window)
# ============================================================

def parse_worksheet_file(filepath: str, expert_name: str) -> list[ExpertRecord]:
    """Parse a worksheet-format file (10cases, 13cases, 快速通道, 快速通道_v2)."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    filename = os.path.basename(filepath)
    records = []

    # Find all case boundaries
    case_starts = list(re.finditer(r"^##\s*Case\s+(\d+)/(\d+)\s*[:：]", content, re.MULTILINE))

    for idx, m in enumerate(case_starts):
        case_num = m.group(1)
        start_pos = m.end()
        end_pos = case_starts[idx + 1].start() if idx + 1 < len(case_starts) else len(content)
        case_text = content[start_pos:end_pos]

        records.extend(parse_single_worksheet_case(case_text, case_num, filename, expert_name))

    return records


def parse_single_worksheet_case(
    case_text: str, case_num: str, filename: str, expert_name: str
) -> list[ExpertRecord]:
    """Parse a single case from a worksheet file into 1+ window records."""
    records = []

    # Extract title
    title_match = re.search(r"[*]*案例[：:]\s*(.+?)(?:\(ID:|$)", case_text)
    if not title_match:
        title_match = re.search(r"[-*]\s*标题[：:]\s*(.+)", case_text)
    case_title = title_match.group(1).strip() if title_match else f"Case_{case_num}"

    # Extract complete dialogue
    dialogue = ""
    dialogue_match = re.search(r"###\s*💬\s*完整对话\s*\n```\s*\n(.*?)```", case_text, re.DOTALL)
    if dialogue_match:
        dialogue = strip_dialogue_numbers(dialogue_match.group(1))

    # Find all windows
    window_matches = list(WINDOW_HEADER_RE.finditer(case_text))

    for idx, wm in enumerate(window_matches):
        label = wm.group(1)
        sent_start = int(wm.group(2))
        sent_end = int(wm.group(3))
        start_pos = wm.start()

        # Find window end (next window or next case)
        next_window_pos = len(case_text)
        if idx + 1 < len(window_matches):
            next_window_pos = window_matches[idx + 1].start()
        # Also check for next section markers
        for pattern in [r"\n##\s+Case", r"\n---\s*\n"]:
            m = re.search(pattern, case_text[start_pos:])
            if m and start_pos + m.start() < next_window_pos:
                next_window_pos = start_pos + m.start()

        window_text = case_text[start_pos:next_window_pos]

        # Parse system popup
        system_popup = ""
        system_tone = ""
        system_channel = ""
        popup_triggered = True

        popup_match = re.search(
            r"####\s*🤖\s*系统弹窗(?:[（(]句\s*\d+\s*触发[)）])?\s*\n(.*?)(?=####|$)",
            window_text, re.DOTALL,
        )
        if popup_match:
            popup_section = popup_match.group(1)

            # Extract tone/channel metadata
            tone_m = SYSTEM_TONE_RE.search(popup_section)
            if tone_m:
                system_tone = tone_m.group(1)
            channel_m = SYSTEM_CHANNEL_RE.search(popup_section)
            if channel_m:
                system_channel = channel_m.group(1)

            # Extract actual popup text (the blockquote part)
            popup_lines = []
            for line in popup_section.split("\n"):
                stripped = line.strip()
                if stripped.startswith(">") and "ℹ️" not in stripped:
                    popup_text = stripped.lstrip("> ").strip()
                    if popup_text and "语气" not in popup_text and "通道" not in popup_text:
                        popup_lines.append(popup_text)
            system_popup = "\n".join(popup_lines).strip()
        else:
            # Check if system didn't pop up
            if re.search(r"系统未处理|系统判定[：:]\s*不弹窗", window_text):
                popup_triggered = False

        # Parse expert annotation area
        annotation_match = re.search(
            r"####\s*✍️\s*专家标注区\s*\n(.*?)(?=####|$)",
            window_text, re.DOTALL,
        )
        anno_text = annotation_match.group(1) if annotation_match else ""

        # Extract expert fields
        should_popup = None
        tone = ""
        score = None
        trigger_sents = []
        core_issue = ""
        overall_fb = ""
        good_sents = []
        problem_sents = []
        hit_list = []
        forbidden = []
        ref_popup = ""

        # ① Should popup
        should_section = extract_section(anno_text, r"[①1]\s*是否该弹|\[①1\]\s*是否该弹")
        if should_section:
            if re.search(r"该弹|是", should_section):
                should_popup = True
            elif re.search(r"不该弹|否|不弹", should_section):
                should_popup = False

        # ② Trigger sentence
        trigger_section = extract_section(anno_text, r"[②2]\s*应该")
        if trigger_section:
            trigger_sents = extract_trigger_sentences(trigger_section)

        # ③ Tone
        tone_section = extract_section(anno_text, r"[③3]\s*应该弹什么口吻|弹窗口吻")
        if tone_section:
            if "诊断" in tone_section:
                tone = "诊断式"
            elif "鼓励" in tone_section:
                tone = "鼓励式"

        # ④ Sentence feedback
        fb_section = extract_section(anno_text, r"[④4]\s*弹窗内容句级反馈")
        if fb_section:
            good_sents, problem_sents = extract_good_problem_sentences(fb_section)

        # ⑤ Overall score
        score_section = extract_section(anno_text, r"[⑤5]\s*整体打分")
        if score_section:
            score = extract_score(score_section)

        # ⑥ Overall feedback
        overall_fb = extract_section(anno_text, r"[⑥6]\s*整体反馈")

        # ⑦ Core blind spot / pain point
        core_section = extract_section(anno_text, r"[⑦7]\s*(核心痛点标注|主要矛盾标注)")
        if core_section:
            core_issue = re.sub(r"^盲区[：:]?\s*", "", core_section.strip())

        # ⑧ Hit checklist
        hit_section = extract_section(anno_text, r"[⑧8]\s*命中清单")
        if hit_section:
            hit_list = extract_list_items(hit_section)

        # ⑨ Forbidden list
        forbidden_section = extract_section(anno_text, r"[⑨9]\s*禁止清单")
        if forbidden_section:
            for line in forbidden_section.strip().split("\n"):
                line = line.strip()
                # Match "禁止：text" or "[x] 禁止：text"
                m = re.match(r"(?:\[[x·\-\s]\]\s*)?禁止[：:]\s*(.+)", line)
                if m and "_______" not in m.group(1):
                    forbidden.append(m.group(1).strip())

        # ⑩ Reference popup (expert rewritten)
        ref_section = extract_section(anno_text, r"参考弹窗全文")
        if ref_section:
            ref_popup = ref_section.strip()

        # Also try extracting reference popup from the full window text
        if not ref_popup:
            ref_match = re.search(
                r"参考弹窗全文[（(]请专家手写弹窗正文[)）]?\s*\n>\s*(.+?)(?:\n\n|\n#{2,4}|\Z)",
                window_text, re.DOTALL,
            )
            if ref_match:
                ref_lines = []
                for line in ref_match.group(1).split("\n"):
                    stripped = line.strip()
                    if stripped.startswith(">"):
                        ref_lines.append(stripped.lstrip("> ").strip())
                ref_popup = "\n".join(ref_lines).strip()

        record_id = f"{filename.replace('.md', '')}_case{case_num}_w{label}"
        records.append(ExpertRecord(
            id=record_id,
            source_file=filename,
            case_title=case_title,
            window_label=label,
            sentence_range=[sent_start, sent_end],
            dialogue=dialogue,
            system_popup=system_popup,
            system_tone=system_tone,
            system_channel=system_channel,
            expert_score=score,
            expert_tone=tone,
            should_popup=should_popup,
            trigger_sentences=trigger_sents,
            core_blind_spot=core_issue,
            good_sentences=good_sents,
            problem_sentences=problem_sents,
            hit_checklist=hit_list,
            forbidden_list=forbidden,
            reference_popup=ref_popup,
            overall_feedback=overall_fb,
            expert_name=expert_name,
        ))

    return records


# ============================================================
# Format 3: Manual review parser
# ============================================================

def parse_manual_review_file(filepath: str) -> list[ExpertRecord]:
    """Parse manual_review_20260626.md — sentence-level annotations, no dialogue."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    filename = os.path.basename(filepath)
    records = []

    # Split by GB_XXX section headers
    sections = re.split(r"\n(?=## GB_)", content)

    for section in sections:
        case_match = re.match(r"##\s*(GB_\d+)", section)
        if not case_match:
            continue
        case_id = case_match.group(1)

        # Extract title
        title_match = re.match(r"##\s*GB_\d+[：:]\s*(.+)", section.split("\n")[0])
        case_title = title_match.group(1).strip() if title_match else case_id

        # Extract classification
        class_match = re.search(r"校标分类[：:]\s*(.+)", section)
        classification = class_match.group(1).strip() if class_match else ""

        # Extract popup text
        popup_match = re.search(r"###\s*弹窗全文\s*\n```\s*\n(.*?)```", section, re.DOTALL)
        system_popup = popup_match.group(1).strip() if popup_match else ""

        # Extract sentence annotations
        good_sents = []
        problem_sents = []
        sent_pattern = re.compile(
            r"\*\*句\s*(\d+)\*\*[：:]\s*(.+?)\n"
            r"(?:.*?\n)*?"
            r"-\s*\[\s*[x]?\s*\]\s*★\s*好句子[—\-]*\s*理由[：:]\s*(.*?)\n"
            r"-\s*\[\s*[x]?\s*\]\s*⚠\s*问题句[—\-]*\s*理由[：:]\s*(.*?)(?:\n|$)",
            re.DOTALL,
        )
        # Simpler approach: extract sentence text and check marks
        sent_blocks = re.split(r"\*\*句\s*\d+\*\*", section)[1:]
        for block in sent_blocks:
            lines = block.strip().split("\n")
            text = lines[0].strip().lstrip("：:").strip() if lines else ""
            has_gold = any("★ 好句子" in line and "[x]" in line for line in lines)
            has_problem = any("⚠ 问题句" in line and "[x]" in line for line in lines)

            gold_reason = ""
            problem_reason = ""
            for line in lines:
                if "★ 好句子" in line and "理由" in line:
                    gold_reason = re.sub(r".*理由[：:]\s*", "", line).strip()
                if "⚠ 问题句" in line and "理由" in line:
                    problem_reason = re.sub(r".*理由[：:]\s*", "", line).strip()

            if has_gold:
                good_sents.append({"text": text, "reason": gold_reason})
            if has_problem:
                problem_sents.append({"text": text, "reason": problem_reason})

        # Extract overall judgment
        overall_match = re.search(
            r"###\s*整体判断\s*\n(.*?)(?:\n##|\Z)", section, re.DOTALL
        )
        overall_judgment = ""
        if overall_match:
            overall_text = overall_match.group(1)
            if re.search(r"\[x\]\s*整体合格", overall_text):
                overall_judgment = "整体合格"
                should_popup = True
            elif re.search(r"\[x\]\s*需要修改", overall_text):
                overall_judgment = "需要修改"
                should_popup = True
            elif re.search(r"\[x\]\s*系统正确", overall_text):
                overall_judgment = "系统正确"
                should_popup = False
            elif re.search(r"\[x\]\s*系统错误", overall_text):
                overall_judgment = "系统错误"
                should_popup = True
            else:
                should_popup = None
        else:
            should_popup = None

        # Determine tone from classification
        tone = ""
        if "鼓励" in classification:
            tone = "鼓励式"
        elif "诊断" in classification:
            tone = "诊断式"

        records.append(ExpertRecord(
            id=case_id,
            source_file=filename,
            case_title=case_title,
            dialogue="",  # No dialogue in this format
            system_popup=system_popup,
            expert_tone=tone,
            should_popup=should_popup,
            good_sentences=good_sents,
            problem_sentences=problem_sents,
            overall_feedback=overall_judgment,
            expert_name="系统评审",
        ))

    return records


# ============================================================
# Main
# ============================================================

def main():
    all_records = []

    # File → expert name mapping
    files_config = [
        ("worksheet_10cases-晓浩已手写改完.md", "晓浩"),
        ("worksheet_13cases_专家打标_20260710_114609-子阳手动改写13用例.md", "子阳"),
        ("worksheet_快速通道_20260712_222129OK-廖老师打标.md", "廖老师"),
        ("worksheet_快速通道_v2_20260713_184046-子阳打标4用例.md", "子阳"),
    ]

    for filename, expert in files_config:
        filepath = os.path.join(INPUT_DIR, filename)
        if not os.path.exists(filepath):
            print(f"⚠ 文件不存在，跳过: {filepath}")
            continue
        print(f"解析: {filename} (标注人: {expert})")
        records = parse_worksheet_file(filepath, expert)
        print(f"  → {len(records)} 条窗口级记录")
        all_records.extend(records)

    # Manual review (special format)
    manual_path = os.path.join(INPUT_DIR, "manual_review_20260626.md")
    if os.path.exists(manual_path):
        print(f"解析: manual_review_20260626.md (旧格式)")
        manual_records = parse_manual_review_file(manual_path)
        print(f"  → {len(manual_records)} 条记录")
        all_records.extend(manual_records)

    # Deduplicate by dialogue hash (keep first occurrence)
    seen_hashes = set()
    deduped = []
    for r in all_records:
        h = hash(r.dialogue + r.system_popup) if (r.dialogue or r.system_popup) else id(r)
        if h not in seen_hashes:
            seen_hashes.add(h)
            deduped.append(r)

    print(f"\n总计: {len(all_records)} 条 (去重后: {len(deduped)} 条)")

    # Serialize
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output = [asdict(r) for r in deduped]
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Summary stats
    has_dialogue = sum(1 for r in deduped if r.dialogue)
    has_popup = sum(1 for r in deduped if r.system_popup)
    has_score = sum(1 for r in deduped if r.expert_score is not None)
    has_ref_popup = sum(1 for r in deduped if r.reference_popup)

    print(f"\n数据集统计:")
    print(f"  有对话文本: {has_dialogue}")
    print(f"  有系统弹窗: {has_popup}")
    print(f"  有专家评分 (1-10): {has_score}")
    print(f"  有专家手写参考弹窗: {has_ref_popup}")
    print(f"\n输出: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
