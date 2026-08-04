"""演示：快慢通道窗口逻辑（严格按统一规格）。

把「亲子沟通洞见弹窗」触发层的窗口逻辑做成一个可独立运行的演示脚本：
  - 慢通道：缓冲凑满 300 字才分析
  - 快通道 critical（severity≥4）：命中当下就弹，向前取最多 300 字，<80 取消
  - 快通道一般严重：向前取 250 字，命中后等 50 字再试图分析，总字数 <80 取消

这是 @persistent 演示/测试工具，参数统一取自 channel_spec（唯一权威）。

用法：
    cd use-cases/parent-child-coach
    python demos/run_demo.py
"""

import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

# 让本脚本能 import 同目录上级的 channel_spec
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from channel_spec import (
    CRITICAL_SEVERITY_MIN,
    SLOW_THRESHOLD_CHARS,
    FAST_CRITICAL_FORWARD,
    FAST_GENERAL_FORWARD,
    FAST_GENERAL_WAIT,
    FAST_MIN_CHARS,
    is_critical,
)


@dataclass
class Keyword:
    """一个关键词及其严重度（1-5）。"""
    text: str
    severity: int


@dataclass
class WindowResult:
    """一次快通道窗口截取的决策结果。"""
    triggered: bool = False
    channel: str = ""              # "critical" | "general" | "slow"
    context_window: str = ""
    char_count: int = 0
    reason: str = ""


def _snap_start_to_boundary(buffer: str, start: int, limit: int) -> int:
    """把窗口起点向后对齐到最近的句子边界，避免从句子中间截断。"""
    for i in range(start, min(start + 30, limit)):
        if i < len(buffer) and buffer[i] in {"。", "！", "？", "\n", "，"}:
            return i + 1
    return start


def extract_window(
    buffer: str,
    trigger_pos: int,
    severity: int,
    pending: bool = False,
) -> WindowResult:
    """按统一规格做快通道窗口截取。

    触发点（trigger_pos）为命中关键词在 buffer 中的绝对位置。
    需要「等 50 字」而未凑够时，由调用方（simulate_pipeline）负责挂起。

    Args:
        buffer: 当前完整缓冲
        trigger_pos: 关键词命中位置（绝对索引）
        severity: 命中关键词的严重度（1-5）
        pending: 是否为挂起解析（一般严重已等够 50 字）

    Returns:
        WindowResult
    """
    if is_critical(severity):
        # critical：当下就弹，仅向前取窗
        start = max(0, trigger_pos - FAST_CRITICAL_FORWARD)
        start = _snap_start_to_boundary(buffer, start, trigger_pos)
        ctx = buffer[start:trigger_pos]
        if len(ctx) < FAST_MIN_CHARS:
            return WindowResult(triggered=False, channel="critical",
                                reason=f"critical <{FAST_MIN_CHARS}字取消")
        return WindowResult(triggered=True, channel="critical",
                            context_window=ctx, char_count=len(ctx))

    # 一般严重：向前 FAST_GENERAL_FORWARD + 向后等 FAST_GENERAL_WAIT
    start = max(0, trigger_pos - FAST_GENERAL_FORWARD)
    start = _snap_start_to_boundary(buffer, start, trigger_pos)
    end = min(len(buffer), trigger_pos + FAST_GENERAL_WAIT)
    ctx = buffer[start:end]
    if len(ctx) < FAST_MIN_CHARS:
        return WindowResult(triggered=False, channel="general",
                            reason=f"general <{FAST_MIN_CHARS}字取消")
    return WindowResult(triggered=True, channel="general",
                        context_window=ctx, char_count=len(ctx))


@dataclass
class SlowWindow:
    """一个慢通道 300 字分析窗口。"""
    index: int
    context_window: str
    char_count: int


def _first_keyword(text: str, keywords: List[Keyword]) -> Optional[tuple]:
    """在新到达文本中找第一个命中关键词，返回 (keyword, idx, severity)。

    只匹配新文本（不做全量重扫），与生产 SUT 的逐 feed 去重语义一致，
    避免同一关键词在后续 chunk 重复触发。
    """
    for kw in sorted(keywords, key=lambda k: len(k.text), reverse=True):
        idx = text.find(kw.text)
        if idx >= 0:
            return (kw.text, idx, kw.severity)
    return None


@dataclass
class Popup:
    """一次弹窗输出。"""
    channel: str
    trigger_type: str      # "关键词触发" | "字数触发"
    tone: str
    context_window: str
    char_count: int


def simulate_pipeline(
    text: str,
    keywords: List[Keyword],
    chunk_size: int = 40,
) -> List[Popup]:
    """按统一规格模拟流式触发管线，输出弹窗序列。

    - 快通道：critical 当下弹；一般严重命中后挂起，等缓冲再进 50 字后解析
    - 慢通道：缓冲凑满 300 字，按 300 字窗口切分分析（累积前文，不冷启动）
    """
    popups: List[Popup] = []
    buffer = ""
    slow_accum: List[SlowWindow] = []
    pending_general: Optional[dict] = None  # {"pos":..,"kw":..,"sev":..}
    offset = 0  # 缓冲中「已扫描过」的位置——只匹配新文本

    # 流式喂入
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        buffer += chunk

        # 快通道：只在本段新文本中找关键词，绝对位置 = 扫描起点 + 段内索引
        found = _first_keyword(chunk, keywords)
        if found:
            kw, idx_in_chunk, sev = found
            pos = offset + idx_in_chunk
            if is_critical(sev):
                r = extract_window(buffer, pos, sev)
                if r.triggered:
                    popups.append(Popup("fast", "关键词触发", "提醒型",
                                        r.context_window, r.char_count))
                pending_general = None
            else:
                # 一般严重：等缓冲够 trigger_pos + 50
                if len(buffer) < pos + FAST_GENERAL_WAIT:
                    pending_general = {"pos": pos, "kw": kw, "sev": sev}
                else:
                    r = extract_window(buffer, pos, sev)
                    if r.triggered:
                        popups.append(Popup("fast", "关键词触发", "提醒型",
                                            r.context_window, r.char_count))
                    pending_general = None

        # 无新命中：解析挂起的一般严重（等够 50 字）
        elif pending_general is not None:
            if len(buffer) >= pending_general["pos"] + FAST_GENERAL_WAIT:
                r = extract_window(buffer, pending_general["pos"],
                                   pending_general["sev"])
                pending_general = None
                if r.triggered:
                    popups.append(Popup("fast", "关键词触发", "提醒型",
                                        r.context_window, r.char_count))

        # 慢通道：缓冲凑满 300 字，按窗口分析（累积前文）
        if len(buffer) >= SLOW_THRESHOLD_CHARS:
            n = len(buffer) // SLOW_THRESHOLD_CHARS
            for k in range(n):
                seg = buffer[k * SLOW_THRESHOLD_CHARS:
                             (k + 1) * SLOW_THRESHOLD_CHARS]
                if seg.strip():
                    slow_accum.append(SlowWindow(len(slow_accum), seg,
                                                 len(seg)))
                    popups.append(Popup("slow", "字数触发", "洞察型",
                                        seg, len(seg)))

        # 本段已扫描，推进扫描起点
        offset += len(chunk)

    return popups


def main():
    """跑两个演示案例，打印快慢通道弹窗序列。"""
    keywords = [
        Keyword("你再哭我就不要你了", 5),   # critical
        Keyword("我像你这么大", 3),         # 一般严重
        Keyword("你不知足", 3),             # 一般严重
        Keyword("你看看人家", 3),           # 一般严重
        Keyword("我说了算", 4),             # critical
    ]

    print("=" * 60)
    print("快慢通道窗口演示（统一规格）")
    print(f"  critical≥{CRITICAL_SEVERITY_MIN} · 慢通道{SLOW_THRESHOLD_CHARS}字"
          f" · 快critical向前{FAST_CRITICAL_FORWARD} · 一般向前{FAST_GENERAL_FORWARD}"
          f" +等{FAST_GENERAL_WAIT} · <{FAST_MIN_CHARS}取消")
    print("=" * 60)

    cases = {
        "案例A（critical·前文≥80 当下弹）": (
            "孩子一放学就把书包扔在地上，作业也不写，趴在沙发上看动画片，"
            "妈妈从厨房出来叫了三遍让他先去洗手吃饭，孩子头也不抬只顾着看，"
            "妈妈越说越生气，走过去一把按掉了电视，又忍不住数落起来，"
            "说他天天就知道看动画，作业一个字都不写，再这样下去干脆别上学了，"
            "你再哭我就不要你了，说着妈妈转身就走，留下孩子一个人愣在原地。"
        ),
        "案例B（一般严重·等50字后弹）": (
            "孩子低着头不说话，眼眶有点发红，两只手紧紧攥着衣角，肩膀微微发抖，"
            "过了好一会儿才小声说，我像你这么大的时候，爷爷对我可比你耐心多了，"
            "从来不会这样催我，说着眼泪就掉下来了，妈妈愣了一下，不知道该说什么才好。"
        ),
        "案例C（critical·前文<80 按规格取消）": (
            "你再哭我就不要你了。"
        ),
    }

    for name, text in cases.items():
        print(f"\n▶ {name}")
        popups = simulate_pipeline(text, keywords)
        for p in popups:
            print(f"  [{p.channel}] {p.trigger_type} · tone={p.tone} · "
                  f"{p.char_count}字: {p.context_window[:40]}...")


if __name__ == "__main__":
    main()
