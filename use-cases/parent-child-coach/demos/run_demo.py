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
    FAST_BACKGROUND,
    FAST_MIN_CHARS,
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
    channel: str = ""              # "fast" | "slow"
    context_window: str = ""
    background: str = ""           # 触发点前 FAST_BACKGROUND 字背景（含分析窗口），仅作理解
    char_count: int = 0
    window_start: int = 0          # 窗口在对话缓冲中的起始位置
    reason: str = ""


def _snap_start_to_boundary(buffer: str, start: int, limit: int = None) -> int:
    """把窗口起点往回（左）对齐到最近的句子边界，不漏句子。"""
    for i in range(start - 1, max(0, start - 30) - 1, -1):
        if i >= 0 and buffer[i] in {"。", "！", "？", "\n", "，"}:
            return i + 1
    return start


def extract_window(
    buffer: str,
    trigger_pos: int,
) -> WindowResult:
    """按统一规格做快通道窗口截取（三类词：严重/警告/机会 统一）。

    触发点（trigger_pos）为命中关键词在 buffer 中的绝对位置，命中当下即刻分析。
    分析窗口 = 触发点前 FAST_CRITICAL_FORWARD(150) 字（对齐句子边界）；
    背景 = 触发点前 FAST_BACKGROUND(900) 字（含分析窗口，仅作理解）。
    窗口有效内容 < FAST_MIN_CHARS(80) 取消。

    Args:
        buffer: 当前完整缓冲
        trigger_pos: 关键词命中位置（绝对索引）

    Returns:
        WindowResult
    """
    start = max(0, trigger_pos - FAST_CRITICAL_FORWARD)
    start = _snap_start_to_boundary(buffer, start, trigger_pos)
    ctx = buffer[start:trigger_pos]
    if len(ctx) < FAST_MIN_CHARS:
        return WindowResult(triggered=False, channel="fast",
                            reason=f"fast <{FAST_MIN_CHARS}字取消")
    bg_start = max(0, trigger_pos - FAST_BACKGROUND)
    return WindowResult(triggered=True, channel="fast",
                        context_window=ctx, char_count=len(ctx),
                        window_start=start,
                        background=buffer[bg_start:start])


@dataclass
class SlowWindow:
    """一个慢通道 300 字分析窗口。"""
    index: int
    context_window: str
    char_count: int


def _first_keyword(
    text: str,
    keywords: List[Keyword],
    min_abs_pos: int = 0,
    base_pos: int = 0,
) -> Optional[tuple]:
    """在新到达文本中找第一个（绝对位置 ≥ min_abs_pos 的）命中关键词。

    只匹配新文本（不做全量重扫），与生产 SUT 的逐 feed 去重语义一致。
    min_abs_pos 用于 80 字保护窗口：命中后其后的关键词被过滤，防同一段重复触发。

    Args:
        text: 当前 chunk 文本
        keywords: 关键词列表
        min_abs_pos: 绝对位置扫描下限（保护窗口过滤）
        base_pos: chunk 起始绝对位置（text[0] 在完整 buffer 中的索引）

    Returns:
        (keyword_text, idx_in_chunk, severity) 或 None
    """
    best = None  # (kw, abs_pos)
    for kw in sorted(keywords, key=lambda k: len(k.text), reverse=True):
        idx = text.find(kw.text)
        if idx >= 0:
            pos = base_pos + idx
            if pos >= min_abs_pos:
                if best is None or pos < best[1]:
                    best = (kw, pos)
    if best:
        kw, pos = best
        return (kw.text, pos - base_pos, kw.severity)
    return None


@dataclass
class Popup:
    """一次弹窗输出。"""
    channel: str
    trigger_type: str      # "关键词触发" | "字数触发"
    tone: str
    context_window: str
    char_count: int
    window_start: int = 0  # 窗口在对话缓冲中的起始位置
    background: str = ""   # 触发点前 FAST_BACKGROUND 字背景（仅作理解，非生成依据）


def simulate_pipeline(
    text: str,
    keywords: List[Keyword],
    chunk_size: int = 40,
    slow_threshold: int = None,
) -> List[Popup]:
    """按统一规格模拟流式触发管线，输出弹窗序列。

    - 快通道：critical 当下弹；一般严重命中后挂起，等缓冲再进 50 字后解析
    - 慢通道：缓冲凑满 slow_threshold 字，按 slow_threshold 字窗口切分分析

    Args:
        text: 完整对话文本
        keywords: 关键词列表
        chunk_size: 流式喂入的 chunk 大小
        slow_threshold: 慢通道触发字数（默认取 channel_spec.slow_threshold）
    """
    if slow_threshold is None:
        slow_threshold = SLOW_THRESHOLD_CHARS
    popups: List[Popup] = []
    buffer = ""
    slow_accum: List[SlowWindow] = []
    slow_cursor = 0  # 慢通道已处理到的位置，快通道触发后重置
    guard_until = 0  # 80 字保护窗口：命中后该绝对位置内的新关键词不再触发
    offset = 0  # 缓冲中「已扫描过」的位置——只匹配新文本

    # 流式喂入
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        buffer += chunk

        fast_fired = False  # 本轮是否有快通道触发

        # 快通道（三类词统一）：只在本段新文本中找关键词，绝对位置 = 扫描起点 + 段内索引。
        # 命中后设置保护窗口 guard_until = pos + FAST_MIN_CHARS，其内新关键词被过滤。
        found = _first_keyword(chunk, keywords,
                               min_abs_pos=guard_until, base_pos=offset)
        if found:
            kw, idx_in_chunk, _ = found
            pos = offset + idx_in_chunk
            r = extract_window(buffer, pos)
            if r.triggered:
                popups.append(Popup("fast", "关键词触发", "提醒型",
                                    r.context_window, r.char_count,
                                    window_start=r.window_start,
                                    background=r.background))
                fast_fired = True
                guard_until = pos + FAST_MIN_CHARS  # 保护窗口：该位置内不再扫描

        # 快慢互斥：快通道触发后慢通道重新计数
        if fast_fired:
            slow_cursor = len(buffer)
        else:
            # 慢通道：从未处理位置开始，凑满 300 字才分析
            unprocessed = buffer[slow_cursor:]
            if len(unprocessed) >= slow_threshold:
                n = len(unprocessed) // slow_threshold
                for k in range(n):
                    seg = unprocessed[k * slow_threshold:
                                     (k + 1) * slow_threshold]
                    if seg.strip():
                        slow_accum.append(SlowWindow(len(slow_accum), seg,
                                                     len(seg)))
                        popups.append(Popup("slow", "字数触发", "洞察型",
                                            seg, len(seg),
                                            window_start=slow_cursor + k * slow_threshold))
                slow_cursor += n * slow_threshold

        # 本段已扫描，推进扫描起点
        offset += len(chunk)

    # 流式结束后：剩余 ≥80 字的未处理文本，作为最后一个慢通道窗口
    # 这是慢通道的自然尾部处理，不是兜底——300 字窗口的余数部分本身就应该分析
    remaining = buffer[slow_cursor:]
    if len(remaining) >= FAST_MIN_CHARS:
        popups.append(Popup("slow", "字数触发", "洞察型", remaining, len(remaining),
                            window_start=slow_cursor))

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
          f" · 快统一向前{FAST_CRITICAL_FORWARD} · 背景{FAST_BACKGROUND}"
          f" · <{FAST_MIN_CHARS}取消")
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
