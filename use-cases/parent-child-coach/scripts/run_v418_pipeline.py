"""v4.0.18 快慢通道流水线测试 —— 严格按统一规格。

对每个案例喂入流式文本，驱动快慢双通道，输出完整执行链路：

- 快通道 critical（severity≥4）：命中当下就弹，向前取最多 300 字，<80 取消
- 快通道一般严重：向前取 250 字，命中后等 50 字再试图分析，总字数 <80 取消
- 慢通道：缓冲凑满 300 字才分析；窗口采用「最近 300 字」滑动窗，
  天然携带前文（不冷启动、不跨窗口丢上下文）

参数统一取自 channel_spec（唯一权威），与 SUT、realtime、run_demo 保持一致。

用法：
    cd use-cases/parent-child-coach
    python scripts/run_v418_pipeline.py
"""

import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

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
    text: str
    severity: int


@dataclass
class Window:
    """一次快通道窗口决策。"""
    window_index: int
    channel: str              # "fast_critical" | "fast_general" | "slow"
    trigger_type: str
    tone: str
    context_window: str
    char_count: int
    cancelled: bool = False
    reason: str = ""


@dataclass
class CaseResult:
    case_id: str
    windows: List[Window] = field(default_factory=list)


def _snap_start_to_boundary(buffer: str, start: int, limit: int) -> int:
    for i in range(start, min(start + 30, limit)):
        if i < len(buffer) and buffer[i] in {"。", "！", "？", "\n", "，"}:
            return i + 1
    return start


def _fast_window(buffer: str, trigger_pos: int, severity: int) -> Window:
    """快通道窗口截取（统一规格）。"""
    if is_critical(severity):
        start = max(0, trigger_pos - FAST_CRITICAL_FORWARD)
        start = _snap_start_to_boundary(buffer, start, trigger_pos)
        ctx = buffer[start:trigger_pos]
        if len(ctx) < FAST_MIN_CHARS:
            return Window(0, "fast_critical", "关键词触发", "提醒型",
                          "", 0, cancelled=True,
                          reason=f"critical <{FAST_MIN_CHARS}字取消")
        return Window(0, "fast_critical", "关键词触发", "提醒型",
                      ctx, len(ctx))
    start = max(0, trigger_pos - FAST_GENERAL_FORWARD)
    start = _snap_start_to_boundary(buffer, start, trigger_pos)
    end = min(len(buffer), trigger_pos + FAST_GENERAL_WAIT)
    ctx = buffer[start:end]
    if len(ctx) < FAST_MIN_CHARS:
        return Window(0, "fast_general", "关键词触发", "提醒型",
                      "", 0, cancelled=True,
                      reason=f"general <{FAST_MIN_CHARS}字取消")
    return Window(0, "fast_general", "关键词触发", "提醒型", ctx, len(ctx))


def _first_keyword(text: str, keywords: List[Keyword]):
    """在文本中找第一个命中关键词（长词优先）。"""
    for kw in sorted(keywords, key=lambda k: len(k.text), reverse=True):
        idx = text.find(kw.text)
        if idx >= 0:
            return (kw.text, idx, kw.severity)
    return None


def run_case(case_id: str, text: str, keywords: List[Keyword],
             chunk_size: int = 40) -> CaseResult:
    """跑一个案例，输出快慢通道完整执行链路。"""
    res = CaseResult(case_id)
    buffer = ""
    offset = 0
    slow_fired_at = 0  # 已触发慢通道的累积字数
    pending_general: Optional[dict] = None
    reported_until = 0  # 缓冲中已触发上报的位置（防跨块重复）
    w_idx = 0
    max_kw = max(len(k.text) for k in keywords)

    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        buffer += chunk

        # ── 快通道：在「新文本 + 上一段尾部重叠」中匹配（捕捉跨块关键词） ──
        scan_base = max(0, offset - (max_kw - 1))
        found = _first_keyword(buffer[scan_base:], keywords)
        if found:
            kw, idx_in, sev = found
            pos = scan_base + idx_in
            # 跨块重叠扫描可能重复命中同一关键词——只上报延伸到新区域的一次
            if pos + len(kw) <= reported_until:
                offset += len(chunk)
                continue
            reported_until = pos + len(kw)
            if is_critical(sev):
                w = _fast_window(buffer, pos, sev)
                w.window_index = w_idx; w_idx += 1
                res.windows.append(w)
                pending_general = None
            else:
                if len(buffer) < pos + FAST_GENERAL_WAIT:
                    pending_general = {"pos": pos, "sev": sev}
                else:
                    w = _fast_window(buffer, pos, sev)
                    w.window_index = w_idx; w_idx += 1
                    res.windows.append(w)
                    pending_general = None
        elif pending_general is not None:
            # 挂起的一般严重：等够 50 字后解析
            if len(buffer) >= pending_general["pos"] + FAST_GENERAL_WAIT:
                w = _fast_window(buffer, pending_general["pos"],
                                 pending_general["sev"])
                w.window_index = w_idx; w_idx += 1
                res.windows.append(w)
                pending_general = None

        # ── 慢通道：累积凑满 300 字才分析，窗口=最近300字（携带前文） ──
        if len(buffer) - slow_fired_at >= SLOW_THRESHOLD_CHARS:
            seg = buffer[-SLOW_THRESHOLD_CHARS:]
            res.windows.append(Window(
                w_idx, "slow", "字数触发", "洞察型",
                seg, len(seg)))
            w_idx += 1
            slow_fired_at = len(buffer)

        offset += len(chunk)

    return res


# ── 测试案例（覆盖快慢双通道 + 取消分支） ──

KEYWORDS = [
    Keyword("你再哭我就不要你了", 5),
    Keyword("我说了算", 4),
    Keyword("我像你这么大", 3),
    Keyword("你不知足", 3),
    Keyword("你看看人家", 3),
]

CASES = {
    "C1_快critical即时弹": (
        "孩子放学回家把书包往地上一扔，连鞋都没换就趴在沙发上打起游戏来，"
        "作业本丢在桌角碰也不碰，妈妈从厨房出来喊了他整整五遍让他先洗手吃饭，"
        "他头也不抬地说再玩一局，妈妈端着菜站在饭桌前，越听越气，"
        "走过去一把把手机抢过来摔在茶几上，你再哭我就不要你了，"
        "撂下这句话妈妈转身就进了厨房，孩子愣在原地，眼眶一下子红了，"
        "他没想到妈妈会这么说，眼泪开始一颗一颗往下掉，肩膀也抖了起来。"
    ),
    "C2_一般严重等50字": (
        "孩子低着头站在墙角，不说话，两只手紧紧攥着衣角，肩膀微微发抖，"
        "过了好一会儿，他才鼓起勇气小声开口，我像你这么大的时候，"
        "爷爷对我可比你耐心多了，从来不会像这样吼我，说着眼泪就掉下来了，"
        "妈妈愣了一下，张了张嘴，却不知道该说什么才好，屋子里一下子安静下来，"
        "只剩下孩子低低的抽泣声，和墙上时钟滴答滴答的走动声。"
    ),
    "C3_短句critical取消": (
        "你再哭我就不要你了。"
    ),
    "C4_慢通道300字": (
        "这是一段超过三百字的日常对话文本，用来验证慢通道要凑满三百字才分析。"
        "家长和孩子在饭桌上聊今天学校里发生的事，孩子说同桌小明今天上课被老师表扬了，"
        "因为他把一道很难的数学题做对了，孩子自己却因为粗心算错被留堂了，"
        "妈妈听了没有责备他，只是问他错在哪里，孩子说是因为太着急没看清题目，"
        "妈妈点点头说下次看清楚再动笔，孩子松了一口气，说以后会注意的，"
        "妈妈拍了拍他的肩膀，说没关系，慢慢来，孩子这才露出笑容，"
        "继续夹菜吃饭，一家人又恢复了平常的温馨气氛，饭桌上有说有笑，"
        "吃完饭后孩子主动帮妈妈收拾碗筷，妈妈夸他长大了，他心里很高兴，"
        "觉得自己今天虽然犯了错，但妈妈没有骂他，反而教他下次要细心，"
        "他暗暗下决心以后做数学题一定要先审题再计算，不能再因为粗心丢分了，"
        "想到这里他抬头看了看妈妈，妈妈也正好看过来，冲他温柔地笑了一下。"
    ),
}


def print_record(res: CaseResult):
    print(f"\n▶ {res.case_id}  →  {len(res.windows)} 个窗口")
    for w in res.windows:
        if w.cancelled:
            print(f"  ✗ [{w.channel}] {w.trigger_type} · {w.reason}")
        else:
            print(f"  ✓ [{w.channel}] {w.trigger_type} · tone={w.tone} · "
                  f"{w.char_count}字: {w.context_window[:44]}...")


def main():
    print("=" * 64)
    print("v4.0.18 快慢通道流水线测试（统一规格）")
    print(f"  critical≥{CRITICAL_SEVERITY_MIN} · 慢通道{SLOW_THRESHOLD_CHARS}字"
          f" · 快crit向前{FAST_CRITICAL_FORWARD} · 一般向前{FAST_GENERAL_FORWARD}"
          f" +等{FAST_GENERAL_WAIT} · <{FAST_MIN_CHARS}取消")
    print("=" * 64)
    for cid, text in CASES.items():
        res = run_case(cid, text, KEYWORDS)
        print_record(res)


if __name__ == "__main__":
    main()
