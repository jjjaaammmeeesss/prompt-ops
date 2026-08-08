# @transient — 保留到 v4.0.19 验收通过后删除（预计 2026-08-10）
"""v4.0.19 真实管线批量测试器 —— 300字窗口 + 快速通道 + Codex 裁判 + child_insight。

与之前所有测试脚本的关键区别：
- 之前：整段对话一次塞给 DeepSeek（假测试）
- 现在：先模拟快慢通道互斥窗口切分，每个窗口独立调 LLM（= 生产行为）

用法:
  python scripts/run_v4019_pipeline.py                          # 跑全部 12 题
  python scripts/run_v4019_pipeline.py --cases C5-004,DS_001    # 只跑指定 case
  python scripts/run_v4019_pipeline.py --no-judge               # 只生成，不裁判
"""

# 版本号跟随 prompt 走（用户 2026-08-06 定）：格式 v{prompt版本}-e{自身迭代}。
# __version__ 前缀必须 == PROMPT_VERSION（契约闸门一强制校验，防版本漂移）。
# 例：绑定 v4.0.19、自身迭代到第 4 版 → v4.0.19-e4
__version__ = "v4.0.19-e6"
# 自身迭代 changelog（e1→e2→…→e6，不随 prompt 版本跳号）：
# e1 — 初始版：快慢窗口模拟 + 裸调 Stage 2
# e2 — 9 缺口修复：生产级 Stage 2（zhouyi/debounce/FC_TONE_OFF/FC_STALE/P2/window_size）
# e3 — 完整模拟真实调用链：真实 ZhouYiAnalyzer(Stage1) + P0硬拦截 + 真实 DebounceGate，
#       窗口保持 300+900 死规定不碰；analyzer=None 时 mock 回退供历史对比脚本
# e4 — 1:1 复刻生产滑动窗口：删除「绝对前900字从0累积」拼接，改按生产 TextBuffer
#       （window_size + lookback 相对滑动）构造每窗喂入，并区分
#       「背景（供理解，非生成依据）」与「本轮重点分析（弹窗生成依据）」两段标注；
#       analyzer=None mock 回退路径保持兼容
# e5 — 快速通道禁引背景素材（配合 v4.0.21 prompt「背景段识别与禁引」规则）：
#       fast 通道背景段标注「严禁引用其原句」——背景仅作理解、其"家长原话/孩子行为"
#       禁进弹窗，弹窗只准基于本轮窗口；慢通道保持「供理解」标注不变
# e6 — 背景泄漏硬门禁（用户 2026-08-07 定「字符串聚合≥6字」，替代 LLM judge）：
#       detect_background_leak 确定性检测 fast 弹窗是否泄漏背景段原句（累计背景独有、
#       且不在本轮窗口的片段≥6字判违规），写入 background_leak / bg_leak_chars /
#       bg_leak_frags；slow 记数据不判违规
PROMPT_VERSION = "v4.0.19"   # 适配的 prompt 版本（默认；可 --prompt 切换，见 PROMPT_MAP）

import argparse
import difflib
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import litellm
from dotenv import load_dotenv

litellm.suppress_debug_info = True

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
load_dotenv(PROJECT / ".env")

# 把 PROJECT 根加入 sys.path 以便直接 import 生产执行器（v4.0.19 原始执行器）
sys.path.insert(0, str(PROJECT))
# 把 demos/ 加入 sys.path 以便导入管线函数
sys.path.insert(0, str(PROJECT / "demos"))
from run_demo import simulate_pipeline, Keyword, Popup

# ── 直接复用生产执行器（realtime/popup_generator.py · PROMPT_VERSION=v4.0.19）
# 铁律：测试执行器 = 生产执行器，弹窗内容完全由生产 PopupGenerator 决定，
# 不再走下方复刻的 generate_popup（P2/字数门/FC_STALE/FC_TONE_OFF 全部由生产内部闭环）。
from realtime.popup_generator import PopupGenerator  # noqa: E402
from realtime.zhouyi_analyzer import ZhouYiAnalyzer  # noqa: E402
from realtime.stream_orchestrator import TextBuffer, DebounceGate  # noqa: E402
from realtime.output_schemas import (  # noqa: E402
    ZhouYiState, PopupTone, Trigram, YaoState,
)

# severity 映射：critical→5, warning→3, opportunity→2
SEVERITY_MAP = {"critical": 5, "warning": 3, "opportunity": 2}

# ── 生产 TextBuffer 滑动窗口权威参数（realtime/stream_orchestrator.py:50 默认值）──
# 1:1 复刻生产滑动窗口语义：window_size + lookback + _last_window_end 游标相对滑动。
# 每窗只回看 lookback 字重叠（相对滑动），不随绝对位置从0无限累积。
# 修复原「绝对前900字从0累积」拼接 → 相邻窗口喂入趋同 → LLM 输出雷同弹窗。
PROD_WINDOW_SIZE = 3000
PROD_LOOKBACK = 500

# ═══════════════════════════════════════════════════════════════════════════════
# 生产等价函数 — 从 realtime/popup_generator.py 移植，确保测试=生产行为
# ═══════════════════════════════════════════════════════════════════════════════

# === FC_TONE_OFF: 家长行为 tone override（与 popup_generator.py 完全一致）===

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
    """扫描对话文本，命中家长行为 tone override 关键词时返回命中的类别名。"""
    if not dialogue:
        return None
    for category, keywords in PARENT_OVERRIDE_KEYWORDS.items():
        for kw in keywords:
            if kw in dialogue:
                return category
    return None


# === v4.0.19: child_insight 检测（与 popup_generator.py 一致）===

# 各类型字数上限（与 popup_generator.py 一致，用于字数门重试）
DIAGNOSTIC_MAX_CHARS = 200
ENCOURAGING_MAX_CHARS = 100

CHILD_EXPRESSION_SIGNALS = [
    "我觉得", "我想", "我喜欢", "我不喜欢", "我怕", "我担心",
    "我发现了", "我知道了", "我自己", "我来", "我能", "我会",
    "因为", "所以", "但是我不", "可是我",
]


def detect_child_insight_opportunity(dialogue: str) -> bool:
    """检测对话窗口是否适合使用 child_insight 弹窗（与 popup_generator 一致）。"""
    if not dialogue:
        return False
    lines = [l.strip() for l in dialogue.split("\n") if l.strip()]
    if len(lines) < 3:
        return False
    child_lines = 0
    parent_lines = 0
    for line in lines:
        if line[0].isdigit():
            if any(sig in line for sig in CHILD_EXPRESSION_SIGNALS):
                child_lines += 1
            elif any(kw in line for kw in ["快点", "不许", "必须", "给我", "你应该", "你怎麼", "你怎么"]):
                parent_lines += 1
            else:
                child_lines += 1
    total = child_lines + parent_lines
    if total == 0:
        return False
    return child_lines / total >= 0.30


# === P2: parent-quotable repair phrase 检测 ===

_QUOTABLE_PHRASE_RE = re.compile(r'[「『“"]([^」』”"]{4,})[」』”"]')


def has_quotable_phrase(text: str) -> bool:
    """检测文本中是否含至少一句引号内的可直接引用话术（≥4字）。"""
    return bool(_QUOTABLE_PHRASE_RE.search(text or ""))


# === FC_STALE: 跨窗口语义去重 ===

def semantic_similarity(a: str, b: str) -> float:
    """计算两段文本的相似度（difflib SequenceMatcher ratio）。"""
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()

def load_keywords() -> list:
    """从 keyword_config.json 加载关键词，返回 List[Keyword]。"""
    kw_path = PROJECT / "realtime" / "keyword_config.json"
    with open(kw_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    keywords = []
    for level, sev in SEVERITY_MAP.items():
        for word in config.get(level, []):
            keywords.append(Keyword(text=word, severity=sev))
    return keywords

# ── 路径 ──
PROMPT_MAP = {
    "v4.0.12": PROJECT / "system_prompt_v4.0.12.txt",
    "v4.0.18": PROJECT / "system_prompt_v4.0.18.txt",
    "v4.0.19": PROJECT / "system_prompt_v4.0.19.txt",
}
DATASET_PATH = PROJECT / "data" / "new_12_independent.json"
RESULTS_DIR = PROJECT / "results" / "pipeline_tests"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── 生成模型 ──
GEN_API_KEY = os.getenv("GEN_API_KEY", os.getenv("DEEPSEEK_API_KEY", os.getenv("ZHIPUAI_API_KEY", "")))
GEN_API_BASE = os.getenv("GEN_API_BASE", os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
_raw_model = os.getenv("GEN_MODEL", os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
GEN_MODEL = _raw_model if "/" in _raw_model else f"deepseek/{_raw_model}"


class _ProductionAdapter:
    """生产 LiteLLMModelAdapter 的最小等价适配器。

    生产 PopupGenerator._call_llm 优先调用 adapter.generate_with_chat_format(messages)，
    这里用测试同一套 GEN_MODEL/GEN_API_KEY/GEN_API_BASE（deepseek · v4.0.19 生产模型）
    透传完整 messages（含 system role），保证与真实生产调用路径一致。
    """

    def generate_with_chat_format(
        self, messages, temperature, max_tokens,
    ) -> str:
        resp = litellm.completion(
            model=GEN_MODEL,
            messages=messages,
            api_key=GEN_API_KEY,
            api_base=GEN_API_BASE,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=180,
        )
        return (resp.choices[0].message.content or "").strip()

    def generate(self, prompt, temperature, max_tokens) -> str:
        # 兜底：若生产改走纯文本 prompt 路径
        resp = litellm.completion(
            model=GEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            api_key=GEN_API_KEY,
            api_base=GEN_API_BASE,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=180,
        )
        return (resp.choices[0].message.content or "").strip()


def _make_default_zhouyi_state() -> ZhouYiState:
    """构造测试默认 Stage 1 状态（不跑真实卦象分析）。

    生产 generate() 以 suggested_tone 作 soft-bias 起点 + FC_TONE_OFF / child_insight
    倾向信号，最终由 LLM 结合对话全文裁决——此处给中性起点即可，与生产一致。
    """
    t = Trigram.KAN  # 波动修复型（中性起点）
    return ZhouYiState(
        trigram=t,
        lower_yao=t.lower,
        middle_yao=t.middle,
        upper_yao=t.upper,
        container_status="不适用",
        risk_level="低",
        suggested_tone=PopupTone.DIAGNOSTIC,
        confidence=1.0,
        brief_reason="测试默认（不跑 Stage1）",
    )


def run_case(
    case: dict,
    generator: PopupGenerator,
    system_prompt: str,
    keywords: list,
    window_size: int = 300,
    debounce_enabled: bool = True,
    analyzer: Optional[ZhouYiAnalyzer] = None,
    debounce: Optional[DebounceGate] = None,
) -> dict:
    """对单个 case 跑完整管线（快通道 + 慢通道），完整模拟真实调用逻辑。

    逐窗口真实链路（与生产 StreamOrchestrator 对齐，窗口喂入 = 生产 TextBuffer 相对滑动）：
      1. 窗口切分：simulate_pipeline（300 字本片段，快慢通道互斥，不碰）
         喂入构造：生产 TextBuffer 1:1 复刻 —— 背景取 lookback 回看重叠（相对滑动、
         非从0累积），焦点 = 触发点核心窗口 p.context_window（弹窗生成依据）
      2. 真实 Stage 1：analyzer.analyze(dialogue_for_gen) → 真实卦象/风险/tone
      3. P0 硬拦截：risk=低 + 坤卦 + 容器=不适用 → 抑制（对齐 stream_orchestrator）
      4. 真实去抖：debounce.should_show(zhouyi_state, now=sim_now)
         sim_now 每窗 +20s（>冷却15s），使去抖按卦象/风险/类型维度生效，
         不被纯时间冷却卡死（离线测试无真实时间流）
      5. 生产 Stage 2：generator.generate(dialogue_for_gen, zhouyi_state, previous_popups)

    弹窗内容完全由生产 PopupGenerator.generate() 生成，其内部闭环 FC_TONE_OFF /
    child_insight / P2 话术门 / 字数门 / FC_STALE 跨窗去重。

    向后兼容：analyzer/debounce 传 None 时回退到旧 mock（默认卦象 + trigger_key 去抖），
    供历史对比脚本（compare_v4012_v4019.py）使用。
    """
    case_id = case.get("case_id", "unknown")
    dialogue = case.get("question", "")

    # 清除行号前缀（"1.xxx\n2.xxx" → "xxx\nxxx"）
    dialogue = re.sub(r"^\d+\.", "", dialogue, flags=re.MULTILINE).strip()

    dlen = len(dialogue)

    # ── 管线模拟：流式快慢通道互斥（传入 window_size）──
    popups: list = simulate_pipeline(dialogue, keywords, slow_threshold=window_size)

    # ── 生产 TextBuffer 滑动窗口上下文缓冲（1:1 复刻生产相对滑动语义）──
    # 直接复用生产 TextBuffer：window_size + lookback + _last_window_end 游标相对滑动。
    # 每窗把自上一窗边界以来到本窗触发点的新文本喂入，背景取其 lookback 回看重叠，
    # 不再用「绝对前900字从0累积」——保证相邻窗口喂入文本不随绝对位置趋同。
    context_buf = TextBuffer(window_size=PROD_WINDOW_SIZE, lookback=PROD_LOOKBACK)

    use_real = analyzer is not None  # 真实调用模式：传 analyzer → 走完整链路

    # ── 去抖 + 逐窗调用状态 ──
    previous_popups: list = []   # 生产 Popup 对象（供生产 generate 跨窗语义去重）
    last_trigger: dict = {}      # {channel, trigger_type} 仅 mock 回退去抖用
    fast_popups: list[dict] = []
    slow_popups: list[dict] = []
    suppressed_count = 0
    sim_now = 0.0                # 递增模拟时间戳（真实去抖用）

    # 每 case 重置去抖（对齐生产 StreamOrchestrator.reset → debounce.reset）。
    # 否则跨 case 共享 DebounceGate 时，_last_popup_time 记录上个 case 的 sim_now，
    # 新 case 的 elapsed = 20 - 旧值 变负 → 全部被"绝对最小间隔未到"误拦（负时间 bug）。
    if use_real and debounce is not None:
        debounce.reset()

    for p_idx, p in enumerate(popups):
        # ── 简化去抖：仅 mock 回退路径使用（真实路径用 DebounceGate）──
        if debounce_enabled and not use_real:
            trigger_key = (p.channel, p.trigger_type)
            if trigger_key == last_trigger.get("key"):
                suppressed_count += 1
                last_trigger["count"] = last_trigger.get("count", 0) + 1
                continue
            last_trigger = {"key": trigger_key, "count": 1}

        # ── 1:1 复刻生产 TextBuffer 相对滑动喂入，区分「背景」vs「本轮重点分析」──
        # 生产 realtime/stream_orchestrator.py TextBuffer：每窗只回看 lookback 字重叠
        # （current_window.start = _last_window_end - lookback），非从0累积。这里背景 =
        # 本窗触发点之前 lookback 字（= 生产回看重叠，供理解上下文，非生成依据）；
        # 焦点 = 触发点附近核心窗口 p.context_window（simulate_pipeline 产出的
        # "向前取300字/向后取50字"聚焦窗口 = 生产 TextBuffer 当前新增窗口，弹窗生成依据）。
        if p.window_start > context_buf.total_chars:
            context_buf.append(dialogue[context_buf.total_chars:p.window_start])
            context_buf.mark_window_analyzed()  # 推进相对滑动游标（非绝对位置）
        background = dialogue[
            max(0, p.window_start - context_buf.lookback):p.window_start
        ]

        # 快速通道：背景段标注「严禁引用其原句」——prompt 据此收紧，背景仅作理解、
        # 其"家长原话/孩子行为"禁进弹窗，弹窗只准基于本轮窗口（配合 v4.0.21 prompt 规则）。
        # 慢通道：保持「供理解」标注不变（慢通道不作改动）。
        if p.channel == "fast":
            bg_label = "## 背景（仅作理解，严禁引用其原句）"
        else:
            bg_label = "## 背景（供理解，非生成依据）"
        dialogue_for_gen = (
            f"{bg_label}\n{background}\n\n"
            f"## 本轮重点分析（弹窗生成依据）\n{p.context_window}"
            if background.strip() else p.context_window
        )

        # ════════ 真实调用链路（analyzer 传入时）════════
        if use_real:
            # 1) 真实 Stage 1：周爻分析（内部 try/except，失败回退默认坤卦）
            zhouyi_state = analyzer.analyze(dialogue_for_gen)

            # 2) P0 硬拦截：稳态日常对话不弹（对齐 stream_orchestrator.py v4.0.20）
            #    v4.0.20 收紧：仅当 Stage1 建议类型也为「不弹窗」才拦；
            #    Stage1 判了诊断式/鼓励式/看见孩子（识别到情绪信号/教育契机）则放行。
            if (
                zhouyi_state.risk_level == "低"
                and zhouyi_state.trigram == Trigram.KUN
                and zhouyi_state.container_status == "不适用"
                and "不弹窗" in (getattr(zhouyi_state, "suggestion", "") or "")
            ):
                suppressed_count += 1
                entry = {
                    "channel": p.channel, "trigger_type": p.trigger_type,
                    "window_chars": p.char_count, "window_start": p.window_start,
                    "prior_chars": len(background), "popup_order": p_idx,
                    "tone": zhouyi_state.suggested_tone.value, "tone_override": "",
                    "source": "production_popup_generator_v4.0.19",
                    "popup": "", "raw": "", "has_pre_analysis": False,
                    "separator_missing": False, "error": "p0_blocked",
                    "prod_tone": zhouyi_state.suggested_tone.value,
                    "debounce_reason": "P0 硬拦截（稳态日常 + Stage1建议类型=不弹窗）",
                    "zhouyi": _zhouyi_meta(zhouyi_state),
                }
                if p.channel == "fast":
                    fast_popups.append(entry)
                else:
                    slow_popups.append(entry)
                continue

            # 3) 真实去抖：注入递增模拟时间戳，使卦象/风险/类型逻辑生效
            sim_now += 20.0
            should_show, reason = debounce.should_show(zhouyi_state, now=sim_now)
            if not should_show:
                suppressed_count += 1
                entry = {
                    "channel": p.channel, "trigger_type": p.trigger_type,
                    "window_chars": p.char_count, "window_start": p.window_start,
                    "prior_chars": len(background), "popup_order": p_idx,
                    "tone": zhouyi_state.suggested_tone.value, "tone_override": "",
                    "source": "production_popup_generator_v4.0.19",
                    "popup": "", "raw": "", "has_pre_analysis": False,
                    "separator_missing": False, "error": "debounced",
                    "prod_tone": zhouyi_state.suggested_tone.value,
                    "debounce_reason": reason,
                    "zhouyi": _zhouyi_meta(zhouyi_state),
                }
                if p.channel == "fast":
                    fast_popups.append(entry)
                else:
                    slow_popups.append(entry)
                continue
        else:
            # mock 回退：默认卦象（历史对比脚本用）
            zhouyi_state = _make_default_zhouyi_state()

        # ── Stage 2：调用生产执行器（v4.0.19）生成弹窗 ──
        # 生产 generate() 内部闭环 FC_TONE_OFF soft-bias / child_insight / P2 话术门 /
        # 字数门 / FC_STALE 跨窗去重——测试不再自建复刻逻辑，弹窗内容完全由生产决定。
        try:
            popup = generator.generate(
                dialogue_for_gen, zhouyi_state, previous_popups,
            )
            if popup and popup.should_popup and popup.full_text:
                gen = {
                    "popup": popup.full_text,
                    "raw": "",
                    "has_pre_analysis": False,
                    "separator_missing": False,
                    "error": None,
                    "prod_tone": popup.tone.value,
                }
                previous_popups.append(popup)  # 生产对象，供后续跨窗去重
                if use_real:
                    debounce.record_popup(popup)  # 真实去抖状态更新
            else:
                # 生产拒绝（P2 缺话术 / FC_STALE 去重 / 字数门仍超 / should_popup=False）
                gen = {
                    "popup": "",
                    "raw": "",
                    "has_pre_analysis": False,
                    "separator_missing": False,
                    "error": "rejected_by_production",
                    "prod_tone": popup.tone.value if popup else "auto",
                }
                suppressed_count += 1
        except Exception as e:
            gen = {
                "popup": "",
                "raw": "",
                "has_pre_analysis": False,
                "separator_missing": False,
                "error": str(e),
                "prod_tone": "auto",
            }

        tone = gen.get("prod_tone", "auto")

        entry = {
            "channel": p.channel,
            "trigger_type": p.trigger_type,
            "window_chars": p.char_count,
            "window_start": p.window_start,
            "prior_chars": len(background),
            "popup_order": p_idx,  # 保留原始时间顺序
            "tone": tone,
            "tone_override": "",  # FC_TONE_OFF 由生产内部 soft-bias 处理
            "source": "production_popup_generator_v4.0.19",
            **gen,
        }
        if use_real:
            entry["zhouyi"] = _zhouyi_meta(zhouyi_state)

        # ── 背景泄漏硬门禁（e6，快速通道禁引背景素材）──
        # 只对 fast 判违规（用户 2026-08-07：慢通道暂不改动）；
        # slow 记录 bg_leak 数据但不判 background_leak，供参考不拦截。
        pop_text = gen.get("popup", "")
        if p.channel == "fast":
            bg_leak_chars, bg_leak_frags = detect_background_leak(
                pop_text, background, dialogue[p.window_start:p.window_start + p.char_count]
            )
            entry["background_leak"] = bg_leak_chars >= 6
            entry["bg_leak_chars"] = bg_leak_chars
            entry["bg_leak_frags"] = bg_leak_frags
        else:
            entry["background_leak"] = False
            entry["bg_leak_chars"] = 0
            entry["bg_leak_frags"] = []

        if p.channel == "fast":
            fast_popups.append(entry)
        else:
            slow_popups.append(entry)

    result = {
        "case_id": case_id,
        "dialogue_chars": dlen,
        "fast_triggers": sum(1 for p in popups if p.channel == "fast"),
        "slow_windows": sum(1 for p in popups if p.channel == "slow"),
        "total_windows": len(popups),
        "suppressed": suppressed_count,
        "fast_popups": fast_popups,
        "slow_popups": slow_popups,
        "total_popups": sum(1 for e in fast_popups + slow_popups if e.get("popup")),
        "_zhouyi_source": "production-analyzer" if use_real else "test-default",
    }

    return result


def _zhouyi_meta(state: ZhouYiState) -> dict:
    """提取 ZhouYiState 的可序列化元信息，供结果记录。"""
    return {
        "trigram": state.trigram.chinese_name,
        "symbol": state.trigram.symbol,
        "risk_level": state.risk_level,
        "suggested_tone": state.suggested_tone.value,
        "container_status": state.container_status,
        "confidence": state.confidence,
    }


def detect_background_leak(pop_text: str, bg: str, win: str, min_total: int = 6) -> tuple:
    """背景泄漏硬门禁（快速通道用，用户 2026-08-07 定「字符串聚合 ≥6字」）。

    background_leak 判定：累计『背景段独有、且不在本轮窗口』的连续片段字数 ≥ min_total。
    背景独有 = 片段在背景段、但本轮窗口没有 → 弹窗把背景当素材引用（违规）；
    片段同时在本轮窗口 = 合法引用（窗口内本来就该基于它生成），不算泄漏。

    确定性字符串匹配（SequenceMatcher），不引入 LLM judge（零噪声、可复现）。
    实测区分度：f#3 多片段累计13字→违规；f#4 单片段5字→放过（合理转述）；f#0/f#1 0字→OK。
    """
    if not pop_text or not bg:
        return 0, []
    from difflib import SequenceMatcher
    frags, total = [], 0
    sm = SequenceMatcher(None, pop_text, bg, autojunk=False)
    for m in sm.get_matching_blocks():
        sub = pop_text[m.a:m.a + m.size]
        if len(sub) >= 3 and sub not in win and sub not in frags:
            frags.append(sub)
            total += len(sub)
    return total, frags


def run_codex_batch_judge(results: list, output_file: str) -> list:
    """批量 Codex 裁判：一次 codex exec 调用评完所有弹窗。

    返回更新后的 results（含 codex_score / codex_reason）。
    """
    import subprocess

    # 构建批量裁判 prompt —— 逐弹窗：每个非空弹窗作为独立条目
    cases_text = ""
    for i, r in enumerate(results):
        dialogue = r.get("_dialogue", "")
        all_pops = [p.get("popup") for p in
                    list(r.get("fast_popups", [])) + list(r.get("slow_popups", []))
                    if p.get("popup")]
        for j, popup in enumerate(all_pops):
            cases_text += f"""
---
## Case {i + 1}.{j + 1}: {r['case_id']} 弹窗#{j + 1}（对话{len(dialogue)}字）
对话: {dialogue}
弹窗: {popup}
---"""

    judge_prompt = f"""你是亲子沟通弹窗质量裁判。请对以下每个 case 的每条弹窗打分（0-10分），输出 JSON 数组。

评分维度：
- 洞察深度（揭示家长行为背后的心理机制）
- 具体性（锚定到对话中的具体言行）
- 人称准确性（指代孩子时用"ta"而非"她/他"）
- 可用建议（suggestion 具体可操作）
- 术语泄漏（出现"多极/在场/内生性/关系根/双向"等内部术语则扣分）
- F1 原文锚定（是否引用对话中的具体原话）
- F2 反话检测（如有讽刺/阴阳怪气，是否正确识别并触发防御模式）

{cases_text}

请输出严格 JSON 数组，每项包含 case_id, score, reason:
[{{"case_id": "C5-004", "score": 8.5, "reason": "一句话理由"}}, ...]"""

    temp_dir = PROJECT / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = temp_dir / "codex_judge_prompt.txt"
    prompt_file.write_text(judge_prompt, encoding="utf-8")

    out_file = temp_dir / "codex_judge_output.json"
    n_total = sum(
        1 for r in results
        for p in list(r.get("fast_popups", [])) + list(r.get("slow_popups", []))
        if p.get("popup"))
    print(f"  📋 批量裁判 {n_total} 条弹窗...")

    try:
        result = subprocess.run(
            ["D:/root/.npm-global/codex.cmd", "exec", "--ephemeral", "--json",
             "-o", str(out_file)],
            input=judge_prompt,
            capture_output=True, text=True, timeout=300,
            cwd=str(PROJECT),
            env={**os.environ, "CODEX_NO_COLOR": "1"},
        )

        # 解析 JSONL 输出，取最后一条 agent_message
        raw_output = result.stdout
        last_text = ""
        for line in raw_output.strip().split("\n"):
            try:
                obj = json.loads(line)
                if obj.get("item", {}).get("type") == "agent_message":
                    last_text = obj["item"].get("text", "")
            except json.JSONDecodeError:
                pass

        if not last_text and out_file.exists():
            # 尝试从输出文件读取
            out_content = out_file.read_text(encoding="utf-8")
            for line in out_content.strip().split("\n"):
                try:
                    obj = json.loads(line)
                    if obj.get("item", {}).get("type") == "agent_message":
                        last_text = obj["item"].get("text", "")
                except json.JSONDecodeError:
                    pass

        # 解析 JSON 数组
        if last_text:
            # 提取 JSON 数组
            arr_match = re.search(r"\[.*\]", last_text, re.DOTALL)
            if arr_match:
                scores = json.loads(arr_match.group(0))
                score_map = {s["case_id"]: s for s in scores}
                for r in results:
                    if r["case_id"] in score_map:
                        r["codex_score"] = score_map[r["case_id"]]["score"]
                        r["codex_reason"] = score_map[r["case_id"]]["reason"]
                return results

        print(f"  ⚠️ 无法解析 Codex 批量输出，前 500 字: {raw_output[:500]}")
    except Exception as e:
        print(f"  ❌ Codex 批量裁判失败: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="真实管线批量测试（支持多版本 prompt）")
    parser.add_argument("--prompt", type=str, default="v4.0.19",
                        help="提示词版本: v4.0.12 / v4.0.18 / v4.0.19（默认 v4.0.19）")
    parser.add_argument("--cases", type=str, default="",
                        help="逗号分隔的 case_id 列表（默认：全部）")
    parser.add_argument("--dataset", type=str, default=str(DATASET_PATH),
                        help="数据集路径")
    parser.add_argument("--no-judge", action="store_true",
                        help="跳过 Codex 裁判")
    parser.add_argument("--window-size", type=int, default=300,
                        help="慢通道窗口大小（默认 300）")
    args = parser.parse_args()

    # ── 版本解析 ──
    prompt_version = args.prompt
    prompt_path = PROMPT_MAP.get(prompt_version)
    if not prompt_path:
        print(f"❌ 未知版本: {prompt_version}，可用: {list(PROMPT_MAP.keys())}")
        sys.exit(1)
    prompt_label = prompt_version.replace(".", "").replace("v", "v")  # "v4.0.18" → "v4018"

    # ── 加载 ──
    print("=" * 70)
    print(f"{prompt_version} 真实管线批量测试（快通道 + 300字窗口慢通道）")
    print("=" * 70)

    system_prompt = prompt_path.read_text(encoding="utf-8")
    print(f"Prompt: {len(system_prompt)} 字, {system_prompt.count(chr(10))} 行")

    # ── 构造生产执行器（v4.0.19 原始 PopupGenerator + 真实 Stage1 + 真实去抖）──
    # 复用同一个 _ProductionAdapter（deepseek v4.0.19），接口兼容 ZhouYiAnalyzer 与 PopupGenerator，
    # 保证 Stage 1 / Stage 2 走同一条真实生产 LLM 调用路径。
    adapter = _ProductionAdapter()
    analyzer = ZhouYiAnalyzer(model_adapter=adapter)   # 真实 Stage 1
    debounce = DebounceGate()                          # 真实去抖（生产默认参数）
    generator = PopupGenerator(
        model_adapter=adapter,
        system_prompt_path=str(prompt_path),
        dedup_config={
            "enabled": True,
            "semantic_similarity_threshold": 0.70,
            "history_window": 5,
        },
    )

    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    keywords = load_keywords()

    # 筛选 cases
    if args.cases:
        target_ids = set(args.cases.split(","))
        dataset = [c for c in dataset if c.get("case_id") in target_ids]
        print(f"筛选: {len(dataset)} 题 ({', '.join(c['case_id'] for c in dataset)})")
    else:
        print(f"数据集: {len(dataset)} 题")

    print(f"生成模型: {GEN_MODEL}")
    print(f"Stage 1: 真实 ZhouYiAnalyzer（每窗真实卦象分析）")
    print(f"去抖: 真实 DebounceGate（P0硬拦截 + 卦象/风险/类型去抖）")
    print(f"窗口大小: {args.window_size} 字 + 生产 TextBuffer 相对滑动背景"
          f"（lookback={PROD_LOOKBACK}字，非从0累积）")
    print(f"快速通道关键词: {len(keywords)} 个")
    print(f"Codex 裁判: {'❌ 跳过' if args.no_judge else '✅ 启用'}")
    print()

    # ── 逐题测试 ──
    all_results = []
    for idx, case in enumerate(dataset):
        case_id = case.get("case_id", f"case_{idx}")
        d_short = case.get("question", "")[:60].replace("\n", " ")
        print(f"[{idx + 1}/{len(dataset)}] {case_id}: {d_short}...")

        result = run_case(
            case, generator, system_prompt, keywords, args.window_size,
            analyzer=analyzer, debounce=debounce,
        )
        result["_dialogue"] = case.get("question", "")  # 保留原对话供裁判使用

        print(f"  快通道: {result['fast_triggers']} 触发, "
              f"慢通道: {result['slow_windows']} 窗口, "
              f"弹窗: {result['total_popups']} 条")

        # 检查预分析分隔
        for i, p in enumerate(result["fast_popups"]):
            err = p.get("error")
            trigger = p.get("trigger_type", "?")
            wc = p.get("window_chars", "?")
            plen = len(p.get("popup", ""))
            if err:
                print(f"    ⚠️ 快#{i} [{trigger}] 窗口{wc}字: {err}")
            elif p.get("has_pre_analysis"):
                print(f"    快#{i} [{trigger}] 窗口{wc}字 → 弹窗{plen}字 ✓")
            else:
                print(f"    快#{i} [{trigger}] 窗口{wc}字 → 弹窗{plen}字 ⚠️ 未检测到预分析分隔")

        for i, p in enumerate(result["slow_popups"]):
            err = p.get("error")
            wc = p.get("window_chars", "?")
            plen = len(p.get("popup", ""))
            if err:
                print(f"    ⚠️ 慢#{i} [窗口{wc}字]: {err}")
            elif p.get("has_pre_analysis"):
                print(f"    慢#{i} [窗口{wc}字] → 弹窗{plen}字 ✓")
            else:
                print(f"    慢#{i} [窗口{wc}字] → 弹窗{plen}字 ⚠️")

        all_results.append(result)
        time.sleep(0.5)  # API 限流

    # ── 批量 Codex 裁判（所有 case 一次调用）──
    if not args.no_judge:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        all_results = run_codex_batch_judge(all_results, str(RESULTS_DIR / f"codex_judge_{timestamp}.json"))
        for r in all_results:
            score = f"{r['codex_score']:.1f}" if r.get("codex_score") is not None else "N/A"
            reason = r.get("codex_reason", "")[:80] if r.get("codex_reason") else ""
            print(f"  🧑‍⚖️ {r['case_id']}: {score}/10 — {reason}")

    # ── 汇总 ──
    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)

    for r in all_results:
        cid = r["case_id"]
        n_popups = r["total_popups"]
        first = next((p.get("popup", "")[:80] for p in
                      list(r.get("fast_popups", [])) + list(r.get("slow_popups", []))
                      if p.get("popup")), "(无弹窗)")
        score = f"{r['codex_score']:.1f}" if r.get("codex_score") is not None else "N/A"
        print(f"  {cid}: {n_popups}条弹窗 | Codex={score} | {first}...")

    # ── 保存 ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cases_tag = f"_{args.cases.replace(',', '_')}" if args.cases else "_all12"
    out_path = RESULTS_DIR / f"{prompt_label}_pipeline{cases_tag}_{timestamp}.json"

    output = {
        "config": {
            "prompt": f"system_prompt_{prompt_version}.txt",
            "prompt_chars": len(system_prompt),
            "gen_model": GEN_MODEL,
            "window_size": args.window_size,
            "judge": "codex" if not args.no_judge else "none",
            "timestamp": timestamp,
        },
        "summary": {
            "total_cases": len(all_results),
            "cases_with_popup": sum(1 for r in all_results if r["total_popups"] > 0),
            "total_popups": sum(r["total_popups"] for r in all_results),
            "avg_codex_score": (
                sum(r["codex_score"] for r in all_results if r.get("codex_score") is not None)
                / max(1, sum(1 for r in all_results if r.get("codex_score") is not None))
                if not args.no_judge else None
            ),
        },
        "results": all_results,
    }
    json.dump(output, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")
    return out_path


if __name__ == "__main__":
    main()
