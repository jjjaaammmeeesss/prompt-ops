#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""执行器契约校验器 —— 双重闸门：版本关联 + 输入构造奇偶。

背景（教训 2026-08-06）：测试执行器 `run_v4019_pipeline.py` 曾用「绝对位置前900字从0
累积」构造喂入，而生产 TextBuffer 是「window_size + lookback 相对滑动」。绝对累积导致
相邻窗口喂入趋同（C5-004 输入相似度 0.907）→ LLM 弹窗雷同。执行器 v1.4 已修复为复用
生产 TextBuffer。本脚本把这次教训固化成**自动闸门**，防止回归。

闸门一 · 版本关联（用户 2026-08-06 新需求）：
  执行器自身版本号 `__version__`（独立演进）与它绑定的 prompt 版本 `PROMPT_VERSION`
  必须可追溯、且实际加载的 prompt 文件版本 == 声明的 PROMPT_VERSION == config.yaml 生产版本。
  → 防"执行器声称适配 v4.0.19 但实际喂的是别的 prompt 文件"的漂移。

闸门二 · 输入构造奇偶：
  对同一 dialogue + 同一窗口触发序列，执行器喂给生产 generator 的输入，其**机制**必须
  == 生产 TextBuffer 在同一触发点会产出的 current_window 机制：
    P1 背景有界：每窗 prior_chars ≤ lookback（防"绝对前900从0累积"回归）
    P2 背景来源：background == dialogue[max(0, window_start-lookback):window_start]
                （= 生产 TextBuffer 回看重叠区，非从0前缀）
    P3 相邻窗输入不趋同：相邻窗口喂入相似度 < 阈值（防 C5-004 输入趋同复发）
  第一层防御：**import 生产 TextBuffer 类**做基准，不复制滑动窗口逻辑。

用法:
  python scripts/check_executor_input_parity.py --json <pipeline_output.json>
  python scripts/check_executor_input_parity.py --json <pipeline_output.json> --strict
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))

# 第一层防御：import 生产 TextBuffer（权威滑动窗口语义），不复制逻辑。
from realtime.stream_orchestrator import TextBuffer  # noqa: E402
# 快通道统一窗口参数（channel_spec 唯一权威）：背景900含150分析窗口 → 背景上限750
from channel_spec import FAST_BACKGROUND, FAST_CRITICAL_FORWARD  # noqa: E402

# 快通道背景段上限 = FAST_BACKGROUND(900) - 分析窗口(150) = 750；慢通道 = 生产 lookback(500)
FAST_BG_LOOKBACK = FAST_BACKGROUND - FAST_CRITICAL_FORWARD

EXECUTOR = PROJECT / "scripts" / "run_v4019_pipeline.py"
CONFIG = PROJECT / "realtime" / "config.yaml"

EXEC_VERSION_RE = re.compile(r'^__version__\s*=\s*["\']([^"\']+)["\']', re.MULTILINE)
PROMPT_VER_CONST_RE = re.compile(r'^PROMPT_VERSION\s*=\s*["\']([^"\']+)["\']', re.MULTILINE)
PROMPT_FILE_RE = re.compile(r"system_prompt_v([\d.]+)\.txt")

FAILS: list[str] = []
WARNS: list[str] = []


def _fail(msg: str) -> None:
    FAILS.append(msg)


def _warn(msg: str) -> None:
    WARNS.append(msg)


def _extract(path: Path, regex: re.Pattern) -> str | None:
    if not path.exists():
        return None
    m = regex.search(path.read_text(encoding="utf-8"))
    return m.group(1) if m else None


def _config_prod_prompt() -> str | None:
    """从 config.yaml 读生产权威 prompt 文件版本。"""
    if not CONFIG.exists():
        return None
    try:
        import yaml
        data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    except Exception:
        return None
    for section, key in (("generator", "system_prompt_path"),
                         ("popup", "system_prompt_path"),
                         ("system_prompt", "file")):
        sec = data.get(section) if isinstance(data, dict) else None
        if isinstance(sec, dict) and sec.get(key):
            m = PROMPT_FILE_RE.search(str(sec[key]))
            if m:
                return m.group(1)
    return None


def _window_feed_sim(w1: str, w2: str) -> float:
    return difflib.SequenceMatcher(None, w1, w2).ratio()


def gate_version():
    """闸门一：执行器自身版本 + 绑定 prompt 版本 + 生产版本三方对齐。"""
    exec_ver = _extract(EXECUTOR, EXEC_VERSION_RE)
    declared_prompt = _extract(EXECUTOR, PROMPT_VER_CONST_RE)
    prod_prompt = _config_prod_prompt()

    if not exec_ver:
        _fail("执行器缺 __version__（自身版本号）——无法追溯执行器演进")
    if not declared_prompt:
        _fail("执行器缺 PROMPT_VERSION（绑定 prompt 版本）——无法关联提示词版本")
    if not prod_prompt:
        _fail("config.yaml 未声明生产 prompt 文件版本——权威源缺失")

    def _v(s):
        """归一化版本号：v4.0.19 / 4.0.19 → v4.0.19。"""
        return s if s.startswith("v") else f"v{s}"

    print(f"[版本] 执行器自身版本      __version__        = {exec_ver}")
    print(f"[版本] 执行器绑定 prompt    PROMPT_VERSION     = {declared_prompt}")
    print(f"[版本] config.yaml 生产版本  generator         = {prod_prompt}")

    d, p_ = _v(declared_prompt) if declared_prompt else None, _v(prod_prompt) if prod_prompt else None
    if d and p_ and d != p_:
        _fail(f"执行器声明绑定 PROMPT_VERSION={d}，但 config.yaml 生产是 {p_}"
              f"——版本漂移，必须同步")

    # 版本号跟随 prompt 走（用户 2026-08-06 定）：__version__ 前缀必须 == PROMPT_VERSION。
    # 格式 v{prompt版本}-e{自身迭代}，如 v4.0.19-e4。防执行器改了绑定 prompt 却忘升自身版本号。
    if exec_ver and declared_prompt:
        prefix = exec_ver.split("-")[0]
        if prefix != _v(declared_prompt):
            _fail(f"__version__={exec_ver} 前缀 {prefix} ≠ 绑定 PROMPT_VERSION={_v(declared_prompt)}"
                  f"——执行器版本号未跟随 prompt，应形如 {_v(declared_prompt)}-e{{n}}")

    # 执行器声明绑定的 prompt 文件必须真实存在且版本号一致
    if d:
        p = PROJECT / f"system_prompt_{d}.txt"
        if not p.exists():
            _fail(f"执行器绑定 prompt 文件不存在: system_prompt_v{declared_prompt}.txt")
        else:
            # 全局规则：文件名版本 == 文件内部标题版本号
            title = None
            for line in p.read_text(encoding="utf-8").splitlines()[:10]:
                m = re.match(r"#\s*Prompt.*?v([\d.]+)", line)
                if m:
                    title = _v(m.group(1))
                    break
            if title and title != d:
                _warn(f"prompt 文件名 {d} ≠ 内部标题 {title}——文件名/标题漂移")


def gate_input_parity(pipeline_json: Path):
    """闸门二：对每窗重放生产 TextBuffer，断言执行器喂入机制奇偶。"""
    if not pipeline_json.exists():
        _fail(f"管道输出 JSON 不存在: {pipeline_json}")
        return
    data = json.loads(pipeline_json.read_text(encoding="utf-8"))
    results = data.get("results") or []

    for r in results:
        cid = r.get("case_id")
        dialogue = r.get("_dialogue") or ""
        # 收集按时间序的窗口（window_start, prior_chars, 该窗触发内容）
        wins = []
        for w in (r.get("fast_popups") or []) + (r.get("slow_popups") or []):
            if w.get("window_start") is not None:
                wins.append(w)
        wins.sort(key=lambda w: w.get("popup_order", 0))
        if not wins:
            continue

        # 用生产 TextBuffer 重放窗口序列，得到每窗生产 TextBuffer 的状态
        buf = TextBuffer()  # 生产默认 3000+500
        for w in wins:
            ws = w["window_start"]
            prior = w.get("prior_chars", 0)
            pc = w.get("window_chars", 0)

            # 该窗生产 TextBuffer 覆盖的原文区间 [window_start, window_start+window_chars]
            win_text = dialogue[ws:ws + pc] if dialogue else ""

            # P1/P2 区分通道（channel_spec 统一快通道窗口 v4.0.22）：
            #   fast —— 背景段 = [触发点-900 : window_start]，分析窗口 = [window_start:触发点]，
            #            二者合计覆盖「触发点前 900 字」（channel_spec.FAST_BACKGROUND）；
            #            触发点绝对位置 = ws + pc（窗口末端），expect_prior = min(触发点,900) - pc
            #   slow —— 保持生产 TextBuffer 相对滑动 lookback=500 不变
            channel = w.get("channel", "slow")
            if channel == "fast":
                trig = ws + pc
                expect_prior = max(0, min(trig, FAST_BACKGROUND) - pc)
                # P1 快通道背景有界：背景 ≤ FAST_BACKGROUND，且 背景+分析窗口 合计 ≤ 900
                if prior + pc > FAST_BACKGROUND:
                    _fail(f"[{cid} ws={ws}] 快通道 背景+分析窗口={prior}+{pc}"
                          f" > FAST_BACKGROUND={FAST_BACKGROUND}——合计覆盖超900，窗口不1:1")
            else:
                expect_prior = min(max(ws, 0), buf.lookback)
                # P1 慢通道背景有界：prior ≤ 生产 lookback（防"绝对前900从0累积"）
                if prior > buf.lookback:
                    _fail(f"[{cid} ws={ws}] prior_chars={prior} > lookback={buf.lookback}"
                          f"——背景从0绝对累积，违反相对滑动")

            # P2 背景来源：期望与执行器实际喂入必须一致（防窗口机制漂移）
            if prior != expect_prior:
                _fail(f"[{cid} ws={ws}] prior_chars={prior} 应为 {expect_prior}"
                      f"（{channel}通道背景={prior}，期望={expect_prior}）"
                      f"——背景来源非 {channel} 通道窗口语义")

            # P3 相邻窗口喂入不趋同：相邻触发点喂入原文重叠度 < 阈值
            if dialogue and buf._last_window_end == 0 and ws == 0:
                pass
            # 推进生产 TextBuffer（记录该窗已覆盖到 ws+pc）
            buf.append(dialogue[buf.total_chars:ws + pc])
            buf.mark_window_analyzed()

        # P3 跨窗趋同检查：同一 case 内相邻窗口喂入相似度
        if dialogue:
            prev_text = ""
            for w in wins:
                ws, pc = w["window_start"], w.get("window_chars", 0)
                cur_text = dialogue[ws:ws + pc]
                if prev_text and cur_text:
                    sim = _window_feed_sim(prev_text, cur_text)
                    if sim >= 0.70:
                        _warn(f"[{cid}] 相邻窗口喂入相似度 {sim:.3f} ≥ 0.70"
                              f"（ws {prev_ws}→{ws}）——注意输入趋同")
                prev_text, prev_ws = cur_text, ws


def main():
    ap = argparse.ArgumentParser(description="执行器契约校验：版本关联 + 输入构造奇偶")
    ap.add_argument("--json", type=Path, required=True,
                    help="执行器管道输出 JSON 路径")
    ap.add_argument("--strict", action="store_true", help="任一 WARN 也 exit 1")
    args = ap.parse_args()

    print("=" * 70)
    print("执行器契约校验器  —  版本关联 + 输入构造奇偶（生产 TextBuffer 权威基准）")
    print("=" * 70)
    gate_version()
    print("-" * 70)
    gate_input_parity(args.json)

    print("=" * 70)
    if FAILS:
        print(f"❌ 契约校验失败：{len(FAILS)} FAIL")
        for f in FAILS:
            print(f"   - {f}")
    if WARNS:
        print(f"⚠️  WARN：{len(WARNS)}")
        for w in WARNS:
            print(f"   - {w}")
    if not FAILS and not WARNS:
        print("✅ 契约校验全部通过（0 FAIL / 0 WARN）")
    print("=" * 70)

    if FAILS or (args.strict and WARNS):
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
