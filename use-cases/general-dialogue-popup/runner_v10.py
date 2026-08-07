"""v1.0 独立运行器 — 零依赖部署，同事可直接使用。

自包含设计：内联所有需要的 Pydantic 模型，不依赖测试智能体的 src/ 树。

Usage:
    # 命令行：分析一段对话
    python runner_v10.py --dialogue "阿琳：方案改完了吗？\n老周：还在弄……"

    # 命令行：从文件读取对话
    python runner_v10.py --file dialogue.txt

    # Python 调用
    from runner_v10 import V10Runner
    runner = V10Runner()
    result = runner.run(dialogue_text="阿琳：你怎么又忘了回我消息？\n老周：我刚看到！")

依赖：pip install litellm pydantic pyyaml

v1.0 = 一般场景弹窗首版（借鉴亲子场景 v2.3 单 prompt 基线改写）。
架构：窗口模式（~300字切窗逐窗分析，LLM 自主判断洞察视角：自己/对方/模式）。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import re
import time
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

# ── 内联模型（等价于 src/models.py 中的 TestWindow / TestCase / SUTOutput）──


class TestWindow(BaseModel):
    """对话中的单句（一个窗口）。"""
    window_index: int = Field(default=0, ge=0)
    speaker: str = Field(default="")
    text: str = Field(default="")


class TestCase(BaseModel):
    """完整对话测试用例。"""
    case_id: str = Field(default="adhoc", min_length=1)
    title: str = Field(default="")
    source: str = Field(default="")
    windows: list[TestWindow] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class SUTOutput(BaseModel):
    """被测系统在单个窗口的输出。"""
    should_popup: bool = Field(default=False)
    tone: Optional[str] = Field(default=None)
    popup_insight: Optional[str] = Field(default=None)
    popup_suggestion: Optional[str] = Field(default=None)
    popup_text: Optional[str] = Field(default=None)
    channel: Optional[str] = Field(default=None)
    trigger_reason: Optional[str] = Field(default=None)
    suppressed: bool = Field(default=False)
    suppress_reason: Optional[str] = Field(default=None)
    delay_ms: Optional[float] = Field(default=None)
    token_usage: dict = Field(default_factory=dict)
    raw_response: dict = Field(default_factory=dict)


# ── V10Runner ────────────────────────────────────────────────────────────────


class V10Runner:
    """v1.0 独立运行器 — 单 prompt + 窗口模式（~300字切窗逐窗分析）。

    用法：
        runner = V10Runner()
        result = runner.run(dialogue_text="...")
        for popup in result["popups"]:
            print(popup["window_range"], popup["lens"], popup["text"])
    """

    _LENS_INSTRUCTIONS: dict[str, str] = {
        "自己": (
            "请聚焦**看清自己**的视角生成弹窗（60-180字）。"
            "必须：指出账号持有者此刻可能没察觉的盲区（话说重了 / 边界在丢 / 情绪上来了）"
            "→ 给出一个具体可做的说法或做法。"
        ),
        "对方": (
            "请聚焦**看见对方**的视角生成弹窗（60-180字）。"
            "必须：洞察对方没说出口的言下之意或未被表达的需求"
            "→ 给出一个具体可做的说法或做法。"
        ),
        "模式": (
            "请聚焦**看见模式**的视角生成弹窗（60-180字）。"
            "必须：点出两人之间正在重复的互动循环（如逼迫—退让、追—躲）"
            "→ 给出一个具体可做的说法或做法。"
        ),
    }

    def __init__(
        self,
        prompt_path: str | None = None,
        model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        window_size: int = 300,
        config_path: str | None = None,
        keyword_config_path: str | None = None,
        critical_min_context: int = 80,
        warning_backward_chars: int = 250,
        warning_forward_chars: int = 50,
    ):
        self._logger = logging.getLogger(self.__class__.__name__)

        # ── 加载配置 ──
        config: dict = {}
        if config_path:
            config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        elif Path("config.yaml").exists():
            config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8")) or {}

        cfg = config.get("models", {})
        self.model = model or os.getenv("LLM_MODEL") or cfg.get("gen") or "deepseek/deepseek-v4-pro"
        self.api_base = api_base or os.getenv("LLM_API_BASE") or cfg.get("gen_api_base")
        api_key_env = cfg.get("gen_api_key_env")
        self.api_key = (
            api_key
            or os.getenv("LLM_API_KEY")
            or (os.getenv(api_key_env) if api_key_env else None)
            or cfg.get("gen_api_key")
        )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.window_size = window_size
        self.critical_min_context = critical_min_context
        self.warning_backward_chars = warning_backward_chars
        self.warning_forward_chars = warning_forward_chars

        # ── 加载系统 prompt ──
        prompt_file = prompt_path or "system_prompt_v1.4.txt"
        if not Path(prompt_file).exists():
            raise FileNotFoundError(f"找不到系统提示词文件: {prompt_file}")
        self._system_prompt = Path(prompt_file).read_text(encoding="utf-8")

        # ── 加载关键词配置 ──
        keyword_path = keyword_config_path or "keyword_config.json"
        if not Path(keyword_path).exists():
            raise FileNotFoundError(f"找不到关键词配置文件: {keyword_path}")
        self._keywords = json.loads(Path(keyword_path).read_text(encoding="utf-8"))

    # ── 公共 API ────────────────────────────────────────────────────────

    def run(self, dialogue_text: str, lens: str | None = None) -> dict:
        """分析对话文本，返回弹窗列表。

        Args:
            dialogue_text: 完整对话文本，每行一句。
            lens: 可选，强制指定洞察视角（"自己" / "对方" / "模式"）。
                  不指定时 LLM 自主判断。

        Returns:
            dict with:
                - popups: list of {window_range, lens, text, raw_response}
                - windows: list of {index, speaker, text} (解析后的窗口)
                - raw_dialogue: 原始输入
        """
        # ── 解析对话为窗口 ──
        windows = self._parse_dialogue(dialogue_text)
        if not windows:
            return {"popups": [], "windows": [], "raw_dialogue": dialogue_text}

        # ── 切窗 ──
        chunks = self._chunk_windows(windows)
        if not chunks:
            return {"popups": [], "windows": self._windows_to_dicts(windows), "raw_dialogue": dialogue_text}

        # ── 逐窗调用 LLM ──
        popups: list[dict] = []
        for chunk_idx, (chunk_text, window_indices, trigger) in enumerate(chunks):
            self._logger.debug(
                "窗 %d/%d: %d字, 覆盖窗口 %s, 触发=%s",
                chunk_idx + 1, len(chunks), len(chunk_text), window_indices, trigger,
            )

            try:
                llm_output = self._call_llm(chunk_text, lens)
            except Exception as exc:
                self._logger.error("窗 %d LLM 调用失败: %s", chunk_idx + 1, exc)
                continue

            should_popup, detected_lens, popup_text = self._parse_llm_output(llm_output)

            if should_popup and popup_text:
                popups.append({
                    "window_range": f"{window_indices[0]}-{window_indices[-1]}",
                    "window_indices": window_indices,
                    "lens": detected_lens or "auto",
                    "trigger": trigger,
                    "text": popup_text.strip(),
                    "raw_response": llm_output,
                })

        return {
            "popups": popups,
            "windows": self._windows_to_dicts(windows),
            "raw_dialogue": dialogue_text,
        }

    # ── 对话解析 ────────────────────────────────────────────────────────

    @staticmethod
    def _parse_dialogue(dialogue_text: str) -> list[TestWindow]:
        """将对话文本解析为窗口列表。

        支持格式：
          - "妈妈：xxx" / "1妈妈：xxx"（带说话人前缀）
          - 纯文本（每行一句，按行切）
        """
        lines = [l.strip() for l in dialogue_text.strip().split("\n") if l.strip()]
        windows: list[TestWindow] = []

        for i, line in enumerate(lines):
            # 尝试解析 "说话人：内容" 格式
            if "：" in line or ":" in line:
                sep = "：" if "：" in line else ":"
                parts = line.split(sep, 1)
                speaker = parts[0].strip()
                # 去掉可能的数字前缀
                if speaker and speaker[0].isdigit():
                    # e.g. "1妈妈" → "妈妈"
                    speaker_clean = speaker.lstrip("0123456789. )）-")
                    if speaker_clean:
                        speaker = speaker_clean
                text = parts[1].strip() if len(parts) > 1 else ""
            else:
                speaker = ""
                text = line

            windows.append(TestWindow(
                window_index=i,
                speaker=speaker,
                text=text,
            ))

        return windows

    @staticmethod
    def _windows_to_dicts(windows: list[TestWindow]) -> list[dict]:
        return [{"index": w.window_index, "speaker": w.speaker, "text": w.text} for w in windows]

    # ── 切窗逻辑 ────────────────────────────────────────────────────────

    def _match_critical(self, text: str) -> str | None:
        """检查是否命中 critical 关键词。"""
        for kw in self._keywords["critical"]:
            if kw and kw in text:
                return f"keyword:critical:{kw}"
        return None

    def _match_pattern(self, text: str) -> str | None:
        """逐类检查 warning 级正则句式；仅匹配当前句。

        基于《关键对话》沉默/暴力框架，捕获结构性破坏句式：
        stealth_but（假性认同）、masking（沉默面具）、
        helpless_story（无助叙事）、controlling（绝对化控制）。
        """
        for category, patterns in self._keywords.get("patterns", {}).items():
            if category == "_note":
                continue
            for pattern in patterns:
                if pattern and re.search(pattern, text):
                    return f"pattern:warning:{category}:{pattern}"
        return None

    def _match_warning(self, text: str) -> str | None:
        """检查是否命中 warning 关键词。"""
        for kw in self._keywords["warning"]:
            if kw and kw in text:
                return f"keyword:warning:{kw}"
        return None

    def _chunk_windows(self, windows: list[TestWindow]) -> list[tuple[str, list[int], str]]:
        """切窗并触发关键词/句式提前分析。

        规则（优先级从高到低）：
          - critical keyword：最高优先级，沿用 critical 上下文策略（前 300 后补足 80）
          - warning pattern：逐句 re.search 正则句式匹配；命中后跳过 warning keyword
          - warning keyword：仅在 critical/pattern 均未命中时检查
          - pattern 与 warning keyword 均向前取 250 字、再向后等 50 字
          - 正常窗口：无触发时按 ~window_size 切窗

        Returns:
            list of (chunk_text, [window_indices], trigger)
        """
        sentences: list[tuple[str, int]] = []
        for w in windows:
            text = w.text.strip()
            if not text:
                continue
            sentence = f"{w.speaker}：{text}" if w.speaker else text
            sentences.append((sentence, w.window_index))

        n = len(sentences)
        if n == 0:
            return []

        lengths = [len(s) for s, _ in sentences]
        prefix = [0]
        for L in lengths:
            prefix.append(prefix[-1] + L)

        def range_len(l: int, r: int) -> int:
            """sum lengths[l..r] inclusive"""
            if l > r:
                return 0
            return prefix[r + 1] - prefix[l]

        def find_start(i: int, target_back: int, last_end: int) -> int:
            """向前取不超过 target_back 字的最早起始句（但不早于 last_end+1）。"""
            low = last_end + 1
            if i == 0 or range_len(low, i - 1) <= target_back:
                return low
            lo, hi = low, i - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if range_len(mid, i - 1) <= target_back:
                    hi = mid
                else:
                    lo = mid + 1
            return lo

        chunks: list[tuple[str, list[int], str]] = []
        last_end = -1
        i = 0
        while i < n:
            sentence, _ = sentences[i]
            crit = self._match_critical(sentence)
            pattern = self._match_pattern(sentence) if not crit else None
            warn = self._match_warning(sentence) if not crit and not pattern else None
            warning_trigger = pattern or warn

            if crit:
                # critical：向前最多 300，最少总上下文 80
                start = find_start(i, 300, last_end)
                end = i
                total = range_len(start, end)
                while end + 1 < n and total < self.critical_min_context:
                    end += 1
                    total += lengths[end]
                chunk_text = "\n".join(s for s, _ in sentences[start:end + 1])
                chunk_indices = [idx for _, idx in sentences[start:end + 1]]
                chunks.append((chunk_text, chunk_indices, crit))
                last_end = end
                i = end + 1
                continue

            if warning_trigger:
                # pattern / warning：向前 250，向后等 50
                start = find_start(i, self.warning_backward_chars, last_end)
                end = i
                after = 0
                while end + 1 < n and after < self.warning_forward_chars:
                    end += 1
                    after += lengths[end]
                chunk_text = "\n".join(s for s, _ in sentences[start:end + 1])
                chunk_indices = [idx for _, idx in sentences[start:end + 1]]
                chunks.append((chunk_text, chunk_indices, warning_trigger))
                last_end = end
                i = end + 1
                continue

            # 正常窗口：从 last_end+1 到 i 累计达 window_size 时 flush
            start = last_end + 1
            current_len = prefix[i + 1] - prefix[start]
            if current_len >= self.window_size or i == n - 1:
                chunk_text = "\n".join(s for s, _ in sentences[start:i + 1])
                chunk_indices = [idx for _, idx in sentences[start:i + 1]]
                chunks.append((chunk_text, chunk_indices, "window"))
                last_end = i
                i += 1
            else:
                i += 1

        return chunks

    # ── LLM 调用 ────────────────────────────────────────────────────────

    def _call_llm(self, chunk_text: str, lens: str | None = None) -> str:
        """调用 LLM 生成弹窗。"""
        import litellm

        user_msg = f"当前对话：\n{chunk_text}"
        if lens and lens in self._LENS_INSTRUCTIONS:
            user_msg += f"\n\n{self._LENS_INSTRUCTIONS[lens]}\n\n请直接输出弹窗全文（不附加解释、不输出JSON、不输出\"弹窗：\"等前缀）："

        kwargs: dict = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=120,
        )
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key:
            kwargs["api_key"] = self.api_key

        resp = litellm.completion(**kwargs)
        return (resp.choices[0].message.content or "").strip()

    @staticmethod
    def _parse_llm_output(raw: str) -> tuple[bool, str | None, str | None]:
        """解析 LLM 输出，判断是否弹窗、视角、弹窗文本。

        启发式规则：
          - 文本长度 > 10 字 → 视为弹窗
          - 按关键词粗判洞察视角（自己/对方/模式），仅作展示参考
        """
        text = raw.strip()
        if not text or len(text) <= 10:
            return False, None, None

        # 简单视角判断（仅展示用，不影响生成）
        if any(kw in text for kw in ["循环", "模式", "每次", "总是", "一方", "互动"]):
            lens = "模式"
        elif any(kw in text for kw in ["对方", "他其实", "她其实", "言下之意", "没说出口", "需求"]):
            lens = "对方"
        elif any(kw in text for kw in ["你也许", "你可能", "你没注意", "你自己", "边界", "情绪"]):
            lens = "自己"
        else:
            lens = None

        return True, lens, text


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="v1.0 沟通现场弹窗 · 一般场景运行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python runner_v10.py --dialogue "阿琳：方案改完了吗？\\n老周：还在弄……"
  python runner_v10.py --file dialogue.txt
  python runner_v10.py --file dialogue.txt --lens 模式
  python runner_v10.py --file dialogue.txt --json  # JSON 输出
        """,
    )
    parser.add_argument(
        "--dialogue", "-d",
        help="对话文本（命令行直接输入）",
    )
    parser.add_argument(
        "--file", "-f",
        help="从文件读取对话文本",
    )
    parser.add_argument(
        "--lens", "-l",
        choices=["自己", "对方", "模式"],
        default=None,
        help="强制指定洞察视角（不指定则 LLM 自主判断）",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="以 JSON 格式输出",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="LLM 模型名（覆盖配置）",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API Key（覆盖配置）",
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="API Base URL（覆盖配置）",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="config.yaml 路径",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细日志",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

    # ── 获取对话文本 ──
    if args.dialogue:
        dialogue_text = args.dialogue.replace("\\n", "\n")
    elif args.file:
        dialogue_text = Path(args.file).read_text(encoding="utf-8").strip()
    else:
        # 尝试从 stdin 读取
        if not sys.stdin.isatty():
            dialogue_text = sys.stdin.read().strip()
        else:
            print("❌ 请提供 --dialogue 或 --file 参数，或通过管道输入", file=sys.stderr)
            parser.print_help()
            sys.exit(1)

    if not dialogue_text:
        print("❌ 对话文本为空", file=sys.stderr)
        sys.exit(1)

    # ── 初始化运行器 ──
    runner = V10Runner(
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        config_path=args.config,
    )

    # ── 执行 ──
    print(f"🔍 分析对话（{len(dialogue_text)} 字）...", file=sys.stderr)
    t0 = time.perf_counter()
    result = runner.run(dialogue_text, lens=args.lens)
    elapsed = time.perf_counter() - t0

    if args.json:
        import json
        result["elapsed_seconds"] = round(elapsed, 2)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        # ── 人类可读输出 ──
        print()
        print("=" * 60)
        print("  📋 v1.0 弹窗分析结果")
        print("=" * 60)
        print(f"  对话: {len(result['windows'])} 句, {len(dialogue_text)} 字")
        print(f"  耗时: {elapsed:.1f}s")
        print(f"  弹窗数: {len(result['popups'])}")
        print()

        if not result["popups"]:
            print("  (无弹窗 — LLM 判断当前对话不需要介入)")
        else:
            for i, popup in enumerate(result["popups"], 1):
                print(f"─── 弹窗 {i} ───")
                print(f"  窗口范围: {popup['window_range']}")
                print(f"  视角: {popup['lens']}")
                print(f"  弹窗文本:")
                for line in popup["text"].split("\n"):
                    print(f"    {line}")
                print()

        print("=" * 60)


if __name__ == "__main__":
    main()
