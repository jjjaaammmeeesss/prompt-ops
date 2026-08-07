"""v2.3 独立运行器 — 零依赖部署，同事可直接使用。

自包含设计：内联所有需要的 Pydantic 模型，不依赖测试智能体的 src/ 树。

Usage:
    # 命令行：分析一段对话
    python runner_v23.py --dialogue "妈妈：作业写完了吗？\n孩子：还没……"

    # 命令行：从文件读取对话
    python runner_v23.py --file dialogue.txt

    # Python 调用
    from runner_v23 import V23Runner
    runner = V23Runner()
    result = runner.run(dialogue_text="妈妈：你怎么又在玩手机？\n孩子：我刚拿起来！")

依赖：pip install litellm pydantic pyyaml

v2.3 = 2025 年三位专家手工标注的单 prompt 基线。
架构：窗口模式（~300字切窗逐窗分析，LLM 自主判断 diagnostic/empowering tone）。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
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


# ── V23Runner ────────────────────────────────────────────────────────────────


class V23Runner:
    """v2.3 独立运行器 — 单 prompt + 窗口模式（~300字切窗逐窗分析）。

    用法：
        runner = V23Runner()
        result = runner.run(dialogue_text="...")
        for popup in result["popups"]:
            print(popup["window_range"], popup["tone"], popup["text"])
    """

    _TONE_INSTRUCTIONS: dict[str, str] = {
        "诊断式": (
            "请生成**诊断式弹窗**（100-200字）。"
            "必须：先承认发心 → 揭示具体模式 → 给出一个微小可做的尝试。"
        ),
        "鼓励式": (
            "请生成**鼓励式弹窗**（30-60字）。"
            "必须：具体点出家长刚展现的积极模式 → 简短有力。"
        ),
    }

    def __init__(
        self,
        prompt_path: str | None = None,
        model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 640,
        window_size: int = 300,
        config_path: str | None = None,
    ):
        """初始化 v2.3 运行器。

        Args:
            prompt_path: system_prompt_v2.3.txt 路径。默认在同目录下查找。
            model: LLM 模型名（如 "deepseek/deepseek-chat"）。
            api_base: API base URL。
            api_key: API key。
            temperature: 生成温度。
            max_tokens: 最大输出 token 数。
            window_size: 切窗字数（默认 300）。
            config_path: config.yaml 路径。如提供，从其中读取模型配置。
        """
        self._logger = logging.getLogger("V23Runner")

        # ── 加载配置 ──
        cfg = {}
        if config_path is None:
            # 默认：同目录下的 config.yaml
            default_cfg = Path(__file__).parent / "config.yaml"
            if default_cfg.exists():
                config_path = str(default_cfg)

        if config_path:
            cfg_path = Path(config_path)
            if cfg_path.exists():
                cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

        # ── resolve prompt path ──
        if prompt_path is None:
            prompt_path = str(Path(__file__).parent / "system_prompt_v2.3.txt")

        self.prompt_path = Path(prompt_path)
        if not self.prompt_path.exists():
            raise FileNotFoundError(f"Prompt 文件不存在: {self.prompt_path}")

        self._system_prompt = self.prompt_path.read_text(encoding="utf-8").strip()

        # ── resolve model config ──
        models_cfg = cfg.get("models", cfg)
        self.model = model or os.environ.get("V23_MODEL") or models_cfg.get("gen", "deepseek/deepseek-chat")
        self.api_base = api_base or os.environ.get("V23_API_BASE") or models_cfg.get("gen_api_base", "")
        self.api_key = api_key or os.environ.get("V23_API_KEY") or ""

        # 如果 api_key 引用环境变量名（如 "${DEEPSEEK_API_KEY}"），解析它
        if self.api_key.startswith("${") and self.api_key.endswith("}"):
            env_var = self.api_key[2:-1]
            self.api_key = os.environ.get(env_var, "")

        gen_key_env = models_cfg.get("gen_api_key_env", "")
        if gen_key_env and not self.api_key:
            self.api_key = os.environ.get(gen_key_env, "")

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.window_size = window_size

        self._logger.info(
            "V23Runner 初始化完成: prompt=%s (%d字), model=%s, window_size=%d",
            self.prompt_path.name,
            len(self._system_prompt),
            self.model,
            self.window_size,
        )

    # ── 公共 API ────────────────────────────────────────────────────────

    def run(self, dialogue_text: str, tone: str | None = None) -> dict:
        """分析对话文本，返回弹窗列表。

        Args:
            dialogue_text: 完整对话文本，每行一句。
            tone: 可选，强制指定 tone（"诊断式" 或 "鼓励式"）。
                  不指定时 LLM 自主判断。

        Returns:
            dict with:
                - popups: list of {window_range, tone, text, raw_response}
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
        for chunk_idx, (chunk_text, window_indices) in enumerate(chunks):
            self._logger.debug(
                "窗 %d/%d: %d字, 覆盖窗口 %s",
                chunk_idx + 1, len(chunks), len(chunk_text), window_indices,
            )

            try:
                llm_output = self._call_llm(chunk_text, tone)
            except Exception as exc:
                self._logger.error("窗 %d LLM 调用失败: %s", chunk_idx + 1, exc)
                continue

            should_popup, detected_tone, popup_text = self._parse_llm_output(llm_output)

            if should_popup and popup_text:
                popups.append({
                    "window_range": f"{window_indices[0]}-{window_indices[-1]}",
                    "window_indices": window_indices,
                    "tone": detected_tone or "auto",
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

    def _chunk_windows(self, windows: list[TestWindow]) -> list[tuple[str, list[int]]]:
        """按 window_size 切窗，在句边界断开。

        Returns:
            list of (chunk_text, [覆盖的 window_indices])
        """
        # 构建带说话人的句子
        sentences: list[tuple[str, int]] = []
        for w in windows:
            text = w.text.strip()
            if not text:
                continue
            if w.speaker:
                sentence = f"{w.speaker}：{text}"
            else:
                sentence = text
            sentences.append((sentence, w.window_index))

        if not sentences:
            return []

        chunks: list[tuple[str, list[int]]] = []
        buf = sentences[0][0]
        buf_indices = [sentences[0][1]]

        for sentence, wi in sentences[1:]:
            sep_len = 1
            if len(buf) + sep_len + len(sentence) <= self.window_size:
                buf += "\n" + sentence
                buf_indices.append(wi)
            else:
                chunks.append((buf, buf_indices))
                buf = sentence
                buf_indices = [wi]

        if buf:
            chunks.append((buf, buf_indices))

        return chunks

    # ── LLM 调用 ────────────────────────────────────────────────────────

    def _call_llm(self, chunk_text: str, tone: str | None = None) -> str:
        """调用 LLM 生成弹窗。"""
        import litellm

        user_msg = f"当前对话：\n{chunk_text}"
        if tone and tone in self._TONE_INSTRUCTIONS:
            user_msg += f"\n\n{self._TONE_INSTRUCTIONS[tone]}\n\n请直接输出弹窗全文（不附加解释、不输出JSON、不输出\"弹窗：\"等前缀）："

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
        """解析 LLM 输出，判断是否弹窗、tone、弹窗文本。

        启发式规则：
          - 文本长度 > 10 字 → 视为弹窗
          - 包含"鼓励"关键词 → empowering tone
          - 包含"诊断"/"看见"/"注意" → diagnostic tone
        """
        text = raw.strip()
        if not text or len(text) <= 10:
            return False, None, None

        # 简单 tone 判断
        if any(kw in text for kw in ["鼓励", "做对了", "很棒", "真不错"]):
            tone = "鼓励式"
        elif any(kw in text for kw in ["诊断", "看见", "注意", "也许", "可能还有", "没注意到"]):
            tone = "诊断式"
        else:
            tone = None

        return True, tone, text


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="v2.3 亲子沟通教练 · 现场弹窗运行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python runner_v23.py --dialogue "妈妈：作业写完了吗？\\n孩子：还没……"
  python runner_v23.py --file dialogue.txt
  python runner_v23.py --file dialogue.txt --tone 诊断式
  python runner_v23.py --file dialogue.txt --json  # JSON 输出
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
        "--tone", "-t",
        choices=["诊断式", "鼓励式"],
        default=None,
        help="强制指定弹窗 tone（不指定则 LLM 自主判断）",
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
    runner = V23Runner(
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        config_path=args.config,
    )

    # ── 执行 ──
    print(f"🔍 分析对话（{len(dialogue_text)} 字）...", file=sys.stderr)
    t0 = time.perf_counter()
    result = runner.run(dialogue_text, tone=args.tone)
    elapsed = time.perf_counter() - t0

    if args.json:
        import json
        result["elapsed_seconds"] = round(elapsed, 2)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        # ── 人类可读输出 ──
        print()
        print("=" * 60)
        print("  📋 v2.3 弹窗分析结果")
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
                print(f"  Tone: {popup['tone']}")
                print(f"  弹窗文本:")
                for line in popup["text"].split("\n"):
                    print(f"    {line}")
                print()

        print("=" * 60)


if __name__ == "__main__":
    main()
