"""周易八卦 · 亲子沟通实时弹窗系统 — CLI 演示工具。

支持两种输入模式：
  1. 文件模式: python cli_demo.py --input dialogue.txt
  2. 数据集模式: python cli_demo.py --dataset  # 从 dataset 中随机选取对话测试

Usage:
  python -m use-cases.parent-child-coach.realtime.cli_demo --input sample.txt
  python -m use-cases.parent-child-coach.realtime.cli_demo --dataset --index 0
"""

import asyncio
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import yaml

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("prompt_ops.realtime.demo")

# 确保项目路径可导入
_project_root = Path(__file__).resolve().parents[2]  # prompt-ops/
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# 加载 .env
from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parents[1] / ".env"  # parent-child-coach/.env
if _env_path.exists():
    load_dotenv(_env_path)

from prompt_ops.core.model import LiteLLMModelAdapter

# 支持直接运行（python cli_demo.py）和包导入（python -m realtime.cli_demo）
# 直接运行时，将 realtime/ 的父目录加入 sys.path，使 realtime 成为可导入的包
_realtime_parent = str(Path(__file__).resolve().parents[1])  # parent-child-coach/
if _realtime_parent not in sys.path:
    sys.path.insert(0, _realtime_parent)

from realtime.output_schemas import Popup, ZhouYiState
from realtime.zhouyi_analyzer import ZhouYiAnalyzer
from realtime.popup_generator import PopupGenerator
from realtime.stream_orchestrator import StreamOrchestrator


# ============================================================
# 配置加载
# ============================================================

def load_config(config_path: str = None) -> dict:
    """加载实时弹窗系统配置。"""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        logger.warning(f"Config not found at {config_path}, using defaults")
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from {config_path}")
    return config or {}


def setup_models(config: dict) -> tuple:
    """根据配置创建 Stage 1 和 Stage 2 的模型适配器。"""
    api_key = os.getenv("DEEPSEEK_API_KEY")

    analyzer_cfg = config.get("analyzer", {})
    generator_cfg = config.get("generator", {})

    stage1_model = LiteLLMModelAdapter(
        model_name=analyzer_cfg.get("model", "deepseek/deepseek-chat"),
        api_key=api_key,
        temperature=analyzer_cfg.get("temperature", 0.0),
        max_tokens=analyzer_cfg.get("max_tokens", 256),
    )

    stage2_model = LiteLLMModelAdapter(
        model_name=generator_cfg.get("model", "deepseek/deepseek-chat"),
        api_key=api_key,
        temperature=generator_cfg.get("temperature", 0.3),
        max_tokens=generator_cfg.get("max_tokens", 640),
    )

    logger.info(
        f"Models initialized:\n"
        f"  Stage 1 (analyzer): {stage1_model.model_name}\n"
        f"  Stage 2 (generator): {stage2_model.model_name}"
    )

    return stage1_model, stage2_model


# ============================================================
# 演示输出格式化
# ============================================================

class DemoDisplay:
    """格式化演示输出。"""

    HEADER = "\033[1;36m"  # 青色加粗
    POPUP_BG = "\033[1;33m"  # 黄色
    DIAGNOSTIC = "\033[1;31m"  # 红色（诊断）
    ENCOURAGING = "\033[1;32m"  # 绿色（鼓励）
    DIM = "\033[2m"  # 暗色
    RESET = "\033[0m"

    @staticmethod
    def print_header(text: str):
        print(f"\n{DemoDisplay.HEADER}{'='*60}{DemoDisplay.RESET}")
        print(f"{DemoDisplay.HEADER}  {text}{DemoDisplay.RESET}")
        print(f"{DemoDisplay.HEADER}{'='*60}{DemoDisplay.RESET}")

    @staticmethod
    def print_analysis(state: ZhouYiState, trigger_reason: str):
        """打印 Stage 1 分析结果。"""
        print(f"\n{DemoDisplay.DIM}─── 分析触发 ({trigger_reason}) ───{DemoDisplay.RESET}")
        print(f"  卦象: {state.trigram.symbol} {state.trigram.chinese_name} "
              f"({state.trigram.yao_pattern}) — {state.trigram.description}")
        print(f"  风险: {state.risk_level} | 容器: {state.container_status}")
        print(f"  置信度: {state.confidence:.0%}")
        print(f"  💡 {state.brief_reason}")

    @staticmethod
    def print_popup(popup: Popup):
        """打印弹窗内容。"""
        color = DemoDisplay.DIAGNOSTIC if popup.tone.value == "diagnostic" else DemoDisplay.ENCOURAGING
        symbol = popup.zhouyi_context.trigram.symbol if popup.zhouyi_context else ""

        print(f"\n{color}┌{'─'*50}┐{DemoDisplay.RESET}")
        print(f"{color}│ {symbol}  {popup.tone.value.upper()} 弹窗 ({popup.char_count}字){DemoDisplay.RESET}")
        print(f"{color}├{'─'*50}┤{DemoDisplay.RESET}")

        # 主文本
        for line in popup.popup_insight.split("\n"):
            print(f"{color}│ {line}{DemoDisplay.RESET}")

        if popup.popup_suggestion:
            print(f"{color}│{DemoDisplay.RESET}")
            print(f"{color}│ ——{DemoDisplay.RESET}")
            for line in popup.popup_suggestion.split("\n"):
                print(f"{color}│ {line}{DemoDisplay.RESET}")

        print(f"{color}└{'─'*50}┘{DemoDisplay.RESET}")

    @staticmethod
    def print_suppressed(state: ZhouYiState, reason: str):
        """打印被抑制的触发信息。"""
        print(f"  {DemoDisplay.DIM}⊘ 抑制: {reason}{DemoDisplay.RESET}")

    @staticmethod
    def print_summary(stats: dict):
        """打印运行统计摘要。"""
        print(f"\n{DemoDisplay.HEADER}{'='*60}{DemoDisplay.RESET}")
        print(f"{DemoDisplay.HEADER}  运行统计摘要{ DemoDisplay.RESET}")
        print(f"{DemoDisplay.HEADER}{'='*60}{DemoDisplay.RESET}")
        print(f"  对话总字数: {stats['total_chars']}")
        print(f"  分析次数: {stats['analysis_count']}")
        print(f"  弹窗次数: {stats['popup_count']}")
        print(f"  抑制次数: {stats['suppressed_count']}")
        print(f"  抑制率: {stats['suppression_rate']:.1%}")
        if stats.get("last_state"):
            s = stats["last_state"]
            print(f"  最终卦象: {s.get('trigram_symbol', '?')} {s.get('trigram_name', '?')}")


# ============================================================
# 主演示函数
# ============================================================

async def demo_from_text(
    text: str,
    orchestrator: StreamOrchestrator,
    display: DemoDisplay,
    config: dict,
) -> list:
    """从完整文本模拟流式输入，逐块处理。"""
    demo_cfg = config.get("demo", {})
    chunk_size = config.get("buffer", {}).get("chunk_size", 60)
    chunk_delay = demo_cfg.get("chunk_delay_ms", 50) / 1000.0
    show_suppressed = demo_cfg.get("show_suppressed", True)

    display.print_header("开始实时分析（模拟流式输入）")
    print(f"  文本长度: {len(text)} 字符 | 块大小: {chunk_size} 字符 | "
          f"块延迟: {chunk_delay*1000:.0f}ms")
    print()

    popups = []

    # 按 chunk_size 分块
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])

    for i, chunk in enumerate(chunks):
        # 实时显示文本
        print(chunk, end="", flush=True)

        popup = await orchestrator.process_chunk(chunk)

        if popup:
            popups.append(popup)
            display.print_popup(popup)
        elif show_suppressed and orchestrator._analysis_count > len(popups):
            pass  # suppress 日志已在 orchestrator 内部打印

        if chunk_delay > 0:
            await asyncio.sleep(chunk_delay)

    print("\n")
    display.print_summary(orchestrator.stats)
    return popups


async def demo_from_dataset(
    orchestrator: StreamOrchestrator,
    display: DemoDisplay,
    config: dict,
    index: int = None,
    dataset_path: str = None,
) -> list:
    """从数据集中选取对话进行演示。"""
    # 查找数据集
    if dataset_path is None:
        dataset_path = Path(__file__).resolve().parents[1] / "data" / "dataset_merged_train.json"
    else:
        dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        print(f"错误: 数据集文件不存在: {dataset_path}")
        return []

    data = json.loads(dataset_path.read_text(encoding="utf-8"))

    if index is not None:
        if index >= len(data):
            print(f"错误: 索引 {index} 超出范围 (共 {len(data)} 条)")
            return []
        item = data[index]
    else:
        item = random.choice(data)

    dialogue = item.get("question", "")
    dialogue_id = item.get("id", index or "?")

    display.print_header(f"数据集演示 — ID: {dialogue_id}")
    return await demo_from_text(dialogue, orchestrator, display, config)


async def demo_interactive(
    orchestrator: StreamOrchestrator,
    display: DemoDisplay,
    config: dict,
):
    """交互式逐行输入模式。"""
    display.print_header("交互模式 — 逐行输入对话")
    print("  输入对话内容，每次回车触发分析检查。")
    print("  输入空行结束。")
    print("  输入 :stats 查看统计。")
    print("  输入 :reset 重置对话。")
    print("  输入 :quit 退出。")
    print()

    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            print("\n结束。")
            break

        if line == ":quit":
            break
        elif line == ":stats":
            stats = orchestrator.stats
            print(f"  字数: {stats['total_chars']} | "
                  f"分析: {stats['analysis_count']} | "
                  f"弹窗: {stats['popup_count']} | "
                  f"抑制: {stats['suppressed_count']}")
            continue
        elif line == ":reset":
            orchestrator.reset()
            print("  已重置。")
            continue
        elif line == "":
            # 空行也触发分析（可能包含重要信号）
            continue

        chunk = line + "\n"
        print(f"\033[2m  ...\033[0m", end="")  # 提示正在处理
        popup = await orchestrator.process_chunk(chunk)

        if popup:
            display.print_popup(popup)

    display.print_summary(orchestrator.stats)


# ============================================================
# CLI 入口（同步包装器）
# ============================================================

def main(
    input_file: str = None,
    dataset: bool = False,
    dataset_index: int = None,
    interactive: bool = False,
    config_path: str = None,
):
    """CLI 主入口。

    Args:
        input_file: 输入文本文件路径
        dataset: 是否使用数据集模式
        dataset_index: 数据集索引（None 表示随机）
        interactive: 交互模式
        config_path: 配置文件路径
    """
    # 加载配置
    config = load_config(config_path)

    # 初始化模型
    stage1_model, stage2_model = setup_models(config)

    # 初始化组件
    trigger_cfg = config.get("trigger", {})
    debounce_cfg = config.get("debounce", {})
    buffer_cfg = config.get("buffer", {})
    generator_cfg = config.get("generator", {})

    analyzer = ZhouYiAnalyzer(
        model_adapter=stage1_model,
        temperature=config.get("analyzer", {}).get("temperature", 0.0),
        max_tokens=config.get("analyzer", {}).get("max_tokens", 256),
        timeout=config.get("analyzer", {}).get("timeout", 15.0),
    )

    generator = PopupGenerator(
        model_adapter=stage2_model,
        system_prompt_path=generator_cfg.get("system_prompt_path"),
        temperature=generator_cfg.get("temperature", 0.3),
        max_tokens=generator_cfg.get("max_tokens", 640),
        dedup_config=config.get("dedup", {}),
    )

    orchestrator = StreamOrchestrator(
        analyzer=analyzer,
        generator=generator,
        char_trigger=trigger_cfg.get("char_trigger", 120),
        min_chars_for_analysis=trigger_cfg.get("min_chars_for_analysis", 60),
        min_interval_ms=trigger_cfg.get("min_interval_ms", 3000),
        keyword_file=trigger_cfg.get("keyword_file", "keyword_config.json"),
        cooldown_seconds=debounce_cfg.get("cooldown_seconds", 15.0),
        same_state_max_repeats=debounce_cfg.get("same_state_max_repeats", 2),
        window_size=buffer_cfg.get("window_size", 3000),
        lookback=buffer_cfg.get("lookback", 500),
        stable_block_enabled=config.get("daily_conversation_skip", {}).get(
            "enabled", True
        ),
    )

    display = DemoDisplay()
    show_suppressed = config.get("demo", {}).get("show_suppressed", True)
    show_analysis = config.get("demo", {}).get("show_analysis_log", True)

    if not show_analysis:
        logging.getLogger("prompt_ops.realtime").setLevel(logging.WARNING)

    # 运行
    if interactive:
        asyncio.run(demo_interactive(orchestrator, display, config))

    elif dataset:
        asyncio.run(
            demo_from_dataset(orchestrator, display, config, index=dataset_index)
        )

    elif input_file:
        input_path = Path(input_file)
        if not input_path.exists():
            print(f"错误: 文件不存在: {input_file}")
            return 1

        text = input_path.read_text(encoding="utf-8")
        asyncio.run(demo_from_text(text, orchestrator, display, config))

    else:
        # 默认：数据集随机一条
        asyncio.run(
            demo_from_dataset(orchestrator, display, config)
        )

    return 0


# ============================================================
# 快速测试函数
# ============================================================

def quick_test():
    """快速测试：用一段内置对话测试完整流程。"""
    test_dialogue = """妈妈我回来了
回来啦 今天学校怎么样
还行吧
还行是什么意思 看你好像不太高兴
没什么
你先喝点水 坐下来慢慢说
我今天不想说
不想说就不说 妈妈不逼你 但如果你想说了 我在这儿
真的吗
当然真的
其实今天跟小明吵架了
为什么事吵架
他说我画的画很难看
你觉得呢
我觉得还不错
那就够了 你觉得不错就好
但是同学们都笑我
被很多人笑是挺难受的
嗯
你画的什么
我画了一只蓝色的猫
蓝色的猫 挺有创意的啊
你也觉得有创意吗
当然 谁说猫一定要画成橘色的
妈妈你真的这么想
真的 你小时候画的画我都留着呢 每一张都特别
那你最喜欢哪一张
最喜欢你画的全家福 把爸爸画成了一个球
哈哈哈 因为爸爸肚子大
你下次画小明的时候 也给他画成动物试试
好啊 那我把他画成一只猪
别 画成他喜欢的 比如他喜欢恐龙
好吧 那我画个剑龙
这就对了 你比小明大度
妈妈我爱你
我也爱你 去洗手吧 一会儿吃饭了"""

    config = load_config()
    stage1, stage2 = setup_models(config)

    analyzer = ZhouYiAnalyzer(model_adapter=stage1)
    generator = PopupGenerator(
        model_adapter=stage2,
        system_prompt_path=config.get("generator", {}).get("system_prompt_path"),
    )

    orchestrator = StreamOrchestrator(
        analyzer=analyzer,
        generator=generator,
        output_callback=None,  # 由 DemoDisplay 统一处理输出
        char_trigger=120,
        keyword_file="keyword_config.json",
        cooldown_seconds=15.0,
    )

    display = DemoDisplay()
    asyncio.run(demo_from_text(test_dialogue, orchestrator, display, config))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="周易八卦 · 亲子沟通实时弹窗系统 — CLI 演示"
    )
    parser.add_argument(
        "--input", "-i", type=str, help="输入文本文件路径"
    )
    parser.add_argument(
        "--dataset", "-d", action="store_true", help="从数据集中选取对话"
    )
    parser.add_argument(
        "--index", type=int, default=None, help="数据集索引（配合 --dataset）"
    )
    parser.add_argument(
        "--interactive", action="store_true", help="交互式逐行输入模式"
    )
    parser.add_argument(
        "--config", "-c", type=str, default=None, help="配置文件路径"
    )
    parser.add_argument(
        "--quick-test", action="store_true", help="用内置测试对话快速验证"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="显示详细日志"
    )

    args = parser.parse_args()

    if not args.verbose:
        logging.getLogger("prompt_ops.realtime").setLevel(logging.WARNING)

    if args.quick_test:
        quick_test()
    else:
        sys.exit(
            main(
                input_file=args.input,
                dataset=args.dataset,
                dataset_index=args.index,
                interactive=args.interactive,
                config_path=args.config,
            )
        )
