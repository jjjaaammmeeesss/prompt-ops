"""周易八卦 · 亲子沟通实时弹窗子系统。

基于师父的周易三爻八卦亲子沟通模型，在对话进行中实时分析家长的沟通状态，
识别八卦卦象，决定是否弹窗干预，并生成诊断式或鼓励式弹窗内容。

核心组件:
  - ZhouYiAnalyzer:      Stage 1 — 周易八卦状态分类
  - PopupGenerator:      Stage 2 — 弹窗内容生成
  - StreamOrchestrator:  流式编排器（触发 + 去抖 + 串联）
  - TextBuffer:          滑动窗口文本缓冲
  - TriggerEngine:       字数 + 关键词触发管理
  - DebounceGate:        弹窗去抖门控

Quickstart:
  from realtime import StreamOrchestrator, ZhouYiAnalyzer, PopupGenerator
  from prompt_ops.core.model import LiteLLMModelAdapter

  stage1 = LiteLLMModelAdapter(model_name="deepseek/deepseek-chat", api_key=key)
  stage2 = LiteLLMModelAdapter(model_name="deepseek/deepseek-chat", api_key=key)

  orchestrator = StreamOrchestrator(
      analyzer=ZhouYiAnalyzer(stage1),
      generator=PopupGenerator(stage2),
      keyword_file="keyword_config.json",
  )

  async for popup in orchestrator.process_chunk(text_chunk):
      if popup:
          print(popup.full_text)
"""

from .output_schemas import (
    YaoState,
    Trigram,
    PopupTone,
    ZhouYiState,
    Popup,
    TriggerEvent,
)
from .zhouyi_analyzer import ZhouYiAnalyzer
from .popup_generator import PopupGenerator
from .stream_orchestrator import (
    StreamOrchestrator,
    TextBuffer,
    TriggerEngine,
    DebounceGate,
)

__all__ = [
    # 数据模型
    "YaoState",
    "Trigram",
    "PopupTone",
    "ZhouYiState",
    "Popup",
    "TriggerEvent",
    # Stage 1 & 2
    "ZhouYiAnalyzer",
    "PopupGenerator",
    # 编排器
    "StreamOrchestrator",
    "TextBuffer",
    "TriggerEngine",
    "DebounceGate",
]
