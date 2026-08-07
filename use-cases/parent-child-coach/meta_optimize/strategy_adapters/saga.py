"""SAGA 异步补偿适配器（同步 MVP 版）：Fast Path → Deep Review → Compensation。

LLM 调用: Fast Path (3 calls) + Deep Review (2 calls) = 5 calls/窗 (同步版)
首弹窗延迟: +0 (Fast Path 不变)
预期 M5: 短期 +0-5pp, 长期 +15-25pp
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

XINGLING_ROOT = Path("D:/星灵-soul-手搓/亲子沟通洞见/路线A_自上而下法_鼓励版")
sys.path.insert(0, str(XINGLING_ROOT))

from src.case_memory import CaseMemory, PerceptionReport, MasterDecision, PopupDraft
from src.multi_agent_orchestrator import MultiAgentResult


# ── SAGA 数据结构 ───────────────────────────────────────

@dataclass
class DeepReviewResult:
    window_id: str
    fast_path_tone: str
    deep_tone: str          # Deep Review 判断应该使用的 tone
    is_correction_needed: bool
    evidence: str = ""
    confidence: float = 0.0


@dataclass
class CompensationState:
    """跨窗口补偿状态。"""
    pending_correction: bool = False
    last_correction_tone: str = ""
    narrative_bias: str = ""  # 叙事偏斜方向，影响下窗 tone 选择


# ── Default prompts ─────────────────────────────────────

DEFAULT_DEEP_REVIEW_PROMPT = (
    "你是深度审视 agent。你收到的材料包括：当前窗口原文、Fast Path 的首弹窗 + tone、"
    "以及前序弹窗的叙事历史。\n\n"
    "你的任务是深度检查 Fast Path 的 tone 判定是否正确。你有三个 Fast Path 没有的优势：\n"
    "1. 你可以看到多窗口上下文（而非仅当前 300 字）\n"
    "2. 你没有 500ms 延迟限制，可以做更深入的分析\n"
    "3. 你可以识别跨窗口的叙事偏斜\n\n"
    "判断标准：\n"
    "- Fast Path tone 正确 → correction_needed=false\n"
    "- Fast Path tone 错误 → correction_needed=true，给出正确的 tone 和证据\n\n"
    "特别注意：\n"
    "- 假 empowering: 家长表面妥协但内心放弃 → 应该是 diagnostic\n"
    "- 假 diagnostic: 家长痛苦回避被误判为盲区 → 应该是 empowering\n\n"
    "输出JSON: {\n"
    '  "deep_tone": "diagnostic|empowering",\n'
    '  "is_correction_needed": true/false,\n'
    '  "evidence": "证据引用",\n'
    '  "confidence": 0.0-1.0,\n'
    '  "narrative_impact": "如果 tone 错了，对后续叙事的影响是什么"\n'
    "}"
)


class SAGAAdapter:
    """SAGA 异步补偿适配器（同步 MVP 版）。

    流程:
      Fast Path (当前流程) → 首弹窗
      Deep Review (同步运行在 MVP) → 检查 Fast Path tone
      如果需要修正 → 更新 CompensationState → 影响下一窗口的 tone 决策
    """

    def __init__(self, llm_client, model: str = "deepseek-v4-pro",
                 harness_dir: str = "", prompt_base_dir: str = ""):
        self._client = llm_client
        self._model = model
        self._harness_dir = Path(harness_dir) if harness_dir else None
        self._prompt_base = Path(prompt_base_dir) if prompt_base_dir else (
            XINGLING_ROOT / "prompts"
        )

        self._deep_review_prompt = self._load_deep_review_prompt()

        # 复用现有 agents
        from src.perception_agent import PerceptionAgent
        from src.master_agent import MasterAgent
        from src.production_agent import ProductionAgent

        perception_path = str(self._prompt_base / "prompt_感知层_v3.1.md")
        master_path = str(self._prompt_base / "prompt_总控_v3.1.md")
        production_path = str(self._prompt_base / "prompt_生产层_v3.1.md")

        self._perception = PerceptionAgent(llm_client, model, temperature=0.3, prompt_path=perception_path)
        self._master = MasterAgent(llm_client, model, temperature=0.4, prompt_path=master_path)
        self._production = ProductionAgent(llm_client, model, temperature=0.5, prompt_path=production_path)

        self._memories: dict[str, CaseMemory] = {}
        self._compensations: dict[str, CompensationState] = {}
        self._popup_history: dict[str, list[dict]] = {}  # family → [{"text":..., "tone":...}]
        self._total_windows = 0
        self._total_popups = 0

    def _load_deep_review_prompt(self) -> str:
        if self._harness_dir:
            md_path = self._harness_dir / "harness.md"
            if md_path.exists():
                text = md_path.read_text(encoding="utf-8")
                for section in text.split("\n## "):
                    if "Deep Review" in section or "深度审视" in section:
                        return section
        return DEFAULT_DEEP_REVIEW_PROMPT

    def _deep_review(self, window_text: str, fast_path_tone: str, family: str) -> DeepReviewResult:
        """Deep Review: 深度审视 Fast Path 的 tone 判定。"""
        history = self._popup_history.get(family, [])
        history_text = "\n".join(
            f"窗口{p['window']}: tone={p['tone']}, 弹窗={p['text'][:100]}"
            for p in history[-3:]  # 最近 3 个窗口
        )

        user_prompt = (
            f"## 当前窗口原文\n{window_text[:800]}\n\n"
            f"## Fast Path 判定\ntone: {fast_path_tone}\n\n"
            f"## 前序弹窗历史\n{history_text or '（首窗）'}"
        )

        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self._deep_review_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.3, timeout=60,
            )
            raw = resp.choices[0].message.content
            data = json.loads(raw) if raw else {}
        except Exception:
            data = {}

        return DeepReviewResult(
            window_id=str(self._total_windows),
            fast_path_tone=fast_path_tone,
            deep_tone=data.get("deep_tone", fast_path_tone),
            is_correction_needed=data.get("is_correction_needed", False),
            evidence=data.get("evidence", ""),
            confidence=float(data.get("confidence", 0.0)),
        )

    def process_window(self, window_text: str, family: str = "") -> MultiAgentResult:
        """SAGA 流程：Fast Path → Deep Review → Compensation。"""
        if not window_text or len(window_text.strip()) < 20:
            return MultiAgentResult(should_popup=False)

        self._total_windows += 1
        memory = self._get_memory(family)
        comp = self._get_compensation(family)

        # ── Fast Path（当前流程，不变）──────────────────
        perception = self._perception.perceive(window_text)

        # 如果有待处理的叙事修正，传递给 Master
        if comp.pending_correction:
            # 在 perception 上附加修正信号
            perception.response_need = (
                f"needs_{comp.last_correction_tone}"
                if comp.last_correction_tone in ("diagnostic", "empowering")
                else perception.response_need
            )

        decision = self._master.decide(perception, memory, window_text)

        if not decision.should_popup:
            comp.pending_correction = False  # 清除修正等待
            return MultiAgentResult(
                should_popup=False,
                perception=perception,
                decision=decision,
            )

        # Production (Fast Path)
        prev_text = ""
        if memory.last_popup:
            p = memory.last_popup
            prev_text = p.popup_insight

        draft = self._production.produce(
            decision, perception, window_text,
            decision.story_arc, prev_text,
        )

        # 记录弹窗历史
        if family not in self._popup_history:
            self._popup_history[family] = []
        self._popup_history[family].append({
            "window": self._total_windows,
            "text": draft.popup_insight,
            "tone": decision.direction,
        })

        # ── Deep Review（同步版 MVP）────────────────────
        deep_result = self._deep_review(window_text, decision.direction, family)

        # ── Compensation ────────────────────────────────
        if deep_result.is_correction_needed and deep_result.confidence >= 0.7:
            comp.pending_correction = True
            comp.last_correction_tone = deep_result.deep_tone
            comp.narrative_bias = (
                f"上一窗(Fast Path={deep_result.fast_path_tone}, "
                f"Deep Review={deep_result.deep_tone}): {deep_result.evidence[:100]}"
            )

            # 将补偿信息写入 decision 供审计
            decision.contradiction_flag = (
                f"[SAGA] Fast Path={deep_result.fast_path_tone} → "
                f"Deep Review={deep_result.deep_tone}, "
                f"correction pending for next window"
            )
        else:
            comp.pending_correction = False

        self._total_popups += 1

        return MultiAgentResult(
            popup_text=draft.popup_insight,
            should_popup=True,
            tone=decision.direction,
            main_contradiction=decision.main_contradiction,
            story_arc=decision.story_arc,
            route_a_insight=f"[Fast Path] {decision.route_a_insight}",
            route_b_insight=f"[Deep Review] correction={deep_result.is_correction_needed}, "
                           f"deep_tone={deep_result.deep_tone}, conf={deep_result.confidence:.2f}",
            perception=perception,
            decision=decision,
            draft=draft,
        )

    def _get_memory(self, family: str) -> CaseMemory:
        if family not in self._memories:
            self._memories[family] = CaseMemory()
        return self._memories[family]

    def _get_compensation(self, family: str) -> CompensationState:
        if family not in self._compensations:
            self._compensations[family] = CompensationState()
        return self._compensations[family]
