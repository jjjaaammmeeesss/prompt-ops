"""Senate 元老院适配器：3专家并行独立审视 → Speaker综合裁决。

LLM 调用: 3专家(并行/串行) + Speaker + Production = 5 calls/窗 (vs 当前3 calls)
预期 M5: 75-85% | 延迟 +1-2s
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# 星灵项目路径
XINGLING_ROOT = Path("D:/星灵-soul-手搓/亲子沟通洞见/路线A_自上而下法_鼓励版")
sys.path.insert(0, str(XINGLING_ROOT))

from src.case_memory import CaseMemory, PerceptionReport, MasterDecision, PopupDraft
from src.multi_agent_orchestrator import MultiAgentResult


# ── Senate 数据结构 ──────────────────────────────────────

@dataclass
class ExpertOpinion:
    expert_id: str        # "emotion" | "needs" | "development"
    tone: str             # "diagnostic" | "empowering"
    evidence: str         # 原文证据
    confidence: float     # 0.0-1.0


@dataclass
class SenateDecision:
    final_tone: str       # "diagnostic" | "empowering"
    votes: dict           # {"diagnostic": N, "empowering": N}
    resolution: str       # "unanimous" | "majority" | "speaker_tiebreak"
    reasoning: str


# ── Expert prompts（从 harness 文件加载，这里是 fallback） ──

DEFAULT_EXPERT_PROMPTS = {
    "emotion": (
        "你是一位情感视角的亲子沟通专家。\n"
        "观察这段对话中家长的情绪状态和孩子的情感回应。\n"
        "判断这个时刻最需要的是什么：\n"
        "- diagnostic（诊断式）：家长存在情绪盲区，需要被看见、被澄清\n"
        "- empowering（鼓励式）：家长做出了值得肯定的情感回应，需要被强化\n\n"
        "输出JSON: {\"tone\": \"diagnostic|empowering\", \"evidence\": \"原文引用\", "
        "\"confidence\": 0.0-1.0, \"reasoning\": \"判断理由\"}"
    ),
    "needs": (
        "你是一位需求视角的亲子沟通专家。\n"
        "观察这段对话中家长是否回应了孩子的核心心理需求（被理解、被接纳、安全感）。\n"
        "判断这个时刻最需要的是什么：\n"
        "- diagnostic（诊断式）：家长误解了孩子的需求，或忽视了关键信号\n"
        "- empowering（鼓励式）：家长准确识别并回应了孩子的需求\n\n"
        "输出JSON: {\"tone\": \"diagnostic|empowering\", \"evidence\": \"原文引用\", "
        "\"confidence\": 0.0-1.0, \"reasoning\": \"判断理由\"}"
    ),
    "development": (
        "你是一位发展心理学视角的亲子沟通专家。\n"
        "观察这段对话对孩子的长期发展（自主性、自我认知、情绪调节能力）有什么影响。\n"
        "判断这个时刻最需要的是什么：\n"
        "- diagnostic（诊断式）：家长的回应方式可能阻碍孩子的发展需求\n"
        "- empowering（鼓励式）：家长的回应方式支持了孩子的发展需求\n\n"
        "输出JSON: {\"tone\": \"diagnostic|empowering\", \"evidence\": \"原文引用\", "
        "\"confidence\": 0.0-1.0, \"reasoning\": \"判断理由\"}"
    ),
}

DEFAULT_SPEAKER_PROMPT = (
    "你是 Senate 议长。三位专家各自从不同视角（情感/需求/发展）审视了一段亲子对话，"
    "并给出了 tone 建议 + 证据 + 置信度。\n\n"
    "你需要综合裁决最终的 tone 方向。规则：\n"
    "1. 三方一致 → 直接通过（unanimous）\n"
    "2. 两方一致 → majority vote，但必须检查少数方的证据是否有说服力\n"
    "3. 三方各不同 → speaker tiebreak，优先置信度最高的专家，但说明理由\n"
    "4. 如果所有专家置信度都 < 0.5 → 输出 'diagnostic' 作为安全 fallback\n\n"
    "输出JSON: {\"final_tone\": \"diagnostic|empowering\", "
    "\"resolution\": \"unanimous|majority|speaker_tiebreak\", "
    "\"reasoning\": \"裁决理由，引用专家证据\"}"
)


class SenateAdapter:
    """Senate 元老院适配器。

    复用 PerceptionAgent 做输入分析（保留五维洞察作为 Speaker 的额外上下文），
    但 tone 决策不由单个 MasterAgent 做出——由 3 个专家 + Speaker 裁决。
    """

    def __init__(self, llm_client, model: str = "deepseek-v4-pro",
                 harness_dir: str = "", prompt_base_dir: str = ""):
        self._client = llm_client
        self._model = model
        self._harness_dir = Path(harness_dir) if harness_dir else None
        self._prompt_base = Path(prompt_base_dir) if prompt_base_dir else (
            XINGLING_ROOT / "prompts"
        )

        # 加载 expert & speaker prompts
        self._expert_prompts = self._load_expert_prompts()
        self._speaker_prompt = self._load_speaker_prompt()

        # 复用 PerceptionAgent 做输入分析
        from src.perception_agent import PerceptionAgent
        perception_path = str(self._prompt_base / "prompt_感知层_v3.1.md")
        self._perception = PerceptionAgent(llm_client, model, temperature=0.3, prompt_path=perception_path)

        # 复用 ProductionAgent
        from src.production_agent import ProductionAgent
        production_path = str(self._prompt_base / "prompt_生产层_v3.1.md")
        self._production = ProductionAgent(llm_client, model, temperature=0.5, prompt_path=production_path)

        # 简单记忆（每个 family 一个）
        self._memories: dict[str, CaseMemory] = {}
        self._total_windows = 0
        self._total_popups = 0

    def _load_expert_prompts(self) -> dict[str, str]:
        """从 harness_dir 或 defaults 加载专家 prompt。"""
        if self._harness_dir:
            md_path = self._harness_dir / "harness.md"
            if md_path.exists():
                text = md_path.read_text(encoding="utf-8")
                return self._parse_expert_prompts_from_md(text)
        return dict(DEFAULT_EXPERT_PROMPTS)

    def _parse_expert_prompts_from_md(self, text: str) -> dict[str, str]:
        """从 harness.md 解析三个专家 prompt。"""
        prompts = dict(DEFAULT_EXPERT_PROMPTS)
        sections = text.split("\n## ")
        for section in sections:
            for expert_id in ("emotion", "needs", "development"):
                if section.startswith(f"Expert {expert_id}") or section.startswith(f"专家 {expert_id}"):
                    prompts[expert_id] = section
        return prompts

    def _load_speaker_prompt(self) -> str:
        """从 harness_dir 或 defaults 加载 Speaker prompt。"""
        if self._harness_dir:
            md_path = self._harness_dir / "harness.md"
            if md_path.exists():
                text = md_path.read_text(encoding="utf-8")
                # 找 Speaker 部分
                for section in text.split("\n## "):
                    if "Speaker" in section or "议长" in section:
                        return section
        return DEFAULT_SPEAKER_PROMPT

    def _call_expert(self, expert_id: str, window_text: str) -> ExpertOpinion:
        """调用单个专家。"""
        system_prompt = self._expert_prompts.get(expert_id, DEFAULT_EXPERT_PROMPTS[expert_id])
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"对话内容：\n{window_text}"},
                ],
                response_format={"type": "json_object"},
                temperature=0.3, timeout=60,
            )
            raw = resp.choices[0].message.content
            data = json.loads(raw) if raw else {}
        except Exception:
            data = {}

        return ExpertOpinion(
            expert_id=expert_id,
            tone=data.get("tone", "diagnostic"),
            evidence=data.get("evidence", ""),
            confidence=float(data.get("confidence", 0.5)),
        )

    def _speaker_adjudicate(self, opinions: list[ExpertOpinion]) -> SenateDecision:
        """Speaker 裁决（简单多数投票，MVP 版不调 LLM）。"""
        votes = {"diagnostic": 0, "empowering": 0}
        for o in opinions:
            if o.tone in votes:
                votes[o.tone] += 1

        if votes["diagnostic"] >= 3:
            resolution = "unanimous"
            final_tone = "diagnostic"
        elif votes["empowering"] >= 3:
            resolution = "unanimous"
            final_tone = "empowering"
        elif votes["diagnostic"] >= 2:
            resolution = "majority"
            final_tone = "diagnostic"
        elif votes["empowering"] >= 2:
            resolution = "majority"
            final_tone = "empowering"
        else:
            # tiebreak: 取置信度加权最高的
            resolution = "speaker_tiebreak"
            diag_conf = sum(o.confidence for o in opinions if o.tone == "diagnostic")
            emp_conf = sum(o.confidence for o in opinions if o.tone == "empowering")
            final_tone = "diagnostic" if diag_conf >= emp_conf else "empowering"

        reasoning_parts = [f"{o.expert_id}={o.tone}(conf={o.confidence:.1f})" for o in opinions]
        return SenateDecision(
            final_tone=final_tone,
            votes=votes,
            resolution=resolution,
            reasoning=f"[{resolution}] " + ", ".join(reasoning_parts),
        )

    def process_window(self, window_text: str, family: str = "") -> MultiAgentResult:
        """Senate 流程：3专家 → Speaker → Production。"""
        if not window_text or len(window_text.strip()) < 20:
            return MultiAgentResult(should_popup=False)

        self._total_windows += 1

        # Step 1: 感知层（保留，用于 Production 的上下文，但不参与 tone 决策）
        perception = self._perception.perceive(window_text)

        # Step 2: 3 专家并行审视（MVP 串行，后续可改并行）
        opinions = [
            self._call_expert("emotion", window_text),
            self._call_expert("needs", window_text),
            self._call_expert("development", window_text),
        ]

        # Step 3: Speaker 裁决
        senate_decision = self._speaker_adjudicate(opinions)
        tone = senate_decision.final_tone

        # 构建 MasterDecision（兼容 Production 接口）
        decision = MasterDecision(
            direction=tone,
            should_popup=True,
            main_contradiction=senate_decision.reasoning,
            story_arc="",
            tone_reasoning=senate_decision.reasoning,
            route_a_insight=f"[Senate] {opinions[0].evidence}" if opinions else "",
            route_b_insight=f"[Senate] {opinions[1].evidence}" if len(opinions) > 1 else "",
        )

        # Step 4: 生产弹窗
        draft = self._production.produce(
            decision, perception, window_text,
            story_arc="", previous_popup_text="",
        )

        self._total_popups += 1

        return MultiAgentResult(
            popup_text=draft.popup_insight,
            should_popup=True,
            tone=tone,
            main_contradiction=senate_decision.reasoning,
            story_arc="",
            route_a_insight=f"[Senate vote: {senate_decision.votes}]",
            route_b_insight=f"[Resolution: {senate_decision.resolution}]",
            perception=perception,
            decision=decision,
            draft=draft,
        )
