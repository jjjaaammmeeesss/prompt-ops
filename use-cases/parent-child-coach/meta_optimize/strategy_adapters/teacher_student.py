"""Teacher-Student 反馈环适配器：Teacher→Student(Production)→Validator→[feedback/重试]。

LLM 调用: Perception + Master + Production + Validator = 4 calls (重试时 +2)
预期 M5: 70-80% | 延迟 +0.5s
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


# ── Validator 数据结构 ──────────────────────────────────

@dataclass
class ValidationResult:
    is_consistent: bool      # tone 与弹窗内容一致？
    actual_tone_in_draft: str  # 弹窗实际体现的 tone
    mismatch_evidence: str = ""  # 不一致时，证据引用
    suggestion: str = ""        # 修正建议


# ── Default prompts ─────────────────────────────────────

DEFAULT_VALIDATOR_PROMPT = (
    "你是弹窗 tone 一致性验证器。你的任务是检查「弹窗草稿的内容」是否与「指定的 tone 方向」一致。\n\n"
    "评判标准：\n"
    "- diagnostic（诊断式）弹窗：帮助家长看到自己的盲区或误解，语气温和但不回避问题\n"
    "- empowering（鼓励式）弹窗：具体肯定家长的做法，不空洞、不敷衍\n\n"
    "你只需做「一致性判断」——比较弹窗内容和 tone 标签，而不需要从原文重新判断应该用什么 tone。\n"
    "这比你直接判断原文的 tone 更简单——你只需比较，不需创作。\n\n"
    "输出JSON: {\n"
    '  "is_consistent": true/false,\n'
    '  "actual_tone_in_draft": "diagnostic|empowering|mixed",\n'
    '  "mismatch_evidence": "如果不一致，引用弹窗中的具体语句",\n'
    '  "suggestion": "修正建议（如有）"\n'
    "}"
)

MAX_RETRIES = 2


class TeacherStudentAdapter:
    """Teacher-Student 反馈环适配器。

    流程: Perception → Master(Teacher) → Production(Student) → Validator
          如果不一致 → feedback → Teacher 重试（最多2次）
    """

    def __init__(self, llm_client, model: str = "deepseek-v4-pro",
                 harness_dir: str = "", prompt_base_dir: str = ""):
        self._client = llm_client
        self._model = model
        self._harness_dir = Path(harness_dir) if harness_dir else None
        self._prompt_base = Path(prompt_base_dir) if prompt_base_dir else (
            XINGLING_ROOT / "prompts"
        )

        # 加载 Validator prompt
        self._validator_prompt = self._load_validator_prompt()

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
        self._total_windows = 0
        self._total_popups = 0

    def _load_validator_prompt(self) -> str:
        if self._harness_dir:
            md_path = self._harness_dir / "harness.md"
            if md_path.exists():
                text = md_path.read_text(encoding="utf-8")
                for section in text.split("\n## "):
                    if "Validator" in section or "验证器" in section:
                        return section
        return DEFAULT_VALIDATOR_PROMPT

    def _validate(self, window_text: str, tone: str, popup_text: str) -> ValidationResult:
        """Validator: 判断弹窗内容与 tone 标签是否一致。"""
        user_prompt = (
            f"## 原文\n{window_text[:800]}\n\n"
            f"## 指定 tone\n{tone}\n\n"
            f"## 弹窗草稿\n{popup_text[:600]}"
        )
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self._validator_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2, timeout=60,
            )
            raw = resp.choices[0].message.content
            data = json.loads(raw) if raw else {}
        except Exception:
            data = {}

        return ValidationResult(
            is_consistent=data.get("is_consistent", True),
            actual_tone_in_draft=data.get("actual_tone_in_draft", tone),
            mismatch_evidence=data.get("mismatch_evidence", ""),
            suggestion=data.get("suggestion", ""),
        )

    def process_window(self, window_text: str, family: str = "") -> MultiAgentResult:
        """Teacher-Student 流程：Perception→Teacher→Student→Validator→[feedback]。"""
        if not window_text or len(window_text.strip()) < 20:
            return MultiAgentResult(should_popup=False)

        self._total_windows += 1
        memory = self._get_memory(family)

        # Step 1: Perception（保持不变）
        perception = self._perception.perceive(window_text)

        # Step 2: Master = Teacher（tone 初判）
        decision = self._master.decide(perception, memory, window_text)

        if not decision.should_popup:
            return MultiAgentResult(
                should_popup=False,
                perception=perception,
                decision=decision,
            )

        # Step 3-5: Production → Validate → [Retry]
        retry_context = ""
        for attempt in range(1 + MAX_RETRIES):  # 1 first + 2 retries
            # 如果是重试，给 Teacher 加上 Validator 反馈
            if attempt > 0 and retry_context:
                # 用 Validator 反馈补充 perception 上下文 → 重新决策
                decision.direction = self._reverse_tone(decision.direction) if attempt == 1 else decision.direction
                # 简单翻转：第一次重试翻转 tone，第二次重试用 Validator 的建议

            # Production (Student)
            prev_text = ""
            if memory.last_popup:
                p = memory.last_popup
                prev_text = p.popup_insight

            draft = self._production.produce(
                decision, perception, window_text,
                decision.story_arc, prev_text,
            )
            popup_text = draft.popup_insight

            # Validator
            validation = self._validate(window_text, decision.direction, popup_text)

            if validation.is_consistent:
                # 通过：输出弹窗
                break
            elif attempt < MAX_RETRIES:
                # 不通过但有重试配额 → 收集 Validator 反馈用于下次
                retry_context = (
                    f"[Validator反馈] tone={decision.direction} 与弹窗内容不一致。"
                    f"弹窗实际体现: {validation.actual_tone_in_draft}。"
                    f"证据: {validation.mismatch_evidence}。"
                    f"建议: {validation.suggestion}"
                )
                decision.tone_reasoning = retry_context
            # else: 最后一次重试仍失败 → 输出，记录分歧

        self._total_popups += 1

        return MultiAgentResult(
            popup_text=draft.popup_insight,
            should_popup=True,
            tone=decision.direction,
            main_contradiction=decision.main_contradiction,
            story_arc=decision.story_arc,
            route_a_insight=decision.route_a_insight,
            route_b_insight=f"[TS feedback] attempt={attempt + 1}, "
                           f"validator_consistent={validation.is_consistent}",
            perception=perception,
            decision=decision,
            draft=draft,
        )

    def _get_memory(self, family: str) -> CaseMemory:
        if family not in self._memories:
            self._memories[family] = CaseMemory()
        return self._memories[family]

    @staticmethod
    def _reverse_tone(tone: str) -> str:
        if tone == "diagnostic":
            return "empowering"
        elif tone == "empowering":
            return "diagnostic"
        return tone
