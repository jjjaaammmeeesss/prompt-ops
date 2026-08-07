"""Teacher-Student 反馈环适配器 v2：Teacher→Student→Validator→[feedback→Teacher重新决策]。

与 v1 的区别：v1 在重试时只是盲翻 tone（diagnostic↔empowering），
Validator 的反馈只传给 Production 而非 Master。v2 实现真正的反馈环：
Validator 的判定 + 证据 → 回到 Master → Master 重新审视 tone 决策 → 重新 Production。

LLM 调用: Perception + Master + Production + Validator = 4 calls (重试时 +2 per retry)
预期 M5: 70-80% | 延迟 +0.5-1.0s
"""

import json
import sys
from dataclasses import dataclass, field
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


# ── 跨窗口学习状态 ──────────────────────────────────────

@dataclass
class FeedbackMemory:
    """记录 Teacher 在哪些类型的 case 上容易判错，用于后续窗口的决策校准。"""
    recent_failures: list[dict] = field(default_factory=list)  # 最近 5 次失败
    tone_bias_count: dict = field(default_factory=lambda: {"diagnostic": 0, "empowering": 0})
    # 被 Validator 纠正过的次数统计

    def record_failure(self, teacher_tone: str, validator_tone: str, evidence: str):
        self.recent_failures.append({
            "teacher_tone": teacher_tone,
            "validator_tone": validator_tone,
            "evidence": evidence[:200],
        })
        if len(self.recent_failures) > 5:
            self.recent_failures.pop(0)
        if teacher_tone in self.tone_bias_count:
            self.tone_bias_count[teacher_tone] += 1

    @property
    def bias_summary(self) -> str:
        """生成当前偏斜总结，供 Master 反思时参考。"""
        if not self.recent_failures:
            return ""
        d_count = self.tone_bias_count.get("diagnostic", 0)
        e_count = self.tone_bias_count.get("empowering", 0)
        summary = f"历史纠正记录: diagnostic判错{d_count}次, empowering判错{e_count}次。\n"
        if self.recent_failures:
            last = self.recent_failures[-1]
            summary += f"最近一次: Teacher判{last['teacher_tone']} → Validator认为是{last['validator_tone']}。"
        return summary


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

# Teacher 反思 prompt：收到 Validator 反馈后，要求 Master 重新审视
TEACHER_RETHINK_PREFIX = (
    "[反馈环] 你上一轮判定的 tone 被验证器驳回了。"
    "请仔细阅读验证器的反馈，重新审视这段对话，"
    "然后给出新的 tone 判定。不要重复上一轮的错误。\n\n"
)

MAX_RETRIES = 2


class TeacherStudentAdapter:
    """Teacher-Student 反馈环适配器 v2。

    流程: Perception → Master(Teacher) → Production(Student) → Validator
          如果不一致 → Validator 反馈回到 Master → Master 重新决策 → 重新 Production
          （v2: Master 真正参与反馈环，而非盲翻 tone）
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
        self._feedback_memories: dict[str, FeedbackMemory] = {}  # 跨窗口学习
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

    def _master_rethink(
        self,
        perception: PerceptionReport,
        memory: CaseMemory,
        window_text: str,
        original_tone: str,
        validator_feedback: str,
        feedback_memory: FeedbackMemory,
    ) -> MasterDecision:
        """v2 核心：Teacher 收到 Validator 反馈后重新审视 tone 决策。

        与 v1 盲翻 tone 不同，这里真正调用 Master，将 Validator 的证据
        和跨窗口学习记忆作为上下文注入，让 Master 自己重新判断。
        """
        # 构建反思上下文
        rethink_context = (
            f"{TEACHER_RETHINK_PREFIX}"
            f"## 你上一轮的 tone 判定\n{original_tone}\n\n"
            f"## 验证器反馈\n{validator_feedback}\n\n"
        )
        # 附加跨窗口学习记忆
        bias = feedback_memory.bias_summary
        if bias:
            rethink_context += f"## 历史纠正记录（跨窗口）\n{bias}\n\n"

        rethink_context += f"## 原文（重新审视）\n{window_text}"

        # 让 Master 重新决策（将反思上下文作为 window_text 传入，
        # 这样 Master 的 prompt 模板中的 {window_text} 包含完整反馈）
        decision = self._master.decide(perception, memory, rethink_context)

        # 记录：这是经过反馈环修正的决策
        decision.tone_reasoning = (
            f"[反馈环 v2] 原判={original_tone}, "
            f"Validator反馈={validator_feedback[:150]}, "
            f"反思后重判={decision.direction}"
        )
        return decision

    def process_window(self, window_text: str, family: str = "") -> MultiAgentResult:
        """Teacher-Student v2 流程：Perception→Teacher→Student→Validator→[反馈回 Teacher]。

        v2 的关键改变：当 Validator 判定不一致时，
        - 不再盲翻 tone
        - 而是将 Validator 的证据 + 跨窗口学习记忆 → 送回 Teacher
        - Teacher 重新审视 → 重新决策 tone → 重新 Production
        """
        if not window_text or len(window_text.strip()) < 20:
            return MultiAgentResult(should_popup=False)

        self._total_windows += 1
        memory = self._get_memory(family)
        fb_memory = self._get_feedback_memory(family)

        # Step 1: Perception（保持不变）
        perception = self._perception.perceive(window_text)

        # Step 2: Master = Teacher（tone 初判）
        decision = self._master.decide(perception, memory, window_text)
        original_tone = decision.direction  # 记录初判，用于审计

        if not decision.should_popup:
            return MultiAgentResult(
                should_popup=False,
                perception=perception,
                decision=decision,
            )

        # Step 3-5: Production → Validate → [反馈回 Teacher]
        retry_context = ""
        popup_text = ""
        validation = None
        all_attempts: list[dict] = []  # 审计记录

        for attempt in range(1 + MAX_RETRIES):  # 1 first + 2 retries
            # ── v2 核心：Teacher 收到反馈后重新决策 ──
            if attempt > 0 and retry_context:
                # 真正的反馈环：Validator → Teacher
                decision = self._master_rethink(
                    perception, memory, window_text,
                    original_tone=decision.direction,  # 上一轮的 tone
                    validator_feedback=retry_context,
                    feedback_memory=fb_memory,
                )

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

            all_attempts.append({
                "attempt": attempt + 1,
                "tone": decision.direction,
                "validator_consistent": validation.is_consistent,
                "validator_actual_tone": validation.actual_tone_in_draft,
            })

            if validation.is_consistent:
                # 通过：输出弹窗
                # 如果经过反馈环修正才通过，记录跨窗口学习
                if attempt > 0:
                    fb_memory.record_failure(
                        teacher_tone=original_tone,
                        validator_tone=validation.actual_tone_in_draft,
                        evidence=validation.mismatch_evidence,
                    )
                break
            elif attempt < MAX_RETRIES:
                # 不通过但有重试配额 → 构建反馈上下文给 Teacher
                retry_context = (
                    f"[Validator反馈] tone={decision.direction} 与弹窗内容不一致。\n"
                    f"弹窗实际体现: {validation.actual_tone_in_draft}。\n"
                    f"证据: {validation.mismatch_evidence}。\n"
                    f"建议: {validation.suggestion}\n"
                    f"请重新审视原文，判断正确的 tone 应该是什么。"
                )
            # else: 最后一次重试仍失败 → 输出，但记录分歧 + 学习

        # 最后一次仍失败 → 记录到跨窗口学习记忆
        if not validation.is_consistent:
            fb_memory.record_failure(
                teacher_tone=decision.direction,
                validator_tone=validation.actual_tone_in_draft,
                evidence=validation.mismatch_evidence,
            )

        self._total_popups += 1

        # 构建审计信息
        audit_lines = [f"[TS v2] 初判={original_tone}, 终判={decision.direction}, "
                       f"attempts={len(all_attempts)}"]
        for a in all_attempts:
            audit_lines.append(
                f"  attempt={a['attempt']}: tone={a['tone']}, "
                f"consistent={a['validator_consistent']}, "
                f"actual={a['validator_actual_tone']}"
            )
        if fb_memory.bias_summary:
            audit_lines.append(f"  跨窗口学习: {fb_memory.bias_summary[:200]}")

        return MultiAgentResult(
            popup_text=popup_text,
            should_popup=True,
            tone=decision.direction,
            main_contradiction=decision.main_contradiction,
            story_arc=decision.story_arc,
            route_a_insight=decision.route_a_insight,
            route_b_insight=" | ".join(audit_lines),
            perception=perception,
            decision=decision,
            draft=draft,
        )

    def _get_memory(self, family: str) -> CaseMemory:
        if family not in self._memories:
            self._memories[family] = CaseMemory()
        return self._memories[family]

    def _get_feedback_memory(self, family: str) -> FeedbackMemory:
        if family not in self._feedback_memories:
            self._feedback_memories[family] = FeedbackMemory()
        return self._feedback_memories[family]
