"""Proposer: LLM 读 trace → 诊断失败源 → 生成 harness 修改提案。

设计（来自 Bilevel Autoresearch）：
  - 先分类失败源（prompt/judge/dataset/search/model），再决定是否可修
  - 四轮协议适配：每轮的 Explore→Critique 结果注入 proposer context
  - Karpathy 约束：单一可变面，evidence-driven
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from candidate_store import (
    TraceEntry,
    MetricsSnapshot,
    load_trace,
    load_metrics,
    load_proposal,
    get_m5_failures,
)


# ═══════════════════════════════════════════════════════════════
# 失败源分类
# ═══════════════════════════════════════════════════════════════

FAILURE_SOURCE_HINTS = {
    "prompt": (
        "prompt 措辞/结构导致 tone 误判。症状：system prompt 中对 'diagnostic vs empowering' 的判定标准模糊、"
        "边界 case 覆盖不足、或者 prompt 指令自相矛盾。可修——proposer 可以改 prompt 文本。"
    ),
    "judge": (
        "M5 gold label 本身可能有问题。症状：系统输出的 tone 在语义上合理，但 gold label 标记为不匹配。"
        "例如窗口文本本身很模糊、专家标注可能有分歧。不可修——标记为 'judge noise'，不计入优化目标。"
    ),
    "dataset": (
        "该 case 的窗口文本本身模糊，任何模型都难以判断 tone。症状：窗口文本截断了关键上下文、"
        "对话处于 tone 切换的临界点、或 case 本身存在标注歧义。不可修——标记为 'hard case'，降低权重。"
    ),
    "search": (
        "proposer 陷入局部最优。症状：连续多次修改方向相同但无改善，或反复在 2-3 个版本间振荡。"
        "可修——切换可变参数、加大变异幅度、或从不同角度切入。"
    ),
    "model": (
        "DeepSeek 对该类语义存在稳定的错误边界。症状：同一 case 在不同 prompt 版本下都稳定输出错误 tone。"
        "这是模型能力天花板——prompt 修不了，需要架构级改动（如引入 multi-agent）。不可修在 L1 层面。"
    ),
}


def classify_failure_source(entry: TraceEntry, candidate_history: list[dict]) -> str:
    """对单个 M5 失败 case 做五分类。

    启发式规则（快速路径，不调 LLM）：
      1. 如果该 case 在历史中 >=3 次被标记为 'model' → 直接返回 'model'
      2. 如果该 case 在历史中 >=2 次被标记为 'judge' → 直接返回 'judge'
      3. 如果历史中从未有过 M5=1.0 → 倾向 'dataset' 或 'model'
      4. 如果该 case 有时对有时错 → 倾向 'prompt' 或 'search'
      5. 默认返回 'prompt'（可修）

    Args:
        entry: M5 失败的 trace entry
        candidate_history: 历史候选的 metrics/trace 摘要列表
    """
    case_id = entry.case_id

    # 统计历史表现
    ever_correct = False
    history_fail_sources = []
    for h in candidate_history:
        for tc in h.get("trace_summary", []):
            if tc["case_id"] == case_id:
                if tc.get("m5_tone_match") == 1.0:
                    ever_correct = True
                if tc.get("failure_source"):
                    history_fail_sources.append(tc["failure_source"])

    # Rule 1: 历史稳定错误 → model
    model_count = history_fail_sources.count("model")
    if model_count >= 3:
        return "model"

    # Rule 2: 历史标记为 judge noise → judge
    judge_count = history_fail_sources.count("judge")
    if judge_count >= 2:
        return "judge"

    # Rule 3: 从未正确过 → 可能是 dataset 或 model
    if not ever_correct and len(history_fail_sources) >= 2:
        # 如果所有历史都归因为 model → model
        if all(s == "model" for s in history_fail_sources):
            return "model"
        # 否则 → dataset（可能是文本本身模糊）
        return "dataset"

    # Rule 4: 有时对有时错 → prompt 可修
    if ever_correct:
        return "prompt"

    # Rule 5: 只有一次失败记录，默认 prompt
    return "prompt"


# ═══════════════════════════════════════════════════════════════
# Proposal 数据结构
# ═══════════════════════════════════════════════════════════════

@dataclass
class EditProposal:
    """单条修改提案。"""
    target_file: str           # "harness.md" | "harness.py"
    before: str                # 原文本（用于精确匹配替换）
    after: str                 # 新文本
    reason: str                # 修改原因，引用 trace 证据
    affected_cases: list[str] = field(default_factory=list)  # 预期影响的 case_id


@dataclass
class MutationProposal:
    """一轮变异提案（1-3 条 edit）。"""
    strategy: str
    candidate_id: str
    failure_analysis: dict[str, list[str]]  # failure_source → case_id 列表
    edits: list[EditProposal] = field(default_factory=list)
    rationale: str = ""         # 整体策略思路
    risks: str = ""             # 风险评估

    def to_markdown(self) -> str:
        """生成 proposal.md 内容。"""
        lines = [
            f"# Mutation Proposal · {self.strategy} · {self.candidate_id}",
            "",
            "## 失败源分析",
            "",
        ]
        for source, cases in self.failure_analysis.items():
            lines.append(f"### {source} ({len(cases)} cases)")
            lines.append(f"症状：{FAILURE_SOURCE_HINTS.get(source, '')}")
            lines.append(f"涉及：{', '.join(cases[:8])}")
            lines.append("")

        lines.append("## 整体策略")
        lines.append(self.rationale)
        lines.append("")

        lines.append("## 修改列表")
        for i, edit in enumerate(self.edits, 1):
            lines.append(f"### Edit {i}: {edit.target_file}")
            lines.append(f"**原因**：{edit.reason}")
            lines.append(f"**影响 case**：{', '.join(edit.affected_cases[:5])}")
            lines.append("")
            lines.append("**Before**:")
            lines.append("```")
            lines.append(edit.before[:500])
            lines.append("```")
            lines.append("")
            lines.append("**After**:")
            lines.append("```")
            lines.append(edit.after[:500])
            lines.append("```")
            lines.append("")

        lines.append("## 风险")
        lines.append(self.risks)

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# LLM Proposer
# ═══════════════════════════════════════════════════════════════

PROPOSER_SYSTEM_PROMPT = """你是亲子沟通弹窗系统的 harness 优化器。你的任务是根据评估 trace 诊断 tone 匹配失败的原因，并提出精确的 prompt/代码修改。

## 核心原则（Karpathy 约束）

1. **单一可变面**：每轮只改一个文件（harness.md 或 harness.py 中的一个）
2. **证据驱动**：每条修改必须有 trace 证据引用（具体 case_id + 失败症状）
3. **最简修改**：改最小范围，不动无关内容
4. **M5 优先**：tone matching 是最核心的优化目标

## 失败源五分类

修改前，先判断每个 M5 失败 case 属于哪类：
- **prompt**: prompt 措辞问题 → 可修
- **judge**: gold label 可能有问题 → 标记，不计入优化
- **dataset**: 窗口文本本身模糊 → 标记为 hard case，降低权重
- **search**: 局部最优 → 换角度/大变异
- **model**: DeepSeek 天花板 → prompt 修不了，需架构改动

## 输出格式

严格 JSON：
```json
{
  "failure_analysis": {
    "prompt": ["C10-002", ...],
    "judge": [...],
    "dataset": [...],
    "search": [],
    "model": ["C11-006", "C11-009"]
  },
  "edits": [
    {
      "target_file": "harness.md",
      "before": "原始文本片段（精确匹配）",
      "after": "修改后文本片段",
      "reason": "修改原因，引用具体 case_id 和失败模式",
      "affected_cases": ["C10-002"]
    }
  ],
  "rationale": "整体策略思路",
  "risks": "可能的风险"
}
```

注意：
- `before` 必须是 harness.md/harness.py 中的精确原文片段，方便程序做字符串替换
- 如果某个失败源没有对应 case，使用空数组 []
- edits 数量 1-3 条
"""


def build_proposer_prompt(
    strategy: str,
    trace_entries: list[TraceEntry],
    current_harness_md: str,
    current_harness_py: str,
    metrics: MetricsSnapshot,
    history_summary: str = "",
    strategy_context: str = "",
) -> str:
    """构建 proposer 的 user prompt。

    Args:
        strategy: 策略名
        trace_entries: 当前候选的 trace
        current_harness_md: 当前 harness.md 内容
        current_harness_py: 当前 harness.py 内容
        metrics: 当前指标
        history_summary: 历史候选摘要
        strategy_context: 策略特定上下文（从 config 注入）
    """
    m5_failures = [e for e in trace_entries if e.m5_tone_match == 0.0]
    m5_successes = [e for e in trace_entries if e.m5_tone_match == 1.0]

    prompt = f"""## 当前策略: {strategy}

{strategy_context}

## 当前指标

- Overall: {metrics.overall_score:.3f}
- M5 tone match: {metrics.aggregate_m5:.1%}
- M6 insight: {metrics.aggregate_m6:.2f}/5
- M7 safety: {metrics.aggregate_m7:.2f}/5
- M5 失败 case 数: {len(m5_failures)} / {metrics.n_cases}

## M5 失败 case

"""
    for e in m5_failures[:20]:  # 最多显示 20 个失败
        prompt += (f"- {e.case_id} (w{e.window_index}): "
                   f"sys={e.sys_tone}, gold={e.gold_tone}, "
                   f"M6={e.m6_insight_score}, M7={e.m7_safety_score}\n")
        if e.sys_popup_text:
            # 截取弹窗文本前 150 字
            popup_snippet = e.sys_popup_text[:150].replace("\n", " ")
            prompt += f"  popup: {popup_snippet}...\n"

    if m5_successes:
        prompt += f"\n## M5 正确 case (对比参考)\n"
        for e in m5_successes[:5]:
            prompt += f"- {e.case_id}: sys={e.sys_tone}, gold={e.gold_tone}\n"

    if history_summary:
        prompt += f"\n## 历史候选摘要\n{history_summary}\n"

    prompt += f"""
## 当前 harness.md

```markdown
{current_harness_md[:5000]}
```

## 当前 harness.py

```python
{current_harness_py[:3000]}
```

请分析 M5 失败的原因，按五分类归类，并提出 1-3 条精确修改。
"""
    return prompt


def parse_proposer_response(raw: str) -> MutationProposal | None:
    """解析 proposer LLM 返回的 JSON。"""
    try:
        # 清理可能的 markdown 包裹
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1])
        data = json.loads(text)
        return MutationProposal(
            strategy="",
            candidate_id="",
            failure_analysis=data.get("failure_analysis", {}),
            edits=[EditProposal(**e) for e in data.get("edits", [])],
            rationale=data.get("rationale", ""),
            risks=data.get("risks", ""),
        )
    except (json.JSONDecodeError, TypeError) as e:
        print(f"[proposer] JSON 解析失败: {e}")
        print(f"[proposer] 原始返回:\n{raw[:500]}")
        return None


# ═══════════════════════════════════════════════════════════════
# 启发式 Proposer（不调 LLM，用于快速验证循环逻辑）
# ═══════════════════════════════════════════════════════════════

def heuristic_classify_failures(trace_entries: list[TraceEntry]) -> dict[str, list[str]]:
    """纯规则版失败源分类。按优先级：prompt > model > dataset > judge。"""
    classified: dict[str, list[str]] = {
        "prompt": [], "judge": [], "dataset": [], "search": [], "model": [],
    }
    for e in trace_entries:
        if e.m5_tone_match == 1.0:
            continue
        # 简单启发式：sys=empowering gold=diagnostic → fake empowering (prompt)
        if e.sys_tone == "empowering" and e.gold_tone == "diagnostic":
            classified["prompt"].append(e.case_id)
        # sys=diagnostic gold=empowering → fake diagnostic (prompt)
        elif e.sys_tone == "diagnostic" and e.gold_tone == "empowering":
            classified["prompt"].append(e.case_id)
        # 系统输出空 → model 天花板
        elif not e.sys_tone:
            classified["model"].append(e.case_id)
        # 其他 → dataset 模糊
        else:
            classified["dataset"].append(e.case_id)
    return classified


def heuristic_propose(
    strategy: str,
    candidate_id: str,
    trace_entries: list[TraceEntry],
    current_harness_md: str,
    metrics: MetricsSnapshot,
) -> MutationProposal:
    """启发式 proposer：不调 LLM，基于规则生成简单变异。

    诊断模式（按优先级）：
    1. 系统性 bias：全部/绝大多数失败都是同一方向 → 调整 fallback 方向
    2. 缺少区分标准：prompt 中没有明确的 diagnostic/empowering 边界
    3. Speaker 偏向：Senate 策略特有，安全 fallback 过于保守
    """
    m5_failures = get_m5_failures_for_entries(trace_entries)
    classified = heuristic_classify_failures(trace_entries)

    prop = MutationProposal(
        strategy=strategy,
        candidate_id=candidate_id,
        failure_analysis=classified,
        rationale="",
        risks="启发式变异，未调 LLM。仅用于循环逻辑验证。",
    )

    prompt_cases = classified["prompt"]
    if not prompt_cases:
        prop.rationale = "无明确 prompt 可修项。检查是否需要架构级改动（Level 2）。"
        return prop

    # ── 统计分析失败方向 ──────────────────────────
    diag_as_emp = [e for e in m5_failures
                   if e.sys_tone == "diagnostic" and e.gold_tone == "empowering"]
    emp_as_diag = [e for e in m5_failures
                   if e.sys_tone == "empowering" and e.gold_tone == "diagnostic"]
    n_diag_bias = len(diag_as_emp)
    n_emp_bias = len(emp_as_diag)

    # ── 规则 1: 系统性 bias 检测（≥70% 失败偏向同一方向）──
    total = n_diag_bias + n_emp_bias
    if total > 0:
        diag_ratio = n_diag_bias / total if total > 0 else 0
        if diag_ratio >= 0.70 and n_diag_bias >= 3:
            # Senate: 修改 Speaker 安全 fallback 方向
            if strategy == "senate" and "默认 diagnostic" in current_harness_md:
                prop.edits.append(EditProposal(
                    target_file="harness.md",
                    before="4. 所有专家置信度 < 0.5 → 默认 diagnostic（安全 fallback）",
                    after="4. 所有专家置信度 < 0.5 → 标记为 uncertain，交由 Master 基于上下文判断",
                    reason=f"Senate Speaker 安全 fallback 偏向 diagnostic，导致 {n_diag_bias}/{total} "
                           f"({diag_ratio:.0%}) 的 M5 失败都是 diagnostic→empowering 误判。"
                           f"改为交由 Master 上下文判断，而非硬编码 fallback。",
                    affected_cases=[e.case_id for e in diag_as_emp[:3]],
                ))
                prop.rationale = (
                    f"Senate 系统性 diagnostic bias: {n_diag_bias} 个 case 将 empowering 误判为 diagnostic。"
                    f"根因: Speaker 规则 4 的 '默认 diagnostic' 安全 fallback 过于保守。"
                    f"修改: 移除硬编码 fallback，改由 Master 基于上下文判断。"
                )
            else:
                # 通用：加 bias 提醒
                biased_dir = "diagnostic" if diag_ratio >= 0.70 else "empowering"
                opposite = "empowering" if biased_dir == "diagnostic" else "diagnostic"
                if f"不要过度使用 {biased_dir}" not in current_harness_md:
                    prop.edits.append(EditProposal(
                        target_file="harness.md",
                        before="## Speaker（议长裁决规则）" if strategy == "senate" else "输出JSON:",
                        after=(
                            f"## ⚠️ 系统性 bias 警告\n"
                            f"当前版本过度偏向 {biased_dir}（{diag_ratio:.0%} 的 M5 失败都是 "
                            f"将 {opposite} 误判为 {biased_dir}）。\n"
                            f"请在所有裁决中优先考虑：是否有证据支持 {opposite} 方向？\n\n"
                            f"## Speaker（议长裁决规则）" if strategy == "senate" else
                            f"⚠️ 注意：不要过度使用 {biased_dir}。优先检查是否有 {opposite} 的证据。\n\n"
                            f"输出JSON:"
                        ),
                        reason=f"系统性 {biased_dir} bias: {diag_ratio:.0%} M5 失败都是 "
                               f"{opposite}→{biased_dir} 误判",
                        affected_cases=[e.case_id for e in
                                       (diag_as_emp if biased_dir == "diagnostic" else emp_as_diag)[:3]],
                    ))
                    prop.rationale = (
                        f"系统性 {biased_dir} bias ({diag_ratio:.0%})。在 prompt 中加入反 bias 提醒。"
                    )

        elif n_emp_bias >= 2 and n_emp_bias / total >= 0.70:
            # 反向 bias（少见）
            pass

    # ── 规则 2: 缺少区分标准 ─────────────────────
    if not prop.edits and "核心区分标准" not in current_harness_md:
        prop.edits.append(EditProposal(
            target_file="harness.md",
            before="## 口吻方向规则",
            after="## 口吻方向规则\n\n"
                   "### 核心区分标准（区分 diagnostic vs empowering）\n"
                   "- diagnostic = 家长存在盲区或误解，弹窗指出盲区、帮助看清真相（不审判）\n"
                   "- empowering = 家长做对了但可能不自知，弹窗肯定具体做法、强化内在动机\n"
                   "- 当窗口文本同时包含 '家长努力' 和 '家长误解' 时，优先取主导方向（占文本 60%+ 的内容方向）",
            reason=f"当前 prompt 缺少明确的 diagnostic/empowering 区分标准。"
                   f"M5 失败的 prompt 类 cases: {prompt_cases[:5]}",
            affected_cases=prompt_cases[:3],
        ))
        prop.rationale = f"基于 {len(prompt_cases)} 个 prompt 类失败 case，添加 tone 区分标准。"

    if not prop.edits:
        prop.rationale = "无明确 prompt 可修项。检查是否需要架构级改动（Level 2）。"

    return prop


def get_m5_failures_for_entries(entries: list[TraceEntry]) -> list[TraceEntry]:
    return [e for e in entries if e.m5_tone_match == 0.0]
