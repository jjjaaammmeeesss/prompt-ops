---
title: "LLM-as-Judge Metric Replaces Default Exact-Match Metric for MIPROv2 Prompt Optimization"
date: 2026-07-08
category: workflow-issues
module: prompt-ops
problem_type: workflow_issue
component: tooling
severity: medium
applies_when:
  - "Using DSPy MIPROv2 or similar Bayesian optimizers to optimize prompts for subjective or qualitative tasks where output quality cannot be measured via exact string matching"
  - "The default metric (e.g., selected_fields_comparison) rewards format adherence rather than task quality, causing the optimizer to converge on worse prompts"
  - "Prompt optimization produces results that score worse than the hand-written baseline despite the optimizer reporting improvement"
  - "The task involves nuanced evaluation dimensions (tone, insight quality, actionability) that require LLM judgment rather than programmatic comparison"
tags:
  - dspy
  - miprov2
  - llm-as-judge
  - metric-design
  - prompt-optimization
  - coaching
  - workflow-issue
  - prompt-ops
---

# LLM-as-Judge Metric Replaces Default Exact-Match Metric for MIPROv2 Prompt Optimization

## Context

When optimizing a prompt with Meta's prompt-ops framework (MIPROv2), the default `StandardJSONMetric` with `selected_fields_comparison` performs exact string matching on structured output fields -- `should_popup`, `tone`, `popup_insight` -- and rewards prompts that reproduce the surface form of a reference output. This is the wrong objective for coaching-quality tasks, where the space of good responses is large, the surface form is incidental, and what matters is whether the advice is actually helpful to a parent.

The project had a parent-child coaching prompt at version 1.7. Previous MIPROv2 optimization runs failed: the optimized prompts scored worse than v1.7 under manual quality review, even though MIPROv2 reported improvement. The root cause was metric-target misalignment. The exact-string-matching metric was teaching the optimizer to chase JSON formatting patterns rather than produce better coaching advice. Short of paying domain experts to label every candidate output -- which is slow, expensive, and does not scale across iterations -- there was no obvious way to give MIPROv2 a quality signal it could optimize against.

The gap: MIPROv2 needs a scalar score per candidate to guide its search, and the only scoring function available was a format checker wearing a metric costume.

## Guidance

Replace the format-matching metric with a custom LLM-as-Judge metric that evaluates the output on the dimensions that actually define quality for the task. In DSPy terms, this means writing a class that implements `__call__(gold, pred, trace)` and returns a `float` in `[0, 1]`. For prompt-ops compatibility, the class must inherit from `MetricBase`.

### 1. Define a scoring framework with weighted dimensions

Before writing any code, define what "good" means for this task. For the parent-child coaching case, the rubric had seven dimensions with explicit weights:

| Dimension | Weight | What it measures |
|---|---|---|
| acknowledgment | 0.20 | Sees the parent's good intentions before offering critique |
| insight_accuracy | 0.20 | Identifies the real pain point in the dialogue |
| pattern_revelation | 0.10 | Connects dots across the conversation to reveal a pattern |
| invitational_tone | 0.10 | Uses "maybe" / "could it be" rather than declarative judgments |
| actionability | 0.15 | Concrete suggestions; N/A for diagnostic-only popups |
| naturalness | 0.15 | Conversational, no preaching or textbook tone |
| focus | 0.10 | One core issue, deep rather than broad |

The weights encode domain knowledge about what makes coaching effective. Acknowledgment and insight accuracy together account for 40% because building trust and diagnosing correctly are prerequisites to everything else.

A critical design choice: actionability supports N/A. Some coaching popups are purely diagnostic -- they identify a problem without prescribing a fix. When the output is diagnostic-only, actionability is marked N/A and its weight is redistributed proportionally across the remaining dimensions, so the judge is not penalizing a response for correctly choosing not to give advice.

### 2. Implement the metric class

```python
import json
import os
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI
from dspy import MetricBase

# System prompt encoding the 7-dimension framework
LLM_JUDGE_SYSTEM_PROMPT_V4 = """
You are an expert evaluator of parent-child coaching popups.
Score the popup on these 7 dimensions:

1. acknowledgment (weight 0.20): Does the popup explicitly acknowledge the parent's good intentions or effort before offering critique?
2. insight_accuracy (weight 0.20): Does the popup correctly identify the core coaching issue in this dialogue?
3. pattern_revelation (weight 0.10): Does the popup connect the parent's specific words to a broader coaching pattern?
4. invitational_tone (weight 0.10): Does the popup use invitational language (maybe, could it be, sometimes) rather than declarative/accusatory language?
5. actionability (weight 0.15): If the popup gives advice, are the suggestions concrete and actionable? If diagnostic-only, mark N/A.
6. naturalness (weight 0.15): Is the language natural and conversational, not preachy or textbook-like?
7. focus (weight 0.10): Does the popup focus on one core issue deeply rather than listing many issues shallowly?

Output ONLY a valid JSON object:
{
  "acknowledgment": <0.0-1.0 or "N/A">,
  "insight_accuracy": <0.0-1.0 or "N/A">,
  "pattern_revelation": <0.0-1.0 or "N/A">,
  "invitational_tone": <0.0-1.0 or "N/A">,
  "actionability": <0.0-1.0 or "N/A">,
  "naturalness": <0.0-1.0 or "N/A">,
  "focus": <0.0-1.0 or "N/A">
}

Weight redistribution for N/A dimensions:
- If actionability is N/A, redistribute its 0.15 proportionally to the other 6 dimensions.
- Do NOT redistribute weight to other N/A dimensions.
"""


@dataclass
class LLMJudgeMetricConfig:
    model: str = "deepseek-chat"
    api_key: str = field(default_factory=lambda: os.environ.get("DEEPSEEK_API_KEY", ""))
    base_url: str = "https://api.deepseek.com/v1"
    weights: dict = field(default_factory=lambda: {
        "acknowledgment": 0.20,
        "insight_accuracy": 0.20,
        "pattern_revelation": 0.10,
        "invitational_tone": 0.10,
        "actionability": 0.15,
        "naturalness": 0.15,
        "focus": 0.10,
    })


class LLMJudgeMetric(MetricBase):
    """LLM-as-Judge metric for parent-child coaching quality evaluation.
    
    Inherits MetricBase for prompt-ops file-path class loader compatibility.
    Uses DeepSeek-chat via OpenAI-compatible API as the judge model.
    """

    def __init__(self, config: LLMJudgeMetricConfig | None = None):
        super().__init__()
        self.config = config or LLMJudgeMetricConfig()
        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )

    def __call__(
        self,
        gold: Any,
        pred: Any,
        trace: Any = None,
    ) -> float:
        """Score a single prediction against the coaching quality rubric.
        
        Args:
            gold: Contains `gold.question` (the parent-child dialogue).
            pred: Contains `pred.answer` (the model's coaching popup output).
            trace: Unused; required by DSPy signature.
        
        Returns:
            Float in [0, 1], the weighted average score across all dimensions.
        """
        dialogue = gold.question
        popup_output = pred.answer

        scores = self._judge(dialogue, popup_output)
        return self._compute_weighted_score(scores)

    def _judge(self, dialogue: str, popup_output: str) -> dict[str, float]:
        """Call the judge LLM and parse the dimension scores."""
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": LLM_JUDGE_SYSTEM_PROMPT_V4},
                {
                    "role": "user",
                    "content": (
                        f"Dialogue:\n{dialogue}\n\n"
                        f"Popup Output:\n{popup_output}"
                    ),
                },
            ],
            temperature=0.0,
        )
        raw = response.choices[0].message.content
        return json.loads(raw)

    def _compute_weighted_score(self, scores: dict[str, float]) -> float:
        """Compute weighted average, redistributing N/A dimension weights."""
        active_weights = dict(self.config.weights)
        na_dims = [
            dim for dim, val in scores.items()
            if isinstance(val, str) and val.upper() == "N/A"
        ]

        # Remove N/A dimensions from consideration
        for dim in na_dims:
            del active_weights[dim]

        if not active_weights:
            return 0.0

        # Redistribute: normalize remaining weights to sum to 1.0
        total_weight = sum(active_weights.values())
        normalized = {k: v / total_weight for k, v in active_weights.items()}

        score = 0.0
        for dim, weight in normalized.items():
            val = scores.get(dim, 0.0)
            if isinstance(val, (int, float)):
                score += weight * float(val)

        return score
```

### 3. Wire it into prompt-ops config

The `config.yaml` metric section uses file-path mode so the prompt-ops class loader can find the metric without installing it as a package:

```yaml
metric:
  class: "scripts/llm_judge_metric.py"  # file-path mode
  config:
    model: "deepseek-chat"
    api_key: "${DEEPSEEK_API_KEY}"
    base_url: "https://api.deepseek.com/v1"
```

### 4. Bugs you will hit and how to fix them

Two bugs surfaced during integration that are likely to affect anyone doing this:

**Bug 1: `map_auto_mode_to_dspy(None)` returns `"light"` instead of `None`.** When the config has no `mode` field, the utility function in `strategy_utils.py` maps `None` to the string `"light"` rather than passing `None` through. This forces MIPROv2 into light mode even when the user did not request it. Fix by adding an early return when the input is `None`:

```python
def map_auto_mode_to_dspy(mode: str | None) -> str | None:
    if mode is None:
        return None
    # ... rest of mapping logic
```

**Bug 2: `auto` and `num_candidates` are mutually exclusive in DSPy 3.x but both were being passed.** The `prompt_strategies.py` file was sending `auto="light"` alongside `num_candidates=10` to the MIPROv2 optimizer constructor. DSPy 3.x treats these as conflicting parameters. Fix by ensuring only one is passed based on whether the mode is explicitly set.

## Why This Matters

### The format-matching trap

The `selected_fields_comparison` metric is dangerous because it reports improvement numbers that are real but meaningless. MIPROv2 will reliably find prompts that produce output more similar to a reference, and the metric score will increase. The problem is that the optimizer is solving the wrong optimization problem. In the parent-child coaching case, MIPROv2 was discovering that stripping nuance and using formulaic phrasing made the JSON output more predictable -- which the metric rewarded -- while making the coaching advice worse -- which was invisible to the metric.

This is not a MIPROv2 bug. It is a metric specification bug. The optimizer optimizes what you tell it to optimize. If you tell it to optimize exact string matching, it will.

### LLM-as-Judge closes the feedback loop

A good LLM-judge metric directly evaluates the thing you care about -- coaching quality -- rather than a proxy. The key insight is that domain experts encode their knowledge into the rubric once (the seven dimensions and their weights), and then the judge LLM applies that rubric consistently across every candidate. This turns a problem that would otherwise require human labeling into an automated signal MIPROv2 can use.

The evidence from the parent-child coaching case:

| Metric | v1.7 Baseline | Optimized | Delta | p-value | Cohen's d |
|---|---|---|---|---|---|
| LLM Judge (7-dim) | 0.628 +/- 0.147 | 0.830 +/- 0.063 | +20.2 pp | 0.0094 | 1.93 |

A Cohen's d of 1.93 is a large effect. The p-value of 0.0094 on a paired t-test confirms the improvement is not noise. The optimized prompt was substantially shorter than v1.7 -- it kept core concepts and dropped detailed methodology paragraphs that the judge model did not find essential.

### The N/A escape hatch prevents metric gaming

Without the N/A mechanism, a prompt that always gives concrete advice would score higher on actionability than one that correctly identifies when advice is inappropriate. Over enough MIPROv2 iterations, the optimizer would learn to always produce actionable-sounding output, even when the right coaching move is to shut up and reflect. The N/A dimension with weight redistribution prevents this -- it makes "correctly choosing not to give advice" score-neutral rather than score-negative.

## When to Apply

This pattern applies when all of the following are true:

1. **The output quality is multi-dimensional and subjective.** If your task has a single unambiguous correct answer (e.g., math problems, factual Q&A), exact-match metrics work fine and an LLM judge is overkill. Apply this when "good" means "this feels right to a human practitioner" across several soft dimensions.

2. **You are using MIPROv2 or another optimizer that needs a scalar score per candidate.** The `__call__` returning `float` pattern is DSPy's contract for metrics consumed by optimizers.

3. **You have domain expertise to encode into a rubric but not the budget to label every candidate.** The rubric is written once. The LLM applies it thousands of times. This is the efficiency gain over human evaluation.

4. **Some output dimensions may legitimately be N/A for certain inputs.** If every dimension always applies, you can skip the N/A logic. But in coaching, diagnostic, and advisory tasks, "correctly choosing not to act" is often the right behavior and the metric must not penalize it.

5. **You are using prompt-ops with its file-path class loader.** Inheriting from `MetricBase` is required for the `class: "scripts/llm_judge_metric.py"` config syntax to work.

## Examples

### Before: Format-matching metric (what was failing)

```python
from prompt_ops.metrics import StandardJSONMetric

metric = StandardJSONMetric(
    comparison_mode="selected_fields_comparison",
    fields=["should_popup", "tone", "popup_insight"],
)
```

MIPROv2 would see two outputs:

- Output A: `{"should_popup": true, "tone": "warm", "popup_insight": "You validated their feelings"}` -- score 1.0 (perfect match)
- Output B: `{"should_popup": true, "tone": "gentle", "popup_insight": "You noticed they felt unheard before offering advice"}` -- score 0.0 (no field matches)

Output B is better coaching advice, but the metric says it is worse. MIPROv2 converges toward Output-A-style responses: formulaic, predictable, mechanically correct, and substantively worse.

### After: LLM-as-Judge metric (what worked)

The same two outputs under `LLMJudgeMetric`:

Output A scores low on naturalness (formulaic phrasing) and insight_accuracy (generic insight). Weighted score: approximately 0.45.

Output B scores high on acknowledgment (notices the validation), insight_accuracy (identifies the specific dynamic), and naturalness (conversational phrasing). Weighted score: approximately 0.82.

MIPROv2 now has the right gradient. It converges toward prompts that produce output like B.

### The optimizer's choice under the correct metric

MIPROv2, guided by the LLM judge, selected a prompt that was approximately 60% shorter than v1.7. It preserved the core coaching framework concepts -- acknowledgment, pattern recognition, invitational language -- and dropped several paragraphs of detailed methodology explanation. The judge model did not find that detail useful for producing high-quality coaching popups, and MIPROv2 correctly identified that the extra verbiage was diluting the prompt's effectiveness.

This is a concrete example of why the metric matters: with the format-matching metric, MIPROv2 would have converged toward verbose, template-like prompts that reliably produced matching JSON fields. With the LLM judge, it converged toward a leaner prompt that produced better coaching advice. Same optimizer, different metric, fundamentally different outcome.

## Related

- [`scripts/llm_judge_metric.py`](../../use-cases/parent-child-coach/scripts/llm_judge_metric.py) — LLMJudgeMetric 实现，继承 MetricBase，使用 DeepSeek API 进行 7 维度评分
- [`scripts/compare_prompts.py`](../../use-cases/parent-child-coach/scripts/compare_prompts.py) — 统计显著性对比脚本（配对 t 检验 + Bootstrap CI）
- [`config.yaml`](../../use-cases/parent-child-coach/config.yaml) — prompt-ops 配置文件，使用 file-path 模式引用 LLMJudgeMetric
- [`docs/metric_selection_guide.md`](../metric_selection_guide.md) — Metric 选择指南（含 MetricBase 子类化创建自定义 metric 的说明）
- [`src/prompt_ops/core/metrics.py`](../../src/prompt_ops/core/metrics.py) — 核心 Metrics 模块（MetricBase、StandardJSONMetric 等）
