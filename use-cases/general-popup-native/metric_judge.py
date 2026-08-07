"""LLM-as-judge metric for MIPROv2 popup optimization.

Returns a scalar in [0, 1] by asking an LLM to score the popup on six
quality dimensions (1-5 each). MIPROv2 will maximize this score.
"""
import json
import os
import re
from typing import Union, Dict

from openai import OpenAI

from prompt_ops.core.metrics import MetricBase
from prompt_ops.core.utils import extract_value


JUDGE_PROMPT = """You are a strict popup-quality judge. Evaluate the following Chinese dialogue popup on six dimensions, each 1-5 (1=poor, 5=excellent).

Dimensions:
1. Accuracy: insight fits the dialogue, no misidentification or hallucination.
2. Stance: speaks only to "you" (account holder), not both sides.
3. Length: whole popup 60-180 Chinese characters, never over 200.
4. Structure: a standalone `——` line separates insight from exactly one concrete suggestion.
5. Tone: like a friend quietly reminding, no jargon, no moral judgment, no lecturing.
6. Actionability: the suggestion after `——` is specific and executable.

Dialogue:
{dialogue}

Popup:
{popup}

Respond ONLY with a JSON object like:
{{"accuracy": 4, "stance": 5, "length": 5, "structure": 4, "tone": 4, "actionability": 3}}
No other text.
"""


class PopupLLMJudgeMetric(MetricBase):
    def __init__(self, model_name: str = "deepseek-chat", api_base: str = "https://api.deepseek.com/v1", **kwargs):
        super().__init__()
        self.model_name = model_name
        self.api_base = api_base
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY environment variable is required for PopupLLMJudgeMetric")
        self.client = OpenAI(api_key=api_key, base_url=api_base)

    def __call__(self, gold, pred, trace: bool = False, **kwargs) -> Union[float, Dict[str, float]]:
        dialogue = extract_value(gold, "dialogue", default="") or extract_value(gold, "question", default="")
        popup = extract_value(pred, "answer", default="")
        if not popup or len(str(popup).strip()) < 10:
            return {"score": 0.0} if trace else 0.0

        prompt = JUDGE_PROMPT.format(dialogue=dialogue, popup=popup)
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512,
            )
            content = resp.choices[0].message.content.strip()
            # try to extract JSON object
            match = re.search(r"\{.*?\}", content, re.DOTALL)
            if not match:
                score = 0.5
            else:
                scores = json.loads(match.group(0))
                dims = ["accuracy", "stance", "length", "structure", "tone", "actionability"]
                values = [scores.get(d, 3) for d in dims]
                score = sum(values) / (5 * len(dims))
                score = max(0.0, min(1.0, score))
        except Exception:
            score = 0.5

        return {"score": score} if trace else float(score)
