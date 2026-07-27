"""Dummy open-ended metric for PDO popup optimization.

PDO open-ended tasks use pairwise LLM judge, so this metric is only used to
satisfy the CLI/migrator interface and baseline summary. It returns a non-zero
score whenever the model produced a non-empty popup text.
"""
from prompt_ops.core.metrics import MetricBase
from prompt_ops.core.utils import extract_value


class PopupOpenEndedMetric(MetricBase):
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, gold, pred, trace: bool = False, **kwargs):
        pred_text = extract_value(pred, "answer", default="")
        score = 1.0 if pred_text and len(str(pred_text).strip()) > 10 else 0.0
        if trace:
            return {"score": score}
        return float(score)
