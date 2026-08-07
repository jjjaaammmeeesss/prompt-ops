"""候选目录读写（Meta-Harness 契约）。

候选目录结构:
  candidates/
    <strategy>/
      candidate_001/
        harness.md       # 策略核心 prompt
        harness.py       # 策略适配代码
        metrics.json     # search-set 聚合分数
        proposal.md      # proposer 修改说明 + trace 证据引用
        trace.jsonl      # 每 case 一行完整执行记录
        validation.json  # validation-set 分数
        parent.json      # 父候选引用 (baseline 时为空)
      candidate_002/
        ...

trace.jsonl 每行 schema:
{
  "case_id": str,
  "window_index": int | null,
  "sys_tone": str,
  "gold_tone": str,
  "m1_trigger_match": float | null,
  "m5_tone_match": float | null,
  "m6_insight_score": float | null,
  "m7_safety_score": float | null,
  "error": str,
  "sys_popup_text": str,
  "gold_reference_popup": str,
  "failure_source": "prompt" | "judge" | "dataset" | "search" | "model" | null
}
"""

import json
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── 复用 auto_evolve 的数据结构 ──────────────────────────────
import sys
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from auto_evolve.evaluator import EvalResult, BaselineReport  # noqa: E402


CANDIDATES_DIR = Path(__file__).parent / "candidates"


# ═══════════════════════════════════════════════════════════════
# trace.jsonl 数据结构
# ═══════════════════════════════════════════════════════════════

@dataclass
class TraceEntry:
    """单 case 评估 trace 记录。"""
    case_id: str
    window_index: int | None = None
    sys_tone: str = ""
    gold_tone: str = ""
    m1_trigger_match: float | None = None
    m5_tone_match: float | None = None
    m6_insight_score: float | None = None
    m7_safety_score: float | None = None
    error: str = ""
    sys_popup_text: str = ""
    gold_reference_popup: str = ""
    failure_source: str | None = None  # prompt | judge | dataset | search | model

    @classmethod
    def from_eval_result(cls, r: EvalResult) -> "TraceEntry":
        return cls(
            case_id=r.case_id,
            window_index=r.window_index if r.window_index else None,
            sys_tone=r.sys_tone,
            gold_tone=r.gold_tone,
            m1_trigger_match=r.m1_trigger_match,
            m5_tone_match=r.m5_tone_match,
            m6_insight_score=r.m6_insight_score,
            m7_safety_score=r.m7_safety_score,
            error=r.error,
            sys_popup_text=r.sys_popup_text,
            gold_reference_popup=r.gold_reference_popup,
        )


# ═══════════════════════════════════════════════════════════════
# metrics.json 数据结构
# ═══════════════════════════════════════════════════════════════

@dataclass
class MetricsSnapshot:
    """候选评估指标快照。"""
    overall_score: float = 0.0
    aggregate_m1: float = 0.0
    aggregate_m5: float = 0.0
    aggregate_m6: float = 0.0
    aggregate_m7: float = 0.0
    n_cases: int = 0
    n_m5_failures: int = 0
    n_m5_improvements: int = 0
    timestamp: str = ""

    @classmethod
    def from_baseline_report(cls, report: BaselineReport) -> "MetricsSnapshot":
        n_m5_fail = sum(1 for r in report.results if r.m5_tone_match == 0.0)
        return cls(
            overall_score=round(report.overall_score, 4),
            aggregate_m1=round(report.aggregate_m1, 4),
            aggregate_m5=round(report.aggregate_m5, 4),
            aggregate_m6=round(report.aggregate_m6, 2),
            aggregate_m7=round(report.aggregate_m7, 2),
            n_cases=len(report.results),
            n_m5_failures=n_m5_fail,
            timestamp=datetime.now().isoformat(),
        )


# ═══════════════════════════════════════════════════════════════
# parent.json 数据结构
# ═══════════════════════════════════════════════════════════════

@dataclass
class ParentRef:
    """父候选引用。"""
    candidate_id: str = "baseline"
    strategy: str = ""
    overall_score: float = 0.0


# ═══════════════════════════════════════════════════════════════
# 候选目录 I/O
# ═══════════════════════════════════════════════════════════════

def candidate_dir(strategy: str, candidate_id: str) -> Path:
    """返回候选目录路径。"""
    return CANDIDATES_DIR / strategy / candidate_id


def init_candidate(
    strategy: str,
    candidate_id: str,
    harness_md: str = "",
    harness_py: str = "",
    parent: ParentRef | None = None,
) -> Path:
    """初始化候选目录，写入 harness 文件。

    Args:
        strategy: 策略名 (senate / teacher_student / saga)
        candidate_id: 候选 ID (candidate_001, baseline, ...)
        harness_md: prompt markdown 内容
        harness_py: 适配代码
        parent: 父候选引用

    Returns:
        候选目录 Path
    """
    d = candidate_dir(strategy, candidate_id)
    d.mkdir(parents=True, exist_ok=True)

    # 写入 harness 文件
    if harness_md:
        (d / "harness.md").write_text(harness_md, encoding="utf-8")
    if harness_py:
        (d / "harness.py").write_text(harness_py, encoding="utf-8")

    # 写入 parent.json
    parent_data = asdict(parent) if parent else {"candidate_id": "baseline", "strategy": strategy, "overall_score": 0.0}
    (d / "parent.json").write_text(json.dumps(parent_data, ensure_ascii=False, indent=2), encoding="utf-8")

    return d


def save_trace(
    d: Path,
    trace_entries: list[TraceEntry],
) -> None:
    """写入 trace.jsonl。"""
    with open(d / "trace.jsonl", "w", encoding="utf-8") as f:
        for entry in trace_entries:
            f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")


def load_trace(d: Path) -> list[TraceEntry]:
    """加载 trace.jsonl。"""
    entries = []
    trace_path = d / "trace.jsonl"
    if not trace_path.exists():
        return entries
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            data = json.loads(line)
            entries.append(TraceEntry(**data))
    return entries


def get_m5_failures(d: Path) -> list[TraceEntry]:
    """获取 M5 失败的 trace entries。"""
    return [e for e in load_trace(d) if e.m5_tone_match == 0.0]


def save_metrics(d: Path, metrics: MetricsSnapshot) -> None:
    """写入 metrics.json。"""
    (d / "metrics.json").write_text(
        json.dumps(asdict(metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_metrics(d: Path) -> MetricsSnapshot:
    """加载 metrics.json。"""
    p = d / "metrics.json"
    if not p.exists():
        return MetricsSnapshot()
    data = json.loads(p.read_text(encoding="utf-8"))
    return MetricsSnapshot(**data)


def save_proposal(d: Path, proposal_text: str) -> None:
    """写入 proposal.md。"""
    (d / "proposal.md").write_text(proposal_text, encoding="utf-8")


def load_proposal(d: Path) -> str:
    """加载 proposal.md。"""
    p = d / "proposal.md"
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")


def save_validation(d: Path, metrics: MetricsSnapshot) -> None:
    """写入 validation.json。"""
    (d / "validation.json").write_text(
        json.dumps(asdict(metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def list_candidates(strategy: str) -> list[Path]:
    """列出某策略下所有候选目录，按 ID 排序。"""
    d = CANDIDATES_DIR / strategy
    if not d.exists():
        return []
    candidates = sorted(
        [p for p in d.iterdir() if p.is_dir() and p.name.startswith("candidate_")],
        key=lambda p: p.name,
    )
    return candidates


def latest_candidate(strategy: str) -> Path | None:
    """返回最新的候选目录。"""
    candidates = list_candidates(strategy)
    return candidates[-1] if candidates else None


def get_best_candidate(strategy: str) -> Path | None:
    """返回 overall_score 最高的候选目录。"""
    candidates = list_candidates(strategy)
    if not candidates:
        return None
    best = None
    best_score = float("-inf")
    for c in candidates:
        m = load_metrics(c)
        if m.overall_score > best_score:
            best_score = m.overall_score
            best = c
    return best


def archive_candidate(d: Path) -> None:
    """删除候选目录（discard 时调用）。"""
    if d.exists():
        shutil.rmtree(d)


# ═══════════════════════════════════════════════════════════════
# 从 EvalResult 列表构建候选
# ═══════════════════════════════════════════════════════════════

def build_candidate_from_report(
    strategy: str,
    candidate_id: str,
    report: BaselineReport,
    harness_md: str = "",
    harness_py: str = "",
    proposal: str = "",
    parent: ParentRef | None = None,
) -> Path:
    """从 BaselineReport 构建完整候选目录。

    Args:
        strategy: 策略名
        candidate_id: 候选 ID
        report: 评估报告
        harness_md: prompt 内容
        harness_py: 适配代码
        proposal: 修改说明
        parent: 父候选引用

    Returns:
        候选目录 Path
    """
    d = init_candidate(strategy, candidate_id, harness_md, harness_py, parent)

    # trace
    trace = [TraceEntry.from_eval_result(r) for r in report.results]
    save_trace(d, trace)

    # metrics
    save_metrics(d, MetricsSnapshot.from_baseline_report(report))

    # proposal
    if proposal:
        save_proposal(d, proposal)

    return d
