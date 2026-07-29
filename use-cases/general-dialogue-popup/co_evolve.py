#!/usr/bin/env python
"""协同进化循环：v1.5.1 ⇄ v3.0 双轨自进化 · 三层架构.

Layer 1 · 协同进化内循环 (per-round):
  Round 0: 冻结基线
  Round 1..N: 交叉分析 → 并行变异 → 评估 → Keep/Discard → 收敛检查

Layer 2 · 监督层 (per-round):
  每轮后运行健康检查: 进展/变异健康/交叉学习质量/硬规则趋势/退化检测/评判一致性
  严重度: NOMINAL → WARNING → ANOMALY → CRITICAL

Layer 3 · Codex 顾问 (on-demand):
  监督层判定 CRITICAL → 启动深度诊断 → 输出干预方案 → 监督层执行

Usage:
  python co_evolve.py --max-rounds 10 --n 3
  python co_evolve.py --resume state_checkpoint.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results" / "co_evolve"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# API 配置
# ═══════════════════════════════════════════════════════════════════════════════

GEN_MODEL = os.environ.get("GEN_MODEL", "openai/deepseek-v4-pro")
GEN_API_BASE = os.environ.get("GEN_API_BASE", "https://qianfan.baidubce.com/v2")
GEN_API_KEY = os.environ.get("GEN_API_KEY", os.environ.get("DEEPSEEK_API_KEY", os.environ.get("QIANFAN_API_KEY", "")))

# Judge 复用同一个 API
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", GEN_MODEL)
JUDGE_API_BASE = os.environ.get("JUDGE_API_BASE", GEN_API_BASE)
JUDGE_API_KEY = os.environ.get("JUDGE_API_KEY", GEN_API_KEY)

# Mutator 复用同一个 API
MUTATE_MODEL = os.environ.get("MUTATE_MODEL", GEN_MODEL)
MUTATE_API_BASE = os.environ.get("MUTATE_API_BASE", GEN_API_BASE)
MUTATE_API_KEY = os.environ.get("MUTATE_API_KEY", GEN_API_KEY)

# Codex 顾问使用更强的模型
CODEX_MODEL = os.environ.get("CODEX_MODEL", GEN_MODEL)
CODEX_API_BASE = os.environ.get("CODEX_API_BASE", GEN_API_BASE)
CODEX_API_KEY = os.environ.get("CODEX_API_KEY", GEN_API_KEY)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("co_evolve")


# ═══════════════════════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════════════════════

class Severity(Enum):
    NOMINAL = "nominal"       # 🟢 一切正常
    WARNING = "warning"       # 🟡 轻微异常
    ANOMALY = "anomaly"       # 🟠 需要关注
    CRITICAL = "critical"     # 🔴 触发 Codex 顾问


@dataclass
class HealthReport:
    round: int
    severity: Severity
    checks: dict = field(default_factory=dict)       # check_name → (pass: bool, detail: str)
    anomalies: list = field(default_factory=list)
    trends: dict = field(default_factory=dict)        # dimension → "improving"/"stable"/"degrading"
    escalation_needed: bool = False
    escalation_reason: str = ""


@dataclass
class TrackState:
    """单条版本线的进化状态."""
    name: str                          # "v1.5.x" 或 "v3.x"
    lineage: str                       # "v151" 或 "v30"
    prompt_path: Path                  # 当前最优 prompt 文件路径
    current_text: str = ""             # 当前 prompt 文本（缓存）
    version: tuple = ()                # e.g., (1,5,1) or (3,0)
    frozen_baseline: dict = field(default_factory=dict)
    current_best: dict = field(default_factory=dict)
    history_scores: list = field(default_factory=list)
    history_hard_violations: list = field(default_factory=list)
    no_improve: int = 0
    # 本轮结果
    last_eval: dict = field(default_factory=dict)
    last_kept: bool = False
    last_mutation_attempts: int = 0
    last_candidate_path: Optional[Path] = None

    @property
    def aggregate(self) -> float:
        if not self.current_best:
            return 0.0
        return self.current_best.get("mean_aggregate", 0.0)

    @property
    def baseline_aggregate(self) -> float:
        if not self.frozen_baseline:
            return 0.0
        return self.frozen_baseline.get("mean_aggregate", 0.0)

    def next_version(self):
        """递增末位版本号."""
        parts = list(self.version)
        parts[-1] += 1
        self.version = tuple(parts)

    def version_str(self) -> str:
        return "v" + ".".join(str(p) for p in self.version)

    def next_prompt_path(self) -> Path:
        """生成下一版本的 prompt 文件路径."""
        parent = self.prompt_path.parent
        stem = self.prompt_path.stem
        # system_prompt_v1.5.1 → system_prompt_v1.5.2
        new_stem = re.sub(r'v[\d.]+$', self.version_str(), stem)
        return parent / f"{new_stem}.txt"


@dataclass
class CoEvolveState:
    """协同进化全局状态."""
    round: int = 0
    tracks: dict = field(default_factory=dict)        # lineage → TrackState
    deltas: list = field(default_factory=list)         # 逐轮 delta 历史
    converged: bool = False
    converged_reason: str = ""
    mutation_overrides: str = ""                       # Codex 注入的变异约束
    run_dir: Optional[Path] = None                     # 本轮输出目录


# ═══════════════════════════════════════════════════════════════════════════════
# 核心基础设施（从 compare_v151_v30.py 适配）
# ═══════════════════════════════════════════════════════════════════════════════

def generate_popup(system_prompt: str, dialogue: str) -> str | None:
    """调用 LLM 生成弹窗。返回弹窗文本，安静时返回 None。"""
    import litellm

    user_msg = f"当前对话：\n{dialogue}"
    kwargs: dict = dict(
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
        max_tokens=2048,
        timeout=180,
    )
    if GEN_API_BASE:
        kwargs["api_base"] = GEN_API_BASE
    if GEN_API_KEY:
        kwargs["api_key"] = GEN_API_KEY

    for attempt in range(3):
        try:
            resp = litellm.completion(**kwargs)
            text = (resp.choices[0].message.content or "").strip()
            break
        except Exception as e:
            logger.warning("生成尝试 %d/3 失败: %s", attempt + 1, e)
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)

    if not text or len(text) <= 10:
        return None
    if text.strip() == "安好":
        return None
    for prefix in ["弹窗：", "弹窗:", "【弹窗】", "输出："]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return text


def _hard_check(popup_text: str) -> dict:
    """硬规则检查（纯代码）。"""
    violations = []
    text = popup_text.strip()
    n = len(re.sub(r"\s", "", text))

    if n > 200:
        violations.append(f"字数 {n} 超过 200 绝对底线")
    elif n > 180:
        violations.append(f"字数 {n} 超过 180 硬合规线")
    elif n < 60:
        violations.append(f"字数 {n} 低于 60")
    if "——" not in text:
        violations.append("缺少 `——` 功能墙")
    else:
        found_wall = False
        for line in popup_text.splitlines():
            stripped = line.strip().strip("\"'`「」『』")
            if stripped == "——":
                found_wall = True
                break
        if not found_wall:
            violations.append("`——` 未单独成行")
        after = text.split("——")[-1].strip()
        if not after:
            violations.append("`——` 后缺少建议")
    if not text.endswith(("。", "？", "！", "”", '"', "」", "』")):
        violations.append("输出疑似截断")

    return {"pass": not violations, "violations": violations}


_JUDGE_PROMPT = """你是一位苛刻但公正的沟通教练评审。下面是一段两人对话，以及一个"沟通现场弹窗"AI 生成的弹窗。

弹窗的设计意图：帮对话中的一方（分不清是哪方，所以必须对双方都友善）看见此刻还没看到的东西——看清自己 / 看见对方 / 看见两人的互动模式。认知中立、态度是帮助。

【对话】
{dialogue}

【弹窗】
{popup}

请按四个维度打分（1-5 整数，5 最好）：

1. insight 洞察质量（从 5 个子维度综合评估）：
   - 准确性（基础事实-人称角色准确）：基于对话里实际出现的言语/行为，不脑补、不张冠李戴；正确识别"你"是谁
   - 相关性：正好扎在当事人此刻最没看到的那个点，不是重复废话或次要细节
   - 深度：穿过表面内容，点出言下之意 / 未表达的需求 / 两人互动循环
   - 非评判性：不审判、不说对错、不贴标签、不归因于恶意
   - 具体性：绑定到对话中的具体细节，不是模板套话

2. suggestion 建议可用性：`——` 之后的建议是否具体可操作、当场能用？是否符合平等协商原则？

3. non_judgment 非评判：是否不审判、不说对错、不贴标签、不归因于恶意？

4. language 语言质感：是否口语、像朋友现场轻声说的人话？有无术语、说教、模板腔？

再写一段 50 字以内的 comment，指出最主要的问题或亮点。

只输出 JSON：{{"insight": 4, "suggestion": 4, "non_judgment": 5, "language": 4, "comment": "..."}}"""


def _parse_judge_json(raw: str) -> dict | None:
    """从 LLM 输出中提取 JSON。"""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        soft = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    defaults = {"insight": 3, "suggestion": 3, "non_judgment": 3, "language": 3, "comment": ""}
    for k, v in defaults.items():
        if k not in soft:
            soft[k] = v
    for k in ("insight", "suggestion", "non_judgment", "language"):
        try:
            soft[k] = max(1, min(5, int(soft[k])))
        except (TypeError, ValueError):
            soft[k] = 3
    return soft


def _judge_popup(dialogue: str, popup_text: str) -> dict:
    """用 DeepSeek 打分 + 硬规则检查。"""
    import litellm

    prompt = _JUDGE_PROMPT.format(dialogue=dialogue, popup=popup_text)
    for attempt in range(3):
        try:
            resp = litellm.completion(
                model=JUDGE_MODEL,
                api_key=JUDGE_API_KEY,
                api_base=JUDGE_API_BASE,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=4096,
                timeout=180,
            )
            raw = (resp.choices[0].message.content or "").strip()
            break
        except Exception as e:
            logger.warning("judge 尝试 %d/3 失败: %s", attempt + 1, e)
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)

    soft = _parse_judge_json(raw)
    if soft is None:
        logger.warning("judge JSON 解析失败 | raw=%s", raw[:200])
        soft = {"insight": 3, "suggestion": 3, "non_judgment": 3, "language": 3,
                "comment": f"[parse_error] {raw[:80]}"}

    hard = _hard_check(popup_text)
    soft_mean = sum(soft[k] for k in ("insight", "suggestion", "non_judgment", "language")) / 4
    penalty = min(1.5, 0.5 * len(hard["violations"]))
    aggregate = round(max(1.0, soft_mean - penalty), 2)

    return {
        "aggregate": aggregate,
        "soft": soft,
        "hard": hard,
        "comment": soft.get("comment", ""),
        "popup_text": popup_text,
    }


def summarize(results: list[dict], label: str) -> dict:
    """计算汇总统计。"""
    valid = [r for r in results if r.get("popup_text") is not None and r.get("aggregate", 0) > 0]
    all_agg = [r["aggregate"] for r in results if r.get("aggregate", 0) > 0]
    soft_scores = {k: [] for k in ["insight", "suggestion", "non_judgment", "language"]}
    for r in valid:
        if r.get("soft"):
            for k in soft_scores:
                if k in r["soft"] and r["soft"][k] is not None:
                    soft_scores[k].append(r["soft"][k])

    hard_violations = sum(len(r.get("hard", {}).get("violations", [])) for r in results)
    n_should_popup = sum(1 for r in results if r.get("expect") != "安静")
    n_did_popup = sum(1 for r in results if r.get("popup_text") is not None)
    popup_rate = n_did_popup / n_should_popup if n_should_popup > 0 else 0
    quiet_cases = [r for r in results if r.get("expect") == "安静"]
    quiet_misfires = sum(1 for r in quiet_cases if r.get("popup_text") is not None)
    missing = sum(1 for r in results if r.get("expect") != "安静" and r.get("popup_text") is None)

    def mean(vals): return round(sum(vals) / len(vals), 3) if vals else 0

    return {
        "label": label,
        "n_total": len(results),
        "n_valid": len(valid),
        "mean_aggregate": mean(all_agg),
        "soft_means": {k: mean(v) for k, v in soft_scores.items()},
        "hard_violations": hard_violations,
        "popup_rate": round(popup_rate, 3),
        "quiet_misfires": quiet_misfires,
        "quiet_total": len(quiet_cases),
        "missing_popups": missing,
        "aggregates": all_agg,
    }


def run_eval(prompt_path: Path, cases: list[dict], n_runs: int = 3) -> tuple[list[dict], dict]:
    """对每个 case 跑 n 次，返回 (逐案结果, 汇总)。"""
    system_prompt = prompt_path.read_text(encoding="utf-8")
    if not system_prompt.strip():
        raise ValueError(f"Prompt 文件为空: {prompt_path}")

    results = []
    for ci, case in enumerate(cases):
        case_id = case["id"]
        expect = case.get("expect", "")
        dialogue = case["dialogue"]
        logger.info("  [%s] case %d/%d: %s", prompt_path.stem, ci + 1, len(cases), case_id)

        for run_i in range(n_runs):
            try:
                popup_text = generate_popup(system_prompt, dialogue)
            except Exception as e:
                logger.error("  生成失败: %s", e)
                results.append({
                    "case_id": case_id, "run": run_i, "expect": expect,
                    "popup_text": None, "error": str(e),
                    "aggregate": 0, "soft": None,
                    "hard": {"pass": False, "violations": [f"生成异常: {e}"]},
                })
                continue

            if expect == "安静":
                if popup_text is None:
                    results.append({
                        "case_id": case_id, "run": run_i, "expect": expect,
                        "popup_text": None, "aggregate": 5.0, "soft": None,
                        "hard": {"pass": True, "violations": []},
                        "comment": "正确保持安静",
                    })
                else:
                    results.append({
                        "case_id": case_id, "run": run_i, "expect": expect,
                        "popup_text": popup_text, "aggregate": 1.0, "soft": None,
                        "hard": {"pass": False, "violations": ["健康对话误弹"]},
                        "comment": "健康对话不应弹窗",
                    })
                continue

            if popup_text is None:
                results.append({
                    "case_id": case_id, "run": run_i, "expect": expect,
                    "popup_text": None, "aggregate": 1.5, "soft": None,
                    "hard": {"pass": False, "violations": ["应弹未弹"]},
                    "comment": "对话存在盲区但未弹窗",
                })
                continue

            try:
                eval_result = _judge_popup(dialogue, popup_text)
            except Exception as e:
                logger.error("  评审失败: %s", e)
                eval_result = {
                    "aggregate": 0, "soft": None,
                    "hard": {"pass": False, "violations": [f"评审异常: {e}"]},
                    "comment": f"评审异常: {e}", "popup_text": popup_text,
                }

            results.append({
                "case_id": case_id, "run": run_i, "expect": expect,
                "popup_text": popup_text,
                "aggregate": eval_result["aggregate"],
                "soft": eval_result.get("soft"),
                "hard": eval_result["hard"],
                "comment": eval_result.get("comment", ""),
            })

    summary = summarize(results, prompt_path.stem)
    # 附带逐案均值（方便交叉分析）
    summary["case_means"] = _per_case_means(results)
    return results, summary


def _per_case_means(results: list[dict]) -> dict:
    """计算每个 case 的均值聚合分。"""
    from collections import defaultdict
    case_scores = defaultdict(list)
    for r in results:
        if r.get("aggregate", 0) > 0:
            case_scores[r["case_id"]].append(r["aggregate"])
    return {cid: round(sum(s) / len(s), 3) for cid, s in case_scores.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 1 · 交叉分析
# ═══════════════════════════════════════════════════════════════════════════════

def summarize_failures(case_results: list[dict], top_n: int = 5) -> str:
    """从评估结果中提取最差案例的反馈文本。"""
    # 按 case_id 聚合
    from collections import defaultdict
    case_groups = defaultdict(list)
    for r in case_results:
        if r.get("aggregate", 0) > 0:
            case_groups[r["case_id"]].append(r)

    # 计算每个 case 的均值并排序
    case_means = []
    for cid, entries in case_groups.items():
        mean_agg = sum(e["aggregate"] for e in entries) / len(entries)
        case_means.append((cid, mean_agg, entries))
    case_means.sort(key=lambda x: x[1])

    lines = []
    for cid, mean_agg, entries in case_means[:top_n]:
        lines.append(f"### Case {cid} (均值 {mean_agg:.2f})")
        for e in entries[:2]:  # 取前 2 个 run
            popup = e.get('popup_text') or '(无弹窗)'
            lines.append(f"- 弹窗: {popup[:150]}")
            lines.append(f"- 评语: {e.get('comment') or 'N/A'}")
            if e.get("hard", {}).get("violations"):
                lines.append(f"- 违规: {'; '.join(e['hard']['violations'])}")
            if e.get("soft"):
                dims = e["soft"]
                lines.append(f"- 维度: insight={dims.get('insight','?')} "
                           f"suggestion={dims.get('suggestion','?')} "
                           f"non_judgment={dims.get('non_judgment','?')} "
                           f"language={dims.get('language','?')}")
        lines.append("")
    return "\n".join(lines)


def cross_analyze(
    eval_a: dict,
    eval_b: dict,
    label_a: str = "v1.5",
    label_b: str = "v3",
) -> dict:
    """分析双方互鉴素材。

    找出 A 弱 B 强的案例（A 应该学 B）和 B 弱 A 强的案例（B 应该学 A）。
    阈值：对方比自己高 >0.2 分。
    """
    case_means_a = eval_a.get("case_means", {})
    case_means_b = eval_b.get("case_means", {})

    a_strengths = []  # A 比 B 好的案例
    b_strengths = []  # B 比 A 好的案例
    both_weak = []    # 双方都差

    all_cases = set(list(case_means_a.keys()) + list(case_means_b.keys()))
    for cid in sorted(all_cases):
        ma = case_means_a.get(cid, 0)
        mb = case_means_b.get(cid, 0)
        if ma > mb + 0.2:
            a_strengths.append({"case_id": cid, "delta": round(ma - mb, 2), "a_score": ma, "b_score": mb})
        elif mb > ma + 0.2:
            b_strengths.append({"case_id": cid, "delta": round(mb - ma, 2), "a_score": ma, "b_score": mb})
        elif ma < 3.0 and mb < 3.0:
            both_weak.append({"case_id": cid, "a_score": ma, "b_score": mb})

    # 生成互鉴文本
    a_from_b = _build_cross_learning_text(a_strengths, b_strengths, label_a, label_b, "a_from_b")
    b_from_a = _build_cross_learning_text(a_strengths, b_strengths, label_a, label_b, "b_from_a")

    return {
        "a_strengths": a_strengths,
        "b_strengths": b_strengths,
        "both_weak": both_weak,
        "a_from_b": a_from_b,   # v151 可以从 v30 学什么
        "b_from_a": b_from_a,   # v30 可以从 v151 学什么
        "meaningful_diffs": len(a_strengths) + len(b_strengths),
    }


def _build_cross_learning_text(
    a_strengths: list,
    b_strengths: list,
    label_a: str,
    label_b: str,
    direction: str,
) -> str:
    """构造交叉学习提示文本。"""
    if direction == "a_from_b":
        # v151 应该学 v30
        target = b_strengths  # v30 比 v151 好的案例
        source_label = label_b
        target_label = label_a
    else:
        # v30 应该学 v151
        target = a_strengths  # v151 比 v30 好的案例
        source_label = label_a
        target_label = label_b

    if not target:
        return f"本轮 {source_label} 没有明显优于 {target_label} 的案例（delta > 0.2），无需交叉学习。"

    lines = [
        f"以下是 **{source_label}** 明显优于 **{target_label}** 的案例（delta > 0.2）：",
        f"共 {len(target)} 个案例。请分析 {source_label} 在这些案例上的做法，"
        f"思考 {target_label} 的 prompt 应该如何调整以吸收这些优势。",
        "",
    ]
    for item in target:
        lines.append(f"- Case {item['case_id']}: {source_label}={item.get('b_score' if direction == 'a_from_b' else 'a_score', '?')} "
                     f"vs {target_label}={item.get('a_score' if direction == 'a_from_b' else 'b_score', '?')} "
                     f"(Δ={item['delta']})")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 1 · 变异引擎
# ═══════════════════════════════════════════════════════════════════════════════

_CO_MUTATOR_PROMPT = """你是一位提示词工程专家。下面是一个"沟通现场弹窗"AI 的当前系统提示词。

【当前版本】{label}
【当前提示词】
{current_prompt}

【自身最差案例反馈】
{own_failures}

【对标版本的优势 —— 你可以借鉴的】
{cross_learning}

【之前失败的变异方向 —— 不要重复】
{previous_attempts}

【额外约束】
{mutation_overrides}

【改写纪律】
1. 你可以自由决定改动范围——从微调到重构都可以。但必须确保每次改动有明确理由。
2. 必须保留的核心不变项：
   - 角色定位："沟通现场弹窗"AI，不站队、不审判
   - `——` 功能墙（墙前洞察、墙后一句建议）
   - 字数死线：60-180 字，200 字绝对底线
   - "安好"安静信号
   - "只对「你」一个人说话"
3. 改动的目标：让弹窗更准（洞察）、更有用（建议）、更自然（语言）。
4. 如果自身问题在硬规则违规多，优先修格式。
5. 如果对标版本在某个维度显著领先，分析其做法并选择性吸收。
6. 严禁在 prompt 中使用 markdown 代码块、表格、或任何非纯文本格式标记。

请输出改写后的完整提示词全文（不要输出任何解释、不要用 markdown 代码块包裹）。"""


def co_mutate(
    current_prompt: str,
    own_failures: str,
    cross_learning: str,
    previous_attempts: list[str],
    label: str,
    mutation_overrides: str = "",
) -> str:
    """LLM 自由变异。返回变异后的完整 prompt 文本。"""
    import litellm

    prev_text = "\n".join(f"- {a}" for a in previous_attempts[-3:]) if previous_attempts else "（无）"
    overrides = mutation_overrides or "（无额外约束）"

    prompt = _CO_MUTATOR_PROMPT.format(
        label=label,
        current_prompt=current_prompt,
        own_failures=own_failures,
        cross_learning=cross_learning,
        previous_attempts=prev_text,
        mutation_overrides=overrides,
    )

    for attempt in range(3):
        try:
            resp = litellm.completion(
                model=MUTATE_MODEL,
                api_key=MUTATE_API_KEY,
                api_base=MUTATE_API_BASE,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=32000,
                timeout=300,
            )
            text = (resp.choices[0].message.content or "").strip()
            # 去掉可能的 markdown 包裹
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            return text
        except Exception as e:
            logger.warning("变异尝试 %d/3 失败: %s", attempt + 1, e)
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)

    return current_prompt  # fallback


def validate_mutation(prompt_text: str) -> tuple[bool, list[str]]:
    """变异后验证核心不变项是否保留。"""
    violations = []
    if "——" not in prompt_text:
        violations.append("缺少 `——` 功能墙")
    if "安好" not in prompt_text:
        violations.append("缺少 安好 安静信号")
    if "只对" not in prompt_text or "你" not in prompt_text:
        violations.append("缺少'只对「你」一个人说话'约束")
    if "180" not in prompt_text or "60" not in prompt_text:
        violations.append("缺少字数硬合规线（60/180）")
    # 最低长度检查
    if len(prompt_text.strip()) < 100:
        violations.append(f"Prompt 过短 ({len(prompt_text)} 字)，疑似截断")
    return not violations, violations


def mutate_with_retry(
    track: TrackState,
    cross: dict,
    previous_attempts: list[str],
    mutation_overrides: str = "",
    max_retries: int = 3,
) -> tuple[str, int]:
    """变异 + 硬规则验证 + 重试。返回 (mutated_text, attempts_used)。"""
    current_prompt = track.prompt_path.read_text(encoding="utf-8")
    own_failures = summarize_failures(
        track.last_eval.get("cases", track.current_best.get("cases", [])),
        top_n=5,
    )

    xlearn = cross["a_from_b"] if track.lineage == "v151" else cross["b_from_a"]

    for retry in range(max_retries):
        try:
            mutated = co_mutate(
                current_prompt=current_prompt,
                own_failures=own_failures,
                cross_learning=xlearn,
                previous_attempts=previous_attempts,
                label=track.name,
                mutation_overrides=mutation_overrides,
            )
        except Exception as e:
            logger.error("%s 变异 API 失败 (attempt %d): %s", track.name, retry + 1, e)
            continue

        ok, violations = validate_mutation(mutated)
        if ok:
            logger.info("%s 变异通过验证 (attempt %d)", track.name, retry + 1)
            return mutated, retry + 1

        logger.warning("%s 变异验证失败 (attempt %d): %s", track.name, retry + 1, violations)
        # 将 violation 反馈加入下次重试的约束
        mutation_overrides = (
            f"{mutation_overrides}\n"
            f"【上轮验证失败】{'; '.join(violations)}。请确保修改后的 prompt 包含这些要素。"
        )

    logger.error("%s 变异 %d 次均未通过验证", track.name, max_retries)
    return "", max_retries


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 2 · 监督层
# ═══════════════════════════════════════════════════════════════════════════════

class Supervisor:
    """Layer 2: 每轮后运行健康检查，必要时触发 Layer 3 Codex 顾问。"""

    def __init__(self, state: CoEvolveState, codex_enabled: bool = True):
        self.state = state
        self.codex_enabled = codex_enabled
        self.escalation_history: list[dict] = []
        self.interventions_applied: list[str] = []
        # 跟踪连续异常计数
        self._no_progress_streak = 0
        self._mutation_fail_streak = 0
        self._cross_learning_empty_streak = 0
        self._hard_rule_rise_streak = 0
        self._prev_violations: dict[str, int] = {}

    def check(self, round_result: dict) -> HealthReport:
        """每轮后运行完整健康检查。"""
        report = HealthReport(round=self.state.round, severity=Severity.NOMINAL)
        all_severities = []

        # 1. 目标进展
        ok, detail = self._check_progress(round_result)
        report.checks["progress"] = (ok, detail)
        if not ok:
            self._no_progress_streak += 1
        else:
            self._no_progress_streak = 0
        all_severities.append(self._progress_severity())

        # 2. 变异健康
        ok, detail = self._check_mutation_health(round_result)
        report.checks["mutation_health"] = (ok, detail)
        if not ok:
            self._mutation_fail_streak += 1
        else:
            self._mutation_fail_streak = 0
        all_severities.append(self._mutation_severity())

        # 3. 交叉学习质量
        ok, detail = self._check_cross_learning(round_result)
        report.checks["cross_learning"] = (ok, detail)
        if not ok:
            self._cross_learning_empty_streak += 1
        else:
            self._cross_learning_empty_streak = 0
        all_severities.append(self._cross_learning_severity())

        # 4. 硬规则趋势
        ok, detail = self._check_hard_rule_trend(round_result)
        report.checks["hard_rule_trend"] = (ok, detail)
        if not ok:
            self._hard_rule_rise_streak += 1
        else:
            self._hard_rule_rise_streak = 0
        all_severities.append(self._hard_rule_severity())

        # 5. 退化检测
        ok, detail = self._check_degradation(round_result)
        report.checks["degradation"] = (ok, detail)
        all_severities.append(Severity.CRITICAL if not ok else Severity.NOMINAL)

        # 6. 收敛趋势
        ok, detail = self._check_convergence_trend()
        report.checks["convergence_trend"] = (ok, detail)
        report.trends["delta"] = detail

        # 7. 评判一致性
        ok, detail = self._check_judge_consistency(round_result)
        report.checks["judge_consistency"] = (ok, detail)
        all_severities.append(Severity.WARNING if not ok else Severity.NOMINAL)

        # 取最高严重度
        severity_order = [Severity.NOMINAL, Severity.WARNING, Severity.ANOMALY, Severity.CRITICAL]
        report.severity = max(all_severities, key=lambda s: severity_order.index(s))

        # 升级逻辑
        if report.severity == Severity.CRITICAL:
            report.escalation_needed = True
            report.escalation_reason = self._build_escalation_reason()
        elif report.severity == Severity.ANOMALY:
            report.anomalies = [detail for ok, detail in report.checks.values() if not ok]

        return report

    # ── 各项检查 ──

    def _check_progress(self, round_result: dict) -> tuple[bool, str]:
        tracks_data = round_result.get("tracks", {})
        improved = []
        for lineage, data in tracks_data.items():
            track = self.state.tracks.get(lineage)
            if track and data.get("kept"):
                improved.append(lineage)
        if improved:
            return True, f"提升: {', '.join(improved)}"
        return False, "双方均无提升"

    def _progress_severity(self) -> Severity:
        if self._no_progress_streak >= 3:
            return Severity.CRITICAL
        if self._no_progress_streak >= 2:
            return Severity.ANOMALY
        if self._no_progress_streak >= 1:
            return Severity.WARNING
        return Severity.NOMINAL

    def _check_mutation_health(self, round_result: dict) -> tuple[bool, str]:
        tracks_data = round_result.get("tracks", {})
        failed = []
        for lineage, data in tracks_data.items():
            attempts = data.get("mutation_attempts", 0)
            if attempts >= 3 and not data.get("mutated_text"):
                failed.append(lineage)
        if failed:
            return False, f"变异全部失败: {', '.join(failed)}"
        return True, "变异均通过"

    def _mutation_severity(self) -> Severity:
        if self._mutation_fail_streak >= 3:
            return Severity.CRITICAL
        if self._mutation_fail_streak >= 2:
            return Severity.ANOMALY
        if self._mutation_fail_streak >= 1:
            return Severity.WARNING
        return Severity.NOMINAL

    def _check_cross_learning(self, round_result: dict) -> tuple[bool, str]:
        cross = round_result.get("cross_analysis", {})
        n = cross.get("meaningful_diffs", 0)
        if n >= 3:
            return True, f"找到 {n} 个有意义差异"
        if n >= 1:
            return False, f"仅有 {n} 个差异"
        return False, "交叉学习无有意义差异"

    def _cross_learning_severity(self) -> Severity:
        if self._cross_learning_empty_streak >= 3:
            return Severity.CRITICAL
        if self._cross_learning_empty_streak >= 2:
            return Severity.ANOMALY
        if self._cross_learning_empty_streak >= 1:
            return Severity.WARNING
        return Severity.NOMINAL

    def _check_hard_rule_trend(self, round_result: dict) -> tuple[bool, str]:
        tracks_data = round_result.get("tracks", {})
        current = {}
        for lineage, data in tracks_data.items():
            eval_data = data.get("eval", {})
            current[lineage] = eval_data.get("hard_violations", 0)

        if not self._prev_violations:
            self._prev_violations = current
            return True, "首轮，无趋势"

        rising = []
        for lineage, v in current.items():
            prev = self._prev_violations.get(lineage, 0)
            if v > prev:
                rising.append(f"{lineage}: {prev}→{v}")

        self._prev_violations = current
        if rising:
            return False, f"违规上升: {'; '.join(rising)}"
        return True, "违规稳定或下降"

    def _hard_rule_severity(self) -> Severity:
        if self._hard_rule_rise_streak >= 3:
            return Severity.CRITICAL
        if self._hard_rule_rise_streak >= 2:
            return Severity.ANOMALY
        if self._hard_rule_rise_streak >= 1:
            return Severity.WARNING
        return Severity.NOMINAL

    def _check_degradation(self, round_result: dict) -> tuple[bool, str]:
        tracks_data = round_result.get("tracks", {})
        degraded = []
        for lineage, data in tracks_data.items():
            track = self.state.tracks.get(lineage)
            if not track or not track.frozen_baseline:
                continue
            eval_data = data.get("eval", {})
            current_agg = eval_data.get("mean_aggregate", 0)
            baseline_agg = track.baseline_aggregate
            if current_agg < baseline_agg - 0.1:
                degraded.append(f"{lineage}: {baseline_agg:.3f}→{current_agg:.3f}")
        if degraded:
            return False, f"退化: {'; '.join(degraded)}"
        return True, "均高于冻结基线"

    def _check_convergence_trend(self) -> tuple[bool, str]:
        deltas = self.state.deltas
        if len(deltas) < 3:
            return True, f"delta 历史不足 ({len(deltas)} 轮)"
        recent = deltas[-3:]
        spread = max(recent) - min(recent)
        if spread < 0.05:
            return True, f"delta 稳定 (spread={spread:.3f})"
        if recent[-1] < recent[-2]:
            return True, f"delta 收窄中 ({recent[-2]:.3f}→{recent[-1]:.3f})"
        return False, f"delta 扩大 ({recent[-2]:.3f}→{recent[-1]:.3f})"

    def _check_judge_consistency(self, round_result: dict) -> tuple[bool, str]:
        """检查同 case 同 run 的 judge 分数波动。"""
        tracks_data = round_result.get("tracks", {})
        for lineage, data in tracks_data.items():
            eval_data = data.get("eval", {})
            case_results = eval_data.get("cases", [])
            # 按 case_id 分组检查 std
            from collections import defaultdict
            case_scores = defaultdict(list)
            for r in case_results:
                if r.get("aggregate", 0) > 0:
                    case_scores[r["case_id"]].append(r["aggregate"])
            high_std_cases = []
            for cid, scores in case_scores.items():
                if len(scores) >= 2:
                    mean_s = sum(scores) / len(scores)
                    variance = sum((s - mean_s) ** 2 for s in scores) / len(scores)
                    std = variance ** 0.5
                    if std > 1.0:
                        high_std_cases.append(f"{cid}(std={std:.2f})")
            if high_std_cases:
                return False, f"{lineage} 评判波动大: {', '.join(high_std_cases[:3])}"
        return True, "评判一致性正常"

    def _build_escalation_reason(self) -> str:
        reasons = []
        if self._no_progress_streak >= 3:
            reasons.append(f"双方连续 {self._no_progress_streak} 轮无提升")
        if self._mutation_fail_streak >= 3:
            reasons.append(f"变异连续 {self._mutation_fail_streak} 轮全部失败")
        if self._cross_learning_empty_streak >= 3:
            reasons.append(f"交叉学习连续 {self._cross_learning_empty_streak} 轮无差异")
        if self._hard_rule_rise_streak >= 3:
            reasons.append(f"硬规则违规连续 {self._hard_rule_rise_streak} 轮上升")
        return "; ".join(reasons)

    def decide(self, report: HealthReport) -> str:
        if report.severity == Severity.CRITICAL:
            return "escalate"
        if report.severity == Severity.ANOMALY:
            return "pause_observe"
        return "continue"


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 3 · Codex 顾问
# ═══════════════════════════════════════════════════════════════════════════════

_CODEX_DIAGNOSIS_PROMPT = """你是一位提示词进化系统的首席诊断专家。一个双轨协同进化系统遇到了严重问题，需要你深度诊断。

## 系统概况

这是一个"沟通现场弹窗"AI 的 prompt 协同进化系统。两条版本线（v1.5.x 和 v3.x）在并行进化，每轮通过交叉分析互相学习对方的优势。

- v1.5.1: 简洁高效（~10KB），综合分 4.083，擅长 suggestion/language，但弹窗率仅 76.7%
- v3.0: 理论完整但过于庞大（~40KB），综合分 3.808，100% 弹窗率但硬规则违规严重（43 次）

## 进化历史

{round_history}

## 当前 Prompt

### v1.5.x (当前最优)
```
{v151_current_prompt}
```

### v3.x (当前最优)
```
{v30_current_prompt}
```

## 触发诊断的原因

{trigger_reason}

## 诊断要求

请深度分析以下问题：

1. **根因诊断**：导致当前问题的根本原因是什么？（不超过 200 字）
2. **根因分类**：judge_bias / mutation_collapse / cross_learning_failure / prompt_bloat / other
3. **严重程度**：recoverable（可恢复）/ serious（严重）/ fatal（致命）
4. **干预建议**：具体应该采取什么措施？（不超过 200 字）
5. **干预类型**：adjust_mutation / restart_track / prune_prompt / recalibrate_judge / stop / manual_review
6. **是否继续进化**：true/false
7. **置信度**：0.0-1.0

请以 JSON 格式输出：
{{"diagnosis": "...", "root_cause_category": "...", "severity": "...", "recommendation": "...", "intervention_type": "...", "continue_evolution": true/false, "confidence": 0.0}}"""


class CodexAdvisor:
    """Layer 3: 监督层触发 CRITICAL 时，启动深度诊断。

    使用强模型进行根因分析。不直接修改 prompt — 只输出诊断报告和干预建议，
    由 Supervisor 决定是否执行。
    """

    def __init__(self, model: str | None = None):
        self.model = model or CODEX_MODEL
        self.api_base = CODEX_API_BASE
        self.api_key = CODEX_API_KEY
        self.diagnosis_history: list[dict] = []

    def diagnose(
        self,
        state: CoEvolveState,
        trigger_reason: str,
        round_history: list[dict],
    ) -> dict:
        """启动深度诊断，返回干预方案。"""
        import litellm

        context = self._build_diagnosis_context(state, trigger_reason, round_history)
        logger.info("🔍 Codex 顾问开始诊断 (model=%s)...", self.model)

        for attempt in range(3):
            try:
                resp = litellm.completion(
                    model=self.model,
                    api_key=self.api_key,
                    api_base=self.api_base,
                    messages=[{"role": "user", "content": context}],
                    temperature=0.2,
                    max_tokens=4096,
                    timeout=300,
                )
                raw = (resp.choices[0].message.content or "").strip()
                break
            except Exception as e:
                logger.warning("Codex 诊断尝试 %d/3 失败: %s", attempt + 1, e)
                if attempt == 2:
                    # 返回一个保守的诊断
                    return self._fallback_diagnosis(trigger_reason)
                time.sleep(3)

        diagnosis = self._parse_diagnosis(raw)
        if diagnosis is None:
            logger.warning("Codex 诊断 JSON 解析失败，使用 fallback")
            diagnosis = self._fallback_diagnosis(trigger_reason)

        self.diagnosis_history.append(diagnosis)
        logger.info("📋 Codex 诊断: %s (置信度 %.0f%%)",
                    diagnosis["diagnosis"][:100], diagnosis["confidence"] * 100)
        return diagnosis

    def _build_diagnosis_context(
        self, state: CoEvolveState, trigger_reason: str, round_history: list[dict]
    ) -> str:
        # 简化 round history
        history_lines = []
        for entry in round_history[-5:]:  # 最近 5 轮
            rnd = entry.get("round", "?")
            history_lines.append(f"### Round {rnd}")
            for lineage, data in entry.get("tracks", {}).items():
                eval_data = data.get("eval", {})
                history_lines.append(
                    f"- {lineage}: aggregate={eval_data.get('mean_aggregate', '?')}, "
                    f"kept={data.get('kept', '?')}, "
                    f"violations={eval_data.get('hard_violations', '?')}, "
                    f"attempts={data.get('mutation_attempts', '?')}"
                )
            cross = entry.get("cross_analysis", {})
            history_lines.append(f"- 交叉学习: {cross.get('meaningful_diffs', 0)} 个差异")
            history_lines.append("")

        # 截断过长的 prompt
        v151 = state.tracks.get("v151")
        v30 = state.tracks.get("v30")
        v151_text = v151.prompt_path.read_text(encoding="utf-8") if v151 else ""
        v30_text = v30.prompt_path.read_text(encoding="utf-8") if v30 else ""

        max_prompt_len = 6000
        if len(v151_text) > max_prompt_len:
            v151_text = v151_text[:max_prompt_len] + "\n\n... [截断]"
        if len(v30_text) > max_prompt_len:
            v30_text = v30_text[:max_prompt_len] + "\n\n... [截断]"

        return _CODEX_DIAGNOSIS_PROMPT.format(
            round_history="\n".join(history_lines) if history_lines else "（无历史）",
            v151_current_prompt=v151_text,
            v30_current_prompt=v30_text,
            trigger_reason=trigger_reason,
        )

    def _parse_diagnosis(self, raw: str) -> dict | None:
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            d = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None

        return {
            "diagnosis": d.get("diagnosis", "无法解析诊断"),
            "root_cause_category": d.get("root_cause_category", "other"),
            "severity": d.get("severity", "serious"),
            "recommendation": d.get("recommendation", ""),
            "intervention_type": d.get("intervention_type", "manual_review"),
            "continue_evolution": d.get("continue_evolution", True),
            "confidence": max(0.0, min(1.0, float(d.get("confidence", 0.5)))),
        }

    def _fallback_diagnosis(self, trigger_reason: str) -> dict:
        return {
            "diagnosis": f"自动诊断无法完成，触发原因: {trigger_reason}",
            "root_cause_category": "other",
            "severity": "serious",
            "recommendation": "建议暂停进化，人工检查当前 prompt 和评估数据",
            "intervention_type": "manual_review",
            "continue_evolution": False,
            "confidence": 0.3,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 干预执行
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_intervention(diagnosis: dict, state: CoEvolveState):
    """根据 Codex 诊断执行干预。"""
    itype = diagnosis["intervention_type"]

    if itype == "adjust_mutation":
        state.mutation_overrides = diagnosis.get("recommendation", "")
        logger.info("已更新变异约束: %s", state.mutation_overrides[:100])

    elif itype == "restart_track":
        target = diagnosis.get("target_track", "")
        if target in state.tracks:
            track = state.tracks[target]
            track.prompt_path = Path(diagnosis.get("restart_prompt_path",
                                         str(track.prompt_path)))
            track.current_best = track.frozen_baseline
            track.no_improve = 0
            track.history_scores = []
            track.history_hard_violations = []
            logger.info("已重置 %s 到冻结基线", target)

    elif itype == "prune_prompt":
        target = diagnosis.get("target_track", "")
        pruned_text = diagnosis.get("pruned_prompt_text", "")
        if target in state.tracks and pruned_text:
            track = state.tracks[target]
            new_path = track.prompt_path.parent / f"{track.prompt_path.stem}_pruned.txt"
            new_path.write_text(pruned_text, encoding="utf-8")
            track.prompt_path = new_path
            track.current_text = pruned_text
            logger.info("已精简 %s prompt (%d 字)", target, len(pruned_text))

    elif itype == "stop":
        state.converged = True
        state.converged_reason = f"Codex stop: {diagnosis['diagnosis']}"

    elif itype == "manual_review":
        logger.warning("Codex 请求人工审查: %s", diagnosis["recommendation"])
        checkpoint_path = state.run_dir / "manual_review_checkpoint.json" if state.run_dir else None
        if checkpoint_path:
            save_checkpoint(state, checkpoint_path)

    elif itype == "recalibrate_judge":
        logger.warning("Codex 建议重新校准 judge: %s", diagnosis["recommendation"])
        # 保存当前状态供人工审查 judge prompt
        state.converged = True
        state.converged_reason = f"Codex recalibrate: {diagnosis['diagnosis']}"


# ═══════════════════════════════════════════════════════════════════════════════
# 持久化
# ═══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(state: CoEvolveState, path: Path):
    """保存完整状态用于断点续跑。"""
    data = {
        "round": state.round,
        "deltas": state.deltas,
        "converged": state.converged,
        "converged_reason": state.converged_reason,
        "mutation_overrides": state.mutation_overrides,
        "tracks": {},
    }
    for lineage, track in state.tracks.items():
        data["tracks"][lineage] = {
            "name": track.name,
            "lineage": track.lineage,
            "prompt_path": str(track.prompt_path),
            "version": list(track.version),
            "frozen_baseline": track.frozen_baseline,
            "current_best": track.current_best,
            "history_scores": track.history_scores,
            "history_hard_violations": track.history_hard_violations,
            "no_improve": track.no_improve,
        }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_checkpoint(path: Path) -> CoEvolveState:
    """从断点恢复状态。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    state = CoEvolveState(
        round=data["round"],
        deltas=data.get("deltas", []),
        converged=data.get("converged", False),
        converged_reason=data.get("converged_reason", ""),
        mutation_overrides=data.get("mutation_overrides", ""),
    )
    for lineage, td in data.get("tracks", {}).items():
        track = TrackState(
            name=td["name"],
            lineage=td["lineage"],
            prompt_path=Path(td["prompt_path"]),
            version=tuple(td["version"]),
            frozen_baseline=td.get("frozen_baseline", {}),
            current_best=td.get("current_best", {}),
            history_scores=td.get("history_scores", []),
            history_hard_violations=td.get("history_hard_violations", []),
            no_improve=td.get("no_improve", 0),
        )
        state.tracks[lineage] = track
    return state


# ═══════════════════════════════════════════════════════════════════════════════
# 主循环
# ═══════════════════════════════════════════════════════════════════════════════

def main(
    max_rounds: int = 10,
    n_runs: int = 3,
    improve_threshold: float = 0.02,
    delta_threshold: float = 0.05,
    delta_patience: int = 2,
    no_improve_patience: int = 3,
    supervisor_enabled: bool = True,
    codex_enabled: bool = True,
    escalation_cooldown: int = 2,
    resume_from: Optional[Path] = None,
    prompt_v151: Optional[Path] = None,
    prompt_v30: Optional[Path] = None,
    dataset_path: Optional[Path] = None,
):
    # ── 加载数据集 ──
    if dataset_path is None:
        dataset_path = HERE / "data" / "business_dialogues_10.json"
    if not dataset_path.exists():
        logger.error("找不到数据集: %s", dataset_path)
        sys.exit(1)
    cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    logger.info("加载 %d 个职场场景", len(cases))

    # ── API key 检查 ──
    if not GEN_API_KEY:
        logger.error("缺少 API key（请设置 GEN_API_KEY / DEEPSEEK_API_KEY / QIANFAN_API_KEY）")
        sys.exit(1)
    logger.info("生成 API: %s @ %s", GEN_MODEL, GEN_API_BASE)

    # ── 设置运行目录 ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── 初始化或恢复状态 ──
    if resume_from:
        logger.info("从断点恢复: %s", resume_from)
        state = load_checkpoint(resume_from)
        state.run_dir = run_dir
    else:
        # 定位 prompt 文件
        p151 = prompt_v151 or (HERE / "system_prompt_v1.5.1.txt")
        p30 = prompt_v30 or (HERE / "system_prompt_v3.0.txt")
        if not p151.exists():
            logger.error("找不到 v1.5.1 prompt: %s", p151)
            sys.exit(1)
        if not p30.exists():
            logger.error("找不到 v3.0 prompt: %s", p30)
            sys.exit(1)

        v151 = TrackState(
            name="v1.5.x",
            lineage="v151",
            prompt_path=p151,
            current_text=p151.read_text(encoding="utf-8"),
            version=(1, 5, 1),
        )
        v30 = TrackState(
            name="v3.x",
            lineage="v30",
            prompt_path=p30,
            current_text=p30.read_text(encoding="utf-8"),
            version=(3, 0),
        )
        state = CoEvolveState(
            tracks={"v151": v151, "v30": v30},
            run_dir=run_dir,
        )

    supervisor = Supervisor(state, codex_enabled=codex_enabled)
    codex = CodexAdvisor()
    last_escalation_round = -999

    # ═══════════════════════════════════════════════════════════════════
    # Round 0: 冻结基线
    # ═══════════════════════════════════════════════════════════════════

    if state.round == 0:
        logger.info("=" * 60)
        logger.info("Round 0: 冻结基线")
        logger.info("=" * 60)

        for lineage, track in state.tracks.items():
            logger.info("▶ 评估 %s (%s)...", track.name, track.prompt_path.name)
            results, summary = run_eval(track.prompt_path, cases, n_runs)
            track.frozen_baseline = summary
            track.current_best = summary
            track.last_eval = {"cases": results, **summary}
            track.history_scores.append(summary["mean_aggregate"])
            track.history_hard_violations.append(summary["hard_violations"])

            # 保存基线
            baseline_path = run_dir / f"round_000_{lineage}_baseline.json"
            baseline_path.write_text(
                json.dumps({"results": results, "summary": summary}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("  %s 基线: aggregate=%.3f violations=%d popup_rate=%.1f%%",
                       track.name, summary["mean_aggregate"],
                       summary["hard_violations"], summary["popup_rate"] * 100)

        # 初始 delta
        initial_delta = abs(state.tracks["v151"].aggregate - state.tracks["v30"].aggregate)
        state.deltas.append(initial_delta)
        logger.info("初始 delta: %.3f", initial_delta)

    # ═══════════════════════════════════════════════════════════════════
    # Round 1..N: 协同进化
    # ═══════════════════════════════════════════════════════════════════

    prev_delta = state.deltas[-1] if state.deltas else 0
    stable_rounds = 0

    for rnd in range(state.round + 1, max_rounds + 1):
        state.round = rnd
        round_dir = run_dir / f"round_{rnd:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info("Round %d", rnd)
        logger.info("=" * 60)

        # ─── Layer 1: 交叉分析 ───
        cross = cross_analyze(
            state.tracks["v151"].current_best,
            state.tracks["v30"].current_best,
            "v1.5", "v3",
        )
        cross_path = round_dir / "cross_analysis.json"
        cross_path.write_text(json.dumps(cross, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("交叉分析: %d 个有意义差异 (v151→v30: %d, v30→v151: %d)",
                   cross["meaningful_diffs"],
                   len(cross["a_strengths"]), len(cross["b_strengths"]))

        round_result = {"round": rnd, "tracks": {}, "cross_analysis": cross}

        # ─── Layer 1: 并行变异 + 评估 ───
        for lineage in ["v151", "v30"]:
            track = state.tracks[lineage]
            track.next_version()

            # 变异
            mutated_text, attempts = mutate_with_retry(
                track, cross,
                previous_attempts=[
                    f"Round {rnd - i}: mutation failed"
                    for i in range(1, min(4, track.no_improve + 1))
                ],
                mutation_overrides=state.mutation_overrides,
                max_retries=3,
            )

            if not mutated_text:
                logger.error("%s 变异失败，跳过本轮", track.name)
                track.last_kept = False
                track.last_mutation_attempts = attempts
                track.last_eval = track.current_best
                round_result["tracks"][lineage] = {
                    "kept": False,
                    "eval": track.current_best,
                    "mutation_attempts": attempts,
                    "mutated_text": None,
                }
                continue

            # 保存候选
            candidate_path = round_dir / f"{track.version_str()}_candidate.txt"
            candidate_path.write_text(mutated_text, encoding="utf-8")
            logger.info("%s → %s (%d 字, %d attempts)",
                       track.name, track.version_str(), len(mutated_text), attempts)

            # 评估候选
            logger.info("▶ 评估 %s...", candidate_path.name)
            results, summary = run_eval(candidate_path, cases, n_runs)

            # Keep/Discard: 候选 vs 冻结基线
            baseline_agg = track.baseline_aggregate
            candidate_agg = summary["mean_aggregate"]
            kept = candidate_agg > baseline_agg + improve_threshold

            logger.info("  %s: baseline=%.3f candidate=%.3f (Δ=%+.3f) → %s",
                       track.name, baseline_agg, candidate_agg,
                       candidate_agg - baseline_agg,
                       "✅ KEEP" if kept else "❌ DISCARD")

            if kept:
                track.prompt_path = candidate_path
                track.current_text = mutated_text
                track.current_best = summary
                track.last_eval = {"cases": results, **summary}
                track.last_kept = True
                track.no_improve = 0
                track.history_scores.append(candidate_agg)
                track.history_hard_violations.append(summary["hard_violations"])
            else:
                candidate_path.unlink(missing_ok=True)
                track.last_kept = False
                track.last_eval = {"cases": results, **summary}
                track.no_improve += 1

            track.last_mutation_attempts = attempts
            track.last_candidate_path = candidate_path if kept else None

            # 保存评估结果
            eval_path = round_dir / f"{lineage}_eval.json"
            eval_path.write_text(
                json.dumps({"results": results, "summary": summary, "kept": kept},
                          ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            round_result["tracks"][lineage] = {
                "kept": kept,
                "eval": summary,
                "mutation_attempts": attempts,
                "mutated_text": mutated_text,
                "version": track.version_str(),
            }

        # ─── Layer 2: 监督层健康检查 ───
        if supervisor_enabled:
            health = supervisor.check(round_result)
            health_path = round_dir / "health_report.json"
            health_path.write_text(json.dumps({
                "round": health.round,
                "severity": health.severity.value,
                "checks": {k: {"pass": v[0], "detail": v[1]} for k, v in health.checks.items()},
                "anomalies": health.anomalies,
                "trends": health.trends,
                "escalation_needed": health.escalation_needed,
                "escalation_reason": health.escalation_reason,
            }, ensure_ascii=False, indent=2), encoding="utf-8")

            logger.info("🏥 监督层: %s", health.severity.value.upper())

            if health.severity == Severity.CRITICAL:
                rounds_since_last = rnd - last_escalation_round
                if codex_enabled and rounds_since_last >= escalation_cooldown:
                    logger.warning("🔴 触发 Codex 顾问: %s", health.escalation_reason)

                    # ─── Layer 3: Codex 顾问诊断 ───
                    diagnosis = codex.diagnose(
                        state=state,
                        trigger_reason=health.escalation_reason,
                        round_history=_load_round_history(run_dir),
                    )
                    diag_path = round_dir / "codex_diagnosis.json"
                    diag_path.write_text(json.dumps(diagnosis, ensure_ascii=False, indent=2),
                                        encoding="utf-8")

                    supervisor.interventions_applied.append(diagnosis["intervention_type"])
                    last_escalation_round = rnd

                    if not diagnosis["continue_evolution"]:
                        logger.info("Codex 建议停止进化: %s", diagnosis["recommendation"])
                        state.converged = True
                        state.converged_reason = f"Codex stop R{rnd}: {diagnosis['diagnosis']}"
                        break

                    _apply_intervention(diagnosis, state)
                else:
                    logger.info("Codex 冷却中 (上次 R%d, 冷却 %d 轮)，跳过",
                              last_escalation_round, escalation_cooldown)

            elif health.severity == Severity.ANOMALY:
                logger.warning("🟠 监督层 ANOMALY: %s", health.anomalies)

        # ─── 收敛检查 ───
        curr_delta = abs(
            state.tracks["v151"].aggregate - state.tracks["v30"].aggregate
        )
        state.deltas.append(curr_delta)
        delta_change = abs(curr_delta - prev_delta)
        logger.info("delta: %.3f → %.3f (Δ=%.3f)", prev_delta, curr_delta, delta_change)

        if delta_change < delta_threshold:
            stable_rounds += 1
            if stable_rounds >= delta_patience:
                logger.info("✅ delta 收敛: 连续 %d 轮 delta 变化 < %.2f", stable_rounds, delta_threshold)
                state.converged = True
                state.converged_reason = f"delta stabilized at R{rnd}"
                break
        else:
            stable_rounds = 0
        prev_delta = curr_delta

        # 双方都无提升
        if (state.tracks["v151"].no_improve >= no_improve_patience and
                state.tracks["v30"].no_improve >= no_improve_patience):
            logger.info("✅ 双方各 %d 轮无提升，停止", no_improve_patience)
            state.converged = True
            state.converged_reason = f"both tracks stalled at R{rnd}"
            break

        # 保存断点
        save_checkpoint(state, run_dir / "state_checkpoint.json")

        # 保存轮次摘要
        round_summary = {
            "round": rnd,
            "v151_agg": state.tracks["v151"].aggregate,
            "v30_agg": state.tracks["v30"].aggregate,
            "delta": curr_delta,
            "v151_kept": round_result["tracks"]["v151"]["kept"],
            "v30_kept": round_result["tracks"]["v30"]["kept"],
            "supervisor_severity": health.severity.value if supervisor_enabled else "disabled",
        }
        (round_dir / "round_summary.json").write_text(
            json.dumps(round_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ═══════════════════════════════════════════════════════════════════
    # 最终报告
    # ═══════════════════════════════════════════════════════════════════
    _write_final_report(state, supervisor, run_dir)


def _load_round_history(run_dir: Path) -> list[dict]:
    """加载所有轮次的结果。"""
    history = []
    for round_dir in sorted(run_dir.glob("round_*")):
        summary_path = round_dir / "round_summary.json"
        if summary_path.exists():
            entry = json.loads(summary_path.read_text(encoding="utf-8"))
            # 附加更详细的数据
            for lineage in ["v151", "v30"]:
                eval_path = round_dir / f"{lineage}_eval.json"
                if eval_path.exists():
                    eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
                    entry.setdefault("tracks", {})[lineage] = {
                        "kept": eval_data.get("kept", False),
                        "eval": eval_data.get("summary", {}),
                    }
            cross_path = round_dir / "cross_analysis.json"
            if cross_path.exists():
                entry["cross_analysis"] = json.loads(cross_path.read_text(encoding="utf-8"))
            history.append(entry)
    return history


def _write_final_report(state: CoEvolveState, supervisor: Supervisor, run_dir: Path):
    """生成最终汇总报告。"""
    report = {
        "converged": state.converged,
        "converged_reason": state.converged_reason,
        "total_rounds": state.round,
        "final_delta": state.deltas[-1] if state.deltas else 0,
        "deltas": state.deltas,
        "interventions": supervisor.interventions_applied,
        "tracks": {},
    }
    for lineage, track in state.tracks.items():
        report["tracks"][lineage] = {
            "name": track.name,
            "version": track.version_str(),
            "prompt_path": str(track.prompt_path),
            "baseline_aggregate": track.baseline_aggregate,
            "final_aggregate": track.aggregate,
            "improvement": round(track.aggregate - track.baseline_aggregate, 3),
            "best_aggregate": max(track.history_scores) if track.history_scores else track.aggregate,
            "history_scores": track.history_scores,
            "history_hard_violations": track.history_hard_violations,
            "kept_rounds": len([s for i, s in enumerate(track.history_scores)
                              if i > 0 and s > track.history_scores[i - 1]]),
        }

    report_path = run_dir / "final_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 打印摘要
    print("\n" + "=" * 60)
    print("  协同进化完成")
    print("=" * 60)
    print(f"  总轮数: {state.round}")
    print(f"  收敛原因: {state.converged_reason}")
    print(f"  Codex 干预: {len(supervisor.interventions_applied)} 次")
    for lineage, track in state.tracks.items():
        print(f"  {track.name}: {track.baseline_aggregate:.3f} → {track.aggregate:.3f} "
              f"(Δ={track.aggregate - track.baseline_aggregate:+.3f}) "
              f"→ {track.prompt_path.name}")
    print(f"  最终 delta: {state.deltas[-1]:.3f}" if state.deltas else "")
    print(f"  报告: {report_path}")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="协同进化循环：v1.5.1 ⇄ v3.0 双轨自进化 · 三层架构",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples:
          python co_evolve.py --max-rounds 10 --n 3
          python co_evolve.py --resume results/co_evolve/run_xxx/state_checkpoint.json
          python co_evolve.py --no-supervisor  # 仅 Layer 1
          python co_evolve.py --no-codex       # Layer 1 + 2，无 Codex 顾问
        """),
    )
    parser.add_argument("--max-rounds", type=int, default=10, help="最大轮数")
    parser.add_argument("--n", type=int, default=3, dest="n_runs", help="每 case 跑 n 次")
    parser.add_argument("--improve-threshold", type=float, default=0.02, help="保留候选的最小增量")
    parser.add_argument("--delta-threshold", type=float, default=0.05, help="delta 收敛阈值")
    parser.add_argument("--delta-patience", type=int, default=2, help="delta 稳定的连续轮数")
    parser.add_argument("--no-improve-patience", type=int, default=3, help="单版本连续无提升上限")
    parser.add_argument("--gen-model", default=GEN_MODEL, help="生成模型")
    parser.add_argument("--gen-api-base", default=GEN_API_BASE, help="生成 API base")
    parser.add_argument("--no-supervisor", action="store_true", help="禁用监督层（仅 Layer 1）")
    parser.add_argument("--no-codex", action="store_true", help="禁用 Codex 顾问")
    parser.add_argument("--escalation-cooldown", type=int, default=2,
                       help="两次 Codex 调用之间最少间隔轮数")
    parser.add_argument("--resume", type=Path, help="从断点恢复")
    parser.add_argument("--prompt-v151", type=Path, help="v1.5.x prompt 文件路径")
    parser.add_argument("--prompt-v30", type=Path, help="v3.x prompt 文件路径")
    parser.add_argument("--dataset", type=Path, help="数据集路径")

    args = parser.parse_args()

    # 应用 CLI 覆盖的 API 配置
    if args.gen_model:
        GEN_MODEL = args.gen_model
        JUDGE_MODEL = args.gen_model
    if args.gen_api_base:
        GEN_API_BASE = args.gen_api_base
        JUDGE_API_BASE = args.gen_api_base

    main(
        max_rounds=args.max_rounds,
        n_runs=args.n_runs,
        improve_threshold=args.improve_threshold,
        delta_threshold=args.delta_threshold,
        delta_patience=args.delta_patience,
        no_improve_patience=args.no_improve_patience,
        supervisor_enabled=not args.no_supervisor,
        codex_enabled=not args.no_codex,
        escalation_cooldown=args.escalation_cooldown,
        resume_from=args.resume,
        prompt_v151=args.prompt_v151,
        prompt_v30=args.prompt_v30,
        dataset_path=args.dataset,
    )
