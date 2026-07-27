#!/usr/bin/env python
"""双层闭环自进化 · 通用弹窗 Prompt 优化.

架构:
  外层循环 · 元控制器 (max 5 轮):
    ① 失败模式分类 — 取 metric 最低维度
    ② 策略调度     — 失败模式 → 内层策略
    ③ 执行内层策略 — 可插拔 (gepa / gepa_mipro / mipro / pdo)
    ④ 基线守护     — 候选 vs 冻结 origin baseline
    ⑤ 决策         — keep (Δ>0.02 & 约束保留>80%) / discard → 换策略
    ⑥ 收敛检查     — patience=2

  内层策略目录:
    gepa        — GEPA only, 文本反思恢复领域约束
    gepa_mipro  — GEPA → MIPROv2, 默认首轮策略
    mipro       — MIPROv2 only, 仅措辞微调
    pdo         — PDO only, pairwise 对比

  失败模式 → 策略映射:
    约束保真度/硬规则 → gepa
    措辞/字数        → gepa_mipro
    tone 判定        → pdo
    综合偏低          → 换起点

Usage:
  export DEEPSEEK_API_KEY=sk-...
  python self_evolve.py --baseline system_prompt_v1.4.txt
  python self_evolve.py --resume state.json  # 断点续跑
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import dspy
import litellm

logger = logging.getLogger("self_evolve")

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results" / "self_evolve"

# ═══════════════════════════════════════════════════════════════════════════════
# 外层循环 + 收敛常量
# ═══════════════════════════════════════════════════════════════════════════════
MAX_ROUNDS = 12
PATIENCE = 2  # 保留兼容，收敛逻辑已改为策略用尽检测

# ═══════════════════════════════════════════════════════════════════════════════
# 内层策略目录 — 可插拔
# ═══════════════════════════════════════════════════════════════════════════════

STRATEGY_CATALOG: dict[str, dict] = {
    "gepa": {
        "name": "gepa",
        "desc": "GEPA only: 文本反思驱动种群进化, 恢复领域约束",
        "primary": "gepa", "primary_kwargs": {"max_full_evals": 7},
        "secondary": None,
    },
    "gepa_mipro": {
        "name": "gepa_mipro",
        "desc": "GEPA→MIPROv2: 遗传进化改骨架 + 贝叶斯精调措辞",
        "primary": "gepa", "primary_kwargs": {"max_full_evals": 5},
        "secondary": "mipro",
    },
    "mipro": {
        "name": "mipro",
        "desc": "MIPROv2 only: 贝叶斯联合优化指令+few-shot, 仅措辞微调",
        "primary": None,
        "secondary": "mipro",
    },
    # PDO 在 DSPy 3.2.1 已移除，用 gepa 替代
    "gepa_alt": {
        "name": "gepa_alt",
        "desc": "GEPA (替代原 PDO): 文本反思驱动进化, max_full_evals=5",
        "primary": "gepa", "primary_kwargs": {"max_full_evals": 5},
        "secondary": None,
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# 失败模式 → 策略调度映射
# ═══════════════════════════════════════════════════════════════════════════════
# 从 metric 6 维中取最低维度作为失败模式，映射到对应策略。
# 随实验迭代调整，不作为代码常量写死。

# 每个失败模式的策略优先级序列 — 首选项失败后逐个轮换，用尽才收敛
# PDO 在 DSPy 3.2.1 已移除，所有 PDO 槽位替换为 gepa/gepa_mipro
FAILURE_STRATEGY_SEQUENCE: dict[str, list[str]] = {
    "accuracy":      ["gepa", "gepa_mipro", "mipro", "gepa_alt"],
    "stance":        ["gepa", "gepa_mipro", "mipro", "gepa_alt"],
    "length":        ["gepa_mipro", "gepa", "gepa_alt", "mipro"],
    "structure":     ["gepa", "gepa_mipro", "gepa_alt", "mipro"],
    "tone":          ["gepa", "gepa_mipro", "gepa_alt", "mipro"],
    "actionability": ["mipro", "gepa_mipro", "gepa", "gepa_alt"],
}

# 兼容旧引用：取序列首项
FAILURE_TO_STRATEGY: dict[str, str] = {
    k: v[0] for k, v in FAILURE_STRATEGY_SEQUENCE.items()
}


# ═══════════════════════════════════════════════════════════════════════════════
# 硬规则检查（纯代码）
# ═══════════════════════════════════════════════════════════════════════════════


def hard_check(popup_text: str) -> dict:
    violations = []
    text = popup_text.strip()
    n = len(re.sub(r"\s", "", text))

    if n > 200:
        violations.append(f"字数 {n} 超 200 绝对底线")
    elif n > 180:
        violations.append(f"字数 {n} 超 180 硬合规线")
    elif n < 60:
        violations.append(f"字数 {n} 低于 60")

    if "——" not in text:
        violations.append("缺少 `——` 功能墙")
    else:
        found_wall = any(
            line.strip().strip("\"'`「」『』") == "——"
            for line in popup_text.splitlines()
        )
        if not found_wall:
            violations.append("`——` 未单独成行")
        after = text.split("——")[-1].strip()
        if not after:
            violations.append("`——` 后缺少建议")

    if not text.endswith(("。", "？", "！", "”", '"', "」", "』")):
        violations.append("输出疑似截断")

    return {"pass": not violations, "violations": violations}


# ═══════════════════════════════════════════════════════════════════════════════
# LLM Judge
# ═══════════════════════════════════════════════════════════════════════════════

JUDGE_PROMPT = """你是一位苛刻但公正的沟通教练评审。下面是一段两人对话，以及一个"沟通现场弹窗"AI 生成的弹窗。

弹窗的设计意图：帮对话中的一方看见此刻还没看到的东西——看清自己 / 看见对方 / 看见两人的互动模式。认知中立、态度是帮助。

【对话】
{dialogue}

【弹窗】
{popup}

请按 6 个维度打分（1-5 整数，5 最好）：

1. accuracy 准确性：基于对话实际内容，不脑补、不张冠李戴；正确识别"你"是谁。
2. stance 立场：只对"你"一个人说话，不对双方各打五十大板；不站队、不评判对错。
3. length 字数控制：60-180 字，不超 200。
4. structure 结构：`——` 单独成行，墙前全部洞察、墙后只有一句建议。
5. tone 语气：口语化、像朋友轻声提醒、无术语、不说教、不贴标签。
6. actionability 可操作性：墙后建议具体可做，当场能用。

再写一段 80 字以内的 comment，指出最主要的问题或亮点（用于改进提示词，要具体）。
只输出 JSON:
{{"accuracy": 4, "stance": 5, "length": 4, "structure": 4, "tone": 5, "actionability": 3, "comment": "..."}}"""

FIX_JSON_PROMPT = "请把下面这段评审结果整理成严格 JSON 格式。只输出 JSON，不要解释。\n\n"


def judge_popup(
    dialogue: str, popup: str,
    model: str = "deepseek/deepseek-v4-pro",
    api_key: str | None = None,
    api_base: str | None = None,
) -> dict:
    hard = hard_check(popup)

    def _call_judge(prompt: str) -> str:
        kwargs = {"model": model, "temperature": 0.0, "max_tokens": 512, "timeout": 120}
        if api_key:
            kwargs["api_key"] = api_key
        if api_base:
            kwargs["api_base"] = api_base
        resp = litellm.completion(
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return (resp.choices[0].message.content or "").strip()

    def _extract(raw: str) -> dict | None:
        t = raw.strip()
        if t.startswith("```"):
            t = t.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        s, e = t.find("{"), t.rfind("}")
        if s == -1 or e == -1 or e <= s:
            return None
        try:
            return json.loads(t[s:e + 1])
        except json.JSONDecodeError:
            return None

    try:
        raw = _call_judge(JUDGE_PROMPT.format(dialogue=dialogue, popup=popup))
    except Exception as exc:
        logger.warning("judge API 失败: %s", exc)
        return {"score": 0.5, "dimensions": {}, "comment": f"API异常:{exc}",
                "hard": hard, "popup": popup}

    dims = _extract(raw)
    if dims is None:
        try:
            raw = _call_judge(FIX_JSON_PROMPT + raw)
            dims = _extract(raw)
        except Exception:
            dims = None

    if dims is None:
        dims = {"accuracy": 3, "stance": 3, "length": 3, "structure": 3,
                "tone": 3, "actionability": 3,
                "comment": f"[JSON解析失败] {raw[:100]}"}

    dim_keys = ["accuracy", "stance", "length", "structure", "tone", "actionability"]
    soft_mean = sum(dims.get(k, 3) for k in dim_keys) / len(dim_keys)
    penalty = min(1.5, 0.5 * len(hard["violations"]))
    score = round(max(0.0, soft_mean / 5.0 - penalty / 5.0), 4)

    return {
        "score": score,
        "dimensions": {k: dims.get(k, 3) for k in dim_keys},
        "comment": dims.get("comment", ""),
        "hard": hard,
        "popup": popup,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Task Runner
# ═══════════════════════════════════════════════════════════════════════════════


def generate_popup(
    dialogue: str, system_prompt: str,
    model: str = "deepseek/deepseek-v4-pro",
    api_key: str | None = None,
    api_base: str | None = None,
    temperature: float = 0.0, max_tokens: int = 2048,
) -> str:
    try:
        kwargs = {"model": model, "temperature": temperature, "max_tokens": max_tokens, "timeout": 120}
        if api_key:
            kwargs["api_key"] = api_key
        if api_base:
            kwargs["api_base"] = api_base
        resp = litellm.completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"当前对话：\n{dialogue}"},
            ],
            **kwargs,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.error("生成弹窗失败: %s", exc)
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# DSPy 胶水层
# ═══════════════════════════════════════════════════════════════════════════════


class PopupSignature(dspy.Signature):
    """Generate a communication-coach popup for a tense dialogue moment."""
    dialogue: str = dspy.InputField()
    popup: str = dspy.OutputField()


class PopupModule(dspy.Module):
    def __init__(self, system_prompt: str):
        super().__init__()
        self.generate = dspy.Predict(PopupSignature)
        self.generate.signature.instructions = system_prompt

    def forward(self, dialogue: str) -> dspy.Prediction:
        return self.generate(dialogue=dialogue)


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

# 模块级 judge 配置，由 SelfEvolutionRunner 初始化时注入。
# DSPy metric 函数签名固定，无法传额外参数，用此全局配置桥接。
_JUDGE_KWARGS: dict = {}


def _call_metric_judge(dialogue: str, popup: str) -> dict:
    """用注入的 judge 配置评分。"""
    return judge_popup(dialogue, popup, **_JUDGE_KWARGS)


def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """GEPA: 返回 Prediction(score, feedback)。文本反馈驱动 LLM 反思。

    DSPy 3.2 GEPA 要求 5 参数签名: (gold, pred, trace, pred_name, pred_trace)。
    """
    dialogue = gold.dialogue
    popup = getattr(pred, "popup", "") or ""
    if not popup or len(popup.strip()) < 10:
        return dspy.Prediction(score=0.0,
            feedback=f"[弹窗为空]: '{popup}'\n[改进]: 确保 prompt 对每条对话都生成完整弹窗")

    result = _call_metric_judge(dialogue, popup)
    dims = result["dimensions"]
    hard = result["hard"]
    dim_lines = ", ".join(f"{k}={v}" for k, v in dims.items())
    violations = "; ".join(hard["violations"]) if hard["violations"] else "无"

    feedback = (
        f"[弹窗]: {popup}\n"
        f"[6维评分]: {dim_lines}\n"
        f"[硬规则违规]: {violations}\n"
        f"[评语]: {result['comment']}\n"
        f"[改进方向]: "
        + (f"硬规则违规需优先修复: {violations}" if hard["violations"]
           else f"软维度短板: {min(dims, key=dims.get)} 得分最低, 需改进")
    )
    return dspy.Prediction(score=result["score"], feedback=feedback)


def scalar_metric(gold, pred, trace=None):
    """MIPROv2 / PDO: 返回标量分数 0-1。"""
    dialogue = gold.dialogue
    popup = getattr(pred, "popup", "") or ""
    if not popup or len(popup.strip()) < 10:
        return 0.0
    return _call_metric_judge(dialogue, popup)["score"]


# ═══════════════════════════════════════════════════════════════════════════════
# 失败模式分类
# ═══════════════════════════════════════════════════════════════════════════════

def classify_failure_mode(case_results: list[dict]) -> str:
    """从评估结果中提取主要失败模式。

    策略：聚合所有 case 的 6 维分数，返回平均分最低的维度名。
    不调额外 LLM——直接复用 metric 输出。
    """
    dim_sums: dict[str, float] = {}
    dim_counts: dict[str, int] = {}
    for r in case_results:
        for dim, val in r.get("dimensions", {}).items():
            dim_sums[dim] = dim_sums.get(dim, 0.0) + val
            dim_counts[dim] = dim_counts.get(dim, 0) + 1
    if not dim_sums:
        return "accuracy"  # 默认
    dim_avgs = {dim: dim_sums[dim] / dim_counts[dim] for dim in dim_sums}
    worst = min(dim_avgs, key=dim_avgs.get)
    return worst


def schedule_strategy(
    failure_mode: str,
    history: list[dict],
    default: str = "gepa_mipro",
) -> str:
    """失败模式 → 策略名。

    每个失败模式有 4 策略优先级序列。某策略被 discard 后自动轮换到下一个，
    同一失败模式的所有策略用尽后才收敛（而不是三轮就停）。
    """
    sequence = FAILURE_STRATEGY_SEQUENCE.get(failure_mode, [default])

    # 收集此 failure_mode 下已试过且 discard 的策略
    tried_discarded: set[str] = set()
    for h in history:
        if h.get("failure_mode") == failure_mode and not h.get("kept"):
            tried_discarded.add(h.get("strategy", ""))

    # 从序列中挑第一个未被 discard 的策略
    for candidate in sequence:
        if candidate not in tried_discarded:
            return candidate

    # 全部用尽，返回序列最后一项做最后一次尝试
    logger.warning("失败模式 '%s' 下所有策略均已尝试过，用 %s 做最后尝试",
                   failure_mode, sequence[-1])
    return sequence[-1]


@dataclass
class FailureProfile:
    worst_cases: list[dict] = field(default_factory=list)
    persistent_issues: list[str] = field(default_factory=list)
    previous_attempts: list[str] = field(default_factory=list)
    round_scores: list[float] = field(default_factory=list)

    def update(self, case_results: list[dict], notes: str = ""):
        self.worst_cases = sorted(case_results, key=lambda r: r["score"])[:5]
        if notes:
            self.previous_attempts.append(notes)

    def build_failure_report(self) -> str:
        lines = ["## 上一轮失败画像（仅供改进参考，不是 prompt 的一部分）\n"]
        lines.append("以下案例在上一轮评估中得分最低：\n")

        for i, case in enumerate(self.worst_cases, 1):
            lines.append(f"### 案例 {i}: {case.get('case_id', '?')} "
                        f"(得分 {case.get('score', '?')})")
            lines.append(f"- 弹窗: {case.get('popup', '')[:120]}")
            lines.append(f"- 评语: {case.get('comment', '')}")
            if case.get("hard", {}).get("violations"):
                lines.append(f"- 违规: {'; '.join(case['hard']['violations'])}")
            dims = case.get("dimensions", {})
            if dims:
                worst_dim = min(dims, key=dims.get)
                lines.append(f"- 最弱维度: {worst_dim}={dims[worst_dim]}")
            lines.append("")

        if self.previous_attempts:
            lines.append("## 已尝试的修改方向（不要重复！）\n")
            for attempt in self.previous_attempts[-3:]:
                lines.append(f"- {attempt}")
            lines.append("")

        lines.append("## 改进纪律（违反则本轮直接丢弃）\n")
        lines.append("1. 先修硬规则违规（字数/结构/立场），后修软洞察（语气/建议质量）")
        lines.append("2. 只改 1-3 处具体语句/段落，不允许推倒重写")
        lines.append("3. 必须原样保留：生命立场、三种误判场景、`——` 功能墙、安好静音逻辑")
        lines.append("4. 保持全文口语化、第二人称、无术语风格\n")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# SelfEvolutionRunner
# ═══════════════════════════════════════════════════════════════════════════════


class SelfEvolutionRunner:
    """双层闭环自进化运行器。

    外层: 失败模式分类 → 策略调度 → 执行内层 → 基线守护 → 收敛检测
    内层: DSPy 优化器内部各自的搜索循环 (GEPA / PDO / MIPROv2)
    """

    def __init__(
        self,
        baseline_path: str | Path,
        dataset_path: str | Path,
        task_model: str = "deepseek/deepseek-v4-pro",
        task_api_key: str | None = None,
        task_api_base: str | None = None,
        judge_model: str = "deepseek/deepseek-v4-pro",
        judge_api_key: str | None = None,
        judge_api_base: str | None = None,
        max_rounds: int = MAX_ROUNDS,
        patience: int = PATIENCE,
        output_dir: str | Path | None = None,
        seed: int = 42,
    ):
        self.baseline_path = Path(baseline_path)
        self.baseline_text = self.baseline_path.read_text(encoding="utf-8")
        self.dataset_path = Path(dataset_path)
        self.task_model_name = task_model
        self.task_api_key = task_api_key
        self.task_api_base = task_api_base
        self.judge_model_name = judge_model
        self.judge_api_key = judge_api_key
        self.judge_api_base = judge_api_base
        # 注入全局 judge 配置，供模块级 gepa_metric / scalar_metric 使用
        global _JUDGE_KWARGS
        _JUDGE_KWARGS = {"model": judge_model, "api_key": judge_api_key, "api_base": judge_api_base}
        self.max_rounds = max_rounds
        self.patience = patience
        self.seed = seed
        self.output_dir = Path(output_dir) if output_dir else RESULTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 状态
        self.baseline_eval: dict | None = None
        self.failure_profile = FailureProfile()
        self.history: list[dict] = []
        self.best_prompt_text = self.baseline_text
        self.best_score = 0.0
        self.current_round = 0
        self.no_improve = 0

        # 数据集
        raw = json.loads(self.dataset_path.read_text(encoding="utf-8"))
        self.cases = raw if isinstance(raw, list) else raw.get("examples", raw.get("data", []))
        # DSPy Example 集（所有优化器共用）
        self._trainset = [
            dspy.Example(dialogue=c["dialogue"]).with_inputs("dialogue")
            for c in self.cases[:9]
        ]
        self._valset = [
            dspy.Example(dialogue=c["dialogue"]).with_inputs("dialogue")
            for c in self.cases[9:12]
        ]

        self._setup_dspy()

    def _setup_dspy(self):
        lm = dspy.LM(
            model=self.task_model_name,
            temperature=0.0, max_tokens=2048, cache=False,
        )
        dspy.configure(lm=lm)

        # GEPA 反思模型: 使用强模型 (Claude Opus 4.7) 根据反馈提议新指令
        self._reflection_lm = dspy.LM(
            model=self.judge_model_name,
            temperature=1.0, max_tokens=32000, cache=False,
            api_key=self.judge_api_key,
            api_base=self.judge_api_base,
        )

    # ── 评估 ──────────────────────────────────────────────────────────────

    def _evaluate_prompt(self, prompt_text: str) -> dict:
        case_results = []
        for case in self.cases:
            dialogue = case.get("dialogue", "")
            popup = generate_popup(
                dialogue, prompt_text,
                model=self.task_model_name,
                api_key=self.task_api_key,
                api_base=self.task_api_base,
            )
            result = judge_popup(
                dialogue, popup,
                model=self.judge_model_name,
                api_key=self.judge_api_key,
                api_base=self.judge_api_base,
            )
            result["case_id"] = case.get("id", "?")
            result["expect"] = case.get("expect", "")
            case_results.append(result)

        scores = [r["score"] for r in case_results]
        return {
            "mean_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "scores": scores,
            "hard_violations": sum(len(r["hard"]["violations"]) for r in case_results),
            "cases": case_results,
            "prompt_length": len(prompt_text),
        }

    def _evaluate_baseline(self):
        logger.info("═══ Round 0: 基线评估 ═══")
        self.baseline_eval = self._evaluate_prompt(self.baseline_text)
        self.best_score = self.baseline_eval["mean_score"]
        logger.info("基线: mean=%.4f violations=%d length=%d",
                    self.best_score, self.baseline_eval["hard_violations"],
                    self.baseline_eval["prompt_length"])
        self._save_round_result(0, self.baseline_eval, "baseline", "")

    # ── 主优化器调度 ─────────────────────────────────────────────────────

    def _extract_instructions(self, optimized_program) -> str | None:
        """从 DSPy 优化结果中提取指令文本。"""
        try:
            inst = optimized_program.generate.signature.instructions
            if inst and len(inst.strip()) > 50:
                return inst.strip()
        except Exception:
            pass
        return None

    def _run_pdo(self, prompt_text: str) -> str:
        """PDO: pairwise 对决优化, 对 tone 判定更敏感。"""
        logger.info("  [PDO] pairwise 对比优化")
        program = PopupModule(prompt_text)

        try:
            optimizer = dspy.PDO(
                metric=scalar_metric,
                num_threads=2,
                track_stats=False,
                seed=self.seed,
            )
            optimized = optimizer.compile(
                program, trainset=self._trainset,
                eval_kwargs=dict(num_threads=2, display_progress=False),
            )
        except Exception as exc:
            logger.error("  [PDO] 编译失败: %s", exc)
            return prompt_text

        inst = self._extract_instructions(optimized)
        if inst:
            logger.info("  [PDO] 完成, 输出 %d 字", len(inst))
            return inst
        logger.warning("  [PDO] 无法提取指令，回退")
        return prompt_text

    def _run_gepa(self, prompt_text: str, max_full_evals: int = 7) -> str:
        """GEPA: 遗传进化 + LLM 反思。"""
        logger.info("  [GEPA] max_full_evals=%d", max_full_evals)
        program = PopupModule(prompt_text)

        try:
            optimizer = dspy.GEPA(
                metric=gepa_metric,
                reflection_lm=self._reflection_lm,
                max_full_evals=max_full_evals,
                num_threads=2,
                track_stats=False,
                seed=self.seed,
            )
            optimized = optimizer.compile(
                program, trainset=self._trainset, valset=self._valset,
            )
        except Exception as exc:
            logger.error("  [GEPA] 编译失败: %s", exc)
            return prompt_text

        inst = self._extract_instructions(optimized)
        if inst:
            logger.info("  [GEPA] 完成, 输出 %d 字", len(inst))
            return inst
        logger.warning("  [GEPA] 无法提取指令，回退")
        return prompt_text

    def _run_mipro(self, prompt_text: str, auto: str = "light") -> str:
        """MIPROv2: 贝叶斯联合优化指令 + few-shot。"""
        logger.info("  [MIPROv2] auto=%s", auto)
        program = PopupModule(prompt_text)

        try:
            optimizer = dspy.MIPROv2(
                metric=scalar_metric,
                auto=auto,
                max_bootstrapped_demos=4,
                max_labeled_demos=4,
                num_threads=2,
                init_temperature=0.7,
                seed=self.seed,
            )
            optimized = optimizer.compile(
                program,
                trainset=self._trainset,
                valset=self._valset,
                max_bootstrapped_demos=4,
                max_labeled_demos=4,
            )
        except Exception as exc:
            logger.error("  [MIPROv2] 编译失败: %s", exc)
            return prompt_text

        inst = self._extract_instructions(optimized)
        if inst:
            logger.info("  [MIPROv2] 完成, 输出 %d 字", len(inst))
            return inst
        logger.warning("  [MIPROv2] 无法提取指令，回退")
        return prompt_text

    # ── 决策 ──────────────────────────────────────────────────────────────

    def _should_keep(self, candidate_eval: dict) -> tuple[bool, str]:
        base = self.baseline_eval
        delta = round(candidate_eval["mean_score"] - base["mean_score"], 4)

        if delta <= 0.02:
            return False, f"Δ={delta} 未达阈值 0.02"
        if candidate_eval["hard_violations"] > base["hard_violations"]:
            return False, f"违规增加 ({base['hard_violations']}→{candidate_eval['hard_violations']})"
        ratio = candidate_eval["prompt_length"] / max(base["prompt_length"], 1)
        if ratio < 0.6:
            return False, f"长度崩溃 (保留率 {ratio:.0%})"

        return True, f"Δ={delta}, violations OK, length={ratio:.0%}"

    # ── 主循环 ────────────────────────────────────────────────────────────

    def run(self):
        logger.info("=" * 60)
        logger.info("双层闭环自进化 · 失败模式驱动的策略调度")
        logger.info("基线: %s", self.baseline_path.name)
        logger.info("策略目录: %s", ", ".join(STRATEGY_CATALOG))
        logger.info("失败→策略: %s", FAILURE_TO_STRATEGY)
        logger.info("收敛: patience=%d, max_rounds=%d", self.patience, self.max_rounds)
        logger.info("=" * 60)

        # Round 0: 基线
        self._evaluate_baseline()

        # 外层循环 — 失败模式驱动调度
        for rnd in range(1, self.max_rounds + 1):
            self.current_round = rnd
            t_start = time.perf_counter()

            # ① 失败模式分类
            failure_mode = classify_failure_mode(
                self.failure_profile.worst_cases if self.failure_profile.worst_cases
                else self.baseline_eval["cases"]
            )

            # ② 策略调度
            strategy_name = schedule_strategy(failure_mode, self.history)
            strategy = STRATEGY_CATALOG[strategy_name]
            logger.info("═══ Round %d ═══", rnd)
            logger.info("  失败模式: %s → 策略: %s (%s)",
                        failure_mode, strategy_name, strategy["desc"])

            # ③ 执行内层策略
            # DSPy 优化器内部有各自的反馈机制（GEPA 文本反馈 / MIPROv2 贝叶斯），
            # 直接传干净 prompt，不注入失败报告。
            current_prompt = self.best_prompt_text

            # Step 3a: 主优化器
            t0 = time.perf_counter()
            primary_name = strategy.get("primary")
            if primary_name == "gepa":
                kwargs = strategy["primary_kwargs"]
                after_primary = self._run_gepa(
                    current_prompt,
                    max_full_evals=kwargs.get("max_full_evals", 5),
                )
            elif primary_name == "pdo":
                after_primary = self._run_pdo(current_prompt)
            else:
                after_primary = current_prompt  # 跳过主优化器

            primary_elapsed = time.perf_counter() - t0
            primary_path = self.output_dir / f"round_{rnd:03d}_primary.txt"
            primary_path.write_text(after_primary, encoding="utf-8")

            # Step 3b: 第二阶段优化器（可选）
            t0 = time.perf_counter()
            secondary_name = strategy.get("secondary")
            if secondary_name == "mipro":
                after_secondary = self._run_mipro(after_primary, auto="light")
            elif secondary_name == "pdo":
                after_secondary = self._run_pdo(after_primary)
            else:
                after_secondary = after_primary
            secondary_elapsed = time.perf_counter() - t0

            secondary_path = self.output_dir / f"round_{rnd:03d}_secondary.txt"
            secondary_path.write_text(after_secondary, encoding="utf-8")

            # ④ 评估 & 基线守护
            candidate_eval = self._evaluate_prompt(after_secondary)

            # ⑤ 决策
            keep, reason = self._should_keep(candidate_eval)
            total_elapsed = time.perf_counter() - t_start
            logger.info("  mean=%.4f (基线 %.4f) → %s (%s) | %.1fs",
                        candidate_eval["mean_score"],
                        self.baseline_eval["mean_score"],
                        "✅ KEEP" if keep else "❌ DISCARD",
                        reason, total_elapsed)

            # 更新失败画像
            notes = (
                f"Round {rnd} [{strategy_name}]: "
                f"failure={failure_mode}, score={candidate_eval['mean_score']:.4f}, {reason}"
            )
            self.failure_profile.update(candidate_eval["cases"], notes)
            self.failure_profile.round_scores.append(candidate_eval["mean_score"])

            # 保存本轮结果
            round_result = {
                "round": rnd,
                "strategy": strategy_name,
                "strategy_desc": strategy["desc"],
                "failure_mode": failure_mode,
                "candidate_score": candidate_eval["mean_score"],
                "baseline_score": self.baseline_eval["mean_score"],
                "kept": keep,
                "reason": reason,
                "prompt_length": candidate_eval["prompt_length"],
                "hard_violations": candidate_eval["hard_violations"],
                "primary_elapsed_s": round(primary_elapsed, 1),
                "secondary_elapsed_s": round(secondary_elapsed, 1),
                "total_elapsed_s": round(total_elapsed, 1),
            }
            self.history.append(round_result)
            self._save_round_result(rnd, candidate_eval,
                                    "kept" if keep else "discarded",
                                    strategy_name)

            if keep:
                self.best_prompt_text = after_secondary
                self.best_score = candidate_eval["mean_score"]
                self.no_improve = 0
                best_path = self.output_dir / f"best_prompt_r{rnd}.txt"
                best_path.write_text(after_secondary, encoding="utf-8")
            else:
                self.no_improve += 1

            # ⑥ 收敛检查：当前失败模式下所有策略均已尝试？
            sequence = FAILURE_STRATEGY_SEQUENCE.get(failure_mode, [])
            tried_for_mode = {
                h["strategy"] for h in self.history
                if h.get("failure_mode") == failure_mode and not h.get("kept")
            }
            if len(tried_for_mode) >= len(sequence):
                logger.info("失败模式 '%s' 下 %d/%d 策略均已尝试，收敛停止。",
                            failure_mode, len(tried_for_mode), len(sequence))
                break

        self._save_final_report()

    # ── 持久化 ────────────────────────────────────────────────────────────

    def _save_round_result(self, rnd: int, eval_result: dict, tag: str, strategy: str):
        path = self.output_dir / f"round_{rnd:03d}_{tag}.json"
        serializable = {
            "round": rnd, "tag": tag, "strategy": strategy,
            "mean_score": eval_result["mean_score"],
            "hard_violations": eval_result["hard_violations"],
            "prompt_length": eval_result["prompt_length"],
            "cases": [
                {"case_id": c["case_id"], "score": c["score"],
                 "comment": c["comment"], "violations": c["hard"]["violations"]}
                for c in eval_result.get("cases", [])
            ],
        }
        path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2),
                       encoding="utf-8")

    def _save_final_report(self):
        report = {
            "baseline_prompt": str(self.baseline_path),
            "baseline_score": self.baseline_eval["mean_score"] if self.baseline_eval else None,
            "best_score": self.best_score,
            "total_rounds": len(self.history),
            "kept_rounds": sum(1 for h in self.history if h["kept"]),
            "strategy_effectiveness": {},
            "history": self.history,
        }
        # 按策略 + 失败模式统计效果
        for h in self.history:
            s = h["strategy"]
            fm = h.get("failure_mode", "?")
            key = f"{s}[{fm}]"
            if key not in report["strategy_effectiveness"]:
                report["strategy_effectiveness"][key] = {"kept": 0, "total": 0, "total_delta": 0.0}
            report["strategy_effectiveness"][key]["total"] += 1
            if h["kept"]:
                report["strategy_effectiveness"][key]["kept"] += 1
            report["strategy_effectiveness"][key]["total_delta"] += (
                h["candidate_score"] - h["baseline_score"]
            )

        path = self.output_dir / "final_report.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("最终报告: %s", path)

        print("\n" + "=" * 60)
        print("  失败模式驱动 · 自进化完成")
        print("=" * 60)
        print(f"  基线: {report['baseline_score']}")
        print(f"  最优: {report['best_score']}")
        print(f"  有效: {report['kept_rounds']}/{report['total_rounds']} 轮")
        print(f"  策略效果:")
        for key, stats in report["strategy_effectiveness"].items():
            avg_delta = stats["total_delta"] / max(stats["total"], 1)
            print(f"    {key}: {stats['kept']}/{stats['total']} kept, avg Δ={avg_delta:+.4f}")
        for h in self.history:
            flag = "✅" if h["kept"] else "❌"
            fm = h.get("failure_mode", "?")
            print(f"    R{h['round']} [{h['strategy']}] fm={fm}: {h['candidate_score']:.4f} {flag}")

    def save_state(self):
        state = {
            "baseline_path": str(self.baseline_path),
            "dataset_path": str(self.dataset_path),
            "baseline_eval": self.baseline_eval,
            "best_prompt_text": self.best_prompt_text,
            "best_score": self.best_score,
            "current_round": self.current_round,
            "no_improve": self.no_improve,
            "history": self.history,
            "failure_profile": {
                "worst_cases": self.failure_profile.worst_cases,
                "persistent_issues": self.failure_profile.persistent_issues,
                "previous_attempts": self.failure_profile.previous_attempts,
                "round_scores": self.failure_profile.round_scores,
            },
        }
        path = self.output_dir / "state.json"
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("状态: %s", path)
        return path

    @classmethod
    def from_state(cls, state_path: str | Path) -> "SelfEvolutionRunner":
        state = json.loads(Path(state_path).read_text(encoding="utf-8"))
        runner = cls(
            baseline_path=state["baseline_path"],
            dataset_path=state["dataset_path"],
        )
        runner.baseline_eval = state["baseline_eval"]
        runner.best_prompt_text = state["best_prompt_text"]
        runner.best_score = state["best_score"]
        runner.current_round = state["current_round"]
        runner.no_improve = state["no_improve"]
        runner.history = state["history"]
        fp = state["failure_profile"]
        runner.failure_profile = FailureProfile(
            worst_cases=fp["worst_cases"],
            persistent_issues=fp["persistent_issues"],
            previous_attempts=fp["previous_attempts"],
            round_scores=fp["round_scores"],
        )
        logger.info("从 %s 恢复 (round=%d, best=%.4f)",
                    state_path, runner.current_round, runner.best_score)
        return runner


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="双层闭环自进化 · 策略轮换式 Prompt 优化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python self_evolve.py --baseline system_prompt_v1.4.txt
  python self_evolve.py --baseline system_prompt_v1.4.txt --rounds 3
  python self_evolve.py --resume results/self_evolve/state.json
        """,
    )
    parser.add_argument("--baseline", default=str(HERE / "system_prompt_v1.4.txt"))
    parser.add_argument("--dataset", default=str(HERE / "dataset.json"))
    parser.add_argument("--rounds", type=int, default=MAX_ROUNDS,
                        help=f"最大外层轮数 (默认: {MAX_ROUNDS})")
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--task-model", default="deepseek/deepseek-v4-pro")
    parser.add_argument("--task-api-key", default=None, help="生成模型 API key")
    parser.add_argument("--task-api-base", default=None, help="生成模型 API base URL")
    parser.add_argument("--judge-model", default="deepseek/deepseek-v4-pro")
    parser.add_argument("--judge-api-key", default=None, help="裁判模型 API key")
    parser.add_argument("--judge-api-base", default=None, help="裁判模型 API base URL")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default=None, help="从 state.json 断点续跑")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if not os.environ.get("DEEPSEEK_API_KEY"):
        logger.error("缺少 DEEPSEEK_API_KEY 环境变量")
        sys.exit(1)

    if not Path(args.baseline).exists():
        logger.error("基线文件不存在: %s", args.baseline)
        sys.exit(1)
    if not Path(args.dataset).exists():
        logger.error("数据集不存在: %s", args.dataset)
        sys.exit(1)

    if args.resume:
        runner = SelfEvolutionRunner.from_state(args.resume)
    else:
        output_dir = args.output_dir or (
            RESULTS_DIR / time.strftime("%Y%m%d_%H%M%S"))
        # DeepSeek key from env or CLI
        task_api_key = args.task_api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        judge_api_key = args.judge_api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
        runner = SelfEvolutionRunner(
            baseline_path=args.baseline,
            dataset_path=args.dataset,
            task_model=args.task_model,
            task_api_key=task_api_key,
            task_api_base=args.task_api_base,
            judge_model=args.judge_model,
            judge_api_key=judge_api_key,
            judge_api_base=args.judge_api_base,
            max_rounds=args.rounds,
            patience=args.patience,
            output_dir=output_dir,
            seed=args.seed,
        )

    try:
        runner.run()
    except KeyboardInterrupt:
        logger.info("中断, 保存状态...")
        runner.save_state()
        sys.exit(0)
    except Exception as exc:
        logger.exception("异常: %s", exc)
        runner.save_state()
        sys.exit(1)

    runner.save_state()


if __name__ == "__main__":
    main()
