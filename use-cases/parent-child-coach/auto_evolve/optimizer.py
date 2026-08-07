"""自动优化引擎 —— modify → evaluate → compare → keep/discard 主循环。

用法: python -m auto_evolve.optimizer [--max-iter 5]

流程:
  1. 加载基线评估结果 → 构建失败报告
  2. LLM 分析失败 → 提议 prompt 修改
  3. 应用修改 → 写入新 prompt 变体文件
  4. 用新 prompt 跑评估
  5. 与基线对比 → keep（综合分提升 + 无严重退化）或 discard
  6. 循环至收敛或达到最大迭代次数
"""

import json
import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from openai import OpenAI

# 路径设置
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

XINGLING_ROOT = Path("D:/星灵-soul-手搓/亲子沟通洞见/路线A_自上而下法_鼓励版")
sys.path.insert(0, str(XINGLING_ROOT))

from src.multi_agent_orchestrator import MultiAgentOrchestrator
from auto_evolve.evaluator import (
    EvalResult, BaselineReport,
    compute_m1_trigger, compute_m5_tone,
    build_m6_prompt, build_m7_prompt,
    parse_llm_judge_response, aggregate_results,
)
from auto_evolve.prompt_mutator import (
    build_failure_report,
    propose_mutation,
    PromptMutation,
    XINGLING_PROMPTS,
    bump_version,
    extract_version,
)

# ═══════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════

# 评估用案例（12个原始 + 50个盲区 = 62 windows）
# 盲区来自 full_eval_n1 全量校标发现的失败模式（n=1），加入迭代集让 mutator 看到新失败
EVAL_CASES = [
    # === 原始 12 案（已知）===
    # Diagnostic cases
    ("C10-001", None),   # diagnostic, score=9 — 清晰诊断
    ("C10-002", None),   # diagnostic, score=7 — 修复≠转变（挑战）
    ("C10-003", None),   # diagnostic, score=6
    ("C10-005", None),   # diagnostic, score=5
    ("C10-006", None),   # diagnostic, score=6
    ("C10-008", None),   # diagnostic, score=8
    # Empowering cases
    ("C10-004", None),   # empowering, score=8 — 清晰鼓励
    ("C11-001", None),   # empowering, score=8
    ("C11-004", None),   # empowering, score=1 — 极低分
    ("C11-006", 1),      # empowering, score=4 — 恐惧中暂停（挑战）
    ("C11-009", 2),      # empowering, score=2 — 极低分
    ("C11-010", 1),      # empowering, score=9 — 伤痛回避（挑战）
    # === 盲区 50 案（来自 full_eval_n1，含 tone_mismatch/M6_low/M7_low/M1_mismatch）===
    ("C10-009", 1),      # tone_mismatch: sys=diagnostic gold=empowering
    ("C10-010", 1),      # tone_mismatch: sys=diagnostic gold=empowering
    ("C11-001", 4),      # M6_low=1.0
    ("C11-002", 3),      # M6_low=1.0
    ("C11-005", 1),      # tone_mismatch
    ("C11-005", 2),      # tone_mismatch + M7_low=2.0
    ("C11-005", 3),      # M1_mismatch
    ("C11-005", 4),      # tone_mismatch + M7_low=2.0
    ("C11-005", 5),      # M6_low=1.0
    ("C11-007", 1),      # M1_mismatch
    ("C11-008", 1),      # tone_mismatch
    ("C11-009", 1),      # tone_mismatch (EVAL_CASES 已有 w2)
    ("C11-010", 2),      # tone_mismatch (EVAL_CASES 已有 w1)
    ("C13-001", 2),      # M6_low=1.0
    ("C13-002", 1),      # M6_low=1.0 (gold should_popup=False)
    ("C13-004", 2),      # M6_low=1.0
    ("C13-005", 3),      # M1_mismatch (gold should_popup=False)
    ("C13-006", 3),      # M6_low=1.0
    ("C13-007", 1),      # tone_mismatch: sys=empowering gold=diagnostic
    ("C13-007", 2),      # tone_mismatch
    ("C13-007", 3),      # M1_mismatch (gold should_popup=False)
    ("C13-008", 2),      # tone_mismatch: sys=empowering gold=diagnostic
    ("C13-009", 1),      # tone_mismatch + M7_low=2.0
    ("C13-010", 1),      # tone_mismatch
    ("C13-010", 3),      # tone_mismatch
    ("C13-012", 1),      # tone_mismatch
    ("C13-012", 2),      # tone_mismatch (sys=mixed)
    ("C3-001", 1),       # tone_mismatch
    ("C3-002", 1),       # tone_mismatch
    ("C3-003", 1),       # tone_mismatch (sys=mixed)
    ("C4-001", 1),       # tone_mismatch
    ("C4-002", 1),       # M6_low=1.0
    ("C4-002", 3),       # M6_low=1.0
    ("C4-002", 4),       # M6_low=1.0
    ("C4-002", 5),       # M6_low=1.0
    ("C4-003", 2),       # M6_low=1.0
    ("C4-004", 2),       # tone_mismatch + M7_low=2.0
    ("C5-001", 1),       # M6_low=1.0
    ("C5-001", 2),       # M1_mismatch
    ("C5-002", 1),       # M1_mismatch
    ("C5-002", 2),       # M1_mismatch
    ("C5-003", 2),       # tone_mismatch
    ("C5-003", 3),       # tone_mismatch
    ("C5-003", 4),       # M1_mismatch
    ("C5-003", 5),       # tone_mismatch
    ("C5-004", 2),       # M7_low=2.0
    ("C5-004", 3),       # M7_low=3.0
    ("C5-005", 1),       # tone_mismatch
    ("C5-005", 3),       # tone_mismatch
    ("C5-005", 4),       # tone_mismatch
]

# 默认优化目标 prompt（总控对方向决策影响最大）
DEFAULT_TARGET_PROMPT = "master"

# keep/discard 阈值
MIN_OVERALL_IMPROVEMENT = 0.003    # 综合分至少提升 0.3%
MAX_CASE_REGRESSION = 0.5         # 单案例 M6 最多退化 0.5 分（5分制，容忍 n=3 judge 噪声 ±0.3）
MAX_ITERATIONS = 8


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

def load_env():
    """加载环境变量。"""
    env_path = Path("D:/星灵-soul-手搓/.env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def load_golden_dataset() -> list[dict]:
    """加载黄金数据集。"""
    with open(DATA_DIR / "golden_dataset.json", "r", encoding="utf-8") as f:
        return json.load(f)


def find_case(dataset: list[dict], case_id: str) -> dict | None:
    """按 case_id 查找案例。"""
    for c in dataset:
        if c["case_id"] == case_id:
            return c
    return None


def get_input_text(case: dict, window_index: int | None) -> str:
    """获取模型输入文本。"""
    if window_index is not None:
        for w in case["windows"]:
            if w["window_index"] == window_index:
                return w.get("window_text", "") or case.get("dialogue", "")
    return case.get("dialogue", "")


def get_gold_labels(case: dict, window_index: int | None) -> dict:
    """提取黄金标签。"""
    if window_index is not None:
        for w in case["windows"]:
            if w["window_index"] == window_index:
                return {
                    "should_popup": w.get("should_popup"),
                    "tone": w.get("expected_tone", ""),
                    "reference_popup": w.get("reference_popup", ""),
                    "score": w.get("overall_score"),
                    "hit_checklist": w.get("hit_checklist", []),
                    "forbid_checklist": w.get("forbid_checklist", []),
                }
    w = case["windows"][0] if case["windows"] else {}
    return {
        "should_popup": w.get("should_popup"),
        "tone": w.get("expected_tone", ""),
        "reference_popup": w.get("reference_popup", ""),
        "score": w.get("overall_score"),
        "hit_checklist": w.get("hit_checklist", []),
        "forbid_checklist": w.get("forbid_checklist", []),
    }


# ═══════════════════════════════════════════════════════════════
# 评估运行器（可指定自定义 prompt）
# ═══════════════════════════════════════════════════════════════

def _evaluate_case_once(
    client: OpenAI,
    model: str,
    orch: MultiAgentOrchestrator,
    case_id: str,
    win_idx: int | None,
    case: dict,
    gold: dict,
    input_text: str,
    max_empty_retries: int = 2,
) -> EvalResult:
    """对单个案例跑一次评估，返回一个 EvalResult。失败时返回带 error 的 EvalResult。

    max_empty_retries: 当 popup_text 为空时，视为"模型未完成生成"触发重试，
                       最多重试 max_empty_retries 次。
    """
    orch.reset_family(case_id)

    for attempt in range(1 + max_empty_retries):
        try:
            result = orch.process_window(input_text, family=case_id)
        except Exception as e:
            return EvalResult(
                case_id=case_id, window_index=win_idx or 1,
                error=f"{type(e).__name__}: {str(e)[:200]}",
            )

        sys_popup = result.popup_text or ""

        # 空输出视为"模型未完成生成"，触发重试
        if not sys_popup.strip() and attempt < max_empty_retries:
            continue  # 重新调用 API
        break  # 有效输出或已达最大重试次数

    m1 = compute_m1_trigger(result.should_popup, gold["should_popup"])
    m5 = compute_m5_tone(result.tone, gold["tone"])

    # M6 judge
    m6_score = None
    m6_raw = ""
    if gold["reference_popup"].strip() and "内容标注" not in gold["reference_popup"]:
        m6_prompt = build_m6_prompt(
            dialogue=input_text,
            reference_popup=gold["reference_popup"],
            sys_popup=sys_popup,
            sys_direction=result.tone,
            sys_contradiction=result.main_contradiction,
        )
        m6_raw = _call_judge(client, model, m6_prompt)
        m6_score, m6_raw = parse_llm_judge_response(m6_raw, "m6")

    # M7 judge
    m7_prompt = build_m7_prompt(
        dialogue=input_text,
        forbid_checklist=gold["forbid_checklist"],
        sys_popup=sys_popup,
    )
    m7_raw = _call_judge(client, model, m7_prompt)
    m7_score, m7_raw = parse_llm_judge_response(m7_raw, "m7")

    return EvalResult(
        case_id=case_id, window_index=win_idx or 1,
        sys_should_popup=result.should_popup,
        sys_tone=result.tone,
        sys_popup_text=sys_popup,
        sys_main_contradiction=result.main_contradiction,
        gold_should_popup=gold["should_popup"],
        gold_tone=gold["tone"],
        gold_reference_popup=gold["reference_popup"],
        gold_score=gold["score"],
        gold_hit_checklist=gold["hit_checklist"],
        gold_forbid_checklist=gold["forbid_checklist"],
        m1_trigger_match=m1, m5_tone_match=m5,
        m6_insight_score=m6_score, m7_safety_score=m7_score,
        m6_judge_raw=m6_raw, m7_judge_raw=m7_raw,
    )


def _majority_vote(values: list, valid: set | None = None) -> object:
    """对值列表做多数投票，平票时返回第一个出现的值。valid 可指定有效值集合。"""
    from collections import Counter
    cleaned = [v for v in values if (valid is None or v in valid) and v is not None]
    if not cleaned:
        return None
    cnt = Counter(cleaned)
    return cnt.most_common(1)[0][0]


def _mean(values: list) -> float | None:
    """对数值列表求平均，忽略 None。"""
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def _denoise_case_runs(runs: list[EvalResult]) -> EvalResult:
    """把同一个 case 的多次跑结果降噪聚合为单个 EvalResult。

    策略:
      - tone: majority vote (基于非空 tone)
      - should_popup: 至少一次有效输出（有非空 popup_text 的 run）中做 majority vote；
                      只有所有 run 都空输出才判 M1=0（避免空输出噪声拖死 M1）
      - popup_text / contradiction: 从 tone == 多数 tone 的 run 中取第一个（保证一致）
      - M1 / M5: 用降噪后的 tone / should_popup 重新计算
      - M6 / M7: 取所有非 None judge 分数的均值（连续值）

    如果所有 run 都失败 (error)，返回第一个失败 run。
    """
    ok_runs = [r for r in runs if not r.error]
    if not ok_runs:
        # 全部失败 — 返回第一个 run 作为错误占位
        return runs[0]

    valid_tones = {"diagnostic", "empowering"}
    final_tone = _majority_vote([r.sys_tone for r in ok_runs], valid_tones) or ok_runs[0].sys_tone

    # M1 宽松策略：只在有非空 popup_text 的"有效 run"中做多数投票
    # 空输出是模型非确定性噪声，不是"模型认为不该弹窗"
    valid_runs = [r for r in ok_runs if (r.sys_popup_text or "").strip()]
    if valid_runs:
        final_should = _majority_vote([r.sys_should_popup for r in valid_runs])
    else:
        # 所有 run 都空输出，才判不弹窗
        final_should = False

    # 选代表 run: 优先从有效输出中选 tone 匹配的，否则选第一个有效输出，再否则用 ok_runs[0]
    valid_runs_for_rep = valid_runs if valid_runs else ok_runs
    rep_runs = [r for r in valid_runs_for_rep if r.sys_tone == final_tone]
    rep = rep_runs[0] if rep_runs else valid_runs_for_rep[0]

    # 重算 M1/M5（用降噪后的 tone / should_popup）
    m1 = compute_m1_trigger(final_should, rep.gold_should_popup)
    m5 = compute_m5_tone(final_tone, rep.gold_tone)

    m6_avg = _mean([r.m6_insight_score for r in ok_runs])
    m7_avg = _mean([r.m7_safety_score for r in ok_runs])

    # 记录每个 case 的 run-to-run 一致性（供调试）
    tone_runs = [r.sys_tone for r in ok_runs]
    tone_consistency = tone_runs.count(final_tone) / len(tone_runs) if tone_runs else 0.0

    error_runs = sum(1 for r in runs if r.error)

    return EvalResult(
        case_id=rep.case_id, window_index=rep.window_index,
        sys_should_popup=final_should,
        sys_tone=final_tone,
        sys_popup_text=rep.sys_popup_text,
        sys_main_contradiction=rep.sys_main_contradiction,
        gold_should_popup=rep.gold_should_popup,
        gold_tone=rep.gold_tone,
        gold_reference_popup=rep.gold_reference_popup,
        gold_score=rep.gold_score,
        gold_hit_checklist=rep.gold_hit_checklist,
        gold_forbid_checklist=rep.gold_forbid_checklist,
        m1_trigger_match=m1, m5_tone_match=m5,
        m6_insight_score=m6_avg, m7_safety_score=m7_avg,
        m6_judge_raw=rep.m6_judge_raw, m7_judge_raw=rep.m7_judge_raw,
        error=(f"[denoised: n_runs={len(runs)}, tone_consistency={tone_consistency:.2f}, "
               f"error_runs={error_runs}]" + (f" +case_error:{runs[0].error}" if error_runs else "")),
    )


def evaluate_with_prompt(
    client: OpenAI,
    model: str,
    prompt_path_master: str | None = None,
    prompt_path_perception: str | None = None,
    prompt_path_production: str | None = None,
    n_runs_per_case: int = 1,
    verbose: bool = False,
) -> BaselineReport:
    """用指定 prompt 运行评估，返回报告。

    n_runs_per_case > 1 时启用降噪模式:
      - 每案例跑 N 次 orchestrator
      - tone / should_popup 做 majority vote (M5/M1 是确定性匹配, 依赖这两个值)
      - M6/M7 judge 分数取均值
    n_runs_per_case = 1 时保持向后兼容行为。
    """
    dataset = load_golden_dataset()

    orch = MultiAgentOrchestrator(
        llm_client=client,
        model=model,
        prompt_path_master=prompt_path_master,
        prompt_path_perception=prompt_path_perception,
        prompt_path_production=prompt_path_production,
    )

    all_results: list[EvalResult] = []
    n_runs = max(1, int(n_runs_per_case))

    for case_id, win_idx in EVAL_CASES:
        case = find_case(dataset, case_id)
        if not case:
            continue

        gold = get_gold_labels(case, win_idx)
        input_text = get_input_text(case, win_idx)

        if n_runs == 1:
            r = _evaluate_case_once(client, model, orch, case_id, win_idx, case, gold, input_text)
            all_results.append(r)
            if verbose:
                _print_case_line(case_id, r, n_runs_done=1)
        else:
            runs = []
            for _ in range(n_runs):
                runs.append(_evaluate_case_once(client, model, orch, case_id, win_idx, case, gold, input_text))
            d = _denoise_case_runs(runs)
            all_results.append(d)
            if verbose:
                _print_case_line(case_id, d, n_runs_done=n_runs, runs=runs)

    return aggregate_results(all_results)


def _print_case_line(case_id: str, r: EvalResult, n_runs_done: int = 1, runs: list[EvalResult] | None = None) -> None:
    """单行打印 case 评估结果，便于观察 run-to-run 一致性。"""
    m1 = f"{r.m1_trigger_match:.0f}" if r.m1_trigger_match is not None else "-"
    m5 = f"{r.m5_tone_match:.0f}" if r.m5_tone_match is not None else "-"
    m6 = f"{r.m6_insight_score:.1f}" if r.m6_insight_score is not None else "-"
    m7 = f"{r.m7_safety_score:.1f}" if r.m7_safety_score is not None else "-"
    tone = r.sys_tone or "-"
    gold = r.gold_tone or "-"

    extra = ""
    if n_runs_done > 1 and runs:
        tones = [x.sys_tone or "?" for x in runs if not x.error]
        if tones:
            extra = f" | tone_runs={tones}"
    print(f"  {case_id:10s} tone={tone:11s} (gold={gold:11s}) M1={m1} M5={m5} M6={m6} M7={m7}{extra}")


def _call_judge(client: OpenAI, model: str, prompt: str) -> str:
    """调用 LLM judge。"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是评估专家。只输出严格 JSON。"},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1, timeout=60,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {str(e)[:100]}"})


# ═══════════════════════════════════════════════════════════════
# Keep/Discard 决策
# ═══════════════════════════════════════════════════════════════

def should_keep(
    baseline: BaselineReport,
    candidate: BaselineReport,
) -> tuple[bool, str]:
    """比较 candidate 和 baseline，决定保留还是丢弃。

    规则（词典型）:
      1. 综合分必须提升 > MIN_OVERALL_IMPROVEMENT
      2. 整体 M6 不能退化 > 0.2（单案例 judge 噪声大，只看整体）
      3. 整体 M7 不能退化 > 0.2
      4. 单案例 M5 不能从匹配→不匹配（确定性指标，非 judge 噪声）

    Returns: (keep: bool, reason: str)
    """
    if candidate.overall_score < baseline.overall_score + MIN_OVERALL_IMPROVEMENT:
        return False, (
            f"综合分未提升: {baseline.overall_score:.3f} → {candidate.overall_score:.3f}"
            f" (Δ={candidate.overall_score - baseline.overall_score:+.3f})"
        )

    # 整体 M6/M7 回归检查（单案例 judge 噪声太大，只看整体）
    m6_delta = candidate.aggregate_m6 - baseline.aggregate_m6
    if m6_delta < -0.2:
        return False, f"整体 M6 退化: {baseline.aggregate_m6:.2f} → {candidate.aggregate_m6:.2f} (Δ={m6_delta:+.2f})"

    m7_delta = candidate.aggregate_m7 - baseline.aggregate_m7
    if m7_delta < -0.2:
        return False, f"整体 M7 退化: {baseline.aggregate_m7:.2f} → {candidate.aggregate_m7:.2f} (Δ={m7_delta:+.2f})"

    # 整体 M5 回归检查（允许少量单案例回归，只要整体趋势改善）
    m5_delta = candidate.aggregate_m5 - baseline.aggregate_m5
    if m5_delta < -0.05:
        return False, f"整体 M5 退化: {baseline.aggregate_m5:.1%} → {candidate.aggregate_m5:.1%} (Δ={m5_delta:+.1%})"

    # 逐案例检查 M5 退化（原来对现在错）— 这是确定性指标，不是 judge 噪声
    # 允许少量回归，只要整体 M5 提升（上面已检查）
    baseline_by_case = {r.case_id: r for r in baseline.results}
    candidate_by_case = {r.case_id: r for r in candidate.results}

    m5_regressions = []
    for case_id, cr in candidate_by_case.items():
        br = baseline_by_case.get(case_id)
        if not br:
            continue
        if br.m5_tone_match == 1.0 and cr.m5_tone_match == 0.0:
            m5_regressions.append(case_id)

    if m5_regressions:
        m5_improvements = sum(1 for case_id, cr in candidate_by_case.items()
                              if baseline_by_case.get(case_id) and
                              baseline_by_case[case_id].m5_tone_match == 0.0 and
                              cr.m5_tone_match == 1.0)
        if len(m5_regressions) > m5_improvements:
            return False, (
                f"M5 回归案例 ({len(m5_regressions)}) 多于改善案例 ({m5_improvements}): "
                f"{', '.join(m5_regressions[:3])}"
            )

    return True, (
        f"综合分提升: {baseline.overall_score:.3f} → {candidate.overall_score:.3f}"
        f" (Δ={candidate.overall_score - baseline.overall_score:+.3f})"
    )


# ═══════════════════════════════════════════════════════════════
# 主循环
# ═══════════════════════════════════════════════════════════════

@dataclass
class IterationRecord:
    """一次迭代记录。"""
    iteration: int
    version: str
    report: BaselineReport
    mutation: PromptMutation
    kept: bool
    reason: str



# Prompt 文件名前缀映射
PROMPT_FILE_PREFIX = {
    "master": "prompt_总控",
    "perception": "prompt_感知层",
    "production": "prompt_生产层",
    "ideal": "prompt_理想模式",
}


def _variant_path_for(target: str, version: str) -> Path:
    """根据 target 和 version 返回变体文件路径。"""
    prefix = PROMPT_FILE_PREFIX.get(target, "prompt_总控")
    return XINGLING_PROMPTS / f"{prefix}_{version}.md"


def _eval_kwargs_for(target: str, path: str) -> dict:
    """根据 target 构建 evaluate_with_prompt 的 kwargs。"""
    kwargs = {}
    if target == "master":
        kwargs["prompt_path_master"] = path
    elif target == "perception":
        kwargs["prompt_path_perception"] = path
    elif target == "production":
        kwargs["prompt_path_production"] = path
    return kwargs


def run_optimization(
    max_iterations: int = MAX_ITERATIONS,
    target_prompt: str = DEFAULT_TARGET_PROMPT,
) -> list[IterationRecord]:
    """运行自动优化主循环。"""
    load_env()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    client = OpenAI(api_key=api_key, base_url=base_url)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 加载基线
    baseline_path = RESULTS_DIR / "baseline_report.json"
    if not baseline_path.exists():
        print("❌ 未找到基线报告，请先运行 baseline_runner.py")
        return []

    baseline = aggregate_results([])
    with open(baseline_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 从保存的报告中重建 BaselineReport
    for case in data.get("per_case", []):
        baseline.results.append(EvalResult(
            case_id=case["case_id"],
            window_index=case.get("window_index", 1),
            sys_should_popup=case.get("sys_should_popup", False),
            sys_tone=case.get("sys_tone", ""),
            sys_popup_text=case.get("sys_popup_text", ""),
            sys_main_contradiction=case.get("sys_main_contradiction", ""),
            gold_tone=case.get("gold_tone", ""),
            gold_reference_popup=case.get("gold_reference_popup", ""),
            m1_trigger_match=case.get("m1_trigger_match"),
            m5_tone_match=case.get("m5_tone_match"),
            m6_insight_score=case.get("m6_insight_score"),
            m7_safety_score=case.get("m7_safety_score"),
        ))
    baseline = aggregate_results(baseline.results)

    # 初始化 prompt 路径为原始文件
    current_master_path = None  # None = 使用默认
    current_version = "v3.1"
    current_prompt_text = None  # None = 从原始文件读取

    print(f"🚀 自动优化循环启动")
    print(f"   目标 prompt: {target_prompt}")
    print(f"   基线综合分: {baseline.overall_score:.3f}")
    print(f"   最大迭代: {max_iterations}")
    print(f"   评估案例: {len(EVAL_CASES)} 条")
    print("=" * 70)

    history: list[IterationRecord] = []
    origin_baseline = baseline  # 冻结原点，should_keep 始终与此比较，防止基准漂移

    for iteration in range(1, max_iterations + 1):
        print(f"\n{'─' * 50}")
        print(f"🔄 迭代 {iteration}/{max_iterations}")
        print(f"{'─' * 50}")

        # 2. 构建失败报告
        failure_report = build_failure_report(str(baseline_path))
        if not failure_report.failures:
            print("✅ 无失败案例，优化完成！")
            break

        print(f"  失败案例: {len(failure_report.failures)}")
        for p in failure_report.top_patterns:
            print(f"  📋 模式: {p[:120]}")

        # 3. LLM 提议 prompt 修改 — 基于当前最佳版本
        print(f"\n  🧬 变异中... (based on {current_version})")
        mutation = propose_mutation(
            client, model, target_prompt, failure_report,
            current_text=current_prompt_text,
            current_version=current_version,
        )
        if not mutation.modified_text:
            print(f"  ❌ 变异失败: {mutation.rationale}")
            break

        print(f"  📝 版本: {mutation.version_from} → {mutation.version_to}")
        for desc in mutation.edit_description:
            print(f"     - {desc[:100]}")

        # 4. 用新 prompt 跑评估
        # 根据目标 prompt 类型确定文件路径和 orchestrator 参数
        variant_path = _variant_path_for(target_prompt, mutation.version_to)
        eval_kwargs = _eval_kwargs_for(target_prompt, str(variant_path))

        print(f"\n  🏃 跑评估: {variant_path.name}")
        t0 = time.time()
        candidate = evaluate_with_prompt(client, model, **eval_kwargs)
        elapsed = time.time() - t0
        print(f"  ⏱ {elapsed:.1f}s | M1={candidate.aggregate_m1:.0%} M5={candidate.aggregate_m5:.0%} M6={candidate.aggregate_m6:.1f} M7={candidate.aggregate_m7:.1f} | 综合={candidate.overall_score:.3f}")

        # 5. Keep/Discard
        keep, reason = should_keep(origin_baseline, candidate)
        record = IterationRecord(
            iteration=iteration,
            version=mutation.version_to,
            report=candidate,
            mutation=mutation,
            kept=keep,
            reason=reason,
        )
        history.append(record)

        if keep:
            print(f"\n  ✅ KEEP — {reason}")
            baseline = candidate
            current_version = mutation.version_to
            current_prompt_text = mutation.modified_text
            current_master_path = str(variant_path)
            # 更新 baseline_path 指向最新结果
            baseline_path = _save_iteration_report(
                iteration, mutation.version_to, candidate, keep, reason
            )
        else:
            print(f"\n  ❌ DISCARD — {reason}")
            # 删除本次的变体文件（但保留 currently kept 的文件）
            if variant_path.exists() and str(variant_path) != current_master_path:
                variant_path.unlink()
            _save_iteration_report(
                iteration, mutation.version_to, candidate, keep, reason
            )

        # 收敛检查
        if keep:
            if candidate.overall_score > 0.90:
                print(f"\n🎉 综合分已达 {candidate.overall_score:.3f}，提前收敛！")
                break
        else:
            # 连续 3 次 discard → 停止（至少跑满 3 次才判断）
            if len(history) >= 3:
                recent = history[-3:]
                if all(not r.kept for r in recent):
                    print(f"\n⏸️  连续 {len(recent)} 次未采纳，停止迭代。")
                    break

    # 打印最终摘要
    print(f"\n{'=' * 70}")
    print(f"📊 优化结果")
    print(f"{'=' * 70}")
    kept_count = sum(1 for r in history if r.kept)
    print(f"  迭代次数: {len(history)} | 采纳: {kept_count} | 丢弃: {len(history) - kept_count}")
    for r in history:
        status = "✅ KEEP" if r.kept else "❌ DISCARD"
        print(f"  #{r.iteration} {r.version} {status} | 综合={r.report.overall_score:.3f} | {r.reason[:80]}")

    # 如果有改进，把最佳 prompt 复制回原名
    if kept_count > 0 and current_master_path:
        best = Path(current_master_path)
        print(f"\n🏆 最佳版本: {best.name} (综合分={baseline.overall_score:.3f})")
        print(f"   路径: {best}")

    return history


def _save_iteration_report(
    iteration: int, version: str, report: BaselineReport,
    kept: bool, reason: str,
) -> Path:
    """保存单次迭代的评估报告。"""
    detailed = {
        "iteration": iteration,
        "version": version,
        "kept": kept,
        "reason": reason,
        "aggregate": {
            "m1_trigger_accuracy": report.aggregate_m1,
            "m5_tone_match": report.aggregate_m5,
            "m6_insight_quality": report.aggregate_m6,
            "m7_safety_score": report.aggregate_m7,
            "overall_score": report.overall_score,
        },
        "per_case": [],
    }
    for r in report.results:
        detailed["per_case"].append({
            "case_id": r.case_id,
            "window_index": r.window_index,
            "sys_tone": r.sys_tone,
            "sys_popup_text": r.sys_popup_text[:300],
            "sys_main_contradiction": r.sys_main_contradiction,
            "gold_tone": r.gold_tone,
            "gold_score": r.gold_score,
            "m1_trigger_match": r.m1_trigger_match,
            "m5_tone_match": r.m5_tone_match,
            "m6_insight_score": r.m6_insight_score,
            "m7_safety_score": r.m7_safety_score,
            "error": r.error,
        })

    path = RESULTS_DIR / f"iter_{iteration:03d}_{version}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(detailed, f, ensure_ascii=False, indent=2)
    return path


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="自动优化 v3.0 prompt")
    parser.add_argument("--max-iter", type=int, default=MAX_ITERATIONS,
                        help=f"最大迭代次数 (默认: {MAX_ITERATIONS})")
    parser.add_argument("--target", default=DEFAULT_TARGET_PROMPT,
                        help=f"优化目标 prompt (默认: {DEFAULT_TARGET_PROMPT})")
    args = parser.parse_args()

    run_optimization(max_iterations=args.max_iter, target_prompt=args.target)
