"""基线评估运行器 —— 在黄金数据集上跑星灵多智能体 v3.1 架构，输出基线指标。

用法: python -m auto_evolve.baseline_runner

输出:
  - results/baseline_report.json: 详细评估结果
  - results/baseline_summary.json: 聚合指标
"""

import json
import os
import sys
import time
from pathlib import Path
from openai import OpenAI

# 路径设置
ROOT = Path(__file__).parent.parent  # parent-child-coach/
AUTO_EVOLVE = ROOT / "auto_evolve"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

# 将星灵项目 src 加入 sys.path
XINGLING_ROOT = Path("D:/星灵-soul-手搓/亲子沟通洞见/路线A_自上而下法_鼓励版")
sys.path.insert(0, str(XINGLING_ROOT))

from src.multi_agent_orchestrator import MultiAgentOrchestrator
from auto_evolve.evaluator import (
    EvalResult, BaselineReport,
    compute_m1_trigger, compute_m5_tone,
    build_m6_prompt, build_m7_prompt,
    parse_llm_judge_response, aggregate_results,
)

# ═══════════════════════════════════════════════════════════════
# 选定的 5 条基线案例
# ═══════════════════════════════════════════════════════════════

SELECTED_CASES = [
    # case_id, window_index （如果是 C10 案例，window 不适用，用全量对话）
    ("C10-001", None),   # diagnostic, score=9, dlen=327, 晓浩
    ("C10-004", None),   # empowering, score=8, dlen=426, 晓浩
    ("C10-002", None),   # diagnostic, score=7, dlen=515, 晓浩
    ("C11-006", 1),      # empowering, score=4, dlen=370, 廖老师
    ("C11-010", 1),      # empowering, score=9, dlen=483, 廖老师
]

# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

def load_env():
    """加载 .env 环境变量。"""
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
    path = DATA_DIR / "golden_dataset.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_case(dataset: list[dict], case_id: str) -> dict | None:
    """按 case_id 查找案例。"""
    for c in dataset:
        if c["case_id"] == case_id:
            return c
    return None


def get_input_text(case: dict, window_index: int | None) -> str:
    """获取模型的输入文本。

    C10 案例（晓浩标注，window_text 为空）→ 用 case 级别 dialogue
    C11 案例（廖老师标注，有 window_text）→ 用 window_text
    """
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
    # 单窗案例
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
# LLM Judge 调用
# ═══════════════════════════════════════════════════════════════

def call_judge(client: OpenAI, model: str, prompt: str) -> str:
    """调用 LLM judge，返回原始响应文本。"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是评估专家。只输出严格 JSON，不要有其他文字。"},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,  # 低温度确保评估一致性
            timeout=60,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {str(e)[:100]}"})


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def run_baseline():
    """运行基线评估。"""
    load_env()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 加载数据
    dataset = load_golden_dataset()
    print(f"📋 加载黄金数据集: {len(dataset)} 案例")

    # 初始化 LLM client（用于 orchestrator 和 judge）
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

    client = OpenAI(api_key=api_key, base_url=base_url)
    orchestrator = MultiAgentOrchestrator(llm_client=client, model=model)

    print(f"🤖 模型: {model}")
    print(f"📊 基线案例: {len(SELECTED_CASES)} 条")
    print("=" * 70)

    all_results: list[EvalResult] = []

    for case_id, win_idx in SELECTED_CASES:
        case = find_case(dataset, case_id)
        if not case:
            print(f"❌ 找不到案例: {case_id}")
            continue

        gold = get_gold_labels(case, win_idx)
        input_text = get_input_text(case, win_idx)
        win_label = f"w{win_idx}" if win_idx else "full"

        print(f"\n{'─' * 60}")
        print(f"📌 {case_id} {win_label} | annotator={case['annotator']}")
        print(f"   gold_tone={gold['tone']} | gold_score={gold['score']}")
        print(f"   gold_should_popup={gold['should_popup']}")
        print(f"   input_len={len(input_text)} chars")

        # ── 运行 pipeline ──
        t0 = time.time()
        try:
            orchestrator.reset_family(case_id)
            result = orchestrator.process_window(input_text, family=case_id)
        except Exception as e:
            print(f"   ❌ Pipeline 错误: {e}")
            all_results.append(EvalResult(
                case_id=case_id, window_index=win_idx or 1,
                error=f"{type(e).__name__}: {str(e)[:200]}",
            ))
            continue

        elapsed = time.time() - t0
        print(f"   ⏱ {elapsed:.1f}s | sys_should_popup={result.should_popup} | sys_tone={result.tone}")

        # 拼装系统弹窗文本
        sys_popup = result.popup_text or ""

        # ── 计算 deterministic 指标 ──
        m1 = compute_m1_trigger(result.should_popup, gold["should_popup"])
        m5 = compute_m5_tone(result.tone, gold["tone"])

        print(f"   M1(trigger): {m1} | M5(tone): {m5}")

        # ── LLM Judge: M6 Insight Quality ──
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
            m6_raw = call_judge(client, model, m6_prompt)
            m6_score, m6_raw = parse_llm_judge_response(m6_raw, "m6")
            status = "OK" if m6_score is not None else f"PARSE_FAIL(raw={m6_raw[:100]})"
            print(f"   M6(insight): {m6_score} [{status}]")
        else:
            print(f"   M6(insight): SKIP (无有效参考弹窗)")

        # ── LLM Judge: M7 Safety ──
        m7_score = None
        m7_raw = ""
        has_real_forbids = any(
            f.strip() and not f.strip().startswith("_") and "___" not in f
            and f.strip() != "合格弹窗必须覆盖的点（列 2~4 条）："
            for f in gold["forbid_checklist"]
        )
        if has_real_forbids:
            m7_prompt = build_m7_prompt(
                dialogue=input_text,
                forbid_checklist=gold["forbid_checklist"],
                sys_popup=sys_popup,
            )
            m7_raw = call_judge(client, model, m7_prompt)
            m7_score, m7_raw = parse_llm_judge_response(m7_raw, "m7")
            print(f"   M7(safety): {m7_score} (judge)")
        else:
            # 用通用禁止项做 safety check
            m7_prompt = build_m7_prompt(
                dialogue=input_text,
                forbid_checklist=[],  # 会触发默认通用项
                sys_popup=sys_popup,
            )
            m7_raw = call_judge(client, model, m7_prompt)
            m7_score, m7_raw = parse_llm_judge_response(m7_raw, "m7")
            print(f"   M7(safety): {m7_score} (judge, generic)")

        # ── 汇总 ──
        er = EvalResult(
            case_id=case_id,
            window_index=win_idx or 1,
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
            m1_trigger_match=m1,
            m5_tone_match=m5,
            m6_insight_score=m6_score,
            m7_safety_score=m7_score,
            m6_judge_raw=m6_raw,
            m7_judge_raw=m7_raw,
        )
        all_results.append(er)

    # ── 生成报告 ──
    report = aggregate_results(all_results)

    print(f"\n{'=' * 60}")
    print(f"📊 基线评估报告")
    print(f"{'=' * 60}")
    print(f"  M1 触发准确率: {report.aggregate_m1:.2%}")
    print(f"  M5 口吻匹配率: {report.aggregate_m5:.2%}")
    print(f"  M6 洞察质量:   {report.aggregate_m6:.1f}/5")
    print(f"  M7 安全分数:   {report.aggregate_m7:.1f}/5")
    print(f"  {'─' * 40}")
    print(f"  综合得分:      {report.overall_score:.2%}")
    print(f"{'=' * 60}")

    # 保存详细报告
    detailed = {
        "meta": {
            "model": model,
            "num_cases": len(all_results),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "aggregate": {
            "m1_trigger_accuracy": report.aggregate_m1,
            "m5_tone_match": report.aggregate_m5,
            "m6_insight_quality": report.aggregate_m6,
            "m7_safety_score": report.aggregate_m7,
            "overall_score": report.overall_score,
        },
        "per_case": [],
    }
    for r in all_results:
        detailed["per_case"].append({
            "case_id": r.case_id,
            "window_index": r.window_index,
            "sys_should_popup": r.sys_should_popup,
            "sys_tone": r.sys_tone,
            "sys_popup_text": r.sys_popup_text,
            "sys_main_contradiction": r.sys_main_contradiction,
            "gold_should_popup": r.gold_should_popup,
            "gold_tone": r.gold_tone,
            "gold_reference_popup": r.gold_reference_popup[:300],
            "gold_score": r.gold_score,
            "m1_trigger_match": r.m1_trigger_match,
            "m5_tone_match": r.m5_tone_match,
            "m6_insight_score": r.m6_insight_score,
            "m7_safety_score": r.m7_safety_score,
            "error": r.error,
        })

    report_path = RESULTS_DIR / "baseline_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(detailed, f, ensure_ascii=False, indent=2)
    print(f"\n📄 详细报告已保存: {report_path}")

    # 保存简要摘要
    summary_path = RESULTS_DIR / "baseline_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(detailed["aggregate"], f, ensure_ascii=False, indent=2)
    print(f"📄 摘要已保存: {summary_path}")

    return report


if __name__ == "__main__":
    run_baseline()
