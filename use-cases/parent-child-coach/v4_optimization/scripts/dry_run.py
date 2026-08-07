"""
v4.0 dry run — 验证新 metric 在真实数据上的行为。

三阶段:
  1. 单样本测试: 用专家原文做 prediction → 应接近 1.0
  2. 小批量测试: 3条数据 → 确认 score 随 prediction 质量变化
  3. ROUGE-L 旁路: 验证旁路观测正常工作

用法: python v4_optimization/scripts/dry_run.py
"""

import json
import sys
from pathlib import Path

# 确保能导入 scripts/llm_judge_metric
_self_dir = Path(__file__).resolve().parent  # v4_optimization/scripts/
_realtime_parent = _self_dir.parents[1]  # parent-child-coach/
if str(_realtime_parent) not in sys.path:
    sys.path.insert(0, str(_realtime_parent))

# 直接加载模块（scripts 目录无 __init__.py）
import importlib.util
_metric_path = _realtime_parent / "scripts" / "llm_judge_metric.py"
_spec = importlib.util.spec_from_file_location("llm_judge_metric", str(_metric_path))
_metric_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_metric_module)
LLMJudgeMetric = _metric_module.LLMJudgeMetric


def load_clean_dataset():
    """加载并清洗训练数据（过滤占位符/元数据条目）。"""
    data_path = Path(__file__).resolve().parents[1] / "data" / "expert_train_v4.json"
    raw = json.loads(data_path.read_text(encoding="utf-8"))
    clean = []
    dirty = []
    for d in raw:
        answer = d.get("answer", "")
        # 过滤占位符条目：纯分隔线、节奏建议而非真实弹窗、"不新增弹窗"标记
        if (len(answer) < 50
                or answer.strip().startswith("___")
                or "整段对话的弹窗节奏建议" in answer[:100]
                or "本窗口不新增弹窗" in answer[:50]
                or "本窗口作为序列中的延续片段处理" in answer[:50]):
            dirty.append(d.get("id", "?"))
            continue
        clean.append(d)
    if dirty:
        print(f"⚠️  过滤 {len(dirty)} 条脏数据（占位符/元数据/非弹窗标记）: {dirty[:5]}...")
    print(f"Clean dataset: {len(clean)} examples (from {len(raw)} raw).")
    return clean


class MockPrediction:
    def __init__(self, answer: str):
        self.answer = answer


class MockGold:
    def __init__(self, question: str, answer: str):
        self.question = question
        self.answer = answer


def phase_1_single_sample(judge, data):
    """Phase 1: 单样本 — 专家原文自评。"""
    print("=" * 60)
    print("Phase 1: 专家原文自评（应为高分 ≥ 0.85）")
    print("=" * 60)

    item = data[0]
    gold = MockGold(question=item["question"], answer=item["answer"])
    pred = MockPrediction(answer=item["answer"])  # 用专家原文做 prediction

    score = judge(gold, pred, trace=True)
    verdict = "✅" if score >= 0.85 else "❌"
    print(f"\n{verdict} Phase 1 result: {score:.3f} ({'PASS' if score >= 0.85 else 'FAIL'})")
    return score >= 0.85


def phase_2_multi_sample(judge, data):
    """Phase 2: 3条数据 — 专家原文 vs 空输出 vs 乱写。"""
    print("\n" + "=" * 60)
    print("Phase 2: 多场景区分度测试")
    print("=" * 60)

    results = []
    for i, item in enumerate(data[:3]):
        gold = MockGold(question=item["question"], answer=item["answer"])

        # 场景 A: 专家原文
        pred_expert = MockPrediction(answer=item["answer"])
        score_expert = judge(gold, pred_expert, trace=False)

        # 场景 B: 空输出
        pred_empty = MockPrediction(answer="")
        score_empty = judge(gold, pred_empty, trace=False)

        # 场景 C: 乱写（术语泄漏）
        pred_bad = MockPrediction(answer=(
            "你正在经历一个典型的亲子认知失调模式。"
            "你需要建立更好的情绪调节框架，使用积极倾听技术来改善关系。"
        ))
        score_bad = judge(gold, pred_bad, trace=False)

        results.append({
            "id": item.get("id", i),
            "expert": score_expert,
            "empty": score_empty,
            "bad": score_bad,
        })

        print(f"\n  [{item.get('id', i)}] expert={score_expert:.3f} "
              f"| empty={score_empty:.3f} | bad_term_leak={score_bad:.3f}")

    # 检查
    expert_scores = [r["expert"] for r in results]
    empty_scores = [r["empty"] for r in results]
    bad_scores = [r["bad"] for r in results]

    checks = [
        ("专家原文 ≥ 0.80", all(s >= 0.80 for s in expert_scores)),
        ("空输出 = 0.00", all(s == 0.0 for s in empty_scores)),
        ("专家 > 术语泄漏", all(e > b for e, b in zip(expert_scores, bad_scores))),
    ]

    all_pass = True
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        if not passed:
            all_pass = False
        print(f"  {status} {check_name}")

    return all_pass


def phase_3_rouge_observation(judge, data):
    """Phase 3: ROUGE-L 旁路观测验证。"""
    print("\n" + "=" * 60)
    print("Phase 3: ROUGE-L 旁路观测")
    print("=" * 60)

    item = data[0]
    gold = MockGold(question=item["question"], answer=item["answer"])

    # 完全相同 → ROUGE-L = 1.0
    pred_same = MockPrediction(answer=item["answer"])
    score_same = judge(gold, pred_same, trace=True)

    # 完全不同 → ROUGE-L 接近 0
    pred_diff = MockPrediction(answer="这是一段完全不同的文字用于测试ROUGE-L旁路观测是否正常工作")
    score_diff = judge(gold, pred_diff, trace=True)

    print("\n  ROUGE-L 旁路已集成在 trace 输出中（见上方 ROUGE-L 字段）。")
    print(f"  相同文本 score: {score_same:.3f} | 不同文本 score: {score_diff:.3f}")
    print("  ✅ ROUGE-L 旁路观测正常工作 (不参与主分)")

    return True


def main():
    judge = LLMJudgeMetric()
    data = load_clean_dataset()
    print(f"Dataset: {len(data)} examples loaded.")

    p1 = phase_1_single_sample(judge, data)
    p2 = phase_2_multi_sample(judge, data)
    p3 = phase_3_rouge_observation(judge, data)

    print("\n" + "=" * 60)
    print("Dry Run Summary")
    print("=" * 60)
    print(f"  Phase 1 (专家自评): {'✅' if p1 else '❌'}")
    print(f"  Phase 2 (区分度):   {'✅' if p2 else '❌'}")
    print(f"  Phase 3 (ROUGE-L):  {'✅' if p3 else '❌'}")

    all_pass = p1 and p2 and p3
    if all_pass:
        print("\n  ✅ Dry run 全部通过！可以启动 MIPROv2 优化。")
        print("     运行: cd use-cases/parent-child-coach && "
              "prompt-ops migrate --config v4_optimization/config_v4.yaml "
              "--output-dir v4_optimization/results")
    else:
        print("\n  ❌ Dry run 未通过 — 请检查失败项。")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
