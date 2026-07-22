"""测试阶梯脚本 — 渐进式验证 v4.0 弹窗质量。

用法:
  python scripts/test_ladder.py --level 1   # 1 个用例
  python scripts/test_ladder.py --level 2   # 3 个用例
  python scripts/test_ladder.py --level 3   # 9 个用例
  python scripts/test_ladder.py --level 4   # 27 个用例
  python scripts/test_ladder.py --level 5   # 全量

策略：每级全部通过（score >= 0.70, 无 VETO）才进入下一级。
发现失败：修复 prompt → 回到 level 1。
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]
_realtime_parent = Path(__file__).resolve().parent.parent

if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_realtime_parent) not in sys.path:
    sys.path.insert(0, str(_realtime_parent))

from dotenv import load_dotenv
load_dotenv(_realtime_parent / ".env")

from prompt_ops.core.model import LiteLLMModelAdapter

# 加载 judge metric
sys.path.insert(0, str(_realtime_parent / "scripts"))
from llm_judge_metric import LLMJudgeMetric

LEVEL_SIZES = {1: 1, 2: 3, 3: 9, 4: 27, 5: None}  # None = all


def load_dataset(dataset_path: str) -> list:
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_prompt(prompt_path: str) -> str:
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def generate_popup(model, system_prompt: str, dialogue: str, tone: str) -> str:
    """用 v4.0 prompt 生成弹窗（不含周易上下文）。"""
    if tone == "诊断式":
        type_instruction = (
            "请生成**诊断式弹窗**（80-200字）。"
            "必须：先承认发心 → 揭示具体模式 → 给出一个微小可做的尝试。"
        )
    else:
        type_instruction = (
            "请生成**鼓励式弹窗**（30-80字）。"
            "必须：具体点出家长刚展现的积极模式 → 简短有力。"
        )

    user_content = f"""当前对话：
{dialogue}

{type_instruction}

请直接输出弹窗全文（不附加解释、不输出JSON、不输出"弹窗："等前缀）："""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    if hasattr(model, "generate_with_chat_format"):
        raw = model.generate_with_chat_format(
            messages=messages, temperature=0.3, max_tokens=640,
        )
    else:
        combined = "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)
        raw = model.generate(prompt=combined, temperature=0.3, max_tokens=640)

    return raw.strip()


def run_level(level: int, dataset: list, model, system_prompt: str, judge) -> dict:
    """运行一个测试级别。"""
    size = LEVEL_SIZES[level]
    if size is None:
        size = len(dataset)
    cases = dataset[:size]

    print(f"\n{'='*60}")
    print(f"  测试阶梯 L{level}：{len(cases)} 个用例")
    print(f"{'='*60}")

    # 缓存：同一对话 + 同一 tone 只生成一次弹窗
    dialogue_cache = {}  # (dialogue, tone) -> (generated, elapsed)
    results = []

    for i, case in enumerate(cases):
        dialogue = case["question"]
        golden = case["answer"]
        tone = case.get("tone", "诊断式")

        print(f"\n[{i+1}/{len(cases)}] id={case.get('id','?')} | tone={tone}")
        print(f"  对话长度: {len(dialogue)} 字符")

        try:
            cache_key = (dialogue, tone)
            # 同一对话+同一tone复用已生成的弹窗
            if cache_key in dialogue_cache:
                generated, elapsed = dialogue_cache[cache_key]
                print(f"  复用缓存弹窗 ({elapsed:.1f}s)")
            else:
                start = time.time()
                generated = generate_popup(model, system_prompt, dialogue, tone)
                elapsed = time.time() - start
                dialogue_cache[cache_key] = (generated, elapsed)

                # 截断打印
                preview = generated[:120].replace("\n", " ")
                print(f"  生成 ({elapsed:.1f}s): {preview}...")

            # Judge 评分 — 适配 LLMJudgeMetric.__call__ 的 DSPy Example 接口
            from dspy import Example
            gold_ex = Example(question=dialogue, answer=golden)
            pred_ex = Example(answer=generated)

            try:
                weighted = judge(gold_ex, pred_ex, trace=True)
            except Exception as judge_err:
                print(f"  ⚠️ Judge 调用异常: {judge_err}")
                weighted = 0.0

            results.append({
                "id": case.get("id", f"case_{i}"),
                "tone": tone,
                "generated": generated,
                "golden": golden,
                "weighted_score": weighted,
                "elapsed": elapsed,
            })

        except Exception as e:
            print(f"  ❌ 异常: {e}")
            results.append({
                "id": case.get("id", f"case_{i}"),
                "error": str(e),
            })

    # === 窗口级原始结果 ===
    raw_scores = [r["weighted_score"] for r in results if "weighted_score" in r]
    raw_avg = sum(raw_scores) / len(raw_scores) if raw_scores else 0
    raw_passed = sum(1 for s in raw_scores if s >= 0.70)
    raw_failed = len(raw_scores) - raw_passed

    # === 对话+tone 级聚合：同一对话+同一tone，取最佳窗口分 ===
    from collections import defaultdict
    dialogue_tone_groups = defaultdict(list)
    for i, r in enumerate(results):
        if "weighted_score" in r and i < len(cases):
            key = (cases[i]["question"], cases[i].get("tone", "诊断式"))
            dialogue_tone_groups[key].append(r)

    grouped_scores = []
    grouped_details = []
    for (dialogue, tone), group in dialogue_tone_groups.items():
        best = max(group, key=lambda r: r["weighted_score"])
        window_ids = [r["id"] for r in group]
        window_scores = [r["weighted_score"] for r in group]
        grouped_scores.append(best["weighted_score"])
        grouped_details.append({
            "dialogue_preview": dialogue[:80],
            "tone": tone,
            "window_ids": window_ids,
            "window_scores": window_scores,
            "best_score": best["weighted_score"],
            "best_window": best["id"],
        })

    grouped_passed = sum(1 for s in grouped_scores if s >= 0.70)
    grouped_failed = len(grouped_scores) - grouped_passed
    grouped_avg = sum(grouped_scores) / len(grouped_scores) if grouped_scores else 0

    # === 打印汇总 ===
    print(f"\n{'─'*60}")
    print(f"  窗口级(31窗): {raw_passed}✅ / {raw_failed}❌ | 均分: {raw_avg:.3f}")
    print(f"  对话+tone级({len(grouped_scores)}组): {grouped_passed}✅ / {grouped_failed}❌ | 均分: {grouped_avg:.3f}")
    if grouped_failed > 0:
        print(f"  未通过组:")
        for d in grouped_details:
            if d["best_score"] < 0.70:
                print(f"    {d['best_window']}: best={d['best_score']:.3f} (窗口分: {d['window_scores']})")
    print(f"{'─'*60}")

    all_pass = grouped_failed == 0

    return {
        "level": level,
        "total_cases": len(cases),
        "total_groups": len(grouped_scores),
        # 窗口级
        "window_passed": raw_passed,
        "window_failed": raw_failed,
        "window_avg_score": raw_avg,
        # 对话+tone级（聚合后）
        "passed": grouped_passed,
        "failed": grouped_failed,
        "avg_score": grouped_avg,
        "all_pass": all_pass,
        "results": results,
        "dialogue_tone_groups": grouped_details,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, required=True, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--dataset", default="v4_optimization/data/expert_train_v4_clean.json")
    parser.add_argument("--prompt", default="system_prompt_v4.0.txt")
    parser.add_argument("--output-dir", default="results/ladder_tests")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    dataset_path = base / args.dataset
    prompt_path = base / args.prompt
    output_dir = base / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"数据集: {dataset_path}")
    print(f"提示词: {prompt_path}")
    print(f"级别: L{args.level} ({LEVEL_SIZES[args.level] or '全部'} 用例)")

    # 加载
    dataset = load_dataset(str(dataset_path))
    system_prompt = load_prompt(str(prompt_path))
    print(f"加载: {len(dataset)} 条数据, prompt {len(system_prompt)} 字符")

    # 模型
    model = LiteLLMModelAdapter(
        model_name="deepseek/deepseek-chat",
        temperature=0.3,
        max_tokens=640,
    )

    # Judge — 优先 Claude，失败则回退 deepseek
    import os
    backend = os.getenv("JUDGE_BACKEND", "claude")
    try:
        judge = LLMJudgeMetric(judge_backend=backend)
        print(f"Judge 后端: {backend}")
    except RuntimeError:
        print(f"Claude judge 不可用，回退到 deepseek")
        judge = LLMJudgeMetric(judge_backend="deepseek")
        print(f"Judge 后端: deepseek")

    # 运行
    summary = run_level(args.level, dataset, model, system_prompt, judge)

    # 保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"ladder_L{args.level}_{timestamp}.json"
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n结果保存: {output_path}")

    # 返回状态码
    sys.exit(0 if summary["all_pass"] else 1)


if __name__ == "__main__":
    main()
