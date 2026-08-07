"""v1.7 vs v4.0.12 对比测试脚本。

v1.7 是双 prompt 架构（system + user，含 14 few-shot + JSON 输出）。
本脚本提取 v1.7 的两层 prompt，注入对话，调用 API，解析 JSON，
取 popup_insight + popup_suggestion 作为最终弹窗，用相同 LLMJudgeMetric 评分。

用法:
  # 校标集递增
  python scripts/test_v17.py --dataset cal --n 1
  python scripts/test_v17.py --dataset cal --n 3
  python scripts/test_v17.py --dataset cal --n 9
  python scripts/test_v17.py --dataset cal --n 18

  # 独立测试集递增（12 新用例 + 50 题前 43 题，排除 #43-49 与 v1.7 few-shot 重叠）
  python scripts/test_v17.py --dataset independent --n 1
  python scripts/test_v17.py --dataset independent --n 3
  python scripts/test_v17.py --dataset independent --n 9
  python scripts/test_v17.py --dataset independent --n 18

  # 跑 3 轮量化 judge variance
  python scripts/test_v17.py --dataset cal --n 9 --rounds 3
"""

import argparse
import json
import re
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

sys.path.insert(0, str(_realtime_parent / "scripts"))
from llm_judge_metric import LLMJudgeMetric

# v1.7 prompt 文件
V17_PROMPT_PATH = Path(r"D:\星灵-soul-手搓\亲子沟通洞见\路线A_自上而下法_鼓励版\prompts\prompt_A轨_v1.7_修复感知版.md")

# 数据集路径
CAL_DATASET = _realtime_parent / "data" / "expert_dataset_full_71.json"
NEW12_DATASET = _realtime_parent / "data" / "new_12_independent.json"
BLIND50_DATASET = _realtime_parent / "dataset_50_questions.json"

# 与 v1.7 few-shot 重叠的盲测题号（#43-49）
BLIND_EXCLUDE_INDICES = set(range(43, 50))


def extract_v17_prompts(md_path: Path) -> tuple[str, str]:
    """从 v1.7 .md 提取 system prompt 和 user prompt 模板。

    System Prompt: line 28 ``` 到 line 220 ``` 之间（去外层 ```）
    User Prompt: line 226 ```` 到 line 1442 ```` 之间（去外层 ````）
    """
    lines = md_path.read_text(encoding="utf-8").splitlines(keepends=True)

    # System: 找第一个 ``` (line 28) 到对应关闭 ``` (line 220)
    sys_start = None
    sys_end = None
    for i, line in enumerate(lines):
        if line.rstrip() == "```":
            if sys_start is None:
                sys_start = i
            else:
                sys_end = i
                break
    if sys_start is None or sys_end is None:
        raise RuntimeError("无法定位 system prompt 边界")

    system_prompt = "".join(lines[sys_start + 1:sys_end])

    # User: 找 ```` (4 反引号) 开始和结束
    usr_start = None
    usr_end = None
    for i, line in enumerate(lines):
        s = line.rstrip()
        if s == "````":
            if usr_start is None:
                usr_start = i
            else:
                usr_end = i
                break
    if usr_start is None or usr_end is None:
        raise RuntimeError("无法定位 user prompt 边界")

    user_prompt = "".join(lines[usr_start + 1:usr_end])

    return system_prompt, user_prompt


def build_user_prompt(user_template: str, dialogue: str) -> str:
    """注入对话到 {user_input}，填充 {context_block} 和 {profile_context}。"""
    filled = user_template.replace("{user_input}", dialogue)
    filled = filled.replace("{context_block}", "（首次对话，无前序上下文）")
    filled = filled.replace("{profile_context}", "（无用户画像）")
    return filled


def parse_v17_output(raw: str) -> dict:
    """解析 v1.7 的 JSON 输出。

    v1.7 输出严格 JSON，但模型可能加 ```json ``` 包裹或前后空白。
    返回 dict，失败时返回 {"_parse_error": "..."}。
    """
    raw = raw.strip()

    # 去除 ```json ... ``` 包裹
    if raw.startswith("```"):
        # 找第一个换行后的内容
        first_nl = raw.find("\n")
        if first_nl > 0:
            raw = raw[first_nl + 1:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

    # 尝试直接解析
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 尝试提取第一个 { 到最后一个 }
    first_brace = raw.find("{")
    last_brace = raw.rfind("}")
    if first_brace >= 0 and last_brace > first_brace:
        try:
            return json.loads(raw[first_brace:last_brace + 1])
        except json.JSONDecodeError:
            pass

    return {"_parse_error": "JSON 解析失败", "_raw": raw[:500]}


def extract_popup_from_v17_json(obj: dict) -> tuple[str, str]:
    """从 v1.7 JSON 提取最终弹窗文本。

    返回 (popup_text, tone)。
    - should_popup=false → popup_text="", tone="skip"
    - 鼓励式 → popup_text=popup_insight, tone="empowering"
    - 诊断式 → popup_text=popup_insight + popup_suggestion, tone="diagnostic"
    """
    if not obj.get("should_popup", True):
        return "", "skip"

    tone = obj.get("tone", "diagnostic")
    insight = obj.get("popup_insight") or ""
    suggestion = obj.get("popup_suggestion") or ""

    if tone == "empowering":
        # 鼓励式只有 popup_insight
        return insight.strip(), "empowering"
    else:
        # 诊断式 = insight + suggestion
        parts = []
        if insight.strip():
            parts.append(insight.strip())
        if suggestion.strip():
            parts.append(suggestion.strip())
        return "\n\n".join(parts), "diagnostic"


def generate_popup_v17(model, system_prompt: str, user_template: str, dialogue: str) -> tuple[str, dict]:
    """用 v1.7 双 prompt 生成弹窗。返回 (popup_text, meta)。"""
    user_content = build_user_prompt(user_template, dialogue)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    if hasattr(model, "generate_with_chat_format"):
        raw = model.generate_with_chat_format(
            messages=messages, temperature=0.3, max_tokens=2048,
        )
    else:
        combined = f"system: {system_prompt}\n\nuser: {user_content}"
        raw = model.generate(prompt=combined, temperature=0.3, max_tokens=2048)

    obj = parse_v17_output(raw)
    popup, tone = extract_popup_from_v17_json(obj)

    meta = {
        "tone": tone,
        "should_popup": obj.get("should_popup"),
        "skip_reason": obj.get("skip_reason"),
        "parse_error": obj.get("_parse_error"),
        "raw_preview": raw[:300] if not obj.get("_parse_error") else obj.get("_raw", "")[:300],
    }
    return popup, meta


def load_dataset(name: str) -> list:
    """加载数据集。independent = 12 新用例 + 50 题前 43 题（排除 #43-49）。"""
    if name == "cal":
        with open(CAL_DATASET, "r", encoding="utf-8") as f:
            return json.load(f)
    elif name == "independent":
        cases = []
        with open(NEW12_DATASET, "r", encoding="utf-8") as f:
            cases.extend(json.load(f))
        with open(BLIND50_DATASET, "r", encoding="utf-8") as f:
            blind = json.load(f)
        # 排除 #43-49
        for i, c in enumerate(blind):
            if i in BLIND_EXCLUDE_INDICES:
                continue
            # 50 题盲测无 gold answer，跳过（独立测试集必须有 gold）
            if not c.get("answer"):
                continue
            cases.append(c)
        return cases
    else:
        raise ValueError(f"未知数据集: {name}")


def run_test(cases: list, model, system_prompt: str, user_template: str, judge) -> dict:
    """运行测试。"""
    from dspy import Example

    results = []
    dialogue_cache = {}

    for i, case in enumerate(cases):
        dialogue = case["question"]
        golden = case["answer"]
        expected_tone = case.get("tone", "诊断式")

        print(f"\n[{i+1}/{len(cases)}] id={case.get('id','?')} | 期望tone={expected_tone}")
        print(f"  对话长度: {len(dialogue)} 字符")

        try:
            cache_key = dialogue
            if cache_key in dialogue_cache:
                popup, meta, elapsed = dialogue_cache[cache_key]
                print(f"  复用缓存 ({elapsed:.1f}s)")
            else:
                start = time.time()
                popup, meta = generate_popup_v17(
                    model, system_prompt, user_template, dialogue
                )
                elapsed = time.time() - start
                dialogue_cache[cache_key] = (popup, meta, elapsed)

                preview = (popup[:120] if popup else f"[未弹窗] skip_reason={meta.get('skip_reason')}").replace("\n", " ")
                print(f"  生成 ({elapsed:.1f}s, tone={meta['tone']}): {preview}")

            # should_popup=false → 0 分（校标集都是应弹场景）
            if not popup:
                weighted = 0.0
                print(f"  ⚠️ v1.7 判定不弹 → 0 分 (skip_reason={meta.get('skip_reason')})")
            else:
                gold_ex = Example(question=dialogue, answer=golden)
                pred_ex = Example(answer=popup)
                try:
                    weighted = judge(gold_ex, pred_ex, trace=True)
                except Exception as judge_err:
                    print(f"  ⚠️ Judge 异常: {judge_err}")
                    weighted = 0.0

            passed = weighted >= 0.70
            print(f"  分数: {weighted:.3f} {'✅' if passed else '❌'}")

            results.append({
                "id": case.get("id", f"case_{i}"),
                "expected_tone": expected_tone,
                "v17_tone": meta["tone"],
                "should_popup": meta["should_popup"],
                "skip_reason": meta["skip_reason"],
                "parse_error": meta.get("parse_error"),
                "generated": popup,
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

    # 汇总
    scores = [r["weighted_score"] for r in results if "weighted_score" in r]
    avg = sum(scores) / len(scores) if scores else 0
    passed = sum(1 for s in scores if s >= 0.70)
    failed = len(scores) - passed

    # tone 偏移统计
    tone_mismatches = sum(
        1 for r in results
        if "v17_tone" in r and r["v17_tone"] != "skip"
        and r.get("expected_tone") and (
            (r["expected_tone"] == "诊断式" and r["v17_tone"] == "empowering") or
            (r["expected_tone"] == "鼓励式" and r["v17_tone"] == "diagnostic")
        )
    )
    skip_count = sum(1 for r in results if r.get("v17_tone") == "skip")
    parse_errors = sum(1 for r in results if r.get("parse_error"))

    print(f"\n{'─'*60}")
    print(f"  {len(cases)} 题 | {passed}✅ / {failed}❌ | 均分: {avg:.3f}")
    print(f"  tone 偏移: {tone_mismatches} | 判定不弹: {skip_count} | 解析失败: {parse_errors}")
    print(f"{'─'*60}")

    return {
        "total_cases": len(cases),
        "passed": passed,
        "failed": failed,
        "avg_score": avg,
        "tone_mismatches": tone_mismatches,
        "skip_count": skip_count,
        "parse_errors": parse_errors,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["cal", "independent"])
    parser.add_argument("--n", type=int, default=None, help="取前 N 题，默认全部")
    parser.add_argument("--rounds", type=int, default=1, help="重复轮数（量化 judge variance）")
    parser.add_argument("--output-dir", default="results/v17_tests")
    args = parser.parse_args()

    base = _realtime_parent
    output_dir = base / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载 v1.7 prompt
    print(f"加载 v1.7 prompt: {V17_PROMPT_PATH}")
    system_prompt, user_template = extract_v17_prompts(V17_PROMPT_PATH)
    print(f"  system prompt: {len(system_prompt)} 字符")
    print(f"  user prompt 模板: {len(user_template)} 字符")

    # 加载数据集
    dataset = load_dataset(args.dataset)
    print(f"数据集: {args.dataset} ({len(dataset)} 条可用)")

    if args.n:
        cases = dataset[:args.n]
    else:
        cases = dataset
    print(f"本次测试: {len(cases)} 题")

    # 模型
    model = LiteLLMModelAdapter(
        model_name="deepseek/deepseek-chat",
        temperature=0.3,
        max_tokens=2048,
    )

    # Judge
    import os
    backend = os.getenv("JUDGE_BACKEND", "deepseek")
    judge = LLMJudgeMetric(judge_backend=backend)
    print(f"Judge 后端: {backend}")

    # 跑 N 轮
    all_rounds = []
    for r in range(args.rounds):
        print(f"\n{'='*60}")
        print(f"  Round {r+1}/{args.rounds}")
        print(f"{'='*60}")
        summary = run_test(cases, model, system_prompt, user_template, judge)
        all_rounds.append(summary)

    # 多轮汇总
    if args.rounds > 1:
        round_avgs = [s["avg_score"] for s in all_rounds]
        overall_avg = sum(round_avgs) / len(round_avgs)
        variance = max(round_avgs) - min(round_avgs)
        print(f"\n{'='*60}")
        print(f"  {args.rounds} 轮汇总")
        print(f"{'='*60}")
        print(f"  各轮均分: {[f'{a:.3f}' for a in round_avgs]}")
        print(f"  总均分: {overall_avg:.3f} | variance: {variance:.3f}")

    # 保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_tag = f"n{args.n}" if args.n else "nall"
    output_path = output_dir / f"v17_{args.dataset}_{n_tag}_{timestamp}.json"
    output_path.write_text(
        json.dumps(
            {
                "prompt_version": "v1.7",
                "dataset": args.dataset,
                "n": len(cases),
                "rounds": all_rounds,
                "rounds_summary": {
                    "avg_scores": [s["avg_score"] for s in all_rounds],
                    "overall_avg": sum(s["avg_score"] for s in all_rounds) / len(all_rounds),
                    "variance": max(s["avg_score"] for s in all_rounds) - min(s["avg_score"] for s in all_rounds),
                } if len(all_rounds) > 1 else None,
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n结果保存: {output_path}")


if __name__ == "__main__":
    main()
