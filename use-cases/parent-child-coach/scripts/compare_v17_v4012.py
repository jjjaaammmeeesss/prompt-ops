"""v1.7 vs v4.0.12 公平对比脚本（独立测试集 1→3→9 题）。

两个版本用相同数据集、相同 judge、相同轮数，在同一脚本内跑，保证可比。

用法:
  python scripts/compare_v17_v4012.py --n 1 --rounds 3
  python scripts/compare_v17_v4012.py --n 3 --rounds 3
  python scripts/compare_v17_v4012.py --n 9 --rounds 3
"""

import argparse
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

sys.path.insert(0, str(_realtime_parent / "scripts"))
from llm_judge_metric import LLMJudgeMetric

# 路径
V17_PROMPT_PATH = Path(r"D:\星灵-soul-手搓\亲子沟通洞见\路线A_自上而下法_鼓励版\prompts\prompt_A轨_v1.7_修复感知版.md")
V4012_PROMPT_PATH = _realtime_parent / "system_prompt_v4.0.12.txt"
NEW12_DATASET = _realtime_parent / "data" / "new_12_independent.json"


# ===== v1.7 提取与生成 =====

def extract_v17_prompts(md_path: Path) -> tuple[str, str]:
    lines = md_path.read_text(encoding="utf-8").splitlines(keepends=True)
    sys_start = None
    sys_end = None
    for i, line in enumerate(lines):
        if line.rstrip() == "```":
            if sys_start is None:
                sys_start = i
            else:
                sys_end = i
                break
    system_prompt = "".join(lines[sys_start + 1:sys_end])

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
    user_prompt = "".join(lines[usr_start + 1:usr_end])
    return system_prompt, user_prompt


def parse_v17_output(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        first_nl = raw.find("\n")
        if first_nl > 0:
            raw = raw[first_nl + 1:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    first_brace = raw.find("{")
    last_brace = raw.rfind("}")
    if first_brace >= 0 and last_brace > first_brace:
        try:
            return json.loads(raw[first_brace:last_brace + 1])
        except json.JSONDecodeError:
            pass
    return {"_parse_error": "JSON 解析失败", "_raw": raw[:500]}


def generate_v17(model, system_prompt: str, user_template: str, dialogue: str) -> tuple[str, dict]:
    filled = user_template.replace("{user_input}", dialogue)
    filled = filled.replace("{context_block}", "（首次对话，无前序上下文）")
    filled = filled.replace("{profile_context}", "（无用户画像）")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": filled},
    ]
    raw = model.generate_with_chat_format(messages=messages, temperature=0.3, max_tokens=2048)

    obj = parse_v17_output(raw)
    if not obj.get("should_popup", True):
        return "", {"tone": "skip", "should_popup": False, "skip_reason": obj.get("skip_reason")}
    tone = obj.get("tone", "diagnostic")
    insight = obj.get("popup_insight") or ""
    suggestion = obj.get("popup_suggestion") or ""
    if tone == "empowering":
        popup = insight.strip()
    else:
        parts = []
        if insight.strip():
            parts.append(insight.strip())
        if suggestion.strip():
            parts.append(suggestion.strip())
        popup = "\n\n".join(parts)
    return popup, {"tone": tone, "should_popup": True, "skip_reason": None}


# ===== v4.0.12 生成 =====

def generate_v4012(model, system_prompt: str, dialogue: str, tone: str) -> str:
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
    raw = model.generate_with_chat_format(messages=messages, temperature=0.3, max_tokens=640)
    return raw.strip()


# ===== 通用测试 =====

def run_version(name: str, cases: list, gen_fn, model, judge) -> dict:
    from dspy import Example
    results = []
    cache = {}
    for i, case in enumerate(cases):
        dialogue = case["question"]
        golden = case["answer"]
        expected_tone = case.get("tone", "诊断式")
        print(f"\n  [{name}] [{i+1}/{len(cases)}] id={case.get('id','?')} tone={expected_tone}")
        try:
            if dialogue in cache:
                popup, meta, elapsed = cache[dialogue]
                print(f"    复用缓存 ({elapsed:.1f}s)")
            else:
                start = time.time()
                popup, meta = gen_fn(dialogue, expected_tone)
                elapsed = time.time() - start
                cache[dialogue] = (popup, meta, elapsed)

            if not popup:
                weighted = 0.0
                print(f"    ⚠️ 判定不弹 → 0 分")
            else:
                gold_ex = Example(question=dialogue, answer=golden)
                pred_ex = Example(answer=popup)
                try:
                    weighted = judge(gold_ex, pred_ex, trace=False)
                except Exception as e:
                    print(f"    ⚠️ Judge 异常: {e}")
                    weighted = 0.0

            passed = weighted >= 0.70
            print(f"    分数: {weighted:.3f} {'✅' if passed else '❌'} | 实际tone={meta.get('tone')}")
            results.append({
                "id": case.get("id", f"case_{i}"),
                "expected_tone": expected_tone,
                "actual_tone": meta.get("tone"),
                "should_popup": meta.get("should_popup"),
                "skip_reason": meta.get("skip_reason"),
                "generated": popup,
                "golden": golden,
                "weighted_score": weighted,
            })
        except Exception as e:
            print(f"    ❌ 异常: {e}")
            results.append({"id": case.get("id", f"case_{i}"), "error": str(e)})

    scores = [r["weighted_score"] for r in results if "weighted_score" in r]
    avg = sum(scores) / len(scores) if scores else 0
    passed = sum(1 for s in scores if s >= 0.70)
    tone_mismatch = sum(
        1 for r in results
        if r.get("actual_tone") and r["actual_tone"] != "skip"
        and r.get("expected_tone")
        and ((r["expected_tone"] == "诊断式" and r["actual_tone"] == "empowering") or
             (r["expected_tone"] == "鼓励式" and r["actual_tone"] == "diagnostic"))
    )
    skip_count = sum(1 for r in results if r.get("actual_tone") == "skip")
    print(f"\n  [{name}] {len(cases)} 题 | {passed}✅/{len(cases)-passed}❌ | 均分: {avg:.3f} | tone偏移: {tone_mismatch} | 不弹: {skip_count}")
    return {
        "version": name,
        "avg_score": avg,
        "passed": passed,
        "failed": len(cases) - passed,
        "tone_mismatch": tone_mismatch,
        "skip_count": skip_count,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True, choices=[1, 3, 9, 12])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output-dir", default="results/compare_tests")
    args = parser.parse_args()

    base = _realtime_parent
    output_dir = base / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据集
    with open(NEW12_DATASET, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    cases = dataset[:args.n]
    print(f"独立测试集（12新用例）前 {args.n} 题，{args.rounds} 轮")

    # 加载两个 prompt
    v17_sys, v17_usr = extract_v17_prompts(V17_PROMPT_PATH)
    print(f"v1.7: system {len(v17_sys)}字, user {len(v17_usr)}字")
    v4012_prompt = V4012_PROMPT_PATH.read_text(encoding="utf-8")
    print(f"v4.0.12: {len(v4012_prompt)}字")

    # 模型 + judge
    model = LiteLLMModelAdapter(model_name="deepseek/deepseek-chat", temperature=0.3, max_tokens=2048)
    import os
    backend = os.getenv("JUDGE_BACKEND", "deepseek")
    judge = LLMJudgeMetric(judge_backend=backend)
    print(f"Judge: {backend}")

    # 两个版本生成函数
    def gen_v17(dialogue, tone):
        return generate_v17(model, v17_sys, v17_usr, dialogue)

    def gen_v4012(dialogue, tone):
        return generate_v4012(model, v4012_prompt, dialogue, tone), {"tone": tone, "should_popup": True, "skip_reason": None}

    all_rounds_v17 = []
    all_rounds_v4012 = []
    for r in range(args.rounds):
        print(f"\n{'='*60}\n  Round {r+1}/{args.rounds}\n{'='*60}")
        print(f"\n--- v1.7 ---")
        res_v17 = run_version("v1.7", cases, gen_v17, model, judge)
        all_rounds_v17.append(res_v17["avg_score"])
        print(f"\n--- v4.0.12 ---")
        res_v4012 = run_version("v4.0.12", cases, gen_v4012, model, judge)
        all_rounds_v4012.append(res_v4012["avg_score"])

    # 汇总
    v17_avg = sum(all_rounds_v17) / len(all_rounds_v17)
    v4012_avg = sum(all_rounds_v4012) / len(all_rounds_v4012)
    v17_var = max(all_rounds_v17) - min(all_rounds_v17)
    v4012_var = max(all_rounds_v4012) - min(all_rounds_v4012)

    print(f"\n{'='*60}")
    print(f"  对比汇总（独立测试前 {args.n} 题, {args.rounds} 轮）")
    print(f"{'='*60}")
    print(f"  v1.7:    {[f'{a:.3f}' for a in all_rounds_v17]} | 均分 {v17_avg:.3f} | variance {v17_var:.3f}")
    print(f"  v4.0.12: {[f'{a:.3f}' for a in all_rounds_v4012]} | 均分 {v4012_avg:.3f} | variance {v4012_var:.3f}")
    print(f"  差距: v4.0.12 领先 {v4012_avg - v17_avg:+.3f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"compare_n{args.n}_{timestamp}.json"
    output_path.write_text(json.dumps({
        "n": args.n,
        "rounds": args.rounds,
        "v17_rounds": all_rounds_v17,
        "v4012_rounds": all_rounds_v4012,
        "v17_avg": v17_avg,
        "v4012_avg": v4012_avg,
        "v17_variance": v17_var,
        "v4012_variance": v4012_var,
        "diff": v4012_avg - v17_avg,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果保存: {output_path}")


if __name__ == "__main__":
    main()
