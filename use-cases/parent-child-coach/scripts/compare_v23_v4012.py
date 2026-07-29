"""v2.3 vs v4.0.12 公平对比脚本（独立测试集 12 新用例）。

两个版本用相同数据集、相同 judge、相同轮数，在同一脚本内跑，保证可比。

用法:
  python scripts/compare_v23_v4012.py --n 12 --rounds 3
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import litellm
litellm.suppress_debug_info = True

_project_root = Path(__file__).resolve().parents[2]
_realtime_parent = Path(__file__).resolve().parent.parent

if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_realtime_parent) not in sys.path:
    sys.path.insert(0, str(_realtime_parent))

from dotenv import load_dotenv
load_dotenv(_realtime_parent / ".env")

sys.path.insert(0, str(_realtime_parent / "scripts"))
from llm_judge_metric import LLMJudgeMetric

# ── 生成模型：DeepSeek ──
_raw_key = os.getenv("DEEPSEEK_API_KEY", "")
GEN_API_KEY = _raw_key if _raw_key and "PLACEHOLDER" not in _raw_key else "sk-8063c7285a50489e98cf73f50b3c0ec4"
GEN_API_BASE = os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
GEN_MODEL = os.getenv("DEEPSEEK_MODEL") or "deepseek-chat"

# ── 评委模型：XINGLUAN Claude ──
JUDGE_API_KEY = os.getenv("XINGLUAN_AUTH_TOKEN", "06131c3816c4483c8a4da408102d52e3")
JUDGE_API_BASE = "https://luanapi.xingluan.cn/v1"

# 路径
V23_PROMPT_PATH = _realtime_parent / "system_prompt_v2.3.txt"
V4012_PROMPT_PATH = _realtime_parent / "system_prompt_v4.0.12.txt"
NEW12_DATASET = _realtime_parent / "data" / "new_12_independent.json"


# ===== LLM 调用 =====

def call_llm(system_prompt: str, user_content: str, max_tokens: int = 400) -> str:
    resp = litellm.completion(
        model=f"deepseek/{GEN_MODEL}",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.3,
        max_tokens=max_tokens,
        api_key=GEN_API_KEY,
        api_base=GEN_API_BASE,
        timeout=120,
    )
    return (resp.choices[0].message.content or "").strip()


# ===== v2.3 生成 =====

def generate_v23(dialogue: str, tone: str) -> tuple[str, dict]:
    if tone == "诊断式":
        type_instruction = (
            "请生成**诊断式弹窗**（100-200字）。"
            "必须：先承认发心 → 揭示具体模式 → 给出一个微小可做的尝试。"
        )
    else:
        type_instruction = (
            "请生成**鼓励式弹窗**（30-60字）。"
            "必须：具体点出家长刚展现的积极模式 → 简短有力。"
        )
    user_content = f"""当前对话：
{dialogue}

{type_instruction}

请直接输出弹窗全文（不附加解释、不输出JSON、不输出"弹窗："等前缀）："""

    raw = call_llm(v23_prompt_cache, user_content, max_tokens=400)
    return raw, {"tone": tone, "should_popup": True, "skip_reason": None}


# ===== v4.0.12 生成 =====

def generate_v4012(dialogue: str, tone: str) -> tuple[str, dict]:
    if tone == "诊断式":
        type_instruction = (
            "请生成**诊断式弹窗**（100-200字）。"
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

    raw = call_llm(v4012_prompt_cache, user_content, max_tokens=400)
    return raw, {"tone": tone, "should_popup": True, "skip_reason": None}


# 全局缓存，在 main 中赋值
v23_prompt_cache = ""
v4012_prompt_cache = ""


# ===== 通用测试 =====

def run_version(name: str, cases: list, gen_fn, judge) -> dict:
    from dspy import Example
    results = []
    cache = {}
    for i, case in enumerate(cases):
        dialogue = case["question"]
        golden = case["answer"]
        expected_tone = case.get("tone", "诊断式")
        case_id = case.get("case_id", f"case_{i}")
        print(f"\n  [{name}] [{i+1}/{len(cases)}] id={case_id} tone={expected_tone}")
        try:
            if dialogue in cache:
                popup, meta, elapsed = cache[dialogue]
                print(f"    复用缓存 ({elapsed:.1f}s)")
            else:
                start = time.time()
                popup, meta = gen_fn(dialogue, expected_tone)
                elapsed = time.time() - start
                cache[dialogue] = (popup, meta, elapsed)
                print(f"    生成耗时 {elapsed:.1f}s, 输出 {len(popup)} 字")

            if not popup:
                weighted = 0.0
                print(f"    ⚠️ 空输出 → 0 分")
            else:
                gold_ex = Example(question=dialogue, answer=golden)
                pred_ex = Example(answer=popup)
                try:
                    weighted = judge(gold_ex, pred_ex, trace=False)
                except Exception as e:
                    print(f"    ⚠️ Judge 异常: {e}")
                    weighted = 0.0

            passed = weighted >= 0.70
            print(f"    分数: {weighted:.3f} {'✅' if passed else '❌'}")
            results.append({
                "case_id": case_id,
                "expected_tone": expected_tone,
                "actual_tone": meta.get("tone"),
                "generated": popup,
                "golden": golden,
                "weighted_score": weighted,
            })
        except Exception as e:
            print(f"    ❌ 异常: {e}")
            results.append({"case_id": case_id, "error": str(e)})

    scores = [r["weighted_score"] for r in results if "weighted_score" in r]
    avg = sum(scores) / len(scores) if scores else 0
    passed = sum(1 for s in scores if s >= 0.70)
    print(f"\n  [{name}] {len(cases)} 题 | {passed}✅/{len(cases)-passed}❌ | 均分: {avg:.3f}")
    return {
        "version": name,
        "avg_score": avg,
        "passed": passed,
        "failed": len(cases) - passed,
        "results": results,
    }


def main():
    global v23_prompt_cache, v4012_prompt_cache

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
    v23_prompt_cache = V23_PROMPT_PATH.read_text(encoding="utf-8")
    v4012_prompt_cache = V4012_PROMPT_PATH.read_text(encoding="utf-8")
    print(f"v2.3: {len(v23_prompt_cache)} 字")
    print(f"v4.0.12: {len(v4012_prompt_cache)} 字")

    # Judge — override 硬编码的 Claude key/url
    import llm_judge_metric as jmod
    jmod.CLAUDE_KEY = JUDGE_API_KEY
    jmod.CLAUDE_URL = JUDGE_API_BASE + "/messages"
    jmod.CLAUDE_MODEL = "claude-opus-4-7"

    backend = os.getenv("JUDGE_BACKEND", "claude")
    judge = LLMJudgeMetric(judge_backend=backend)
    print(f"生成: deepseek/{GEN_MODEL} | Judge: {backend} @ {jmod.CLAUDE_URL}")

    all_rounds_v23 = []
    all_rounds_v4012 = []
    all_details = []

    for r in range(args.rounds):
        print(f"\n{'='*60}\n  Round {r+1}/{args.rounds}\n{'='*60}")
        print(f"\n--- v2.3 ---")
        res_v23 = run_version("v2.3", cases, generate_v23, judge)
        all_rounds_v23.append(res_v23["avg_score"])
        print(f"\n--- v4.0.12 ---")
        res_v4012 = run_version("v4.0.12", cases, generate_v4012, judge)
        all_rounds_v4012.append(res_v4012["avg_score"])
        all_details.append({
            "round": r + 1,
            "v23": res_v23,
            "v4012": res_v4012,
        })

    # 汇总
    v23_avg = sum(all_rounds_v23) / len(all_rounds_v23)
    v4012_avg = sum(all_rounds_v4012) / len(all_rounds_v4012)
    v23_var = max(all_rounds_v23) - min(all_rounds_v23)
    v4012_var = max(all_rounds_v4012) - min(all_rounds_v4012)

    print(f"\n{'='*60}")
    print(f"  对比汇总（独立测试前 {args.n} 题, {args.rounds} 轮）")
    print(f"{'='*60}")
    print(f"  v2.3:    {[f'{a:.3f}' for a in all_rounds_v23]} | 均分 {v23_avg:.3f} | variance {v23_var:.3f}")
    print(f"  v4.0.12: {[f'{a:.3f}' for a in all_rounds_v4012]} | 均分 {v4012_avg:.3f} | variance {v4012_var:.3f}")
    print(f"  差距: v4.0.12 领先 {v4012_avg - v23_avg:+.3f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"compare_v23_v4012_n{args.n}_{timestamp}.json"
    output_path.write_text(json.dumps({
        "n": args.n,
        "rounds": args.rounds,
        "v23_rounds": all_rounds_v23,
        "v4012_rounds": all_rounds_v4012,
        "v23_avg": v23_avg,
        "v4012_avg": v4012_avg,
        "v23_variance": v23_var,
        "v4012_variance": v4012_var,
        "diff": v4012_avg - v23_avg,
        "details": all_details,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果保存: {output_path}")


if __name__ == "__main__":
    main()
