"""对比 v4.0.18 vs zip v4.0.17 — 12 独立测试题 × 3 轮。

v4.0.18 = zip v4.0.17 + 术语外溢修复（3处）
生成：DeepSeek V4
裁判：Claude via Xingluan

用法:
  python scripts/compare_v418_v417_zip.py
  python scripts/compare_v418_v417_zip.py --n 3  # 每 case 跑 n 轮（默认 3）
"""

import argparse
import json
import os
import statistics
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import litellm
litellm.suppress_debug_info = True

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent

sys.path.insert(0, str(PROJECT / "scripts"))
from llm_judge_metric import LLMJudgeMetric

# ── 路径 ──
V418_PROMPT_PATH = PROJECT / "system_prompt_v4.0.18.txt"
ZIP_PATH = Path("C:/Users/h/Desktop/xingling_local_run_package.zip")
DATASET_PATH = PROJECT / "data" / "new_12_independent.json"
RESULTS_DIR = PROJECT / "results" / "compare_tests"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── 生成模型：DeepSeek V4 ──
GEN_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GEN_API_BASE = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
GEN_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


def load_zip_prompt():
    """从 zip 包中读取 v4.0.17 提示词。"""
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        for name in zf.namelist():
            if name.endswith("system_prompt_v4.0.17.txt"):
                return zf.read(name).decode("utf-8")
    raise FileNotFoundError("zip 中未找到 system_prompt_v4.0.17.txt")


def generate_popup(system_prompt: str, dialogue: str, version_label: str) -> str | None:
    """调用 DeepSeek 生成弹窗。返回弹窗文本，失败返回 None。"""
    user_msg = f"当前对话：\n{dialogue}"
    try:
        resp = litellm.completion(
            model=f"deepseek/{GEN_MODEL}",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=1024,
            api_key=GEN_API_KEY,
            api_base=GEN_API_BASE,
            timeout=180,
        )
        content = (resp.choices[0].message.content or "").strip()
        return content if content else None
    except Exception as e:
        print(f"    ❌ [{version_label}] 生成失败: {e}")
        return None


def judge_popup(judge, dialogue: str, popup: str) -> float:
    """用 Claude Judge 打分。"""
    from dspy import Example
    gold_ex = Example(question=dialogue, answer="")
    pred_ex = Example(answer=popup)
    try:
        return judge(gold_ex, pred_ex, trace=False)
    except Exception as e:
        print(f"    ⚠️ Judge 异常: {e}")
        return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3, help="每 case 跑 n 轮（默认 3）")
    args = parser.parse_args()
    n_rounds = args.n

    # ── 加载 ──
    print("=" * 80)
    print("v4.0.18 vs zip v4.0.17 对比测试")
    print("=" * 80)

    v418_prompt = V418_PROMPT_PATH.read_text(encoding="utf-8")
    v417_zip_prompt = load_zip_prompt()
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    print(f"v4.0.18: {len(v418_prompt)} 字, {v418_prompt.count(chr(10))} 行")
    print(f"v4.0.17(zip): {len(v417_zip_prompt)} 字, {v417_zip_prompt.count(chr(10))} 行")
    print(f"数据集: {len(dataset)} 题 × {n_rounds} 轮 = {len(dataset) * n_rounds} 次生成")
    print(f"生成模型: DeepSeek {GEN_MODEL}")
    print(f"裁判模型: Claude (xingluan)\n")

    judge = LLMJudgeMetric()

    # ── 逐题对比 ──
    all_results = []
    v418_all_scores = []
    v417_all_scores = []

    for case_idx, case in enumerate(dataset):
        case_id = case.get("case_id", f"case_{case_idx}")
        dialogue = case["question"]
        d_short = dialogue[:60].replace("\n", " ")
        print(f"\n{'─' * 60}")
        print(f"[{case_idx + 1}/{len(dataset)}] {case_id}: {d_short}...")

        case_scores_418 = []
        case_scores_417 = []

        for r in range(n_rounds):
            print(f"  轮次 {r + 1}/{n_rounds}:")

            # 交替顺序避免顺序偏差
            if r % 2 == 0:
                popup_418 = generate_popup(v418_prompt, dialogue, "v4.0.18")
                time.sleep(0.5)
                popup_417 = generate_popup(v417_zip_prompt, dialogue, "v4.0.17-zip")
            else:
                popup_417 = generate_popup(v417_zip_prompt, dialogue, "v4.0.17-zip")
                time.sleep(0.5)
                popup_418 = generate_popup(v418_prompt, dialogue, "v4.0.18")

            # 打分
            score_418 = judge_popup(judge, dialogue, popup_418) if popup_418 else 0.0
            score_417 = judge_popup(judge, dialogue, popup_417) if popup_417 else 0.0

            case_scores_418.append(score_418)
            case_scores_417.append(score_417)
            v418_all_scores.append(score_418)
            v417_all_scores.append(score_417)

            delta = score_418 - score_417
            winner = "v4.0.18 ✅" if delta > 0.01 else ("v4.0.17-zip ✅" if delta < -0.01 else "平局 ⚖️")
            print(f"    v4.0.18: {score_418:.3f} | v4.0.17-zip: {score_417:.3f} | Δ={delta:+.3f} | {winner}")

            all_results.append({
                "case_id": case_id,
                "round": r + 1,
                "dialogue": dialogue[:300],
                "popup_v418": popup_418,
                "score_v418": score_418,
                "popup_v417_zip": popup_417,
                "score_v417_zip": score_417,
            })

            time.sleep(0.3)

        # 该 case 汇总
        avg_418 = sum(case_scores_418) / len(case_scores_418)
        avg_417 = sum(case_scores_417) / len(case_scores_417)
        wins_418 = sum(1 for a, b in zip(case_scores_418, case_scores_417) if a > b + 0.01)
        wins_417 = sum(1 for a, b in zip(case_scores_418, case_scores_417) if b > a + 0.01)
        print(f"  → {case_id} 均分: v4.0.18={avg_418:.3f}, v4.0.17-zip={avg_417:.3f} "
              f"| 局数: {wins_418}胜-{wins_417}胜-{n_rounds - wins_418 - wins_417}平")

    # ── 统计分析 ──
    print("\n" + "=" * 80)
    print("统计结果")
    print("=" * 80)

    v418_mean = statistics.mean(v418_all_scores)
    v417_mean = statistics.mean(v417_all_scores)
    v418_std = statistics.stdev(v418_all_scores)
    v417_std = statistics.stdev(v417_all_scores)

    print(f"\nv4.0.18:       {v418_mean:.4f} ± {v418_std:.4f}")
    print(f"v4.0.17-zip:   {v417_mean:.4f} ± {v417_std:.4f}")
    print(f"Δ (v4.0.18 - v4.0.17-zip): {v418_mean - v417_mean:+.4f}")

    # 胜率
    total_rounds = len(all_results)
    wins_418 = sum(1 for r in all_results if r["score_v418"] > r["score_v417_zip"] + 0.01)
    wins_417 = sum(1 for r in all_results if r["score_v417_zip"] > r["score_v418"] + 0.01)
    ties = total_rounds - wins_418 - wins_417
    print(f"\n胜率: v4.0.18 {wins_418}胜 / v4.0.17-zip {wins_417}胜 / 平 {ties}")
    print(f"决胜率: v4.0.18 = {wins_418 / total_rounds * 100:.1f}%, "
          f"v4.0.17-zip = {wins_417 / total_rounds * 100:.1f}%")

    # Per-case 汇总
    print(f"\n{'Case':<12} {'v4.0.18':>8} {'v4.0.17-zip':>12} {'Δ':>8} {'裁决':>6}")
    print("-" * 52)
    case_summaries = {}
    for r in all_results:
        cid = r["case_id"]
        if cid not in case_summaries:
            case_summaries[cid] = {"v418": [], "v417": []}
        case_summaries[cid]["v418"].append(r["score_v418"])
        case_summaries[cid]["v417"].append(r["score_v417_zip"])

    for cid, scores in case_summaries.items():
        avg_418 = statistics.mean(scores["v418"])
        avg_417 = statistics.mean(scores["v417"])
        delta = avg_418 - avg_417
        verdict = "v4.0.18 ✓" if delta > 0.01 else ("v4.0.17 ✓" if delta < -0.01 else "平")
        print(f"{cid:<12} {avg_418:>8.3f} {avg_417:>12.3f} {delta:>+8.3f} {verdict:>6}")

    # ── 保存 ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"compare_v418_v417_zip_n{n_rounds}_{timestamp}.json"
    output = {
        "config": {
            "test_size": len(dataset),
            "rounds": n_rounds,
            "gen_model": f"deepseek/{GEN_MODEL}",
            "judge_model": "claude (xingluan)",
            "timestamp": timestamp,
        },
        "prompts": {
            "v418_lines": v418_prompt.count("\n"),
            "v417_zip_lines": v417_zip_prompt.count("\n"),
        },
        "summary": {
            "v418_mean": v418_mean,
            "v418_std": v418_std,
            "v417_zip_mean": v417_mean,
            "v417_zip_std": v417_std,
            "delta": v418_mean - v417_mean,
            "wins_v418": wins_418,
            "wins_v417_zip": wins_417,
            "ties": ties,
            "win_rate_v418": wins_418 / total_rounds,
            "win_rate_v417_zip": wins_417 / total_rounds,
        },
        "per_case": {cid: {
            "v418_mean": statistics.mean(s["v418"]),
            "v417_zip_mean": statistics.mean(s["v417"]),
            "delta": statistics.mean(s["v418"]) - statistics.mean(s["v417"]),
            "scores_v418": s["v418"],
            "scores_v417_zip": s["v417"],
        } for cid, s in case_summaries.items()},
        "all_rounds": all_results,
    }
    json.dump(output, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存: {out_path}")


if __name__ == "__main__":
    main()
