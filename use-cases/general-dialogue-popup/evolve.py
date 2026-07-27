"""自我迭代主循环 — 一般场景弹窗 prompt 自动优化。

流程（每轮）：
  1. 用当前 prompt 跑全部测试案例（DeepSeek v4 生成弹窗）
  2. GLM-5.2 judge 逐案打分（软维度 + 硬规则）
  3. 汇总最低分案例的反馈，交给 DeepSeek v4 改写 prompt（变异）
  4. 新 prompt 下轮评估，总分更高则保留为新的最优版本，否则丢弃
  5. 连续 2 轮无提升或达到 max_rounds 则停止

Usage:
    export DEEPSEEK_API_KEY=sk-...   # 生成 + 变异
    export GLM_API_KEY=...           # judge
    python evolve.py --rounds 5

每轮结果写入 results/evolve/round_XXX.json，保留的 prompt 存为
system_prompt_v1.<N>.txt（版本号递增）。
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path

from runner_v10 import V10Runner
from judge_glm import GLMJudge
from judge_opus import ClaudeJudge

logger = logging.getLogger("evolve")

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results" / "evolve"

_MUTATOR_PROMPT = """你是一位提示词工程专家。下面是一个"沟通现场弹窗"AI 的当前系统提示词，以及它在测试集上表现最差的案例反馈。

【当前提示词】
{current_prompt}

【最差案例反馈】
{failures}

【改写纪律 —— 违反任何一条都会让这次改动被丢弃】
1. **先修硬规则，再修软洞察**。如果反馈里有字数超限、`——` 未单独成行、墙后多句、健康对话误弹——必须先修这些。
2. **只改 1–3 处具体语句/段落**，不允许推倒重写，不允许大幅调整结构。
3. **必须原样保留**：角色定位、生命立场价值观、隐性判断层、`——` 功能墙、字数死线（60-180，绝不超200）、"安好"安静信号、"只对「你」一个人说话"、发出前三项默查。
4. **保持全文口语化、第二人称、无术语风格**。

请输出改写后的完整提示词全文（不要输出任何解释、不要用 markdown 代码块包裹）。"""


def run_evaluation(runner_args: dict, prompt_path: Path, cases: list[dict], judge: GLMJudge | ClaudeJudge) -> dict:
    """用一个 prompt 跑全部案例并评分。"""
    runner = V10Runner(prompt_path=str(prompt_path), **runner_args)
    case_results = []
    for case in cases:
        try:
            out = runner.run(dialogue_text=case["dialogue"])
        except Exception as exc:
            logger.error("案例 %s 运行失败: %s", case["id"], exc)
            out = {"popups": []}
        try:
            judged = judge.judge_case(case, out["popups"])
        except Exception as exc:
            logger.error("案例 %s 评分失败: %s", case["id"], exc)
            judged = {"aggregate": 1.0, "soft": None,
                      "hard": {"pass": False, "violations": [f"judge异常:{exc}"]},
                      "comment": "judge 异常", "popup_text": None}
        judged["case_id"] = case["id"]
        judged["expect"] = case.get("expect", "")
        judged["n_popups"] = len(out["popups"])
        case_results.append(judged)
        logger.info("  [%s] %s → %.2f (%s)", case["id"], case.get("expect", "?"),
                    judged["aggregate"], judged["comment"][:30])

    aggs = [r["aggregate"] for r in case_results]
    return {
        "mean": round(sum(aggs) / len(aggs), 3),
        "popup_rate": round(sum(1 for r in case_results if r["popup_text"]) / len(case_results), 3),
        "hard_violations": sum(len(r["hard"]["violations"]) for r in case_results),
        "cases": case_results,
    }


def summarize_failures(case_results: list[dict], top_n: int = 5) -> str:
    """取最低分案例，生成给变异器看的反馈摘要。"""
    worst = sorted(case_results, key=lambda r: r["aggregate"])[:top_n]
    lines = []
    for r in worst:
        lines.append(f"- 案例 {r['case_id']}（期望：{r['expect']}）得分 {r['aggregate']}")
        if r["hard"]["violations"]:
            lines.append(f"  硬规则违规: {'; '.join(r['hard']['violations'])}")
        if r["soft"]:
            dims = {k: v for k, v in r["soft"].items() if k != "comment"}
            lines.append(f"  软维度: {dims}")
        lines.append(f"  评语: {r['comment']}")
        if r["popup_text"]:
            lines.append(f"  弹窗原文: {r['popup_text'][:150]}")
    return "\n".join(lines)


def mutate(runner_args: dict, current_prompt: str, failures: str) -> str:
    """DeepSeek v4 改写 prompt。"""
    import litellm

    resp = litellm.completion(
        model=runner_args["model"],
        api_key=runner_args.get("api_key"),
        api_base=runner_args.get("api_base"),
        messages=[{"role": "user", "content": _MUTATOR_PROMPT.format(
            current_prompt=current_prompt, failures=failures)}],
        temperature=0.4,
        max_tokens=6000,
        timeout=300,
    )
    text = (resp.choices[0].message.content or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return text


def next_version_path(directory: Path) -> Path:
    """找下一个 system_prompt_v1.N.txt 版本号。"""
    existing = [int(m.group(1)) for p in directory.glob("system_prompt_v1.*.txt")
                if (m := re.match(r"system_prompt_v1\.(\d+)\.txt", p.name))]
    n = max(existing, default=0) + 1
    return directory / f"system_prompt_v1.{n}.txt"


def main():
    parser = argparse.ArgumentParser(description="一般场景弹窗 prompt 自我迭代")
    parser.add_argument("--rounds", type=int, default=5, help="最大迭代轮数（不含基线轮）")
    parser.add_argument("--data", default=str(HERE / "data" / "test_dialogues.json"))
    parser.add_argument("--prompt", default=str(HERE / "system_prompt_v1.0.txt"))
    parser.add_argument("--model", default="deepseek/deepseek-v4-pro")
    parser.add_argument("--api-base", default="https://api.deepseek.com/v1")
    parser.add_argument("--judge-model", default="glm-5.2")
    parser.add_argument("--judge-base", default="https://qianfan.baidubce.com/v2",
                        help="judge API base（默认百度千帆，智谱官方为 https://open.bigmodel.cn/api/paas/v4）")
    parser.add_argument("--patience", type=int, default=2, help="连续无提升轮数上限")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s: %(message)s")

    import os
    runner_args = {
        "model": args.model,
        "api_base": args.api_base,
        "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
    }
    if "claude" in args.judge_model.lower() or "opus" in args.judge_model.lower():
        judge: GLMJudge | ClaudeJudge = ClaudeJudge(model=args.judge_model, api_base=args.judge_base)
    else:
        judge = GLMJudge(model=args.judge_model, api_base=args.judge_base)

    cases = json.loads(Path(args.data).read_text(encoding="utf-8"))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    best_prompt_path = Path(args.prompt)
    history = []

    # ── 基线轮 ──
    logger.info("═══ 基线轮: %s ═══", best_prompt_path.name)
    best = run_evaluation(runner_args, best_prompt_path, cases, judge)
    best["prompt"] = best_prompt_path.name
    history.append(best)
    (RESULTS_DIR / "round_000_baseline.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("基线: mean=%.3f popup_rate=%.2f hard_violations=%d",
                best["mean"], best["popup_rate"], best["hard_violations"])

    no_improve = 0
    for rnd in range(1, args.rounds + 1):
        logger.info("═══ 第 %d 轮 ═══", rnd)
        current_text = best_prompt_path.read_text(encoding="utf-8")
        failures = summarize_failures(best["cases"])

        candidate_text = mutate(runner_args, current_text, failures)
        candidate_path = next_version_path(HERE)
        candidate_path.write_text(candidate_text, encoding="utf-8")
        logger.info("候选 prompt: %s (%d 字)", candidate_path.name, len(candidate_text))

        result = run_evaluation(runner_args, candidate_path, cases, judge)
        result["prompt"] = candidate_path.name
        result["failures_input"] = failures
        (RESULTS_DIR / f"round_{rnd:03d}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        improved = result["mean"] > best["mean"] + 0.02
        status = "✅ 保留" if improved else "❌ 丢弃"
        logger.info("第 %d 轮: mean=%.3f (基线 %.3f) → %s", rnd, result["mean"], best["mean"], status)

        if improved:
            best = result
            best_prompt_path = candidate_path
            no_improve = 0
        else:
            candidate_path.unlink()  # 丢弃不合格版本
            no_improve += 1
            if no_improve >= args.patience:
                logger.info("连续 %d 轮无提升，停止。", no_improve)
                break

    report = {
        "best_prompt": best_prompt_path.name,
        "best_mean": best["mean"],
        "rounds_run": len(history),
        "history": [{"prompt": h["prompt"], "mean": h["mean"],
                     "popup_rate": h["popup_rate"], "hard_violations": h["hard_violations"]}
                    for h in history + []] ,
    }
    # history 只含基线 + 被保留的版本之外，简单重记全部轮次
    all_rounds = sorted(RESULTS_DIR.glob("round_*.json"))
    report["history"] = [
        {"file": p.name,
         "mean": json.loads(p.read_text(encoding="utf-8"))["mean"]}
        for p in all_rounds
    ]
    (RESULTS_DIR / "final_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "═" * 50)
    print(f"迭代结束。最优 prompt: {report['best_prompt']} (mean={report['best_mean']})")
    for h in report["history"]:
        print(f"  {h['file']}: {h['mean']}")


if __name__ == "__main__":
    main()
