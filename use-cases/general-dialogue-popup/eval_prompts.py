"""手动评估多个 prompt 在测试集上的表现（不迭代，只出对比报告）。"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from runner_v10 import V10Runner
from judge_glm import GLMJudge
from judge_opus import ClaudeJudge

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("eval_prompts")


def evaluate(prompt_path: Path, cases: list[dict], runner_args: dict, judge: GLMJudge) -> dict:
    runner = V10Runner(prompt_path=str(prompt_path), **runner_args)
    results = []
    for case in cases:
        try:
            out = runner.run(dialogue_text=case["dialogue"])
        except Exception as exc:
            logger.error("%s run failed: %s", case["id"], exc)
            out = {"popups": []}
        judged = judge.judge_case(case, out["popups"])
        judged["case_id"] = case["id"]
        judged["expect"] = case.get("expect", "")
        results.append(judged)

    aggs = [r["aggregate"] for r in results]
    return {
        "prompt": prompt_path.name,
        "mean": round(sum(aggs) / len(aggs), 3),
        "popup_rate": round(sum(1 for r in results if r["popup_text"]) / len(results), 3),
        "hard_violations": sum(len(r["hard"]["violations"]) for r in results),
        "cases": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/test_dialogues.json")
    parser.add_argument("--prompts", nargs="+", default=["system_prompt_v1.0.txt", "system_prompt_v1.1.txt", "system_prompt_v1.2.txt"])
    parser.add_argument("--output", default="results/evolve/manual_eval.json")
    parser.add_argument("--model", default="deepseek/deepseek-v4-pro")
    parser.add_argument("--api-base", default="https://api.deepseek.com/v1")
    parser.add_argument("--judge-model", default="glm-5.2")
    parser.add_argument("--judge-base", default="https://qianfan.baidubce.com/v2")
    args = parser.parse_args()

    import os
    runner_args = {"model": args.model, "api_base": args.api_base, "api_key": os.environ.get("DEEPSEEK_API_KEY", "")}
    if "claude" in args.judge_model.lower() or "opus" in args.judge_model.lower():
        judge: GLMJudge | ClaudeJudge = ClaudeJudge(model=args.judge_model, api_base=args.judge_base)
    else:
        judge = GLMJudge(model=args.judge_model, api_base=args.judge_base)

    cases = json.loads(Path(args.data).read_text(encoding="utf-8"))
    report = []
    for p in args.prompts:
        pp = Path(p)
        if not pp.exists():
            logger.warning("跳过不存在的 prompt: %s", p)
            continue
        logger.info("评估 %s ...", pp.name)
        report.append(evaluate(pp, cases, runner_args, judge))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "═" * 60)
    for r in report:
        print(f"{r['prompt']}: mean={r['mean']}, popup_rate={r['popup_rate']}, hard_violations={r['hard_violations']}")


if __name__ == "__main__":
    main()
