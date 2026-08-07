# @transient — v2.0 在 REN-21 手写用例上的评估（14条），跑完诊断后删除
"""v2.0 手写用例评估：用 REN-21 的 14 条手写职场用例评估 v2.0 弹窗质量。

- 生成：popup_generator.generate_popup（v2.0 预分析解析，DeepSeek v4-pro）
- 评审：judge_glm.GLMJudge（GLM 5.2 五维软分 + v2.0 硬规则 + aggregate）
- golden 对照：worksheet 专家标注（打分区间 + 该弹句/盲区）

Usage:
    python eval_handwritten_v20.py            # 全部 14 条 × n 次
    python eval_handwritten_v20.py --n 2      # 每条 n 次（默认 3）
    python eval_handwritten_v20.py --cases zip   # 只看 zip 10 条
    python eval_handwritten_v20.py --only golden # 只看带 golden 的 4 条
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from popup_generator import generate_popup
from judge_glm import GLMJudge

RESULTS_DIR = HERE / "results" / "handwritten_eval"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CASES_DIR = HERE / "data" / "handwritten_cases"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_handwritten")


def load_cases() -> list[dict]:
    return [json.loads(f.read_text(encoding="utf-8")) for f in sorted(CASES_DIR.glob("*.json"))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="每条用例跑 n 次")
    ap.add_argument("--cases", choices=["all", "zip", "golden"], default="all")
    ap.add_argument("--prompt", default=str(HERE / "system_prompt_v2.0.txt"))
    args = ap.parse_args()

    system_prompt = Path(args.prompt).read_text(encoding="utf-8")
    all_cases = load_cases()
    if args.cases == "zip":
        cases = [c for c in all_cases if c["src"] == "zip"]
    elif args.cases == "golden":
        cases = [c for c in all_cases if c["src"] == "worksheet"]
    else:
        cases = all_cases

    # 环境：GLM key 从 parent-child-coach/.env 注入（QIANFAN_API_KEY → GLM_API_KEY）
    if not os.environ.get("GLM_API_KEY"):
        env = HERE.parent / "parent-child-coach" / ".env"
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                if line.strip() and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())
        # QIANFAN_API_KEY 是 GLM 评审 key 的别名 → 映射给 GLM_API_KEY
        if os.environ.get("QIANFAN_API_KEY") and not os.environ.get("GLM_API_KEY"):
            os.environ["GLM_API_KEY"] = os.environ["QIANFAN_API_KEY"]

    judge = GLMJudge()
    logger.info("评审模型: %s", judge.model)
    logger.info("评估 %d 条用例 × n=%d", len(cases), args.n)

    results = {"prompt": args.prompt, "n": args.n, "run_at": datetime.now().isoformat(),
               "cases": []}

    for c in cases:
        cid = c["case_id"]
        dialogue = c["dialogue"]
        if not dialogue:
            logger.warning("跳过 %s（无对话）", cid)
            continue
        logger.info("== %s (%s) ==", cid, c["type"])
        case_out = {"case_id": cid, "src": c["src"], "type": c["type"],
                    "subclass": c["subclass"], "topic": c["topic"],
                    "golden": c.get("golden"), "runs": []}
        for i in range(args.n):
            try:
                popup = generate_popup(system_prompt, dialogue)
            except Exception as e:  # noqa: BLE001
                logger.error("生成失败 %s run%d: %s", cid, i, e)
                popup = None
            if popup is None:
                case_out["runs"].append({"run": i, "popup": None,
                                         "verdict": "silent",
                                         "judge": None})
                continue
            try:
                jr = judge.judge_case(c, [{"text": popup}])
            except Exception as e:  # noqa: BLE001
                logger.error("评审失败 %s run%d: %s", cid, i, e)
                jr = None
            case_out["runs"].append({"run": i, "popup": popup, "verdict": "pop",
                                     "judge": jr})
        results["cases"].append(case_out)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"eval_handwritten_v20_n{args.n}_{ts}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ 结果已写入 {out}")

    # ── 控制台摘要 ──
    print("\n=== 摘要 ===")
    n_total = n_pop = n_silent = 0
    sum_agg = 0.0
    for co in results["cases"]:
        pops = [r for r in co["runs"] if r["verdict"] == "pop"]
        aggs = [r["judge"]["aggregate"] for r in pops if r.get("judge")]
        silent = [r for r in co["runs"] if r["verdict"] == "silent"]
        n_total += len(co["runs"]); n_pop += len(pops); n_silent += len(silent)
        g_scores = (co.get("golden") or {}).get("scores") or []
        g_avg = sum(g_scores) / len(g_scores) if g_scores else None
        agg_avg = sum(aggs) / len(aggs) if aggs else None
        if agg_avg is not None: sum_agg += agg_avg
        golden_tag = f" 专家均分{g_avg:.1f}" if g_avg else ""
        print(f"  {co['case_id'][:38]:38s} 弹窗率{len(pops)}/{len(co['runs'])} "
              f"aggregate均{agg_avg if agg_avg is None else round(agg_avg,2)}{golden_tag}")
    pop_rate = n_pop / n_total * 100 if n_total else 0
    judged = [c for c in results["cases"]
              if any(r.get("judge") for r in c["runs"] if r.get("judge"))]
    agg_runs = [r["judge"]["aggregate"] for c in results["cases"]
                for r in c["runs"] if r.get("judge")]
    avg_agg = sum(agg_runs) / len(agg_runs) if agg_runs else None
    print(f"\n总体: 弹窗率 {pop_rate:.0f}% ({n_pop}/{n_total}) | 评审到 {len(agg_runs)} 个弹窗 | "
          f"平均 aggregate {avg_agg if avg_agg is None else round(avg_agg,2)}")


if __name__ == "__main__":
    main()
