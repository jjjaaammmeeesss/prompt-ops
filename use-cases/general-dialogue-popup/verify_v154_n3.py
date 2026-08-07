#!/usr/bin/env python
"""对 v1.5.4 跑 n=3 验证，确认 4.975 非噪声。"""

import json
import logging
import sys
from pathlib import Path

# 把当前目录加入 sys.path 以便导入 co_evolve
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from co_evolve import run_eval, logger as co_logger

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("verify_v154")

PROMPT_PATH = HERE / "results" / "co_evolve" / "run_20260729_163350" / "round_003" / "v1.5.4_candidate.txt"
DATASET_PATH = HERE / "data" / "business_dialogues_10.json"
OUTPUT_DIR = HERE / "results" / "co_evolve" / "run_20260729_163350" / "verify_v154_n3"
N_RUNS = 3

def main():
    if not PROMPT_PATH.exists():
        logger.error("Prompt 文件不存在: %s", PROMPT_PATH)
        sys.exit(1)

    cases = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    logger.info("加载 %d 个职场场景", len(cases))
    logger.info("Prompt: %s (%d 字)", PROMPT_PATH.name, len(PROMPT_PATH.read_text(encoding="utf-8")))
    logger.info("n=%d, 共 %d 次生成+评审", N_RUNS, len(cases) * N_RUNS)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results, summary = run_eval(PROMPT_PATH, cases, n_runs=N_RUNS)

    # 保存逐案结果
    results_path = OUTPUT_DIR / "v154_n3_eval.json"
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("逐案结果已保存: %s", results_path)

    # 保存汇总
    summary_path = OUTPUT_DIR / "v154_n3_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("汇总已保存: %s", summary_path)

    # 关键指标
    print("\n" + "=" * 60)
    print("  v1.5.4 n=3 验证结果")
    print("=" * 60)
    print(f"  综合分 (aggregate):     {summary['mean_aggregate']:.3f}")
    print(f"  insight:                {summary.get('mean_insight', 'N/A'):.3}" if 'mean_insight' in summary else f"  insight:                N/A")
    if 'dim_means' in summary:
        for dim, val in summary['dim_means'].items():
            print(f"  {dim}:{' ' * (24 - len(dim))}{val:.3f}")
    print(f"  硬规则违规总数:         {summary['hard_violations']}")
    print(f"  弹窗率:                 {summary.get('popup_rate', 'N/A')}")
    print(f"  应弹未弹:               {summary.get('missed_popups', 'N/A')}")
    print(f"  基线 (n=1):             4.975")
    print(f"  偏差:                   {summary['mean_aggregate'] - 4.975:+.3f}")
    print("=" * 60)

    # 对比 n=1 的结果
    n1_path = HERE / "results" / "co_evolve" / "run_20260729_163350" / "round_003" / "v151_eval.json"
    if n1_path.exists():
        n1_data = json.loads(n1_path.read_text(encoding="utf-8"))
        n1_scores = [r['aggregate'] for r in n1_data if r.get('aggregate', 0) > 0]
        if n1_scores:
            n1_mean = sum(n1_scores) / len(n1_scores)
            print(f"\n  n=1 均值 (原始):        {n1_mean:.3f}")
            print(f"  n=3 均值 (验证):        {summary['mean_aggregate']:.3f}")
            print(f"  差异:                   {summary['mean_aggregate'] - n1_mean:+.3f}")
    print()

if __name__ == "__main__":
    main()
