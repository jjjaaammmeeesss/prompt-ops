# @persistent — 对比 system_prompt_v3.1 vs v2.0（用对应执行器 + GLM 五维 + 共享生成）
"""对比 v3.1 vs v2.0 成人通用弹窗 — 10 个职场场景 × n 次。

铁律：用对应的提示词 + 对应的执行器。
    v2.0 → popup_generator.py       (PROMPT_VERSION="v2.0", system_prompt_v2.0.txt)
    v3.1 → popup_generator_v31.py   (PROMPT_VERSION="v3.1", system_prompt_v3.1.txt)

生成：DeepSeek v4-pro（各版本自己的执行器，含预分析解析）
评审：GLM 5.2（百度千帆，五维打分 + 硬规则检查）— Track A
统计：trackB_v20_v31.py 复用本脚本共享生成产物，跑 SystemComparator（bootstrap CI + Wilcoxon）

共享产物：results/compare_tests/_gen_v20_v31_n{n}_{ts}.json（Track B 消费）
输出：results/compare_tests/compare_v20_v31_n{n}_{ts}.json + 一份 scores JSON

Usage:
    python compare_v20_v31.py               # 默认 n=5
    python compare_v20_v31.py --n 3         # 每 case 跑 n 次
    python compare_v20_v31.py --gen-only    # 只生成共享产物，不评审
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results" / "compare_tests"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── 各版本 → 执行器 映射（核心：用对应执行器） ──
sys.path.insert(0, str(HERE))  # 确保能 import 同目录执行器
import popup_generator       # noqa: E402  PROMPT_VERSION="v2.0"
import popup_generator_v31   # noqa: E402  PROMPT_VERSION="v3.1"

VERSIONS = {
    "v2.0": {
        "prompt": HERE / "system_prompt_v2.0.txt",
        "generator": popup_generator.generate_popup,
        "executor_version": popup_generator.__version__,
    },
    "v3.1": {
        "prompt": HERE / "system_prompt_v3.1.txt",
        "generator": popup_generator_v31.generate_popup,
        "executor_version": popup_generator_v31.__version__,
    },
}

# ── 配置 ──
# 生成：DeepSeek v4-pro。直接 API 的 key 当前余额不足，故走百度千帆托管的
# deepseek-v4-pro（同一模型，经 qianfan 可用且 key 有余额）。生成与评审共用 qianfan key。
GEN_MODEL = os.environ.get("GEN_MODEL", "openai/deepseek-v4-pro")
GEN_API_BASE = os.environ.get("GEN_API_BASE", "https://qianfan.baidubce.com/v2")
_TEMPERATURE = float(os.environ.get("GEN_TEMPERATURE", "0.3"))

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "openai/glm-5.2")
JUDGE_API_BASE = os.environ.get("JUDGE_API_BASE", "https://qianfan.baidubce.com/v2")

# 本地开发 key 载体（只在 shell 未设置时兜底加载，不覆盖已有环境）
_KEY_FILES = [
    HERE.parent.parent.parent.parent.parent.parent / "星灵-soul-手搓" / "亲子沟通洞见" / "测试智能体" / ".env",
    HERE.parent.parent.parent / "parent-child-coach" / ".env",
    HERE.parent.parent.parent.parent / ".env",
]


def _load_env_file(path: Path) -> None:
    """极简 .env 解析：把 key=value 灌进 os.environ（不覆盖已有值）。"""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:]
        if "=" in line:
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and v and not os.environ.get(k):
                os.environ[k] = v


def _resolve_keys() -> tuple[str, str]:
    """解析生成 key 与评审 key。返回 (gen_key, judge_key)。

    生成走 qianfan 托管的 deepseek-v4-pro，评审走 qianfan 的 GLM 5.2，
    两者共用 qianfan 系 key（QIANFAN_API_KEY / BAIDU_QIANFAN_KEY / GLM_API_KEY / JUDGE_API_KEY）。
    """
    for f in _KEY_FILES:
        _load_env_file(f)

    qianfan_key = (
        os.environ.get("JUDGE_API_KEY")
        or os.environ.get("GLM_API_KEY")
        or os.environ.get("QIANFAN_API_KEY")
        or os.environ.get("BAIDU_QIANFAN_KEY", "")
    )
    gen_key = qianfan_key or os.environ.get("GEN_API_KEY") or os.environ.get("DEEPSEEK_API_KEY", "")
    return gen_key, qianfan_key


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("compare_v20_v31")


# ── GLM Judge（五维打分 + 硬规则，契约沿用 compare_v20_v21.py） ──

_JUDGE_PROMPT = """你是一位苛刻但公正的沟通教练评审。下面是一段两人对话，以及一个"沟通现场弹窗"AI 生成的弹窗。

弹窗的设计意图：作为一个独立第三方观察者，帮对话双方看见此刻还没看到的东西——纯洞察，不给建议，不站队任何一方，不称"你"而用场景临时功能角色（如讲师/学员、催促方/应允方）。

【对话】
{dialogue}

【弹窗】
{popup}

请按五个维度打分（1-5 整数，5 最好）：

1. insight 洞察质量（从 4 个子维度综合评估）：
   - 准确性：基于对话里实际出现的言语/行为，不脑补、不张冠李戴
   - 相关性：正好扎在对话此刻最没被看到的那个盲点，不是重复废话或次要细节
   - 深度：穿过表面内容，点出言下之意 / 未表达的需求 / 两人互动循环
   - 具体性：绑定到对话中的具体细节，不是模板套话

2. third_party 第三方立场（v2.0 核心新维度）：
   5 分：完全保持独立观察者立场，不把任何一方设为默认服务对象；使用临时功能角色且成对、可原文验证；不站队、不裁判
   4 分：基本保持第三方视角，但有一处轻微偏向某方或角色标签略有猜测成分
   3 分：角色标签存在但单方面（只命名了一方），或叙述中可察觉偏向
   2 分：明确站在某一方角度说话，或角色标签包含价值判断（如"强势者""受害者"）
   1 分：全文以第二人称"你"直接称呼某一方，或明确为某一方辩护/支招

3. language 语言质感：是否口语、像独立观察者在现场轻声说的话？有无术语、说教、模板腔？是否用"也许""可能"等假设语气而非直接宣告？

4. evidence 证据锚定（v1.5.5 起核心能力，v2.0 保留）：
   5 分：洞察紧扣对话中的原句/原词，能在弹窗中找到至少一处三个字以上的原文引用或明确行为锚点
   4 分：洞察有清晰的行为依据（某人说了什么/做了什么），虽未逐字引用但指向明确
   3 分：洞察与对话内容相关，但锚定模糊——说不清到底哪句话触发了这个洞察
   2 分：洞察偏向心理推测（"其实害怕……""内心想要……"），没有言行层面的支撑
   1 分：明显脑补对方内心活动，或编造对话中不存在的情节/动机

5. focus 聚焦度（v1.5.5 起核心能力，v2.0 保留）：
   5 分：弹窗只打一个点，全文围绕一个核心盲区展开，不散不乱；选择依据是"不指出时谁的实际成本更大"
   4 分：有一个主盲区，但末尾轻微带到了第二个点（一笔带过）
   3 分：弹窗混合了两个盲区，各说了一半，没有明确主次
   2 分：弹窗散成三个以上碎片，像检查清单而不是一个聚焦的洞察
   1 分：弹窗是对对话的总结/复述列表，不是任何一个具体的盲区

再写一段 50 字以内的 comment，指出最主要的问题或亮点。

另外检查：弹窗中是否包含任何建议句（"可以试试""下次可以""不妨""建议"等）？如果包含建议句，insight 和 focus 各扣 1 分。

只输出 JSON：{{"insight": 4, "third_party": 5, "language": 4, "evidence": 4, "focus": 4, "comment": "..."}}"""

DIMS = ("insight", "third_party", "language", "evidence", "focus")


def _hard_check(popup_text: str) -> dict:
    """硬规则检查（纯代码）：字数 60-180、无"你"、末尾完整性。"""
    violations = []
    text = popup_text.strip()
    n = len(re.sub(r"\s", "", text))
    if n > 180:
        violations.append(f"字数 {n} 超过 180 硬合规线")
    elif n < 60:
        violations.append(f"字数 {n} 低于 60")
    if "你" in text:
        violations.append("包含禁用字符 `你`")
    if not text.endswith(("。", "？", "！", "”", '"', "」", "』")):
        violations.append("输出疑似截断")
    return {"pass": not violations, "violations": violations}


def _parse_judge_json(raw: str) -> dict | None:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        soft = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    defaults = {"insight": 3, "third_party": 3, "language": 3,
                "evidence": 3, "focus": 3, "comment": ""}
    for k, v in defaults.items():
        soft.setdefault(k, v)
    for k in DIMS:
        try:
            soft[k] = max(1, min(5, int(soft[k])))
        except (TypeError, ValueError):
            soft[k] = 3
    return soft


def judge_popup(dialogue: str, popup_text: str, judge_api_key: str) -> dict:
    """GLM 打分 + 硬规则。返回含 aggregate 的 dict。"""
    import litellm

    prompt = _JUDGE_PROMPT.format(dialogue=dialogue, popup=popup_text)
    raw = ""
    for attempt in range(3):
        try:
            resp = litellm.completion(
                model=JUDGE_MODEL, api_key=judge_api_key, api_base=JUDGE_API_BASE,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=4096, timeout=180,
            )
            raw = (resp.choices[0].message.content or "").strip()
            break
        except Exception as e:
            logger.warning("judge 尝试 %d/3 失败: %s", attempt + 1, e)
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)

    soft = _parse_judge_json(raw)
    if soft is None:
        soft = {d: 3 for d in DIMS}
        soft["comment"] = f"[parse_error] {raw[:80]}"
    hard = _hard_check(popup_text)
    soft_mean = sum(soft[k] for k in DIMS) / len(DIMS)
    penalty = min(1.5, 0.5 * len(hard["violations"]))
    aggregate = round(max(1.0, soft_mean - penalty), 2)
    return {"aggregate": aggregate, "soft": soft, "hard": hard,
            "comment": soft.get("comment", ""), "popup_text": popup_text}


# ── 共享生成：用对应执行器跑各版本 ──

from concurrent.futures import ThreadPoolExecutor, as_completed

MAX_WORKERS = int(os.environ.get("GEN_WORKERS", "4"))


def run_generation(cases: list[dict], n_runs: int, gen_api_key: str) -> dict:
    """对每个 case × 每个版本 × n 次，用各自执行器生成弹窗（线程池并行）。

    返回 {case_id: {id, expect, dialogue, "v2.0": [...], "v3.1": [...]}}，
    并写入共享产物 JSON 供 Track B 消费。
    """
    system_prompts = {v: Path(cfg["prompt"]).read_text(encoding="utf-8")
                      for v, cfg in VERSIONS.items()}
    generators = {v: cfg["generator"] for v, cfg in VERSIONS.items()}

    tasks = []  # (case_id, ver, run_i, case, system_prompt)
    for ci, case in enumerate(cases):
        case_id = case["id"]
        for ver in ("v2.0", "v3.1"):
            for run_i in range(n_runs):
                tasks.append((case_id, ver, run_i, ci, case, system_prompts[ver]))

    results = {}  # (case_id, ver, run_i) -> {"popup":..., "error":...}

    def _one(task):
        case_id, ver, run_i, ci, case, sp = task
        logger.info("[%s %s] run %d/%d (case %d/%d)",
                    ver, case_id, run_i + 1, n_runs, ci + 1, len(cases))
        try:
            popup = generators[ver](
                sp, case["dialogue"],
                model=GEN_MODEL, api_key=gen_api_key, api_base=GEN_API_BASE,
                temperature=_TEMPERATURE, max_tokens=4096,
            )
            return (case_id, ver, run_i), {"popup": popup, "error": None}
        except Exception as e:
            logger.error("生成失败(%s %s): %s", ver, case_id, e)
            return (case_id, ver, run_i), {"popup": None, "error": str(e)}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = [pool.submit(_one, t) for t in tasks]
        for fut in as_completed(futs):
            key, val = fut.result()
            results[key] = val

    gen = {}
    for case in cases:
        case_id = case["id"]
        gen[case_id] = {
            "id": case_id, "expect": case.get("expect", ""),
            "dialogue": case["dialogue"], "v2.0": [], "v3.1": [],
        }
        for ver in ("v2.0", "v3.1"):
            for run_i in range(n_runs):
                gen[case_id][ver].append(results[(case_id, ver, run_i)])

    return gen


# ── 汇总统计 ──

def summarize(results: list[dict], label: str) -> dict:
    valid = [r for r in results if r["popup_text"] is not None and r["aggregate"] > 0]
    all_agg = [r["aggregate"] for r in results if r["aggregate"] > 0]
    soft_scores = {k: [] for k in DIMS}
    for r in valid:
        if r["soft"]:
            for k in DIMS:
                if r["soft"].get(k) is not None:
                    soft_scores[k].append(r["soft"][k])
    hard_violations = sum(len(r.get("hard", {}).get("violations", [])) for r in results)
    n_should_popup = sum(1 for r in results if r["expect"] != "安静")
    n_did_popup = sum(1 for r in results if r["popup_text"] is not None)
    popup_rate = n_did_popup / n_should_popup if n_should_popup > 0 else 0
    missing = sum(1 for r in results if r["expect"] != "安静" and r["popup_text"] is None)

    def mean(vals): return round(sum(vals) / len(vals), 3) if vals else 0

    return {
        "label": label, "n_total": len(results), "n_valid": len(valid),
        "mean_aggregate": mean(all_agg),
        "soft_means": {k: mean(v) for k, v in soft_scores.items()},
        "hard_violations": hard_violations,
        "popup_rate": round(popup_rate, 3), "missing_popups": missing,
        "aggregates": all_agg,
    }


def main(n_runs: int, gen_only: bool = False, from_gen: str | None = None):
    gen_key, judge_key = _resolve_keys()
    if not gen_key:
        logger.error("缺少生成模型 API key（DEEPSEEK_API_KEY / GEN_API_KEY）")
        sys.exit(1)
    if not (gen_only or from_gen) and not judge_key:
        logger.error("缺少评审 key（GLM_API_KEY / QIANFAN_API_KEY / BAIDU_QIANFAN_KEY / JUDGE_API_KEY）")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    t0 = time.perf_counter()

    if from_gen:
        # ── 从既有共享生成产物直接评审（不再生成） ──
        gen = json.loads(Path(from_gen).read_text(encoding="utf-8"))
        gen_out = Path(from_gen)
        n_runs = len(next(iter(gen.values()))["v2.0"])
        logger.info("复用共享生成产物: %s（%d 个 case，n=%d）", from_gen, len(gen), n_runs)
    else:
        dataset_path = HERE / "data" / "business_dialogues_10.json"
        if not dataset_path.exists():
            logger.error("找不到数据集: %s", dataset_path)
            sys.exit(1)
        cases = json.loads(dataset_path.read_text(encoding="utf-8"))
        logger.info("加载 %d 个职场场景；执行器：v2.0=%s / v3.1=%s",
                    len(cases), VERSIONS["v2.0"]["executor_version"],
                    VERSIONS["v3.1"]["executor_version"])

        # ── 共享生成（DeepSeek，用对应执行器） ──
        gen = run_generation(cases, n_runs, gen_key)
        gen_out = RESULTS_DIR / f"_gen_v20_v31_n{n_runs}_{timestamp}.json"
        gen_out.write_text(json.dumps(gen, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("共享生成产物已保存: %s", gen_out)

        if gen_only:
            print(f"\n共享生成完成（仅生成，未评审）：{gen_out}")
            return

    # ── Track A：GLM 五维评审（线程池并行） ──
    scores = {"v2.0": {}, "v3.1": {}}
    results = {"v2.0": [], "v3.1": []}
    judge_tasks = []  # (ver, case_id, dialogue, expect, run)
    for case_id, g in gen.items():
        dialogue = g["dialogue"]
        expect = g["expect"]
        for ver in ("v2.0", "v3.1"):
            for run in g[ver]:
                judge_tasks.append((ver, case_id, dialogue, expect, run))

    def _judge_one(t):
        ver, case_id, dialogue, expect, run = t
        popup = run["popup"]
        if popup is None:
            return ver, {
                "case_id": case_id, "expect": expect, "popup_text": None,
                "aggregate": 0, "soft": None,
                "hard": {"pass": False, "violations": [run["error"] or "生成失败"]},
                "comment": "生成失败或未弹窗", "error": run["error"],
            }
        try:
            er = judge_popup(dialogue, popup, judge_key)
        except Exception as e:
            logger.error("评审失败(%s %s): %s", ver, case_id, e)
            er = {"aggregate": 0, "soft": None,
                  "hard": {"pass": False, "violations": [f"评审异常: {e}"]},
                  "comment": f"评审异常: {e}", "popup_text": popup}
        er["case_id"] = case_id
        er["expect"] = expect
        return ver, er

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = [pool.submit(_judge_one, t) for t in judge_tasks]
        for fut in as_completed(futs):
            ver, er = fut.result()
            results[ver].append(er)
            if er["aggregate"] > 0:
                scores[ver].setdefault(er["case_id"], []).append(er["aggregate"])

    sums = {ver: summarize(results[ver], ver) for ver in ("v2.0", "v3.1")}
    elapsed = time.perf_counter() - t0

    # 每 case 聚合（供 Track B 直接消费）
    case_scores = {}
    for case_id in gen:
        case_scores[case_id] = {
            "v2.0_agg": round(sum(scores["v2.0"].get(case_id, [0])) / max(1, len(scores["v2.0"].get(case_id, []))), 3),
            "v3.1_agg": round(sum(scores["v3.1"].get(case_id, [0])) / max(1, len(scores["v3.1"].get(case_id, []))), 3),
        }

    report = {
        "meta": {
            "timestamp": timestamp, "n_cases": len(gen), "n_runs_per_case": n_runs,
            "gen_model": GEN_MODEL, "judge_model": JUDGE_MODEL,
            "executors": {"v2.0": popup_generator.__version__, "v3.1": popup_generator_v31.__version__},
            "shared_gen_file": gen_out.name, "elapsed_seconds": round(elapsed, 1),
        },
        "summaries": [sums["v2.0"], sums["v3.1"]],
        "per_case_aggregate": case_scores,
        "v2.0": results["v2.0"], "v3.1": results["v3.1"],
    }

    out_path = RESULTS_DIR / f"compare_v20_v31_n{n_runs}_{timestamp}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Track A 报告已保存: %s", out_path)

    # ── 打印 ──
    print("\n" + "=" * 64)
    print("  v2.0 vs v3.1 · Track A 五维对比")
    print("=" * 64)
    print(f"  案例 {len(gen)} × 每 case {n_runs} 次 | 生成 {GEN_MODEL} | 评审 {JUDGE_MODEL}")
    print(f"  执行器版本: v2.0={popup_generator.__version__} / v3.1={popup_generator_v31.__version__}")
    print(f"  耗时 {elapsed:.0f}s")
    print()
    for s in [sums["v2.0"], sums["v3.1"]]:
        print(f"  ── {s['label']} ──")
        print(f"    综合分: {s['mean_aggregate']:.3f}")
        dim_str = "  ".join(f"{d}={s['soft_means'][d]:.3f}" for d in DIMS)
        print(f"    五维: {dim_str}")
        print(f"    硬规则违规: {s['hard_violations']} | 弹窗率 {s['popup_rate']:.1%} | 应弹未弹 {s['missing_popups']}")
        print()
    a, b = sums["v2.0"]["mean_aggregate"], sums["v3.1"]["mean_aggregate"]
    diff = b - a
    winner = "v3.1" if diff > 0 else "v2.0" if diff < 0 else "平手"
    print(f"  Δ aggregate: {diff:+.3f} → {winner} 领先")
    for d in DIMS:
        dd = sums["v3.1"]["soft_means"][d] - sums["v2.0"]["soft_means"][d]
        print(f"  Δ {d}: {dd:+.3f}")
    print(f"\n  报告: {out_path}")
    print(f"  共享生成（Track B 消费）: {gen_out}")
    print("=" * 64)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="对比 v3.1 vs v2.0（对应执行器 + GLM 五维）")
    parser.add_argument("--n", type=int, default=5, help="每 case 跑 n 次（默认 5）")
    parser.add_argument("--gen-only", action="store_true", help="只生成共享产物，不评审")
    parser.add_argument("--from-gen", type=str, default=None,
                        help="复用既有共享生成产物 JSON 路径，跳过生成直接评审")
    args = parser.parse_args()
    main(n_runs=args.n, gen_only=args.gen_only, from_gen=args.from_gen)
