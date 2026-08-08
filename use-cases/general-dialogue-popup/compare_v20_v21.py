"""对比 v2.0 vs v2.1 成人通用弹窗 — 10 个职场场景 × n=3。

生成：DeepSeek v4-pro（国产模型铁律）
评审：GLM 5.2（百度千帆，五维打分 + v2.0 硬规则检查）
维度：insight, third_party, language, evidence, focus
输出：results/compare_tests/compare_v20_v21_n{n}_{timestamp}.json

Usage:
    python compare_v20_v21.py
    python compare_v20_v21.py --n 3  # 每 case 跑 n 次（默认 3）
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

# ── 配置 ──
GEN_MODEL = os.environ.get("GEN_MODEL", "deepseek/deepseek-v4-pro")
GEN_API_BASE = os.environ.get("GEN_API_BASE", "https://api.deepseek.com/v1")
GEN_API_KEY = os.environ.get("GEN_API_KEY", os.environ.get("DEEPSEEK_API_KEY", ""))
_TEMPERATURE = float(os.environ.get("GEN_TEMPERATURE", "0.3"))

# Judge 使用 GLM 5.2（百度千帆）
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "openai/glm-5.2")
JUDGE_API_BASE = os.environ.get("JUDGE_API_BASE", "https://qianfan.baidubce.com/v2")
JUDGE_API_KEY = os.environ.get("JUDGE_API_KEY", os.environ.get("GLM_API_KEY", ""))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("compare")


# ── 简单生成器（全量对话一次调用） ──

def generate_popup(system_prompt: str, dialogue: str, model: str = GEN_MODEL) -> str | None:
    """调用 LLM 生成弹窗。返回弹窗文本，安静时返回 None。"""
    import litellm

    user_msg = f"当前对话：\n{dialogue}"
    kwargs: dict = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=_TEMPERATURE,
        max_tokens=4096,
        timeout=180,
    )
    if GEN_API_BASE:
        kwargs["api_base"] = GEN_API_BASE
    if GEN_API_KEY:
        kwargs["api_key"] = GEN_API_KEY

    for attempt in range(3):
        try:
            resp = litellm.completion(**kwargs)
            text = (resp.choices[0].message.content or "").strip()
            break
        except Exception as e:
            logger.warning("生成尝试 %d/3 失败: %s", attempt + 1, e)
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)

    # 判断是否安静
    if not text or len(text) <= 10:
        return None
    if text.strip() == "安好":
        return None
    # 清理：去掉可能的 "弹窗：" 前缀
    for prefix in ["弹窗：", "弹窗:", "【弹窗】", "输出："]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return text


# ── GLM Judge（五维打分 + v2.0 硬规则） ──

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


def _hard_check(popup_text: str) -> dict:
    """v2.0 硬规则检查（纯代码）。

    规则：
    - 字数 60-180（去空白字符）
    - 全文不出现"你"字
    - 末尾完整性检查
    - 不检查 —— / 功能墙 / 建议（v2.0 已删除这些概念）
    """
    violations = []
    text = popup_text.strip()
    n = len(re.sub(r"\s", "", text))

    # 字数硬合规：60-180，无缓冲区间
    if n > 180:
        violations.append(f"字数 {n} 超过 180 硬合规线")
    elif n < 60:
        violations.append(f"字数 {n} 低于 60")

    # 禁用字符"你"——任意位置即违规
    if "你" in text:
        violations.append("包含禁用字符 `你`")

    # 末尾完整性
    if not text.endswith(("。", "？", "！", "”", '"', "」", "』")):
        violations.append("输出疑似截断")

    return {"pass": not violations, "violations": violations}


def _parse_judge_json(raw: str) -> dict | None:
    """从 LLM 输出中提取 JSON。"""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        soft = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    defaults = {
        "insight": 3, "third_party": 3, "language": 3,
        "evidence": 3, "focus": 3, "comment": "",
    }
    for k, v in defaults.items():
        if k not in soft:
            soft[k] = v
    for k in ("insight", "third_party", "language", "evidence", "focus"):
        try:
            soft[k] = max(1, min(5, int(soft[k])))
        except (TypeError, ValueError):
            soft[k] = 3
    return soft


def _judge_popup(dialogue: str, popup_text: str) -> dict:
    """用 GLM 打分 + v2.0 硬规则检查。"""
    import litellm

    prompt = _JUDGE_PROMPT.format(dialogue=dialogue, popup=popup_text)
    for attempt in range(3):
        try:
            resp = litellm.completion(
                model=JUDGE_MODEL,
                api_key=JUDGE_API_KEY,
                api_base=JUDGE_API_BASE,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=4096,
                timeout=180,
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
        logger.warning("judge JSON 解析失败 | raw=%s", raw[:200])
        soft = {
            "insight": 3, "third_party": 3, "language": 3,
            "evidence": 3, "focus": 3, "comment": f"[parse_error] {raw[:80]}",
        }

    hard = _hard_check(popup_text)
    dims = ("insight", "third_party", "language", "evidence", "focus")
    soft_mean = sum(soft[k] for k in dims) / len(dims)
    penalty = min(1.5, 0.5 * len(hard["violations"]))
    aggregate = round(max(1.0, soft_mean - penalty), 2)

    return {
        "aggregate": aggregate,
        "soft": soft,
        "hard": hard,
        "comment": soft.get("comment", ""),
        "popup_text": popup_text,
    }


# ── 运行单版本 ──

def run_version(prompt_path: str, cases: list[dict], n_runs: int = 5) -> list[dict]:
    """对每个 case 跑 n 次，收集所有弹窗和评分。"""
    system_prompt = Path(prompt_path).read_text(encoding="utf-8")
    if not system_prompt.strip():
        raise ValueError(f"Prompt 文件为空: {prompt_path}")

    results = []

    for ci, case in enumerate(cases):
        case_id = case["id"]
        expect = case.get("expect", "")
        dialogue = case["dialogue"]
        logger.info("[%s] case %d/%d: %s", Path(prompt_path).stem, ci + 1, len(cases), case_id)

        for run_i in range(n_runs):
            logger.info("  run %d/%d", run_i + 1, n_runs)
            try:
                popup_text = generate_popup(system_prompt, dialogue)
            except Exception as e:
                logger.error("  生成失败: %s", e)
                results.append({
                    "case_id": case_id, "run": run_i, "expect": expect,
                    "popup_text": None, "error": str(e),
                    "aggregate": 0, "soft": None,
                    "hard": {"pass": False, "violations": [f"生成异常: {e}"]},
                })
                continue

            # ── 安静案例：不弹为对 ──
            if expect == "安静":
                if popup_text is None:
                    results.append({
                        "case_id": case_id, "run": run_i, "expect": expect,
                        "popup_text": None,
                        "aggregate": 5.0, "soft": None,
                        "hard": {"pass": True, "violations": []},
                        "comment": "正确保持安静",
                    })
                else:
                    results.append({
                        "case_id": case_id, "run": run_i, "expect": expect,
                        "popup_text": popup_text,
                        "aggregate": 1.0, "soft": None,
                        "hard": {"pass": False, "violations": ["健康对话误弹"]},
                        "comment": "健康对话不应弹窗",
                    })
                continue

            # ── 应弹未弹 ──
            if popup_text is None:
                results.append({
                    "case_id": case_id, "run": run_i, "expect": expect,
                    "popup_text": None,
                    "aggregate": 1.5, "soft": None,
                    "hard": {"pass": False, "violations": ["应弹未弹"]},
                    "comment": "对话存在盲区但未弹窗",
                })
                continue

            # ── 正常 judge ──
            try:
                eval_result = _judge_popup(dialogue, popup_text)
            except Exception as e:
                logger.error("  评审失败: %s", e)
                eval_result = {
                    "aggregate": 0, "soft": None,
                    "hard": {"pass": False, "violations": [f"评审异常: {e}"]},
                    "comment": f"评审异常: {e}", "popup_text": popup_text,
                }

            results.append({
                "case_id": case_id, "run": run_i, "expect": expect,
                "popup_text": popup_text,
                "aggregate": eval_result["aggregate"],
                "soft": eval_result.get("soft"),
                "hard": eval_result["hard"],
                "comment": eval_result.get("comment", ""),
            })

    return results


# ── 汇总统计 ──

def summarize(results: list[dict], label: str) -> dict:
    """计算汇总统计。"""
    valid = [r for r in results if r["popup_text"] is not None and r["aggregate"] > 0]
    all_agg = [r["aggregate"] for r in results if r["aggregate"] > 0]
    soft_dims = ["insight", "third_party", "language", "evidence", "focus"]
    soft_scores = {k: [] for k in soft_dims}
    for r in valid:
        if r["soft"]:
            for k in soft_dims:
                if k in r["soft"] and r["soft"][k] is not None:
                    soft_scores[k].append(r["soft"][k])

    hard_violations = sum(len(r.get("hard", {}).get("violations", [])) for r in results)

    # 弹窗率
    n_should_popup = sum(1 for r in results if r["expect"] != "安静")
    n_did_popup = sum(1 for r in results if r["popup_text"] is not None)
    popup_rate = n_did_popup / n_should_popup if n_should_popup > 0 else 0

    # 安静案例误弹率
    quiet_cases = [r for r in results if r["expect"] == "安静"]
    quiet_misfires = sum(1 for r in quiet_cases if r["popup_text"] is not None)

    # 应弹未弹率
    missing = sum(1 for r in results if r["expect"] != "安静" and r["popup_text"] is None)

    def mean(vals): return round(sum(vals) / len(vals), 3) if vals else 0

    return {
        "label": label,
        "n_total": len(results),
        "n_valid": len(valid),
        "mean_aggregate": mean(all_agg),
        "soft_means": {k: mean(v) for k, v in soft_scores.items()},
        "hard_violations": hard_violations,
        "popup_rate": round(popup_rate, 3),
        "quiet_misfires": quiet_misfires,
        "quiet_total": len(quiet_cases),
        "missing_popups": missing,
        "aggregates": all_agg,
    }


# ── 主流程 ──

def main(n_runs: int = 3):
    # 加载数据集
    dataset_path = HERE / "data" / "business_dialogues_10.json"
    if not dataset_path.exists():
        logger.error("找不到数据集: %s", dataset_path)
        sys.exit(1)
    cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    logger.info("加载 %d 个职场场景", len(cases))

    # 检查 API keys
    if not GEN_API_KEY:
        logger.error("缺少生成模型 API key（设置 DEEPSEEK_API_KEY 或 GEN_API_KEY）")
        sys.exit(1)
    if not JUDGE_API_KEY:
        logger.error("缺少 GLM_API_KEY 环境变量（用于评审）")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("开始对比 v2.0 vs v2.1 (n=%d)", n_runs)
    t0 = time.perf_counter()

    # ── 跑 v2.0 ──
    logger.info("=" * 50)
    logger.info("▶ 运行 v2.0（基线）")
    v20_results = run_version(str(HERE / "system_prompt_v2.0.txt"), cases, n_runs)
    v20_summary = summarize(v20_results, "v2.0")

    # ── 跑 v2.1 ──
    logger.info("=" * 50)
    logger.info("▶ 运行 v2.1（候选）")
    v21_results = run_version(str(HERE / "system_prompt_v2.1.txt"), cases, n_runs)
    v21_summary = summarize(v21_results, "v2.1")

    elapsed = time.perf_counter() - t0

    # ── 生成报告 ──
    report = {
        "meta": {
            "timestamp": timestamp,
            "n_cases": len(cases),
            "n_runs_per_case": n_runs,
            "gen_model": GEN_MODEL,
            "judge_model": JUDGE_MODEL,
            "elapsed_seconds": round(elapsed, 1),
        },
        "summaries": [v20_summary, v21_summary],
        "v2.0": v20_results,
        "v2.1": v21_results,
    }

    out_path = RESULTS_DIR / f"compare_v20_v21_n{n_runs}_{timestamp}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("报告已保存: %s", out_path)

    # ── 打印汇总 ──
    print("\n" + "=" * 60)
    print("  v2.0 vs v2.1 对比结果")
    print("=" * 60)
    print(f"  案例数: {len(cases)}, 每案例跑 {n_runs} 次")
    print(f"  生成模型: {GEN_MODEL}")
    print(f"  评审模型: {JUDGE_MODEL}")
    print(f"  耗时: {elapsed:.0f}s")
    print()
    for s in [v20_summary, v21_summary]:
        print(f"  ── {s['label']} ──")
        print(f"    综合分 (aggregate): {s['mean_aggregate']:.3f}")
        dims = ["insight", "third_party", "language", "evidence", "focus"]
        dim_str = "  ".join(f"{d}={s['soft_means'][d]:.3f}" for d in dims)
        print(f"    软维度: {dim_str}")
        print(f"    硬规则违规数: {s['hard_violations']}")
        print(f"    弹窗率: {s['popup_rate']:.1%}  "
              f"安静误弹: {s['quiet_misfires']}/{s['quiet_total']}  "
              f"应弹未弹: {s['missing_popups']}")
        print(f"    有效样本: {s['n_valid']}/{s['n_total']}")
        print()

    # 对比
    v20_agg = v20_summary["mean_aggregate"]
    v21_agg = v21_summary["mean_aggregate"]
    diff = v21_agg - v20_agg
    winner = "v2.1" if diff > 0 else "v2.0" if diff < 0 else "平手"
    print(f"  Δ aggregate: {diff:+.3f} → {winner} 领先")
    print()
    # 关键维度对比
    for d in ["third_party", "evidence", "focus"]:
        d_diff = round(v21_summary["soft_means"][d] - v20_summary["soft_means"][d], 3)
        print(f"  Δ {d}: {d_diff:+.3f}")
    print(f"\n  报告: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="对比 v2.0 vs v2.1")
    parser.add_argument("--n", type=int, default=3, help="每 case 跑 n 次（默认 3）")
    args = parser.parse_args()
    main(n_runs=args.n)
