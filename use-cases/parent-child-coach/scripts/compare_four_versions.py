"""四版本公平对比：v1.11 vs v4.0.12 vs v2.3 vs v1.7

四个版本用相同数据集、相同 judge、相同轮数，在同一脚本内跑，保证可比。

用法:
  python scripts/compare_four_versions.py --n 12 --rounds 3
"""

import argparse
import json
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass
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
GEN_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GEN_API_BASE = os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
GEN_MODEL = os.getenv("DEEPSEEK_MODEL") or "deepseek-chat"

# ── 评委模型：XINGLUAN Claude ──
JUDGE_API_KEY = os.getenv("XINGLUAN_AUTH_TOKEN")
JUDGE_API_BASE = "https://luanapi.xingluan.cn/v1"

# ── 路径 ──
V4012_PROMPT_PATH = _realtime_parent / "system_prompt_v4.0.12.txt"
V23_PROMPT_PATH = _realtime_parent / "system_prompt_v2.3.txt"
V17_PROMPT_PATH = _realtime_parent / "prompts_archive" / "system_prompt_backup_v17.txt"
V111_SYS_PATH = _realtime_parent / "system_prompt_v1.11_sys.txt"
V111_USER_PATH = _realtime_parent / "system_prompt_v1.11_user.txt"
NEW12_DATASET = _realtime_parent / "data" / "new_12_independent.json"

# 加载 v1.11 prompts（由 extract_v111.py 预提取）
V111_SYSTEM_PROMPT = V111_SYS_PATH.read_text(encoding="utf-8")
V111_USER_TEMPLATE = V111_USER_PATH.read_text(encoding="utf-8")

print(f"v1.11 System Prompt: {len(V111_SYSTEM_PROMPT)} 字")
print(f"v1.11 User Template:  {len(V111_USER_TEMPLATE)} 字")


# ===== LLM 调用 =====

def call_llm(system_prompt: str, user_content: str, max_tokens: int = 400,
             temperature: float = 0.3, timeout: int = 180) -> str:
    resp = litellm.completion(
        model=f"deepseek/{GEN_MODEL}",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=GEN_API_KEY,
        api_base=GEN_API_BASE,
        timeout=timeout,
    )
    return (resp.choices[0].message.content or "").strip()


# ===== 各版本生成函数 =====

def generate_v111(dialogue: str, tone: str) -> tuple:
    """v1.11: System Prompt + User Prompt（含 few-shot），解析 JSON 输出。"""
    # 填充 User Prompt 模板
    user_content = V111_USER_TEMPLATE.replace("{user_input}", dialogue)
    user_content = user_content.replace("{profile_context}", "")
    user_content = user_content.replace("{context_block}", "")

    # 添加 tone 指令
    tone_map = {
        "诊断式": "诊断式弹窗（100-200字）",
        "鼓励式": "鼓励式弹窗（30-60字）",
    }
    tone_instruction = (
        f"\n\n【强制指令】请生成**{tone_map.get(tone, tone)}**。"
        f"should_popup 必须为 true，tone 必须为 '{'empowering' if tone == '鼓励式' else 'diagnostic'}'。"
        f"直接输出 JSON，不要其他文字。"
    )
    user_content += tone_instruction

    raw = call_llm(V111_SYSTEM_PROMPT, user_content, max_tokens=1024, temperature=0.3)

    # 解析 JSON
    popup = ""
    meta = {"tone": tone, "should_popup": True, "skip_reason": None}

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # 尝试提取 JSON 块
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                # 无法解析，把原始文本当弹窗
                return raw, {"tone": tone, "should_popup": True, "skip_reason": None, "parse_error": True}
        else:
            return raw, {"tone": tone, "should_popup": True, "skip_reason": None, "parse_error": True}

    meta["should_popup"] = data.get("should_popup", True)
    meta["actual_tone"] = data.get("tone", "")

    insight = (data.get("popup_insight") or "").strip()
    suggestion = (data.get("popup_suggestion") or "").strip()

    if tone == "鼓励式":
        # 鼓励式：只用 insight
        popup = insight
    else:
        # 诊断式：insight + —— + suggestion
        if insight and suggestion:
            popup = insight + "\n——\n" + suggestion
        elif insight:
            popup = insight
        elif suggestion:
            popup = suggestion
        else:
            popup = raw

    if not popup.strip():
        popup = raw

    return popup, meta


def _generate_simple(dialogue: str, tone: str, system_prompt: str, version_label: str) -> tuple:
    """通用生成函数（v4.0.12 / v2.3 / v1.7）：单体 System Prompt + 纯文本输出。"""
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

    raw = call_llm(system_prompt, user_content, max_tokens=400)
    return raw, {"tone": tone, "should_popup": True, "skip_reason": None}


# ── 全局 prompt 缓存 ──
v4012_prompt_cache = ""
v23_prompt_cache = ""
v17_prompt_cache = ""


def generate_v4012(dialogue: str, tone: str) -> tuple:
    return _generate_simple(dialogue, tone, v4012_prompt_cache, "v4.0.12")


def generate_v23(dialogue: str, tone: str) -> tuple:
    return _generate_simple(dialogue, tone, v23_prompt_cache, "v2.3")


def generate_v17(dialogue: str, tone: str) -> tuple:
    return _generate_simple(dialogue, tone, v17_prompt_cache, "v1.7")


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
            cache_key = f"{case_id}_{expected_tone}"
            if cache_key in cache:
                popup, meta, elapsed = cache[cache_key]
                print(f"    复用缓存 ({elapsed:.1f}s)")
            else:
                start = time.time()
                popup, meta = gen_fn(dialogue, expected_tone)
                elapsed = time.time() - start
                cache[cache_key] = (popup, meta, elapsed)
                print(f"    生成耗时 {elapsed:.1f}s, 输出 {len(popup)} 字"
                      f"{' [JSON解析失败]' if meta.get('parse_error') else ''}")

            if not popup or not popup.strip():
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
                "actual_tone": meta.get("tone") or meta.get("actual_tone", ""),
                "generated": popup,
                "golden": golden,
                "weighted_score": weighted,
                "meta": {k: v for k, v in meta.items() if k != "tone"},
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


# ===== 外层监控：目标偏移检测 =====

@dataclass
class RoundSnapshot:
    """单轮对比快照。"""
    round_num: int
    version_scores: dict[str, float]        # version → avg_score
    version_per_case: dict[str, list[float]]  # version → [per-case scores]
    ranking: list[str]                       # version names sorted by score desc


class OuterLoopMonitor:
    """外层循环监控器：定期检测对比目标是否偏移。

    目标（硬编码）：对比 v1.11 / v4.0.12 / v2.3 / v1.7 四个版本，找出最优。

    三种检测：
      1. 排名稳定性 — 最近 stability_rounds 轮 winner 是否一致
      2. 分数漂移 — 任一版本分数异常下滑（模型/Judge 疲劳）
      3. 收敛判断 — 排名稳定 + 分数波动 < threshold → 可提前终止
    """

    def __init__(
        self,
        stability_rounds: int = 2,
        score_drift_threshold: float = 0.08,
        convergence_threshold: float = 0.02,
    ):
        self.stability_rounds = stability_rounds
        self.score_drift_threshold = score_drift_threshold
        self.convergence_threshold = convergence_threshold
        self.snapshots: list[RoundSnapshot] = []
        self.warnings: list[str] = []
        self.corrections: list[str] = []

    def record_round(self, snapshot: RoundSnapshot) -> dict:
        """记录一轮结果，返回检查报告。"""
        self.snapshots.append(snapshot)
        report = {
            "round": snapshot.round_num,
            "ranking": snapshot.ranking,
            "scores": {v: round(s, 3) for v, s in snapshot.version_scores.items()},
            "checks": {},
        }

        # 1. 排名稳定性
        report["checks"]["ranking_stable"] = self._check_ranking_stability()

        # 2. 分数漂移
        drift_warnings = self._check_score_drift()
        report["checks"]["score_drift"] = len(drift_warnings) == 0
        report["checks"]["drift_details"] = drift_warnings

        # 3. 收敛判断
        report["checks"]["converged"] = self._check_convergence()

        # 4. 综合：是否可以提前终止
        can_stop, stop_reason = self._should_stop_early()
        report["checks"]["can_stop_early"] = can_stop
        report["checks"]["stop_reason"] = stop_reason

        return report

    def _check_ranking_stability(self) -> bool:
        """检查最近 stability_rounds 轮 winner 是否一致。"""
        if len(self.snapshots) < self.stability_rounds:
            return False
        recent = self.snapshots[-self.stability_rounds:]
        winners = [s.ranking[0] for s in recent]
        return len(set(winners)) == 1

    def _check_score_drift(self) -> list[str]:
        """检测分数异常漂移：任一版本 vs 前一轮下降超过阈值。"""
        warnings = []
        if len(self.snapshots) < 2:
            return warnings
        prev = self.snapshots[-2]
        curr = self.snapshots[-1]
        for version in curr.version_scores:
            prev_score = prev.version_scores.get(version)
            curr_score = curr.version_scores.get(version)
            if prev_score is not None and curr_score is not None:
                delta = prev_score - curr_score
                if delta > self.score_drift_threshold:
                    warnings.append(
                        f"{version}: {prev_score:.3f} → {curr_score:.3f} "
                        f"(Δ=-{delta:.3f} > {self.score_drift_threshold})"
                    )
        return warnings

    def _check_convergence(self) -> bool:
        """判断是否收敛：排名稳定 + 各版本分数波动 < threshold。"""
        if len(self.snapshots) < self.stability_rounds:
            return False
        if not self._check_ranking_stability():
            return False
        # 检查各版本分数波动
        recent = self.snapshots[-self.stability_rounds:]
        for version in recent[0].version_scores:
            scores = [s.version_scores.get(version, 0) for s in recent]
            if max(scores) - min(scores) > self.convergence_threshold:
                return False
        return True

    def _should_stop_early(self) -> tuple[bool, str]:
        """综合判断是否可以提前终止。

        规则：
          1. 已收敛 → 提前终止（目标达成）
          2. 连续 2 轮有分数漂移警告 → 提前终止（Judge/模型不稳定，继续无意义）
          3. 同一 winner 连续 4 轮 → 提前终止（压倒性优势）
        """
        if self._check_convergence():
            return True, f"排名稳定 + 分数波动 < {self.convergence_threshold}，目标达成"

        if len(self.snapshots) >= 3:
            # 连续漂移：最近 2 轮都有警告
            recent_drifts = []
            for i in range(-2, 0):
                if abs(i) <= len(self.snapshots):
                    drifts = self._check_score_drift_for_snapshot(i)
                    recent_drifts.append(len(drifts) > 0)
            if len(recent_drifts) >= 2 and all(recent_drifts):
                return True, "连续 2 轮分数异常漂移，Judge/模型可能不稳定，建议排查后再继续"

        # 同一 winner 连续 4+ 轮
        if len(self.snapshots) >= 4:
            last_4 = [s.ranking[0] for s in self.snapshots[-4:]]
            if len(set(last_4)) == 1:
                return True, f"{last_4[0]} 连续 4 轮排名第一，压倒性优势，可提前终止"

        return False, ""

    def _check_score_drift_for_snapshot(self, index: int) -> list[str]:
        """检查指定 snapshot 相对前一轮的漂移。"""
        warnings = []
        if abs(index) >= len(self.snapshots):
            return warnings
        idx = index if index >= 0 else len(self.snapshots) + index
        if idx < 1:
            return warnings
        prev = self.snapshots[idx - 1]
        curr = self.snapshots[idx]
        for version in curr.version_scores:
            prev_score = prev.version_scores.get(version)
            curr_score = curr.version_scores.get(version)
            if prev_score is not None and curr_score is not None:
                delta = prev_score - curr_score
                if delta > self.score_drift_threshold:
                    warnings.append(
                        f"{version}: {prev_score:.3f} → {curr_score:.3f} (Δ=-{delta:.3f})"
                    )
        return warnings

    def summarize(self) -> str:
        """生成监控报告文本。"""
        lines = ["", "=" * 60, "  外层监控报告 · 目标偏移检测", "=" * 60]
        lines.append(f"  目标: 对比 v1.11 / v4.0.12 / v2.3 / v1.7，找出最优版本")
        lines.append(f"  总轮数: {len(self.snapshots)}")
        lines.append(f"  警告数: {len(self.warnings)}")
        lines.append(f"  校正数: {len(self.corrections)}")

        if self.snapshots:
            last = self.snapshots[-1]
            lines.append(f"\n  当前排名 ({last.round_num} 轮累计):")
            for i, v in enumerate(last.ranking, 1):
                medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, "  ")
                lines.append(f"    {medal} #{i}: {v} — {last.version_scores[v]:.3f}")

        if self.warnings:
            lines.append(f"\n  ⚠️ 偏移警告:")
            for w in self.warnings[-5:]:  # 最近 5 条
                lines.append(f"    - {w}")

        if self.corrections:
            lines.append(f"\n  🔧 校正记录:")
            for c in self.corrections[-5:]:
                lines.append(f"    - {c}")

        if self._check_convergence():
            lines.append(f"\n  ✅ 目标达成: 排名已收敛，可确定最优版本。")
        elif self.snapshots and self._check_ranking_stability():
            lines.append(f"\n  ✅ 排名稳定: winner 连续 {self.stability_rounds} 轮一致。")

        lines.append("=" * 60)
        return "\n".join(lines)


# ===== 统计辅助 =====

def compute_stats(scores_list: list) -> dict:
    """计算均分、标准差、方差范围。"""
    if not scores_list:
        return {"avg": 0, "stdev": 0, "min": 0, "max": 0}
    import statistics
    return {
        "avg": statistics.mean(scores_list),
        "stdev": statistics.stdev(scores_list) if len(scores_list) > 1 else 0,
        "min": min(scores_list),
        "max": max(scores_list),
    }


def main():
    global v4012_prompt_cache, v23_prompt_cache, v17_prompt_cache

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
    print(f"生成模型: deepseek/{GEN_MODEL}")

    # 加载 prompt
    v4012_prompt_cache = V4012_PROMPT_PATH.read_text(encoding="utf-8")
    v23_prompt_cache = V23_PROMPT_PATH.read_text(encoding="utf-8")
    v17_prompt_cache = V17_PROMPT_PATH.read_text(encoding="utf-8")
    print(f"v1.11: System {len(V111_SYSTEM_PROMPT)} 字 + User {len(V111_USER_TEMPLATE)} 字")
    print(f"v4.0.12: {len(v4012_prompt_cache)} 字")
    print(f"v2.3: {len(v23_prompt_cache)} 字")
    print(f"v1.7: {len(v17_prompt_cache)} 字")

    # Judge — override 硬编码的 Claude key/url
    import llm_judge_metric as jmod
    jmod.CLAUDE_KEY = JUDGE_API_KEY
    jmod.CLAUDE_URL = JUDGE_API_BASE + "/messages"
    jmod.CLAUDE_MODEL = "claude-opus-4-7"

    backend = os.getenv("JUDGE_BACKEND", "claude")
    judge = LLMJudgeMetric(judge_backend=backend)
    print(f"Judge: {backend} @ {jmod.CLAUDE_URL}")

    # ── 版本定义 ──
    versions = [
        ("v1.11", generate_v111),
        ("v4.0.12", generate_v4012),
        ("v2.3", generate_v23),
        ("v1.7", generate_v17),
    ]

    # 收集各轮结果
    all_rounds = {name: [] for name, _ in versions}
    all_details = []

    # 外层监控器
    monitor = OuterLoopMonitor(
        stability_rounds=2,
        score_drift_threshold=0.08,
        convergence_threshold=0.02,
    )

    for r in range(args.rounds):
        print(f"\n{'='*60}\n  Round {r+1}/{args.rounds}\n{'='*60}")
        round_results = {}
        for vname, gen_fn in versions:
            print(f"\n--- {vname} ---")
            res = run_version(vname, cases, gen_fn, judge)
            all_rounds[vname].append(res["avg_score"])
            round_results[vname] = res
        all_details.append({"round": r + 1, "results": round_results})

        # ── 外层监控：记录本轮快照 + 目标偏移检测 ──
        version_scores = {vname: res["avg_score"] for vname, res in round_results.items()}
        version_per_case = {
            vname: [c["weighted_score"] for c in res["results"] if "weighted_score" in c]
            for vname, res in round_results.items()
        }
        ranking = sorted(version_scores.keys(), key=lambda v: version_scores[v], reverse=True)
        snapshot = RoundSnapshot(r + 1, version_scores, version_per_case, ranking)
        monitor_report = monitor.record_round(snapshot)

        checks = monitor_report["checks"]
        stability_icon = "✅" if checks["ranking_stable"] else "❌"
        drift_icon = "✅" if checks["score_drift"] else "⚠️"
        converged_icon = "✅" if checks["converged"] else "❌"
        print(f"\n  [监控] 排名稳定性: {stability_icon} | 分数漂移: {drift_icon} | 收敛: {converged_icon}")
        if checks["drift_details"]:
            for d in checks["drift_details"]:
                print(f"    ⚠️ 漂移: {d}")

        if checks["can_stop_early"]:
            print(f"\n  ⏸️ {checks['stop_reason']}")
            break

    # ── 汇总统计 ──
    print(f"\n{'='*70}")
    print(f"  四版本对比汇总（独立测试前 {args.n} 题, {args.rounds} 轮）")
    print(f"{'='*70}")

    version_stats = {}
    for vname, _ in versions:
        scores = all_rounds[vname]
        stats = compute_stats(scores)
        version_stats[vname] = {
            "rounds": [round(s, 3) for s in scores],
            **{k: round(v, 3) for k, v in stats.items()},
        }
        print(f"  {vname:8s}: {[f'{a:.3f}' for a in scores]} | "
              f"均分 {stats['avg']:.3f} | σ {stats['stdev']:.3f}")

    # ── 排名 ──
    ranking = sorted(version_stats.items(), key=lambda x: x[1]["avg"], reverse=True)
    print(f"\n  🏆 排名:")
    for rank, (vname, stats) in enumerate(ranking, 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "  ")
        print(f"  {medal} #{rank}: {vname} — 均分 {stats['avg']:.3f}")

    # 差距
    best_name, best_stats = ranking[0]
    worst_name, worst_stats = ranking[-1]
    print(f"\n  {best_name} 领先 {worst_name}: {best_stats['avg'] - worst_stats['avg']:+.3f}")

    # ── 外层监控报告 ──
    print(monitor.summarize())

    # ── 保存结果 ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"compare_four_versions_n{args.n}_{timestamp}.json"
    output_path.write_text(json.dumps({
        "n": args.n,
        "rounds": args.rounds,
        "actual_rounds": len(all_details),
        "gen_model": f"deepseek/{GEN_MODEL}",
        "judge_backend": backend,
        "timestamp": timestamp,
        "version_stats": version_stats,
        "ranking": [{"rank": i+1, "version": v, "avg": s["avg"]} for i, (v, s) in enumerate(ranking)],
        "best_version": best_name,
        "outer_loop_monitor": {
            "total_rounds": len(monitor.snapshots),
            "converged": monitor._check_convergence(),
            "ranking_stable": monitor._check_ranking_stability(),
            "warnings": monitor.warnings,
            "corrections": monitor.corrections,
            "snapshots": [
                {
                    "round": s.round_num,
                    "ranking": s.ranking,
                    "scores": {v: round(sc, 3) for v, sc in s.version_scores.items()},
                }
                for s in monitor.snapshots
            ],
        },
        "details": all_details,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果保存: {output_path}")


if __name__ == "__main__":
    main()
