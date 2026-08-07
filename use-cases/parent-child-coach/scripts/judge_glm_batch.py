# @throwaway — GLM-5.2 批裁脚本，对比测试用，跑完即删
"""用 GLM-5.2 (via opencode) 批量裁判弹窗质量。

与 Codex judge (rejudge_full_context.py) 的关键区别：
- Judge 模型：baidu/glm-5.2（替代 Codex）
- 评分维度：5 维度（being_seen, dialogue_fidelity, core_insight, natural_language, warmth）
- 一级否决：事实性错误 / 语气严重误判 → 总分 0

用法:
  python scripts/judge_glm_batch.py results/pipeline_tests/xxx.json
"""

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT / "results" / "pipeline_tests"

DIM_WEIGHTS = [
    ("being_seen", 0.25),
    ("dialogue_fidelity", 0.20),
    ("core_insight", 0.20),
    ("natural_language", 0.20),
    ("warmth", 0.15),
]

JUDGE_TEMPLATE = """你是亲子沟通弹窗评估专家。给弹窗从五个维度打分（1-5整数）。

### 一级否决（任一触发则总分=0）
- **事实性错误**：弹窗编造了对话中不存在的内容
- **语气严重误判**：该鼓励的时刻用了批评/诊断的语气，或反之

### 五个维度
1. being_seen (1-5): 家长读完会不会心里轻轻动一下——"你懂我"？
2. dialogue_fidelity (1-5): 每个判断都能在对话原文中找到确切依据？
3. core_insight (1-5): 是否抓住了这段对话里最该被看见的点？
4. natural_language (1-5): 像真人在耳边说话？没有术语、框架标签、模板套话？
5. warmth (1-5): 整体姿态是盟友还是教师？

### 对话
{DIALOGUE}

### AI弹窗
{POPUP}

严格输出JSON（不要markdown包裹）：
{{"veto":null,"being_seen":1-5,"dialogue_fidelity":1-5,"core_insight":1-5,"natural_language":1-5,"warmth":1-5,"brief":"理由（限20字）"}}"""


def compute_score(scores: dict) -> tuple:
    """计算加权总分（0-10 scale），返回 (score, is_vetoed, veto_reason)。"""
    veto = scores.get("veto")
    if veto and str(veto).strip() not in ("null", "none", ""):
        return 0.0, True, str(veto)

    ws, tw = 0.0, 0.0
    for dk, w in DIM_WEIGHTS:
        v = scores.get(dk)
        if isinstance(v, (int, float)) and 1 <= v <= 5:
            ws += (v - 1) / 4 * w  # normalize to 0-1 per dim
            tw += w
    if tw == 0:
        return 0.0, False, None
    # Convert 0-1 to 0-10
    return round(ws / tw * 10, 1), False, None


def judge_one(dialogue: str, popup: str, max_retries: int = 3) -> dict:
    """用 GLM-5.2 裁判单个弹窗。"""
    prompt = JUDGE_TEMPLATE.format(
        DIALOGUE=dialogue[:3000],
        POPUP=popup[:2000],
    )

    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                ["opencode", "run", "--model", "baidu/glm-5.2"],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=120,
                encoding="utf-8",
                env={**os.environ, "CODEX_NO_COLOR": "1"},
            )
            output = result.stdout.strip()
            # Strip ANSI codes
            clean = re.sub(r'\x1b\[[0-9;]*m', '', output)
            m = re.search(r'\{.*\}', clean, re.DOTALL)
            if m:
                return json.loads(m.group(0))
            print(f"  ⚠️ 无法解析 GLM 输出 (attempt {attempt+1}): {clean[:100]}")
        except subprocess.TimeoutExpired:
            print(f"  ⚠️ GLM 超时 (attempt {attempt+1})")
            if attempt < max_retries - 1:
                time.sleep(3)
        except Exception as e:
            print(f"  ⚠️ GLM 错误 (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(3)

    return {"error": "judge_failed"}


def judge_file(results_file: str):
    """对结果文件中的所有 primary_popup 逐条裁判。"""
    data = json.loads(Path(results_file).read_text(encoding="utf-8"))
    results = data["results"]

    total = 0
    ge7 = 0
    lt4 = 0
    scored_count = 0

    print(f"\n{'='*80}")
    print(f"GLM-5.2 裁判: {Path(results_file).name}")
    print(f"{'='*80}")
    print(f"{'Case':<10} {'字数':<6} {'being':<6} {'fidel':<6} {'insight':<6} {'nat':<6} {'warm':<6} {'score':<7} brief")
    print(f"{'-'*80}")

    for i, r in enumerate(results):
        dialogue = r.get("_dialogue", "")
        popup = r.get("primary_popup", "")
        cid = r["case_id"]
        dlen = r["dialogue_chars"]

        if not popup:
            print(f"{cid:<10} {'—':<6} {'无弹窗，跳过'}")
            continue

        print(f"  [{i+1}/{len(results)}] {cid}...", end=" ", flush=True)
        judge_result = judge_one(dialogue, popup)
        time.sleep(0.5)  # rate limit

        if "error" in judge_result:
            print(f"❌ {judge_result['error']}")
            continue

        score, vetoed, veto_reason = compute_score(judge_result)
        brief = judge_result.get("brief", "")

        # Store scores
        r["glm_score"] = score
        r["glm_vetoed"] = vetoed
        r["glm_veto_reason"] = veto_reason
        r["glm_dimensions"] = {
            dk: judge_result.get(dk) for dk, _ in DIM_WEIGHTS
        }
        r["glm_brief"] = brief

        scored_count += 1
        if not vetoed:
            total += score
            if score >= 7.0:
                ge7 += 1
            if score < 4.0:
                lt4 += 1

        dim_str = " ".join(
            f"{judge_result.get(dk, '?'):<6}" for dk, _ in DIM_WEIGHTS
        )
        veto_flag = "🚫" if vetoed else ""
        print(f"\r{cid:<10} {dlen:<6} {dim_str} {score:<6.1f}{veto_flag} {brief[:40]}")

    avg = total / (scored_count - (len([r for r in results if r.get("glm_vetoed")]))) if scored_count > 0 else 0
    vetoed_count = sum(1 for r in results if r.get("glm_vetoed"))

    # Update summary
    data["summary"]["glm_avg_score"] = round(avg, 2)
    data["summary"]["glm_ge_7"] = ge7
    data["summary"]["glm_lt_4"] = lt4
    data["summary"]["glm_vetoed"] = vetoed_count
    data["summary"]["glm_judge_model"] = "baidu/glm-5.2"
    data["summary"]["glm_scored_count"] = scored_count

    # Save
    out_path = Path(results_file).parent / (Path(results_file).stem + "_glm.json")
    json.dump(data, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"{'-'*80}")
    print(f"均分: {avg:.2f} | ≥7.0: {ge7}/{scored_count} | <4.0: {lt4}/{scored_count} | 否决: {vetoed_count}")
    print(f"结果: {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", help="管线结果 JSON 文件路径")
    args = parser.parse_args()
    judge_file(args.results_file)
