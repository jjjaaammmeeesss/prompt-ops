# @throwaway — 一次性重裁判脚本，跑完即删
"""用完整对话上下文重新 Codex 裁判 v4.0.18 结果。

与 run_v418_pipeline.py 内裁判的关键区别：
- 裁判看到完整对话（不截断 600 字）
- F1 规则：标注"不在原文中"前必须先逐句在原文中搜索确认
"""

import argparse
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT / "results" / "pipeline_tests"


def rejudge(results_file: str):
    """对已有结果文件重新裁判。"""
    data = json.loads(Path(results_file).read_text(encoding="utf-8"))
    results = data["results"]

    # 构建裁判 prompt（完整对话，不截断）
    cases_text = ""
    for i, r in enumerate(results):
        dialogue = r.get("_dialogue", "")
        popup = r.get("primary_popup", "")
        if not popup:
            continue
        cases_text += f"""
---
## Case {i + 1}: {r['case_id']}（对话{len(dialogue)}字）
### 完整对话原文
{dialogue}

### 弹窗
{popup}
---"""

    judge_prompt = f"""你是亲子沟通弹窗质量裁判。请对以下每个 case 的弹窗打分（0-10分），输出 JSON 数组。

## 评分维度
- 洞察深度：揭示家长行为背后的心理机制
- 具体性：锚定到对话中的具体言行
- 人称准确性：指代孩子时用"ta"而非"她/他"
- 可用建议：suggestion 具体可操作
- 术语泄漏：出现"多极/在场/内生性/关系根/双向"等内部术语则扣分
- F1 原文锚定：弹窗中引用的"原话"是否确实出现在对话原文中
- F2 反话检测：如有讽刺/阴阳怪气，是否正确识别

## 关键规则：F1 判定必须先搜索再结论
- 判定弹窗引用的某句原话"不在原文中"之前，**必须**在对话原文中逐句搜索
- 原文可能很长（2000+字），弹窗引用的内容可能在对话后半段
- 只要弹窗引用的意思在原文中出现过（不必字字一致），就视为锚定成立
- 如果判定"不在原文中"，必须在 reason 中写出：你搜索了哪个关键词、在原文中找了几遍、确认没有

{cases_text}

请输出严格 JSON 数组，每项包含 case_id, score, reason:
[{{"case_id": "C5-004", "score": 8.5, "reason": "一句话理由"}}]"""

    temp_dir = PROJECT / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = temp_dir / "rejudge_prompt.txt"
    prompt_file.write_text(judge_prompt, encoding="utf-8")

    print(f"裁判 prompt: {len(judge_prompt)} 字")
    print(f"对话总字数: {sum(len(r.get('_dialogue','')) for r in results)}")

    # Run codex
    result = subprocess.run(
        ["D:/root/.npm-global/codex.cmd", "exec", "--ephemeral", "--json"],
        input=judge_prompt,
        capture_output=True, text=True, timeout=300,
        env={**os.environ, "CODEX_NO_COLOR": "1"},
    )

    raw_output = result.stdout
    last_text = ""
    for line in raw_output.strip().split("\n"):
        try:
            obj = json.loads(line)
            if obj.get("item", {}).get("type") == "agent_message":
                last_text = obj["item"].get("text", "")
        except json.JSONDecodeError:
            pass

    print(f"\n裁判原始输出:\n{last_text}\n")

    # 解析 JSON 数组
    arr_match = re.search(r"\[.*\]", last_text, re.DOTALL)
    if not arr_match:
        print("❌ 无法解析裁判输出")
        return

    scores = json.loads(arr_match.group(0))
    score_map = {s["case_id"]: s for s in scores}

    # 应用分数
    for r in results:
        cid = r["case_id"]
        if cid in score_map:
            r["codex_score"] = score_map[cid]["score"]
            r["codex_reason"] = score_map[cid]["reason"]

    # 更新汇总
    scored = [r for r in results if r.get("codex_score") is not None]
    data["summary"]["avg_codex_score"] = sum(r["codex_score"] for r in scored) / len(scored) if scored else 0
    data["summary"]["score_ge_7"] = sum(1 for r in scored if r["codex_score"] >= 7.0)
    data["summary"]["score_lt_4"] = sum(1 for r in scored if r["codex_score"] < 4.0)
    data["summary"]["judge_mode"] = "full_context"

    # 保存
    out_path = Path(results_file).with_suffix("").with_suffix("")  # strip extensions
    out_path = Path(str(out_path) + "_rejudged.json")
    json.dump(data, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # 打印汇总
    print("=" * 80)
    print("重裁判结果（完整对话上下文）")
    print("=" * 80)
    print(f"{'Case':<10} {'字数':<6} {'原判':<6} {'重判':<6} {'Δ':<8} reason")
    print("-" * 80)

    prev_file = str(results_file).replace("_scored", "")
    prev_data = None
    try:
        prev_data = json.loads(Path(prev_file).read_text(encoding="utf-8"))
    except Exception:
        pass

    prev_map = {}
    if prev_data:
        for r in prev_data.get("results", []):
            prev_map[r["case_id"]] = r.get("codex_score")

    total = 0
    ge7 = 0
    lt4 = 0
    for r in results:
        cid = r["case_id"]
        s = r.get("codex_score")
        prev = prev_map.get(cid)
        if s is not None:
            total += s
            if s >= 7.0: ge7 += 1
            if s < 4.0: lt4 += 1
        delta = f"{s - prev:+.1f}" if s is not None and prev is not None else "N/A"
        reason_short = (r.get("codex_reason", "") or "")[:80]
        print(f"{cid:<10} {r['dialogue_chars']:<6} {prev if prev else 'N/A':<6} {s if s is not None else 'N/A':<6} {delta:<8} {reason_short}")

    avg = total / len(scored) if scored else 0
    print("-" * 80)
    print(f"均分: {avg:.2f} | ≥7.0: {ge7}/{len(scored)} | <4.0: {lt4}/{len(scored)}")
    print(f"\n结果: {out_path}")

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", help="已有结果 JSON 文件路径")
    args = parser.parse_args()
    rejudge(args.results_file)
