"""快速 head-to-head 对比：v1.7 单 prompt vs 星灵多智能体 v3.1

同样的 12 个案例，同样的 M6 judge（deepseek-chat），同一次会话评判——避免 judge 漂移。
单跑 n=1（快速对比，不做降噪）。

输出：每案例的 v1.7 M6 分 vs 3.0 M6 分，总体均值对比，胜负统计。
"""
import json
import os
import re
import sys
import time
from pathlib import Path
from openai import OpenAI

sys.path.insert(0, "D:/prompt-ops/use-cases/parent-child-coach")
sys.path.insert(0, "D:/星灵-soul-手搓/亲子沟通洞见/路线A_自上而下法_鼓励版")

from auto_evolve.optimizer import load_env, EVAL_CASES, load_golden_dataset, find_case, get_input_text, get_gold_labels
from auto_evolve.evaluator import build_m6_prompt
from auto_evolve.optimizer import _call_judge, parse_llm_judge_response

V17_PROMPT_PATH = "D:/星灵-soul-手搓/亲子沟通洞见/路线A_自上而下法_鼓励版/prompts/prompt_A轨_v1.7_修复感知版.md"
RESULTS = Path("D:/prompt-ops/use-cases/parent-child-coach/results")
RESULTS.mkdir(parents=True, exist_ok=True)


def parse_v17_prompt(path: str) -> tuple[str, str]:
    """从 prompt 文件提取 system_prompt 和 user_prompt（用 ## N. System/User Prompt 章节标记定位）。"""
    text = Path(path).read_text(encoding="utf-8")
    # 定位 ## N. System Prompt 和 ## N. User Prompt 章节
    sys_m = re.search(r"^##\s+\d+\.\s+System\s+Prompt\s*$", text, re.MULTILINE)
    usr_m = re.search(r"^##\s+\d+\.\s+User\s+Prompt\s*$", text, re.MULTILINE)
    if not sys_m or not usr_m:
        raise ValueError("找不到 System/User Prompt 章节")
    sys_section = text[sys_m.end():usr_m.start()]
    usr_section = text[usr_m.end():]
    # 各取第一个 ```...``` 代码块
    def first_block(s: str) -> str:
        m = re.search(r"````?\s*\n(.*?)\n````?", s, re.DOTALL)
        if not m:
            raise ValueError("代码块未找到")
        return m.group(1)
    return first_block(sys_section), first_block(usr_section)


def _find_popup_via_tone_dict(data) -> str:
    """找含 type=='diagnostic'|'empowering' 的 dict，取其中最长的字符串字段作为 popup。"""
    valid = {"diagnostic", "empowering"}
    candidates = []
    def walk(obj):
        if isinstance(obj, dict):
            # 检查这个 dict 是否有 type 字段匹配 tone
            has_tone = any(isinstance(v, str) and v.lower().strip() in valid for v in obj.values())
            if has_tone:
                # 取这个 dict 里最长的字符串值（>30 字符，避免短字段干扰）
                strs = [v for v in obj.values() if isinstance(v, str) and len(v) >= 30]
                if strs:
                    candidates.append(max(strs, key=len))
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)
    walk(data)
    if candidates:
        return max(candidates, key=len)
    return ""


def _find_in_tree(data, key_candidates: list[str], min_len: int = 0) -> str:
    """递归在 JSON 树里找第一个匹配 key_candidates 的字符串值。"""
    if isinstance(data, dict):
        for k, v in data.items():
            if k in key_candidates and isinstance(v, str) and len(v) >= min_len and v.strip():
                return v
        for v in data.values():
            r = _find_in_tree(v, key_candidates, min_len)
            if r:
                return r
    elif isinstance(data, list):
        for item in data:
            r = _find_in_tree(item, key_candidates, min_len)
            if r:
                return r
    return ""


def _find_tone_in_tree(data) -> str:
    """找 tone：搜 'diagnostic' 或 'empowering' 字符串值。"""
    valid = {"diagnostic", "empowering"}
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, str) and v.lower().strip() in valid:
                return v.lower().strip()
            r = _find_tone_in_tree(v)
            if r:
                return r
    elif isinstance(data, list):
        for item in data:
            r = _find_tone_in_tree(item)
            if r:
                return r
    return ""


def _find_should_popup(data) -> bool:
    """找 should_popup。"""
    if isinstance(data, dict):
        if "should_popup" in data and isinstance(data["should_popup"], bool):
            return data["should_popup"]
        for v in data.values():
            r = _find_should_popup(v)
            if r is not None:
                return r
    elif isinstance(data, list):
        for item in data:
            r = _find_should_popup(item)
            if r is not None:
                return r
    return None


def run_v17(client: OpenAI, model: str, system_prompt: str, user_template: str, dialogue: str) -> dict:
    """跑一次 v1.7，返回 {tone, popup_text, contradiction, error}。

    DeepSeek 输出结构非确定（step_2/step_3_script/step_0_trigger 等多种命名），
    用递归搜索兜底。
    """
    user_prompt = (user_template
                   .replace("{user_input}", dialogue)
                   .replace("{profile_context}", "")
                   .replace("{context_block}", ""))
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.4,
            timeout=120,
        )
        raw = resp.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            from json_repair import loads as repair_loads
            data = repair_loads(raw)
    except Exception as e:
        return {"tone": "", "popup_text": "", "contradiction": "", "error": str(e)[:120]}

    # should_popup: 递归找
    should = _find_should_popup(data)
    if should is False:
        return {"tone": "", "popup_text": "", "contradiction": "", "error": "should_popup=false"}

    # tone: 递归找 diagnostic/empowering
    tone = _find_tone_in_tree(data)

    # popup: 优先在 tone dict 里找最长字符串；否则搜 popup_*/popup/content
    popup = _find_popup_via_tone_dict(data)
    if not popup:
        popup = _find_in_tree(data, ["popup_text", "popup", "popup_insight", "content", "text", "script"], min_len=30)

    # contradiction: 递归找 diagnosis.description / insight / principal_insight
    contradiction = _find_in_tree(data, ["description", "insight", "principal_insight", "collapsed_layer", "distortion"])

    return {"tone": tone, "popup_text": popup, "contradiction": contradiction, "error": ""}


def judge_m6(client: OpenAI, model: str, dialogue: str, reference: str, sys_popup: str, tone: str, contradiction: str) -> tuple[float | None, str]:
    """跑 M6 judge，返回 (score, raw)。"""
    if not reference.strip() or "内容标注" in reference:
        return None, ""
    if not sys_popup.strip():
        return 1.0, "no popup"
    prompt = build_m6_prompt(dialogue, reference, sys_popup, tone, contradiction)
    raw = _call_judge(client, model, prompt)
    score, raw = parse_llm_judge_response(raw, "m6")
    return score, raw


def main(n_runs: int = 3):
    load_env()
    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )
    model = "deepseek-chat"

    # 1. 加载 v1.7 prompt
    print("加载 v1.7 prompt...")
    v17_sys, v17_user = parse_v17_prompt(V17_PROMPT_PATH)
    print(f"  system: {len(v17_sys)}字, user: {len(v17_user)}字")

    # 2. 加载 3.0 多智能体基线结果（已有 sys_popup_text + tone + contradiction）
    baseline_30 = json.loads((RESULTS / "auto_baseline_v25_full.json").read_text(encoding="utf-8"))
    case_to_30 = {c["case_id"]: c for c in baseline_30["per_case"]}
    print(f"加载 3.0 基线: {len(case_to_30)} cases (n=3 denoised)")

    # 3. 加载 golden dataset
    dataset = load_golden_dataset()

    # 4. 跑 v1.7 n_runs 次 + judge 双方
    print(f"\n🚀 开始对比评估 (12 cases × v1.7 n={n_runs} + 双方 M6 judge)")
    print(f"   start: {time.strftime('%H:%M:%S')}")
    print("=" * 100)

    rows = []
    v17_m6_sum = 0.0
    v30_m6_sum = 0.0
    v17_n = 0
    v30_n = 0
    v17_wins = 0
    v30_wins = 0
    ties = 0

    for i, (case_id, win_idx) in enumerate(EVAL_CASES, 1):
        case = find_case(dataset, case_id)
        if not case:
            print(f"  ❌ {case_id}: 找不到案例")
            continue
        input_text = get_input_text(case, win_idx)
        gold = get_gold_labels(case, win_idx)

        # --- v1.7: 跑 n_runs 次 ---
        v17_runs = []
        for _ in range(n_runs):
            r = run_v17(client, model, v17_sys, v17_user, input_text)
            v17_runs.append(r)

        # 对每次 v1.7 输出跑 judge
        v17_m6_scores = []
        v17_tones = []
        for r in v17_runs:
            m6, _ = judge_m6(
                client, model, input_text, gold["reference_popup"],
                r["popup_text"], r["tone"], r["contradiction"],
            )
            if m6 is not None:
                v17_m6_scores.append(m6)
            if r["tone"]:
                v17_tones.append(r["tone"])
        # denoise: tone majority vote, M6 mean
        from collections import Counter
        v17_tone_final = Counter(v17_tones).most_common(1)[0][0] if v17_tones else ""
        v17_m6_final = sum(v17_m6_scores) / len(v17_m6_scores) if v17_m6_scores else None
        # 取第一个非空 popup 作为代表
        v17_popup_final = next((r["popup_text"] for r in v17_runs if r["popup_text"]), "")

        # --- 3.0 多智能体（从基线取 popup，重跑 judge n_runs 次降噪）---
        c30 = case_to_30.get(case_id, {})
        v30_popup = c30.get("sys_popup_text", "")
        v30_tone = c30.get("sys_tone", "")
        v30_contra = c30.get("sys_main_contradiction", "")
        v30_m6_scores = []
        for _ in range(n_runs):
            m6, _ = judge_m6(
                client, model, input_text, gold["reference_popup"],
                v30_popup, v30_tone, v30_contra,
            )
            if m6 is not None:
                v30_m6_scores.append(m6)
        v30_m6_final = sum(v30_m6_scores) / len(v30_m6_scores) if v30_m6_scores else None

        # M5 tone match
        v17_tone_match = 1.0 if v17_tone_final == gold["tone"] else 0.0
        v30_tone_match = 1.0 if v30_tone == gold["tone"] else 0.0

        # 统计
        if v17_m6_final is not None:
            v17_m6_sum += v17_m6_final
            v17_n += 1
        if v30_m6_final is not None:
            v30_m6_sum += v30_m6_final
            v30_n += 1
        if v17_m6_final is not None and v30_m6_final is not None:
            if v17_m6_final > v30_m6_final:
                v17_wins += 1
            elif v30_m6_final > v17_m6_final:
                v30_wins += 1
            else:
                ties += 1

        rows.append({
            "case_id": case_id,
            "gold_tone": gold["tone"],
            "v17_tone": v17_tone_final,
            "v17_tone_match": v17_tone_match,
            "v17_m6": v17_m6_final,
            "v17_m6_runs": v17_m6_scores,
            "v17_popup": v17_popup_final[:200],
            "v17_run_errors": [r["error"] for r in v17_runs if r["error"]],
            "v30_tone": v30_tone,
            "v30_tone_match": v30_tone_match,
            "v30_m6": v30_m6_final,
            "v30_m6_runs": v30_m6_scores,
            "v30_popup": v30_popup[:200],
        })

        v17m = f"{v17_m6_final:.2f}" if v17_m6_final is not None else "  -  "
        v30m = f"{v30_m6_final:.2f}" if v30_m6_final is not None else "  -  "
        winner = ("v17" if (v17_m6_final or 0) > (v30_m6_final or 0)
                  else ("v30" if (v30_m6_final or 0) > (v17_m6_final or 0) else "tie"))
        print(f"  [{i:2d}/12] {case_id:8s} | v1.7 M6={v17m:>5s} tone={v17_tone_final[:6]:6s} | 3.0 M6={v30m:>5s} tone={v30_tone[:6]:6s} | gold={gold['tone'][:6]:6s} | winner={winner}")

    # 5. 汇总
    v17_avg = v17_m6_sum / v17_n if v17_n else 0
    v30_avg = v30_m6_sum / v30_n if v30_n else 0
    v17_tone_rate = sum(r["v17_tone_match"] for r in rows) / len(rows)
    v30_tone_rate = sum(r["v30_tone_match"] for r in rows) / len(rows)

    print(f"\n{'=' * 100}")
    print(f"📊 对比结果（v1.7 n={n_runs} 降噪 vs 3.0 n=3 基线，同会话 M6 judge）")
    print(f"{'=' * 100}")
    print(f"  {'指标':16s} {'v1.7 单prompt':>16s} {'3.0 多智能体':>16s} {'Δ':>10s}")
    print(f"  {'-'*52}")
    print(f"  {'M6 洞察质量':16s} {v17_avg:>16.2f} {v30_avg:>16.2f} {v30_avg-v17_avg:>+10.2f}")
    print(f"  {'M5 口吻匹配率':16s} {v17_tone_rate:>15.1%} {v30_tone_rate:>15.1%} {v30_tone_rate-v17_tone_rate:>+9.1%}")
    print(f"  {'M6 胜负':16s} {v17_wins:>16d} {v30_wins:>16d} {ties:>6d} ties")
    print(f"  {'有效样本':16s} {v17_n:>16d} {v30_n:>16d}")

    # 6. 保存详细结果
    out = {
        "n_runs": n_runs,
        "summary": {
            "v17_m6_avg": v17_avg,
            "v30_m6_avg": v30_avg,
            "v17_tone_match_rate": v17_tone_rate,
            "v30_tone_match_rate": v30_tone_rate,
            "v17_wins": v17_wins,
            "v30_wins": v30_wins,
            "ties": ties,
        },
        "rows": rows,
    }
    out_path = RESULTS / f"h2h_v17_vs_v30_n{n_runs}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  详细结果: {out_path.name}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-runs", type=int, default=3)
    args = p.parse_args()
    main(n_runs=args.n_runs)
