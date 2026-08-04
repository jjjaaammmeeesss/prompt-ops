# @transient — 保留到 v4.0.18 验收通过后删除（预计 2026-08-10）
"""v4.0.18 真实管线批量测试器 —— 300字窗口 + 快速通道 + Codex 裁判。

与之前所有测试脚本的关键区别：
- 之前：整段对话一次塞给 DeepSeek（假测试）
- 现在：先模拟快慢通道互斥窗口切分，每个窗口独立调 LLM（= 生产行为）

用法:
  python scripts/run_v418_pipeline.py                           # 跑全部 12 题
  python scripts/run_v418_pipeline.py --cases C5-004,DS_001     # 只跑指定 case
  python scripts/run_v418_pipeline.py --no-judge                # 只生成，不裁判
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import litellm

litellm.suppress_debug_info = True

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent

# 把 demos/ 加入 sys.path 以便导入管线函数
sys.path.insert(0, str(PROJECT / "demos"))
from run_demo import simulate_pipeline, Keyword, Popup

# severity 映射：critical→5, warning→3, opportunity→2
SEVERITY_MAP = {"critical": 5, "warning": 3, "opportunity": 2}

def load_keywords() -> list:
    """从 keyword_config.json 加载关键词，返回 List[Keyword]。"""
    kw_path = PROJECT / "realtime" / "keyword_config.json"
    with open(kw_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    keywords = []
    for level, sev in SEVERITY_MAP.items():
        for word in config.get(level, []):
            keywords.append(Keyword(text=word, severity=sev))
    return keywords

# ── 路径 ──
V418_PROMPT_PATH = PROJECT / "system_prompt_v4.0.18.txt"
DATASET_PATH = PROJECT / "data" / "new_12_independent.json"
RESULTS_DIR = PROJECT / "results" / "pipeline_tests"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── 生成模型：DeepSeek V4 ──
GEN_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
GEN_API_BASE = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
GEN_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


def strip_pre_analysis(text: str) -> str:
    """切除预分析部分（`==========` 或 `---` 之前的所有内容），只保留弹窗正文。"""
    if not text:
        return ""
    # 先尝试 v4.0.19 的 ========== 分隔符
    for sep in ["\n==========\n", "\n==========", "==========\n", "=========="]:
        if sep in text:
            parts = text.split(sep, 1)
            after = parts[1].strip() if len(parts) > 1 else ""
            if after:
                return after
    # 回退到旧的 --- 分隔符
    m = re.search(r"\n---\n|\n---$|^---\n|^---$", text, re.MULTILINE)
    if m:
        after = text[m.end():].strip()
        if after:
            return after
    if "---" in text:
        parts = text.split("---", 1)
        after = parts[1].strip() if len(parts) > 1 else text.strip()
        if after and len(after) < len(text) * 0.8:
            return after
    return text.strip()


def generate_popup(system_prompt: str, window_text: str, retries: int = 3) -> dict:
    """调用 DeepSeek 生成弹窗（单窗口、含重试）。"""
    user_msg = f"当前对话：\n{window_text}"
    for attempt in range(retries):
        try:
            resp = litellm.completion(
                model=f"deepseek/{GEN_MODEL}",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.3,
                max_tokens=1024,
                api_key=GEN_API_KEY,
                api_base=GEN_API_BASE,
                timeout=180,
            )
            raw = (resp.choices[0].message.content or "").strip()
            popup_text = strip_pre_analysis(raw)
            return {
                "raw": raw,
                "popup": popup_text,
                "has_pre_analysis": ("==========" in raw or "---" in raw) and popup_text != raw,
                "error": None,
            }
        except Exception as e:
            if attempt < retries - 1:
                wait = (attempt + 1) * 3
                print(f"      ⚠️ 重试 {attempt + 1}/{retries}（{wait}s 后）: {e}")
                time.sleep(wait)
            else:
                return {"raw": "", "popup": "", "has_pre_analysis": False, "error": str(e)}
    return {"raw": "", "popup": "", "has_pre_analysis": False, "error": "unknown"}


def run_case(case: dict, system_prompt: str, keywords: list, window_size: int = 300) -> dict:
    """对单个 case 跑完整管线（快通道 + 慢通道），返回所有窗口的弹窗。"""
    case_id = case.get("case_id", "unknown")
    dialogue = case.get("question", "")

    # 清除行号前缀（"1.xxx\n2.xxx" → "xxx\nxxx"）
    dialogue = re.sub(r"^\d+\.", "", dialogue, flags=re.MULTILINE).strip()

    dlen = len(dialogue)

    # ── 管线模拟：流式快慢通道互斥 ──
    popups: list = simulate_pipeline(dialogue, keywords)

    result = {
        "case_id": case_id,
        "dialogue_chars": dlen,
        "fast_triggers": sum(1 for p in popups if p.channel == "fast"),
        "slow_windows": sum(1 for p in popups if p.channel == "slow"),
        "fast_popups": [],
        "slow_popups": [],
    }

    # ── 按弹窗顺序逐一调 LLM 生成 ──
    for p in popups:
        gen = generate_popup(system_prompt, p.context_window)
        entry = {
            "channel": p.channel,
            "trigger_type": p.trigger_type,
            "window_chars": p.char_count,
            **gen,
        }
        if p.channel == "fast":
            result["fast_popups"].append(entry)
        else:
            result["slow_popups"].append(entry)

    # ── 汇总：取最后一个非空弹窗作为主弹窗 ──
    all_popups = []
    for p in result["fast_popups"]:
        if p["popup"]:
            all_popups.append(p["popup"])
    for p in result["slow_popups"]:
        if p["popup"]:
            all_popups.append(p["popup"])

    result["primary_popup"] = all_popups[-1] if all_popups else ""
    result["total_popups"] = len(all_popups)

    return result


def run_codex_batch_judge(results: list, output_file: str) -> list:
    """批量 Codex 裁判：一次 codex exec 调用评完所有弹窗。

    返回更新后的 results（含 codex_score / codex_reason）。
    """
    import subprocess

    # 构建批量裁判 prompt
    cases_text = ""
    for i, r in enumerate(results):
        dialogue = r.get("_dialogue", "")[:600]
        popup = r.get("primary_popup", "")
        if not popup:
            continue
        cases_text += f"""
---
## Case {i + 1}: {r['case_id']}
对话: {dialogue}{'…' if len(r.get('_dialogue', '')) > 600 else ''}
弹窗: {popup}
---"""

    judge_prompt = f"""你是亲子沟通弹窗质量裁判。请对以下每个 case 的弹窗打分（0-10分），输出 JSON 数组。

评分维度：
- 洞察深度（揭示家长行为背后的心理机制）
- 具体性（锚定到对话中的具体言行）
- 人称准确性（指代孩子时用"ta"而非"她/他"）
- 可用建议（suggestion 具体可操作）
- 术语泄漏（出现"多极/在场/内生性/关系根/双向"等内部术语则扣分）
- F1 原文锚定（是否引用对话中的具体原话）
- F2 反话检测（如有讽刺/阴阳怪气，是否正确识别并触发防御模式）

{cases_text}

请输出严格 JSON 数组，每项包含 case_id, score, reason:
[{{"case_id": "C5-004", "score": 8.5, "reason": "一句话理由"}}, ...]"""

    temp_dir = PROJECT / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = temp_dir / "codex_judge_prompt.txt"
    prompt_file.write_text(judge_prompt, encoding="utf-8")

    out_file = temp_dir / "codex_judge_output.json"
    print(f"  📋 批量裁判 {len([r for r in results if r.get('primary_popup')])} 条弹窗...")

    try:
        result = subprocess.run(
            ["codex", "exec", "--ephemeral", "--json",
             "-o", str(out_file)],
            input=judge_prompt,
            capture_output=True, text=True, timeout=300,
            cwd=str(PROJECT),
            env={**os.environ, "CODEX_NO_COLOR": "1"},
        )

        # 解析 JSONL 输出，取最后一条 agent_message
        raw_output = result.stdout
        last_text = ""
        for line in raw_output.strip().split("\n"):
            try:
                obj = json.loads(line)
                if obj.get("item", {}).get("type") == "agent_message":
                    last_text = obj["item"].get("text", "")
            except json.JSONDecodeError:
                pass

        if not last_text and out_file.exists():
            # 尝试从输出文件读取
            out_content = out_file.read_text(encoding="utf-8")
            for line in out_content.strip().split("\n"):
                try:
                    obj = json.loads(line)
                    if obj.get("item", {}).get("type") == "agent_message":
                        last_text = obj["item"].get("text", "")
                except json.JSONDecodeError:
                    pass

        # 解析 JSON 数组
        if last_text:
            # 提取 JSON 数组
            arr_match = re.search(r"\[.*\]", last_text, re.DOTALL)
            if arr_match:
                scores = json.loads(arr_match.group(0))
                score_map = {s["case_id"]: s for s in scores}
                for r in results:
                    if r["case_id"] in score_map:
                        r["codex_score"] = score_map[r["case_id"]]["score"]
                        r["codex_reason"] = score_map[r["case_id"]]["reason"]
                return results

        print(f"  ⚠️ 无法解析 Codex 批量输出，前 500 字: {raw_output[:500]}")
    except Exception as e:
        print(f"  ❌ Codex 批量裁判失败: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="v4.0.18 真实管线批量测试")
    parser.add_argument("--cases", type=str, default="",
                        help="逗号分隔的 case_id 列表（默认：全部）")
    parser.add_argument("--dataset", type=str, default=str(DATASET_PATH),
                        help="数据集路径")
    parser.add_argument("--no-judge", action="store_true",
                        help="跳过 Codex 裁判")
    parser.add_argument("--window-size", type=int, default=300,
                        help="慢通道窗口大小（默认 300）")
    args = parser.parse_args()

    # ── 加载 ──
    print("=" * 70)
    print("v4.0.18 真实管线批量测试（快通道 + 300字窗口慢通道）")
    print("=" * 70)

    v418_prompt = V418_PROMPT_PATH.read_text(encoding="utf-8")
    print(f"Prompt: {len(v418_prompt)} 字, {v418_prompt.count(chr(10))} 行")

    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    keywords = load_keywords()

    # 筛选 cases
    if args.cases:
        target_ids = set(args.cases.split(","))
        dataset = [c for c in dataset if c.get("case_id") in target_ids]
        print(f"筛选: {len(dataset)} 题 ({', '.join(c['case_id'] for c in dataset)})")
    else:
        print(f"数据集: {len(dataset)} 题")

    print(f"生成模型: DeepSeek {GEN_MODEL}")
    print(f"窗口大小: {args.window_size} 字")
    print(f"快速通道关键词: {len(keywords)} 个")
    print(f"Codex 裁判: {'❌ 跳过' if args.no_judge else '✅ 启用'}")
    print()

    # ── 逐题测试 ──
    all_results = []
    for idx, case in enumerate(dataset):
        case_id = case.get("case_id", f"case_{idx}")
        d_short = case.get("question", "")[:60].replace("\n", " ")
        print(f"[{idx + 1}/{len(dataset)}] {case_id}: {d_short}...")

        result = run_case(case, v418_prompt, keywords, args.window_size)
        result["_dialogue"] = case.get("question", "")  # 保留原对话供裁判使用

        print(f"  快通道: {result['fast_triggers']} 触发, "
              f"慢通道: {result['slow_windows']} 窗口, "
              f"弹窗: {result['total_popups']} 条")

        # 检查预分析分隔
        for i, p in enumerate(result["fast_popups"]):
            err = p.get("error")
            trigger = p.get("trigger_type", "?")
            wc = p.get("window_chars", "?")
            plen = len(p.get("popup", ""))
            if err:
                print(f"    ⚠️ 快#{i} [{trigger}] 窗口{wc}字: {err}")
            elif p.get("has_pre_analysis"):
                print(f"    快#{i} [{trigger}] 窗口{wc}字 → 弹窗{plen}字 ✓")
            else:
                print(f"    快#{i} [{trigger}] 窗口{wc}字 → 弹窗{plen}字 ⚠️ 未检测到预分析分隔")

        for i, p in enumerate(result["slow_popups"]):
            err = p.get("error")
            wc = p.get("window_chars", "?")
            plen = len(p.get("popup", ""))
            if err:
                print(f"    ⚠️ 慢#{i} [窗口{wc}字]: {err}")
            elif p.get("has_pre_analysis"):
                print(f"    慢#{i} [窗口{wc}字] → 弹窗{plen}字 ✓")
            else:
                print(f"    慢#{i} [窗口{wc}字] → 弹窗{plen}字 ⚠️")

        all_results.append(result)
        time.sleep(0.5)  # API 限流

    # ── 批量 Codex 裁判（所有 case 一次调用）──
    if not args.no_judge:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        all_results = run_codex_batch_judge(all_results, str(RESULTS_DIR / f"codex_judge_{timestamp}.json"))
        for r in all_results:
            score = f"{r['codex_score']:.1f}" if r.get("codex_score") is not None else "N/A"
            reason = r.get("codex_reason", "")[:80] if r.get("codex_reason") else ""
            print(f"  🧑‍⚖️ {r['case_id']}: {score}/10 — {reason}")

    # ── 汇总 ──
    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)

    for r in all_results:
        cid = r["case_id"]
        n_popups = r["total_popups"]
        primary = r["primary_popup"][:80].replace("\n", " ") if r["primary_popup"] else "(无弹窗)"
        score = f"{r['codex_score']:.1f}" if r.get("codex_score") is not None else "N/A"
        print(f"  {cid}: {n_popups}条弹窗 | Codex={score} | {primary}...")

    # ── 保存 ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cases_tag = f"_{args.cases.replace(',', '_')}" if args.cases else "_all12"
    out_path = RESULTS_DIR / f"v418_pipeline{cases_tag}_{timestamp}.json"

    output = {
        "config": {
            "prompt": "system_prompt_v4.0.18.txt",
            "prompt_chars": len(v418_prompt),
            "gen_model": f"deepseek/{GEN_MODEL}",
            "window_size": args.window_size,
            "judge": "codex" if not args.no_judge else "none",
            "timestamp": timestamp,
        },
        "summary": {
            "total_cases": len(all_results),
            "cases_with_popup": sum(1 for r in all_results if r["primary_popup"]),
            "avg_codex_score": (
                sum(r["codex_score"] for r in all_results if r.get("codex_score") is not None)
                / max(1, sum(1 for r in all_results if r.get("codex_score") is not None))
                if not args.no_judge else None
            ),
        },
        "results": all_results,
    }
    json.dump(output, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")
    return out_path


if __name__ == "__main__":
    main()
