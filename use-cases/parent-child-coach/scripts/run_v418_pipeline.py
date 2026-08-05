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
import difflib
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import litellm
from dotenv import load_dotenv

litellm.suppress_debug_info = True

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
load_dotenv(PROJECT / ".env")

# 把 demos/ 加入 sys.path 以便导入管线函数
sys.path.insert(0, str(PROJECT / "demos"))
from run_demo import simulate_pipeline, Keyword, Popup

# severity 映射：critical→5, warning→3, opportunity→2
SEVERITY_MAP = {"critical": 5, "warning": 3, "opportunity": 2}

# ═══════════════════════════════════════════════════════════════════════════════
# 生产等价函数 — 从 realtime/popup_generator.py 移植，确保测试=生产行为
# ═══════════════════════════════════════════════════════════════════════════════

# === 默认周易上下文（测试管线不跑真正的 Stage 1 LLM，使用中性默认值）===

def _build_default_zhouyi_context() -> str:
    """构建默认/中性周易卦象上下文。测试管线不跑真正的 ZhouYiAnalyzer，
    用坤卦（全掌控/稳态）作为最中性的默认值，不误导弹窗方向。"""
    return """
## 周易卦象上下文（测试默认值 — 非真实 Stage 1 分析）

当前沟通状态属于 **☷ 坤（控控控）**——稳定承载型。

容器状态：不适用
风险等级：低
分析洞察：测试管线默认值，未执行真正的 ZhouYiAnalyzer 分析。

请参考系统提示词第八节「八卦弹窗策略速查」中对应卦象的指导来生成弹窗。

---
"""

# === FC_TONE_OFF: 家长行为 tone override（与 popup_generator.py 完全一致）===

PARENT_OVERRIDE_KEYWORDS = {
    "催促/打断": [
        "快点", "快一点", "别说了", "行了行了", "行了我知道",
        "别废话", "闭嘴", "你能不能快点", "动作快", "抓紧时间",
        "你快点", "少啰嗦", "有完没完",
    ],
    "评判贴标签": [
        "你就是磨蹭", "你太敏感", "你这个人就是", "你就是个",
        "你太矫情", "你就是太", "矫情", "你就是故意", "你总是",
        "你每次都", "你就是不上心",
    ],
    "命令单向权力": [
        "我让你做你就做", "少废话", "按我说的", "我让你",
        "没有为什么", "我说了算", "听我的", "不许顶嘴",
        "你少跟我", "我是你妈", "我是你爸", "照我说的做",
    ],
    "轻度贬低/否定情绪": [
        "这有什么好哭", "至于吗", "想太多", "无理取闹",
        "小题大做", "娇气", "这有什么", "有什么好哭",
        "别那么娇", "你至于", "哭什么哭", "有什么好闹",
    ],
}


def detect_parent_override(dialogue: str) -> Optional[str]:
    """扫描对话文本，命中家长行为 tone override 关键词时返回命中的类别名。"""
    if not dialogue:
        return None
    for category, keywords in PARENT_OVERRIDE_KEYWORDS.items():
        for kw in keywords:
            if kw in dialogue:
                return category
    return None


# === P2: parent-quotable repair phrase 检测 ===

_QUOTABLE_PHRASE_RE = re.compile(r'[「『“"]([^」』”"]{4,})[」』”"]')


def has_quotable_phrase(text: str) -> bool:
    """检测文本中是否含至少一句引号内的可直接引用话术（≥4字）。"""
    return bool(_QUOTABLE_PHRASE_RE.search(text or ""))


# === FC_STALE: 跨窗口语义去重 ===

def semantic_similarity(a: str, b: str) -> float:
    """计算两段文本的相似度（difflib SequenceMatcher ratio）。"""
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()

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
PROMPT_MAP = {
    "v4.0.12": PROJECT / "system_prompt_v4.0.12.txt",
    "v4.0.18": PROJECT / "system_prompt_v4.0.18.txt",
}
DATASET_PATH = PROJECT / "data" / "new_12_independent.json"
RESULTS_DIR = PROJECT / "results" / "pipeline_tests"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── 生成模型 ──
GEN_API_KEY = os.getenv("GEN_API_KEY", os.getenv("DEEPSEEK_API_KEY", os.getenv("ZHIPUAI_API_KEY", "")))
GEN_API_BASE = os.getenv("GEN_API_BASE", os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
_raw_model = os.getenv("GEN_MODEL", os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
GEN_MODEL = _raw_model if "/" in _raw_model else f"deepseek/{_raw_model}"


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


def generate_popup(
    system_prompt: str,
    window_text: str,
    tone: str = "auto",
    zhouyi_context: str = "",
    extra_instruction: str = "",
    retries: int = 3,
) -> dict:
    """调用 DeepSeek 生成弹窗（单窗口、含重试、生产级消息格式）。

    与生产 popup_generator.py 对齐：
    - system message = zhouyi_context + system_prompt
    - user message = 当前对话 + type_instruction + 输出格式要求
    """
    # ── 构建消息（与 popup_generator._build_messages 对齐）──
    system_content = zhouyi_context + "\n" + system_prompt

    # type_instruction
    if tone == "encouraging":
        type_instruction = (
            "请生成**鼓励式弹窗**（20-80字）。"
            "必须：具体点出家长刚展现的积极模式 → 简短有力。"
            "必须包含至少一句家长可直接引用的话术"
            "（以「你可以这样说：\"……\"」形式给出，引号内为实际措辞）。"
        )
    else:
        type_instruction = (
            "请生成**诊断式弹窗**（80-200字）。"
            "必须：先承认发心 → 揭示具体模式 → 给出一个微小可做的尝试。"
        )

    user_msg = f"""当前对话：
{window_text}

{type_instruction}
{extra_instruction}

请直接输出弹窗全文（不附加解释、不输出JSON、不输出"弹窗："等前缀）："""

    for attempt in range(retries):
        try:
            resp = litellm.completion(
                model=GEN_MODEL,
                messages=[
                    {"role": "system", "content": system_content},
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

            # 缺分隔符时保留全文但标记 warning
            has_sep = "==========" in raw or (
                "---" in raw and len(popup_text) < len(raw) * 0.8
            )

            return {
                "raw": raw,
                "popup": popup_text,
                "has_pre_analysis": has_sep and popup_text != raw,
                "separator_missing": not has_sep,
                "error": None,
            }
        except Exception as e:
            if attempt < retries - 1:
                wait = (attempt + 1) * 3
                print(f"      ⚠️ 重试 {attempt + 1}/{retries}（{wait}s 后）: {e}")
                time.sleep(wait)
            else:
                return {
                    "raw": "", "popup": "", "has_pre_analysis": False,
                    "separator_missing": True, "error": str(e),
                }
    return {"raw": "", "popup": "", "has_pre_analysis": False,
            "separator_missing": True, "error": "unknown"}


def run_case(
    case: dict,
    system_prompt: str,
    keywords: list,
    window_size: int = 300,
    dedup_threshold: float = 0.70,
    debounce_enabled: bool = True,
) -> dict:
    """对单个 case 跑完整管线（快通道 + 慢通道 + 生产级 Stage 2）。

    与生产 stream_orchestrator.process_chunk 对齐的关键行为：
    - FC_TONE_OFF: 调 LLM 前扫描关键词，命中则强制 diagnostic
    - FC_STALE: 调 LLM 后用 difflib 与历史弹窗比对去重
    - P2: 鼓励式弹窗检查 quotable phrase，缺失则重试一次
    - 去抖: 同通道同 trigger 连续出现时跳过后续
    - 窗口参数传递: window_size 传入 simulate_pipeline
    """
    case_id = case.get("case_id", "unknown")
    dialogue = case.get("question", "")

    # 清除行号前缀（"1.xxx\n2.xxx" → "xxx\nxxx"）
    dialogue = re.sub(r"^\d+\.", "", dialogue, flags=re.MULTILINE).strip()

    dlen = len(dialogue)

    # ── 管线模拟：流式快慢通道互斥（传入 window_size）──
    popups: list = simulate_pipeline(dialogue, keywords, slow_threshold=window_size)

    # ── 默认周易上下文（测试管线不跑真正的 Stage 1）──
    zhouyi_context = _build_default_zhouyi_context()

    # ── 去抖 + 逐窗 LLM 调用 ──
    previous_popups: list[str] = []  # 历史弹窗文本（用于语义去重）
    last_trigger: dict = {}           # {channel, trigger_type} 用于去抖
    fast_popups: list[dict] = []
    slow_popups: list[dict] = []
    suppressed_count = 0

    for p_idx, p in enumerate(popups):
        # ── 去抖: 同通道同 trigger 连续出现时跳过 ──
        if debounce_enabled:
            trigger_key = (p.channel, p.trigger_type)
            if trigger_key == last_trigger.get("key"):
                suppressed_count += 1
                last_trigger["count"] = last_trigger.get("count", 0) + 1
                continue
            last_trigger = {"key": trigger_key, "count": 1}

        # ── FC_TONE_OFF: 扫描窗口文本，命中则强制 diagnostic ──
        # 默认 encouraging：无 Stage 1 时仍能覆盖 P2 话术检查 + FC_TONE_OFF 真实切换
        tone = "encouraging"
        override_reason = detect_parent_override(p.context_window)
        if override_reason:
            tone = "diagnostic"

        # ── 调 LLM 生成弹窗 ──
        gen = generate_popup(system_prompt, p.context_window, tone=tone,
                             zhouyi_context=zhouyi_context)

        # ── P2: 鼓励式话术检查 ──
        if tone == "encouraging" and gen["popup"] and not gen["error"]:
            if not has_quotable_phrase(gen["popup"]):
                # 重试一次
                retry = generate_popup(
                    system_prompt, p.context_window, tone=tone,
                    zhouyi_context=zhouyi_context,
                    extra_instruction=(
                        "⚠️ 上一次输出不合格：缺少家长可直接引用的话术。"
                        "必须重新生成，并在弹窗末尾以「你可以这样说：\"……\"」的形式"
                        "给出至少一句家长能脱口说出的完整话术（引号内为实际措辞）。"
                    ),
                )
                if retry["popup"] and not retry["error"]:
                    if has_quotable_phrase(retry["popup"]):
                        gen = retry
                        gen["p2_retry"] = True
                    else:
                        gen["popup"] = ""  # 两次都不合格，拒绝弹窗
                        gen["p2_rejected"] = True
                else:
                    # 重试本身失败（API error 等），保留原文但标记 p2 不合格
                    gen["p2_retry_failed"] = True

        # ── FC_STALE: 语义去重 ──
        if gen["popup"] and previous_popups:
            for prev_text in previous_popups[-5:]:  # 最近5条
                sim = semantic_similarity(gen["popup"], prev_text)
                if sim >= dedup_threshold:
                    gen["dedup_suppressed"] = True
                    gen["dedup_similarity"] = round(sim, 2)
                    gen["popup"] = ""  # 拒绝展示
                    suppressed_count += 1
                    break

        # ── 记录历史（用于后续去重）──
        if gen["popup"]:
            previous_popups.append(gen["popup"])

        entry = {
            "channel": p.channel,
            "trigger_type": p.trigger_type,
            "window_chars": p.char_count,
            "popup_order": p_idx,  # 保留原始时间顺序
            "tone": tone,
            "tone_override": override_reason,
            **gen,
        }
        if p.channel == "fast":
            fast_popups.append(entry)
        else:
            slow_popups.append(entry)

    # ── primary_popup: 按时间线取最后一个非空弹窗 ──
    all_ordered = sorted(
        fast_popups + slow_popups,
        key=lambda x: x.get("popup_order", 0),
    )
    primary_popup = ""
    for p in reversed(all_ordered):
        if p.get("popup"):
            primary_popup = p["popup"]
            break

    result = {
        "case_id": case_id,
        "dialogue_chars": dlen,
        "fast_triggers": sum(1 for p in popups if p.channel == "fast"),
        "slow_windows": sum(1 for p in popups if p.channel == "slow"),
        "total_windows": len(popups),
        "suppressed": suppressed_count,
        "fast_popups": fast_popups,
        "slow_popups": slow_popups,
        "primary_popup": primary_popup,
        "total_popups": sum(1 for e in fast_popups + slow_popups if e.get("popup")),
        "_zhouyi_source": "test-default",
    }

    return result


def run_codex_batch_judge(results: list, output_file: str) -> list:
    """批量 Codex 裁判：一次 codex exec 调用评完所有弹窗。

    返回更新后的 results（含 codex_score / codex_reason）。
    """
    import subprocess

    # 构建批量裁判 prompt
    cases_text = ""
    for i, r in enumerate(results):
        dialogue = r.get("_dialogue", "")
        popup = r.get("primary_popup", "")
        if not popup:
            continue
        cases_text += f"""
---
## Case {i + 1}: {r['case_id']}（对话{len(dialogue)}字）
对话: {dialogue}
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
            ["D:/root/.npm-global/codex.cmd", "exec", "--ephemeral", "--json",
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
    parser = argparse.ArgumentParser(description="真实管线批量测试（支持多版本 prompt）")
    parser.add_argument("--prompt", type=str, default="v4.0.18",
                        help="提示词版本: v4.0.12 或 v4.0.18（默认 v4.0.18）")
    parser.add_argument("--cases", type=str, default="",
                        help="逗号分隔的 case_id 列表（默认：全部）")
    parser.add_argument("--dataset", type=str, default=str(DATASET_PATH),
                        help="数据集路径")
    parser.add_argument("--no-judge", action="store_true",
                        help="跳过 Codex 裁判")
    parser.add_argument("--window-size", type=int, default=300,
                        help="慢通道窗口大小（默认 300）")
    args = parser.parse_args()

    # ── 版本解析 ──
    prompt_version = args.prompt
    prompt_path = PROMPT_MAP.get(prompt_version)
    if not prompt_path:
        print(f"❌ 未知版本: {prompt_version}，可用: {list(PROMPT_MAP.keys())}")
        sys.exit(1)
    prompt_label = prompt_version.replace(".", "").replace("v", "v")  # "v4.0.18" → "v4018"

    # ── 加载 ──
    print("=" * 70)
    print(f"{prompt_version} 真实管线批量测试（快通道 + 300字窗口慢通道）")
    print("=" * 70)

    system_prompt = prompt_path.read_text(encoding="utf-8")
    print(f"Prompt: {len(system_prompt)} 字, {system_prompt.count(chr(10))} 行")

    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    keywords = load_keywords()

    # 筛选 cases
    if args.cases:
        target_ids = set(args.cases.split(","))
        dataset = [c for c in dataset if c.get("case_id") in target_ids]
        print(f"筛选: {len(dataset)} 题 ({', '.join(c['case_id'] for c in dataset)})")
    else:
        print(f"数据集: {len(dataset)} 题")

    print(f"生成模型: {GEN_MODEL}")
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

        result = run_case(case, system_prompt, keywords, args.window_size)
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
    out_path = RESULTS_DIR / f"{prompt_label}_pipeline{cases_tag}_{timestamp}.json"

    output = {
        "config": {
            "prompt": f"system_prompt_{prompt_version}.txt",
            "prompt_chars": len(system_prompt),
            "gen_model": GEN_MODEL,
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
