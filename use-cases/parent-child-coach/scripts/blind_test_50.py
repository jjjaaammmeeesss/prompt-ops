"""v4.0.12 在 50 题盲测集上的泛化评估（支持递增 N 题）

用 H2H 5 维度盲评（being_seen/dialogue_fidelity/core_insight/natural_language/warmth），
不需要 gold answer。Judge 后端用 deepseek-chat（与 test_ladder 生成模型一致）。

生成配置与 test_ladder.py 完全一致：
  - model: deepseek/deepseek-chat (via LiteLLMModelAdapter)
  - temperature: 0.3
  - max_tokens: 640
  - tone-mode:
      * forced-diag: 强制诊断式
      * auto (默认): 让 prompt 按信念维度自判

Judge 配置：
  - model: deepseek-chat (直接 requests 调 API)
  - temperature: 0.0
  - 5 维度盲评，不需要 gold answer
  - VETO 触发则总分=0

用法:
  $env:JUDGE_BACKEND="deepseek"
  # 递增盲测：5 → 10 → 20
  python scripts/blind_test_50.py --n 5
  python scripts/blind_test_50.py --n 10
  python scripts/blind_test_50.py --n 20
  # 全量 50 题
  python scripts/blind_test_50.py
"""

import json
import os
import re
import sys
import time
import argparse
from statistics import mean, stdev
from concurrent.futures import ThreadPoolExecutor
import requests

# ═══════════════════════════════════════════════════════════════
# 路径与 API 配置
# ═══════════════════════════════════════════════════════════════

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT, "results", "blind_tests")
os.makedirs(OUT_DIR, exist_ok=True)

DS_URL = "https://api.deepseek.com/v1/chat/completions"
DS_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DS_KEY:
    env_path = os.path.join(PROJECT, ".env")
    if os.path.exists(env_path):
        for line in open(env_path, encoding="utf-8"):
            if line.startswith("DEEPSEEK_API_KEY="):
                DS_KEY = line.split("=", 1)[1].strip()
                break
if not DS_KEY:
    raise RuntimeError("DEEPSEEK_API_KEY required (env var or .env file)")
# 同步到 os.environ，让 litellm 也能拿到
os.environ["DEEPSEEK_API_KEY"] = DS_KEY

TASK_MODEL = "deepseek-chat"      # 与 test_ladder.py 一致
JUDGE_MODEL = "deepseek-chat"     # Judge 也用 deepseek-chat（claude 会 403）

# ═══════════════════════════════════════════════════════════════
# 盲评 Judge Prompt（来自 h2h_v22_v23.py，5 维度不需要 gold）
# ═══════════════════════════════════════════════════════════════

JUDGE_PROMPT = """你是亲子沟通弹窗评估专家。给弹窗从五个维度打分（1-5整数）。

### 一级否决（任一触发则总分=0）
- **事实性错误**：弹窗编造了对话中不存在的内容（张冠李戴、无中生有）
- **语气严重误判**：该鼓励的时刻弹窗用了批评/诊断的语气，或反之

### 五个维度
1. **看见感** (1-5): 家长读完会不会心里轻轻动一下——"你懂我"？是否先看见了家长的发心和难处？
2. **对话忠实度** (1-5): 每个判断都能在对话原文中找到确切依据？没有脑补、没有过度解读？
3. **命中核心** (1-5): 是否抓住了这段对话里最该被看见的那个点？是挠到痒处还是隔靴搔痒？
4. **人话感** (1-5): 像真人在耳边说话？没有术语、没有框架标签、没有模板套话、没有"你正戴着X的眼镜"这类机械句式？
5. **温度** (1-5): 整体姿态是盟友（并肩看同一个问题）还是教师（居高临下指点）？

### Few-shot 校准参考
- 翻日记对话中，弹窗"信任的门不是撞开的，是你每次蹲下来的时候自己打开的"→ 全5分
- 弹窗含"多极"等术语 + "你正戴着灾难化的眼镜在看孩子" → 人话感=1分

---

### 对话
{dialogue}

### AI弹窗
{popup}

请严格输出JSON（不要markdown包裹，不要额外文字）：
{{"veto":null,"being_seen":1-5,"dialogue_fidelity":1-5,"core_insight":1-5,"natural_language":1-5,"warmth":1-5,"brief_reason":"一句话中文理由"}}"""

DIM_WEIGHTS = [
    ("being_seen", 0.25), ("dialogue_fidelity", 0.20), ("core_insight", 0.20),
    ("natural_language", 0.20), ("warmth", 0.15),
]
DIM_LABELS = {
    "being_seen": "看见感", "dialogue_fidelity": "对话忠实度",
    "core_insight": "命中核心", "natural_language": "人话感", "warmth": "温度",
}

# ═══════════════════════════════════════════════════════════════
# 生成弹窗（与 test_ladder.py 一致：LiteLLMModelAdapter + deepseek-chat）
# ═══════════════════════════════════════════════════════════════

# 复用 prompt_ops 的 LiteLLMModelAdapter，与 test_ladder 完全一致
sys.path.insert(0, os.path.join(PROJECT, "..", "src"))
try:
    from prompt_ops.core.model import LiteLLMModelAdapter
except ImportError:
    # 回退：直接用 requests 调 deepseek API
    LiteLLMModelAdapter = None


def _build_type_instruction(tone: str) -> str:
    """根据 tone 模式构造 type_instruction。
    - "诊断式" / "鼓励式": 强制指定类型（与 test_ladder 一致）
    - "auto": 让 prompt 按其内部规则（v4.0.7 第 207 行）自判类型
    """
    if tone == "auto":
        return (
            "请按系统提示词第六章规则生成弹窗，弹窗类型由你根据信念维度自动判定"
            "（收缩→诊断式 100-200字，打开→鼓励式 30-60字）。"
        )
    elif tone == "鼓励式":
        return (
            "请生成**鼓励式弹窗**（30-80字）。"
            "必须：具体点出家长刚展现的积极模式 → 简短有力。"
        )
    else:  # 诊断式（默认）
        return (
            "请生成**诊断式弹窗**（80-200字）。"
            "必须：先承认发心 → 揭示具体模式 → 给出一个微小可做的尝试。"
        )


def generate_popup_litellm(model, system_prompt: str, dialogue: str, tone: str = "诊断式") -> str:
    """与 test_ladder.generate_popup 完全一致的生成逻辑（tone='auto' 时让 prompt 自判）。"""
    type_instruction = _build_type_instruction(tone)

    user_content = f"""当前对话：
{dialogue}

{type_instruction}

请直接输出弹窗全文（不附加解释、不输出JSON、不输出"弹窗："等前缀）："""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    if hasattr(model, "generate_with_chat_format"):
        raw = model.generate_with_chat_format(
            messages=messages, temperature=0.3, max_tokens=640,
        )
    else:
        combined = "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)
        raw = model.generate(prompt=combined, temperature=0.3, max_tokens=640)

    return raw.strip()


def generate_popup_requests(system_prompt: str, dialogue: str, tone: str = "诊断式") -> str:
    """回退：直接用 requests 调 deepseek API（与 test_ladder 参数一致）。"""
    type_instruction = _build_type_instruction(tone)

    user_content = f"""当前对话：
{dialogue}

{type_instruction}

请直接输出弹窗全文（不附加解释、不输出JSON、不输出"弹窗："等前缀）："""

    headers = {"Authorization": f"Bearer {DS_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": TASK_MODEL,
        "max_tokens": 640,
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }

    for attempt in range(4):
        try:
            resp = requests.post(DS_URL, headers=headers, json=payload, timeout=(30, 120))
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            if len(text) >= 10:
                return text
        except Exception as e:
            if attempt < 3:
                time.sleep(2 ** attempt * 3)
    return "[ERROR] generation failed after 4 attempts"

# ═══════════════════════════════════════════════════════════════
# 盲评 Judge（deepseek-chat 后端）
# ═══════════════════════════════════════════════════════════════


def deepseek_judge(text: str) -> str:
    """用 deepseek-chat 盲评弹窗。"""
    headers = {"Authorization": f"Bearer {DS_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": JUDGE_MODEL,
        "max_tokens": 1024,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": "你是严格的亲子沟通弹窗评估专家。只输出JSON，不要markdown包裹。"},
            {"role": "user", "content": text},
        ],
    }
    for attempt in range(4):
        try:
            resp = requests.post(DS_URL, headers=headers, json=payload, timeout=(30, 90))
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            if attempt < 3:
                time.sleep(2 ** attempt * 3)
    return '{"veto": "judge_error", "brief_reason": "deepseek judge failed after 4 attempts"}'


def parse_json(raw: str) -> dict:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Cannot parse JSON: {raw[:300]}")


def compute_score(scores_dict: dict):
    """计算加权得分。返回 (score, is_veto, veto_reason)。"""
    veto = scores_dict.get("veto")
    if veto and str(veto).strip() not in ("null", "none", ""):
        return 0.0, True, str(veto)
    ws, tw = 0.0, 0.0
    for dk, w in DIM_WEIGHTS:
        v = scores_dict.get(dk)
        if isinstance(v, (int, float)) and 1 <= v <= 5:
            ws += (v - 1) / 4 * w
            tw += w
    return (ws / tw if tw > 0 else 0.0), False, None


def judge_one(dialogue: str, popup: str):
    """评一个弹窗。返回 (score, dims_dict)。"""
    prompt = JUDGE_PROMPT.format(dialogue=dialogue, popup=popup)
    try:
        raw = deepseek_judge(prompt)
        scores = parse_json(raw)
        score, is_veto, veto_reason = compute_score(scores)
        dims = {dk: scores.get(dk) for dk in DIM_LABELS}
        dims["veto"] = veto_reason if is_veto else None
        dims["brief_reason"] = scores.get("brief_reason", "")
        return score, dims
    except Exception as e:
        return 0.0, {"veto": "parse_error", "brief_reason": str(e)[:80]}


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════


def run_blind_test(prompt_path: str, dataset_path: str, output_dir: str, tone_mode: str = "forced-diag", n_questions: int = None):
    """跑 50 题盲测。
    tone_mode:
      - "forced-diag": 强制诊断式（与原 v4.0.7 首跑一致）
      - "auto": 让 prompt 按其内部规则（信念维度）自判诊断式/鼓励式
    n_questions: 可选，只跑前 N 题（用于递增盲测 5→10→20）
    """
    # 映射 tone_mode → 传给生成函数的 tone 字符串
    tone_arg = "auto" if tone_mode == "auto" else "诊断式"
    """跑 50 题盲测。"""
    # 加载 prompt
    with open(prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()
    prompt_name = os.path.basename(prompt_path)

    # 加载数据集
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    if n_questions is not None and n_questions > 0:
        dataset = dataset[:n_questions]
    print(f"  加载数据集: {len(dataset)} 条 ({os.path.basename(dataset_path)})")

    # 初始化生成模型
    use_litelhm = LiteLLMModelAdapter is not None
    if use_litelhm:
        try:
            model = LiteLLMModelAdapter(
                model_name="deepseek/deepseek-chat",
                temperature=0.3,
                max_tokens=640,
            )
            print(f"  生成模型: LiteLLMModelAdapter(deepseek/deepseek-chat, temp=0.3)")
        except Exception as e:
            print(f"  ⚠ LiteLLMModelAdapter 初始化失败 ({e})，回退到 requests 直调")
            use_litelhm = False
            model = None
    if not use_litelhm:
        print(f"  生成模型: requests 直调 deepseek-chat (temp=0.3, max_tokens=640)")

    print(f"\n{'='*60}")
    print(f"  v4.0.7 50 题盲测泛化评估")
    print(f"  Prompt:    {prompt_name}")
    print(f"  数据集:    {len(dataset)} 条")
    print(f"  生成模型:  {TASK_MODEL}")
    print(f"  Tone 模式: {tone_mode}{'（让 prompt 自判）' if tone_mode == 'auto' else '（强制诊断式）'}")
    print(f"  Judge:     {JUDGE_MODEL} (H2H 5 维度盲评, 不需要 gold)")
    print(f"{'='*60}")

    results = []
    veto_count = 0
    all_scores = []
    dim_scores = {dk: [] for dk in DIM_LABELS}

    for i, case in enumerate(dataset):
        dialogue = case.get("question", "")
        case_id = case.get("id", f"blind_{i+1:02d}")

        if len(dialogue.strip()) < 50:
            print(f"  [{i+1}/{len(dataset)}] {case_id}: 对话过短，跳过")
            continue

        # 生成弹窗
        start = time.time()
        if use_litelhm:
            popup = generate_popup_litellm(model, system_prompt, dialogue, tone=tone_arg)
        else:
            popup = generate_popup_requests(system_prompt, dialogue, tone=tone_arg)
        elapsed = time.time() - start

        # 盲评
        score, dims = judge_one(dialogue, popup)

        is_veto = dims.get("veto") is not None
        if is_veto:
            veto_count += 1

        all_scores.append(score)
        for dk in DIM_LABELS:
            v = dims.get(dk)
            if isinstance(v, (int, float)) and 1 <= v <= 5:
                dim_scores[dk].append(v)

        results.append({
            "id": case_id,
            "dialogue": dialogue,
            "generated_popup": popup,
            "score": round(score, 3),
            "is_veto": is_veto,
            "veto_reason": dims.get("veto"),
            "dims": {dk: dims.get(dk) for dk in DIM_LABELS},
            "brief_reason": dims.get("brief_reason", ""),
            "gen_elapsed": round(elapsed, 1),
        })

        status = "VETO" if is_veto else ("FAIL" if score < 0.70 else "PASS")
        print(f"  [{i+1}/{len(dataset)}] {case_id} | {status} | {score:.3f} | {dims.get('brief_reason', '')[:60]}")

    # 汇总统计
    n = len(results)
    avg_score = mean(all_scores) if all_scores else 0
    stdev_score = stdev(all_scores) if len(all_scores) > 1 else 0
    pass_count = sum(1 for s in all_scores if s >= 0.70)
    fail_count = n - pass_count - veto_count

    dim_summary = {}
    for dk, label in DIM_LABELS.items():
        vals = dim_scores[dk]
        if vals:
            dim_summary[dk] = {
                "label": label,
                "mean": round(mean(vals), 2),
                "stdev": round(stdev(vals), 2) if len(vals) > 1 else 0,
                "min": min(vals),
                "max": max(vals),
                "count": len(vals),
            }

    summary = {
        "prompt": prompt_name,
        "dataset": os.path.basename(dataset_path),
        "n_total": n,
        "avg_score": round(avg_score, 3),
        "stdev_score": round(stdev_score, 3),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "veto_count": veto_count,
        "pass_rate": round(pass_count / n, 3) if n else 0,
        "veto_rate": round(veto_count / n, 3) if n else 0,
        "dim_summary": dim_summary,
        "task_model": TASK_MODEL,
        "judge_model": JUDGE_MODEL,
        "judge_dims": "H2H 5 维度盲评 (being_seen/dialogue_fidelity/core_insight/natural_language/warmth)",
        "gen_config": {"temperature": 0.3, "max_tokens": 640, "tone_mode": tone_mode},
    }

    # 打印摘要
    print(f"\n{'='*60}")
    print(f"  盲测结果摘要")
    print(f"{'='*60}")
    print(f"  总数:        {n}")
    print(f"  均分:        {avg_score:.3f} (stdev={stdev_score:.3f})")
    print(f"  通过 (≥0.70): {pass_count}/{n} ({summary['pass_rate']*100:.1f}%)")
    print(f"  失败 (<0.70): {fail_count}/{n}")
    print(f"  VETO:        {veto_count}/{n} ({summary['veto_rate']*100:.1f}%)")
    print(f"\n  5 维度均值:")
    for dk, label in DIM_LABELS.items():
        s = dim_summary.get(dk, {})
        if s:
            print(f"    {label:<8} ({dk}): {s['mean']:.2f} (stdev={s['stdev']:.2f}, min={s['min']}, max={s['max']})")

    # 模式性失败检查（低分 case）
    low_score_cases = sorted([r for r in results if not r["is_veto"] and r["score"] < 0.70],
                              key=lambda x: x["score"])
    if low_score_cases:
        print(f"\n  低分 case (<0.70, 非 VETO) — 检查模式性失败:")
        for r in low_score_cases[:10]:
            print(f"    {r['id']}: {r['score']:.3f} | {r['brief_reason'][:80]}")

    if veto_count > 0:
        print(f"\n  VETO case — 检查事实性错误/语气误判:")
        for r in results:
            if r["is_veto"]:
                print(f"    {r['id']}: {r['veto_reason']} | {r['brief_reason'][:80]}")

    # 保存结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    tone_suffix = "_auto" if tone_mode == "auto" else "_forceddiag"
    out_path = os.path.join(output_dir, f"blind_50_{prompt_name.replace('.txt','')}{tone_suffix}_{timestamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)

    print(f"\n  结果保存: {out_path}")
    return summary, results


def main():
    parser = argparse.ArgumentParser(description="v4.0.19 50 题盲测泛化评估（支持递增 N 题）")
    parser.add_argument("--prompt", default="system_prompt_v4.0.19.txt",
                        help="prompt 文件名 (默认 system_prompt_v4.0.19.txt)")
    parser.add_argument("--dataset", default="dataset_50_questions.json",
                        help="数据集文件名 (默认 dataset_50_questions.json)")
    parser.add_argument("--output-dir", default=OUT_DIR,
                        help=f"输出目录 (默认 {OUT_DIR})")
    parser.add_argument("--tone-mode", choices=["forced-diag", "auto"], default="auto",
                        help="forced-diag=强制诊断式；auto=让 prompt 按信念维度自判（默认）")
    parser.add_argument("--n", type=int, default=None,
                        help="只跑前 N 题（递增盲测：5/10/20）")
    args = parser.parse_args()

    prompt_path = os.path.join(PROJECT, args.prompt) if not os.path.isabs(args.prompt) else args.prompt
    dataset_path = os.path.join(PROJECT, args.dataset) if not os.path.isabs(args.dataset) else args.dataset

    if not os.path.exists(prompt_path):
        print(f"ERROR: prompt 文件不存在: {prompt_path}")
        sys.exit(1)
    if not os.path.exists(dataset_path):
        print(f"ERROR: 数据集文件不存在: {dataset_path}")
        sys.exit(1)

    run_blind_test(prompt_path, dataset_path, args.output_dir, tone_mode=args.tone_mode, n_questions=args.n)


if __name__ == "__main__":
    main()
