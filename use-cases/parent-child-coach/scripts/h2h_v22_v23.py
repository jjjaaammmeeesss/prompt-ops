"""v2.2 vs v2.3 公平头对头盲测

v2.2 (41K chars): 大量 few-shot 示例驱动，CBT 框架
v2.3 (3.8K chars): 原则驱动，无 few-shot，`——`分隔 + 建议句 + 类型边界

用法:
  python scripts/h2h_v22_v23.py -n 30              # 盲测30条
  python scripts/h2h_v22_v23.py -n 30 --test expert_test  # 用 expert_test.json
"""

import json, os, re, sys, time, argparse, random
from statistics import mean, stdev
from concurrent.futures import ThreadPoolExecutor
import requests

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT, "results", "auto_research_judge_v2")
os.makedirs(OUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# API
# ═══════════════════════════════════════════════════════════════

DS_URL = "https://api.deepseek.com/v1/chat/completions"
DS_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DS_KEY:
    env_path = os.path.join(PROJECT, ".env")
    if os.path.exists(env_path):
        for line in open(env_path, encoding="utf-8"):
            if line.startswith("DEEPSEEK_API_KEY="):
                DS_KEY = line.split("=", 1)[1].strip(); break
if not DS_KEY:
    raise RuntimeError("DEEPSEEK_API_KEY required")

CL_URL = "https://s.lconai.com/v1/messages"
CL_KEY = "CLAUDE_API_KEY_PLACEHOLDER"
CL_MODEL = "claude-opus-4-8"
TASK_MODEL = "deepseek-v4-pro"

# ═══════════════════════════════════════════════════════════════
# Judge v2.0 — 与 h2h_r5_v17.py 完全一致，保证可比性
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
# 数据加载
# ═══════════════════════════════════════════════════════════════

def load_test_set(source="expert_dataset", n=30, seed=42):
    """加载测试集。source 可以是文件名或 'expert_test' / 'expert_dataset'。"""
    if source == "expert_test":
        path = os.path.join(PROJECT, "data", "expert_test.json")
    elif source == "expert_dataset":
        path = os.path.join(PROJECT, "data", "expert_dataset.json")
    else:
        path = os.path.join(PROJECT, "data", source)

    data = json.load(open(path, "r", encoding="utf-8"))

    # 过滤空对话
    valid = [e for e in data if len(e.get("dialogue", "").strip()) > 50]
    empty = len(data) - len(valid)
    if empty:
        print(f"  过滤空对话: {empty} 条")

    rng = random.Random(seed)
    sampled = rng.sample(valid, min(n, len(valid)))
    print(f"  测试集: {len(sampled)} 条 (seed={seed})")
    return sampled


# ═══════════════════════════════════════════════════════════════
# API 调用
# ═══════════════════════════════════════════════════════════════

def deepseek(sys_prompt, dialogue, max_tokens=800, temp=0.7):
    """用 DeepSeek 生成弹窗。"""
    headers = {"Authorization": f"Bearer {DS_KEY}", "Content-Type": "application/json"}
    user_msg = f"对话：\n{dialogue}\n\n请生成弹窗："
    for attempt in range(4):
        try:
            resp = requests.post(DS_URL, headers=headers,
                json={"model": TASK_MODEL, "max_tokens": max_tokens, "temperature": temp,
                      "messages": [{"role": "system", "content": sys_prompt},
                                   {"role": "user", "content": user_msg}]},
                timeout=(30, 120))
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            if len(text) >= 20:
                return text
        except Exception as e:
            if attempt < 3:
                time.sleep(2 ** attempt * 3)
    return "[ERROR] DeepSeek failed after 4 attempts"


def claude_judge(text):
    """用 Claude 盲评弹窗。"""
    headers = {"x-api-key": CL_KEY, "anthropic-version": "2023-06-01",
               "Content-Type": "application/json"}
    for attempt in range(4):
        try:
            resp = requests.post(CL_URL, headers=headers,
                json={"model": CL_MODEL, "max_tokens": 1024, "temperature": 0.0,
                      "thinking": {"type": "disabled"},
                      "system": "你是严格的亲子沟通弹窗评估专家。只输出JSON，不要markdown包裹。",
                      "messages": [{"role": "user", "content": text}]},
                timeout=(30, 90))
            resp.raise_for_status()
            for block in resp.json().get("content", []):
                if isinstance(block, dict) and block.get("type") == "text":
                    return block["text"]
        except Exception as e:
            if attempt < 3:
                time.sleep(2 ** attempt * 3)
    return '{"veto": "judge_error", "brief_reason": "Claude failed after 4 attempts"}'


def parse_json(raw):
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


def compute_score(scores_dict):
    veto = scores_dict.get("veto")
    if veto and str(veto).strip() not in ("null", "none", ""):
        return 0.0, True, str(veto)
    ws, tw = 0.0, 0.0
    for dk, w in DIM_WEIGHTS:
        v = scores_dict.get(dk)
        if isinstance(v, (int, float)) and 1 <= v <= 5:
            ws += (v - 1) / 4 * w
            tw += w
    return ws / tw if tw > 0 else 0.0, False, None


def judge_one(dialogue, popup):
    prompt = JUDGE_PROMPT.format(dialogue=dialogue, popup=popup)
    try:
        raw = claude_judge(prompt)
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

def run():
    parser = argparse.ArgumentParser(description="v2.2 vs v2.3 公平盲测")
    parser.add_argument("-n", type=int, default=30, help="测试对话数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--test", type=str, default="expert_dataset",
                        help="测试数据源 (expert_test / expert_dataset / 文件名)")
    args = parser.parse_args()

    # 加载 prompt
    prompt_v22 = open(os.path.join(PROJECT, "system_prompt_v2.2.txt"),
                       "r", encoding="utf-8").read().strip()
    prompt_v23 = open(os.path.join(PROJECT, "system_prompt_v2.3.txt"),
                       "r", encoding="utf-8").read().strip()

    # 加载测试集
    test_set = load_test_set(source=args.test, n=args.n, seed=args.seed)

    print(f"\n{'=' * 80}")
    print(f"v2.2 vs v2.3 · 公平盲测")
    print(f"  测试集:   {len(test_set)} 条 ({args.test}, seed={args.seed})")
    print(f"  v2.2:     {len(prompt_v22)} chars (大量 few-shot，CBT 框架)")
    print(f"  v2.3:     {len(prompt_v23)} chars (原则驱动，——分隔 + 建议句)")
    print(f"  任务模型: {TASK_MODEL}")
    print(f"  Judge:    {CL_MODEL} v2.0 (5-dim + veto, thinking=disabled)")
    print(f"{'=' * 80}")

    results = []
    all_v22, all_v23 = [], []
    wins = {"v2.2": 0, "v2.3": 0, "tie": 0}
    t_start = time.time()

    for di, item in enumerate(test_set):
        dialogue = item["dialogue"]
        eid = item.get("id", f"case_{di}")
        title = item.get("case_title", "") or item.get("title", "")
        d_short = dialogue[:80].replace("\n", " ")

        eta = "" if di == 0 else \
            f" | ETA {((time.time()-t_start)/di*(len(test_set)-di)/60):.0f}m"
        print(f"\n[{di+1}/{len(test_set)}] {eid}", flush=True)
        print(f"  {title}", flush=True)
        print(f"  {d_short}...{eta}", flush=True)

        # 并行生成两个版本的弹窗
        gen_start = time.time()
        with ThreadPoolExecutor(max_workers=2) as pool:
            f22 = pool.submit(deepseek, prompt_v22, dialogue)
            f23 = pool.submit(deepseek, prompt_v23, dialogue)
            popup_v22 = f22.result()
            popup_v23 = f23.result()
        gen_time = time.time() - gen_start

        # 并行盲评（judge 不知道版本）
        judge_start = time.time()
        with ThreadPoolExecutor(max_workers=2) as pool:
            j22 = pool.submit(judge_one, dialogue, popup_v22)
            j23 = pool.submit(judge_one, dialogue, popup_v23)
            score_v22, dims_v22 = j22.result()
            score_v23, dims_v23 = j23.result()
        judge_time = time.time() - judge_start

        v22_veto = dims_v22.get("veto", "")
        v23_veto = dims_v23.get("veto", "")
        v22_flag = f" [VETO:{v22_veto}]" if v22_veto else ""
        v23_flag = f" [VETO:{v23_veto}]" if v23_veto else ""

        delta = score_v23 - score_v22
        if delta > 0.01:
            winner = "v2.3"
        elif delta < -0.01:
            winner = "v2.2"
        else:
            winner = "tie"
        wins[winner] += 1

        print(f"  v2.2: {score_v22:.4f}{v22_flag} | v2.3: {score_v23:.4f}{v23_flag} | "
              f"Δ={delta:+.4f} → {winner}  (gen {gen_time:.0f}s + judge {judge_time:.0f}s)",
              flush=True)
        print(f"    v2.2: {dims_v22.get('brief_reason','')[:120]}", flush=True)
        print(f"    v2.3: {dims_v23.get('brief_reason','')[:120]}", flush=True)

        all_v22.append(score_v22)
        all_v23.append(score_v23)

        results.append({
            "case_id": eid,
            "case_title": title,
            "dialogue": dialogue,
            "dialogue_preview": d_short,
            "v22": {
                "popup": popup_v22,
                "score": score_v22,
                "dims": {DIM_LABELS.get(k, k): dims_v22.get(k) for k in DIM_LABELS},
                "veto": v22_veto,
                "brief_reason": dims_v22.get("brief_reason", ""),
            },
            "v23": {
                "popup": popup_v23,
                "score": score_v23,
                "dims": {DIM_LABELS.get(k, k): dims_v23.get(k) for k in DIM_LABELS},
                "veto": v23_veto,
                "brief_reason": dims_v23.get("brief_reason", ""),
            },
            "delta": delta,
            "winner": winner,
        })

        time.sleep(0.3)  # rate limit

    # ═══════════════════════════════════════════════════════════
    # 汇总
    # ═══════════════════════════════════════════════════════════

    total_time = time.time() - t_start
    n = len(results)
    m22 = mean(all_v22) if all_v22 else 0
    m23 = mean(all_v23) if all_v23 else 0
    s22 = stdev(all_v22) if len(all_v22) > 1 else 0
    s23 = stdev(all_v23) if len(all_v23) > 1 else 0
    vetos_22 = sum(1 for r in results if r["v22"]["veto"])
    vetos_23 = sum(1 for r in results if r["v23"]["veto"])
    delta_mean = m23 - m22

    # 维度聚合
    dim_agg = {"v2.2": {}, "v2.3": {}}
    for version_key, result_key in [("v2.2", "v22"), ("v2.3", "v23")]:
        for dk in DIM_LABELS.values():
            vals = []
            for r in results:
                v = r[result_key]["dims"].get(dk)
                if isinstance(v, (int, float)) and 1 <= v <= 5:
                    vals.append((v - 1) / 4)
            dim_agg[version_key][dk] = {
                "mean": mean(vals) if vals else 0,
                "std": stdev(vals) if len(vals) > 1 else 0,
                "n": len(vals),
            }

    # 打印汇总
    print(f"\n{'=' * 80}")
    print(f"📊 盲测结果")
    print(f"  总耗时: {total_time/60:.1f}m ({total_time/n:.0f}s/sample)")
    print(f"{'=' * 80}")

    print(f"\n  ┌─ 总体对比 ─{'─'*45}")
    print(f"  │ {'':12} {'v2.2 (基线)':<22} {'v2.3 (实验)':<22}")
    print(f"  │ {'─'*12} {'─'*22} {'─'*22}")
    print(f"  │ {'均分':12} {m22:.4f} ± {s22:.4f}     {m23:.4f} ± {s23:.4f}")
    print(f"  │ {'Veto次数':12} {vetos_22:<22} {vetos_23:<22}")
    print(f"  │ {'胜/平/负':12} v2.2={wins['v2.2']} 平={wins['tie']}  v2.3={wins['v2.3']}")
    print(f"  │ {'Δ (v2.3-v2.2)':12} {delta_mean:+.4f}")
    print(f"  └{'─'*55}")

    print(f"\n  ┌─ 维度对比 ─{'─'*45}")
    print(f"  │ {'维度':<10} {'v2.2':<10} {'v2.3':<10} {'Δ':>8}  {'趋势':>6}")
    print(f"  │ {'─'*10} {'─'*10} {'─'*10} {'─'*8}  {'─'*6}")
    for dk in DIM_LABELS.values():
        m1 = dim_agg["v2.2"][dk]["mean"]
        m2 = dim_agg["v2.3"][dk]["mean"]
        d = m2 - m1
        trend = "📈 v2.3↑" if d > 0.02 else ("📉 v2.3↓" if d < -0.02 else "≈ 持平")
        print(f"  │ {dk:<8}  {m1:.4f}    {m2:.4f}    {d:+.4f}   {trend}")
    print(f"  └{'─'*55}")

    # 分数分布
    print(f"\n  ┌─ 分数分布 ─{'─'*45}")
    for label, scores in [("v2.2", all_v22), ("v2.3", all_v23)]:
        buckets = {"[0.0-0.2)": 0, "[0.2-0.4)": 0, "[0.4-0.6)": 0,
                   "[0.6-0.8)": 0, "[0.8-1.0)": 0, "[1.0]": 0}
        for s in scores:
            if s >= 1.0: buckets["[1.0]"] += 1
            elif s >= 0.8: buckets["[0.8-1.0)"] += 1
            elif s >= 0.6: buckets["[0.6-0.8)"] += 1
            elif s >= 0.4: buckets["[0.4-0.6)"] += 1
            elif s >= 0.2: buckets["[0.2-0.4)"] += 1
            else: buckets["[0.0-0.2)"] += 1
        bars = "  ".join(f"{k}: {'█'*v}{v}" for k, v in buckets.items())
        print(f"  │ {label}: {bars}")

    # 逐条对比表
    print(f"\n  ┌─ 逐条对比 ─{'─'*45}")
    print(f"  │ {'#':<3} {'案例':<22} {'v2.2':>6} {'v2.3':>6} {'Δ':>8}  {'胜者'}")
    print(f"  │ {'─'*3} {'─'*22} {'─'*6} {'─'*6} {'─'*8}  {'─'*4}")
    for i, r in enumerate(results):
        title = r["case_title"][:20] if r["case_title"] else r["dialogue_preview"][:20]
        v22s = f"{r['v22']['score']:.4f}"
        v23s = f"{r['v23']['score']:.4f}"
        d = f"{r['delta']:+.4f}"
        w = r["winner"]
        veto_mark = ""
        if r["v22"]["veto"]: veto_mark += " ⚠v2.2"
        if r["v23"]["veto"]: veto_mark += " ⚠v2.3"
        print(f"  │ {i+1:<3} {title:<22} {v22s:>6} {v23s:>6} {d:>8}  {w}{veto_mark}")
    print(f"  └{'─'*55}")

    # 保存
    out = {
        "config": {
            "task_model": TASK_MODEL,
            "judge_model": CL_MODEL,
            "judge_version": "v2.0 (5-dim + veto + few-shot, thinking=disabled)",
            "test_source": args.test,
            "n_dialogues": n,
            "seed": args.seed,
            "v22_chars": len(prompt_v22),
            "v23_chars": len(prompt_v23),
            "v22_file": "system_prompt_v2.2.txt",
            "v23_file": "system_prompt_v2.3.txt",
            "total_time_s": round(total_time, 1),
        },
        "summary": {
            "v22_mean": m22, "v22_std": s22,
            "v23_mean": m23, "v23_std": s23,
            "delta": delta_mean,
            "v22_vetos": vetos_22, "v23_vetos": vetos_23,
            "wins": wins,
            "dim_comparison": dim_agg,
        },
        "results": results,
    }

    out_path = os.path.join(OUT_DIR, "h2h_v22_v23_blind.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 盲测结果 → {out_path}")
    print(f"   v2.2={m22:.4f}  v2.3={m23:.4f}  Δ={delta_mean:+.4f}")
    print(f"   H2H: v2.2={wins['v2.2']}  v2.3={wins['v2.3']}  tie={wins['tie']}")

    return out_path


if __name__ == "__main__":
    run()
