"""r5 vs v1.7 公平头对头比赛

Phase 1 盲测: 从 expert_dataset.json 抽取独立样本，Claude v2.0 盲评
Phase 2 对标: 拿校标（expert_score, reference_popup, hit_checklist）做标准答案对比

用法:
  python scripts/h2h_r5_v17.py -n 30                    # 盲测30条
  python scripts/h2h_r5_v17.py -n 30 --compare           # 盲测 + 对标
  python scripts/h2h_r5_v17.py --compare result.json     # 仅对标已有结果
"""

import json, os, re, sys, time, argparse, random
from statistics import mean, stdev
from concurrent.futures import ThreadPoolExecutor
import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "results", "auto_research_judge_v2")
os.makedirs(OUT_DIR, exist_ok=True)

# ── API Keys ──────────────────────────────────────────────
DS_URL = "https://api.deepseek.com/v1/chat/completions"
DS_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DS_KEY:
    env_path = os.path.join(BASE_DIR, ".env")
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

# ── Load Prompts ──────────────────────────────────────────
PROMPT_V17 = open(os.path.join(BASE_DIR, "system_prompt.txt"), "r", encoding="utf-8").read().strip()
PROMPT_R5  = open(os.path.join(OUT_DIR, "final_best_prompt.txt"), "r", encoding="utf-8").read().strip()

# ── Judge v2.0 ────────────────────────────────────────────
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

# ═══════════════════════════════════════════════════════════
# Dataset Helpers
# ═══════════════════════════════════════════════════════════

def load_expert_test_set(n=30, seed=42):
    """Load expert_dataset.json, filter empty dialogues, random sample.

    GB_xxx entries have empty dialogue fields — excluded as data quality issue.
    Overlap with merged_train is noted but NOT used for exclusion:
    expert_score (校标) was never seen during r5 evolution, so it's a clean signal.
    """
    expert = json.load(open(os.path.join(DATA_DIR, "expert_dataset.json"), "r", encoding="utf-8"))

    # Filter: must have real dialogue content (>50 chars)
    valid = [e for e in expert if len(e.get("dialogue", "").strip()) > 50]
    empty = [e for e in expert if len(e.get("dialogue", "").strip()) <= 50]

    print(f"expert_dataset: {len(expert)} total")
    print(f"  空对话/GB条目 (排除): {len(empty)} 条")
    if empty:
        ids = [e["id"][:20] for e in empty[:5]]
        print(f"  示例: {ids}...")
    print(f"  有效对话: {len(valid)} 条")

    # Random sample from valid entries
    rng = random.Random(seed)
    sampled = rng.sample(valid, min(n, len(valid)))
    print(f"  随机抽样: {len(sampled)} 条 (seed={seed})")
    return sampled, empty


def normalize_dialogue(text):
    """Strip speaker labels, normalize whitespace for comparison."""
    # Remove speaker labels like "1孩子：", "2妈妈："
    text = re.sub(r'\d+(孩子|妈妈|爸爸|儿子|女儿|老师|同学)[：:]', '', text)
    text = re.sub(r'\s+', '', text)
    return text[:100]

# ═══════════════════════════════════════════════════════════
# API Helpers
# ═══════════════════════════════════════════════════════════

def deepseek(sys_prompt, user_text, max_tokens=800, temp=0.7):
    headers = {"Authorization": f"Bearer {DS_KEY}", "Content-Type": "application/json"}
    for attempt in range(4):
        try:
            resp = requests.post(DS_URL, headers=headers,
                json={"model": TASK_MODEL, "max_tokens": max_tokens, "temperature": temp,
                      "messages": [{"role": "system", "content": sys_prompt},
                                   {"role": "user", "content": user_text}]},
                timeout=(30, 120))
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"    ⚠ DS error (attempt {attempt+1}/4): {type(e).__name__}", flush=True)
            time.sleep(2 ** attempt * 3)
    return "[ERROR] DeepSeek failed after 4 attempts"

def claude_judge(text):
    headers = {"x-api-key": CL_KEY, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
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
            print(f"    ⚠ CL error (attempt {attempt+1}/4): {type(e).__name__}", flush=True)
            time.sleep(2 ** attempt * 3)
    return '{"veto": "judge_error", "brief_reason": "Claude failed after 4 attempts"}'

def parse_json(raw):
    raw = raw.strip()
    try: return json.loads(raw)
    except json.JSONDecodeError: pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if m:
        try: return json.loads(m.group(1))
        except json.JSONDecodeError: pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try: return json.loads(m.group(0))
        except json.JSONDecodeError: pass
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
        print(f"    ❌ Judge parse failed: {type(e).__name__}", flush=True)
        return 0.0, {"veto": "parse_error", "brief_reason": str(e)[:80]}

# ═══════════════════════════════════════════════════════════
# Phase 1: Blind H2H
# ═══════════════════════════════════════════════════════════

def run_blind_h2h(n=30, seed=42):
    test_set, overlapped = load_expert_test_set(n=n, seed=seed)

    print(f"\n{'=' * 80}")
    print(f"Phase 1 · r5 vs v1.7 盲测")
    print(f"  测试集: {len(test_set)} 条 (expert_dataset 独立样本, seed={seed})")
    print(f"  v1.7:   {len(PROMPT_V17)} chars")
    print(f"  r5:     {len(PROMPT_R5)} chars")
    print(f"  任务模型: {TASK_MODEL}")
    print(f"  Judge:    {CL_MODEL} v2.0 (5-dim + veto, thinking=disabled)")
    print(f"  盲评:     judge 不知道弹窗来自哪个版本")
    print(f"  ⚠ 不偷看 expert_score / reference_popup")
    print(f"{'=' * 80}")

    results = []
    all_v17, all_r5 = [], []
    wins = {"v1.7": 0, "r5": 0, "tie": 0}
    t_start = time.time()

    for di, item in enumerate(test_set):
        dialogue = item["dialogue"]
        expert_id = item["id"]
        case_title = item.get("case_title", "")
        d_short = dialogue[:70].replace("\n", " ")

        eta = "" if di == 0 else f" | ETA {((time.time()-t_start)/di*(len(test_set)-di)/60):.0f}m"
        print(f"\n[{di+1}/{len(test_set)}] {expert_id}", flush=True)
        print(f"  {case_title}", flush=True)
        print(f"  {d_short}...{eta}", flush=True)

        # ── Generate both popups in parallel ──
        gen_start = time.time()
        with ThreadPoolExecutor(max_workers=2) as pool:
            f17 = pool.submit(deepseek, PROMPT_V17, f"对话：\n{dialogue}\n\n请生成弹窗：")
            fr5 = pool.submit(deepseek, PROMPT_R5, f"对话：\n{dialogue}\n\n请生成弹窗：")
            popup_v17 = f17.result()
            popup_r5 = fr5.result()
        gen_time = time.time() - gen_start

        # ── Judge both independently (blind) ──
        judge_start = time.time()
        with ThreadPoolExecutor(max_workers=2) as pool:
            j17 = pool.submit(judge_one, dialogue, popup_v17)
            jr5 = pool.submit(judge_one, dialogue, popup_r5)
            score_v17, dims_v17 = j17.result()
            score_r5, dims_r5 = jr5.result()
        judge_time = time.time() - judge_start

        v17_veto = dims_v17.get("veto", "")
        r5_veto = dims_r5.get("veto", "")
        v17_flag = f" [VETO:{v17_veto}]" if v17_veto else ""
        r5_flag = f" [VETO:{r5_veto}]" if r5_veto else ""

        delta = score_r5 - score_v17
        if delta > 0.01: winner = "r5"
        elif delta < -0.01: winner = "v1.7"
        else: winner = "tie"
        wins[winner] += 1

        print(f"  v1.7: {score_v17:.4f}{v17_flag} | r5: {score_r5:.4f}{r5_flag} | Δ={delta:+.4f} → {winner}  (gen {gen_time:.0f}s + judge {judge_time:.0f}s)", flush=True)
        print(f"    v1.7: {dims_v17.get('brief_reason','')[:120]}", flush=True)
        print(f"    r5:   {dims_r5.get('brief_reason','')[:120]}", flush=True)

        all_v17.append(score_v17)
        all_r5.append(score_r5)

        results.append({
            "expert_id": expert_id,
            "case_title": case_title,
            "dialogue": dialogue,
            "dialogue_preview": d_short,
            # Store expert metadata for Phase 2 (but don't look at scores!)
            "_expert_score": item.get("expert_score"),
            "_expert_tone": item.get("expert_tone"),
            "_expert_name": item.get("expert_name"),
            "_reference_popup": item.get("reference_popup", ""),
            "_should_popup": item.get("should_popup"),
            "v17": {
                "popup": popup_v17,
                "score": score_v17,
                "dims": {DIM_LABELS.get(k, k): dims_v17.get(k) for k in DIM_LABELS},
                "veto": v17_veto,
                "brief_reason": dims_v17.get("brief_reason", ""),
            },
            "r5": {
                "popup": popup_r5,
                "score": score_r5,
                "dims": {DIM_LABELS.get(k, k): dims_r5.get(k) for k in DIM_LABELS},
                "veto": r5_veto,
                "brief_reason": dims_r5.get("brief_reason", ""),
            },
            "delta": delta,
            "winner": winner,
        })

        time.sleep(0.3)

    # ── Summary ──
    total_time = time.time() - t_start
    n = len(results)
    m17 = mean(all_v17) if all_v17 else 0
    mr5 = mean(all_r5) if all_r5 else 0
    s17 = stdev(all_v17) if len(all_v17) > 1 else 0
    sr5 = stdev(all_r5) if len(all_r5) > 1 else 0
    vetos_17 = sum(1 for r in results if r["v17"]["veto"])
    vetos_r5 = sum(1 for r in results if r["r5"]["veto"])
    delta_mean = mr5 - m17

    # Per-dimension aggregation
    dim_agg = {"v1.7": {}, "r5": {}}
    for version_key, result_key in [("v1.7", "v17"), ("r5", "r5")]:
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

    # Print summary
    print(f"\n{'=' * 80}")
    print(f"📊 Phase 1 · 盲测结果")
    print(f"  总耗时: {total_time/60:.1f}m ({total_time/n:.0f}s/sample)")
    print(f"{'=' * 80}")

    print(f"\n  ┌─ 总体对比 ─{'─'*45}")
    print(f"  │ {'':12} {'v1.7 (基线)':<22} {'r5 (实验)':<22}")
    print(f"  │ {'─'*12} {'─'*22} {'─'*22}")
    print(f"  │ {'均分':12} {m17:.4f} ± {s17:.4f}     {mr5:.4f} ± {sr5:.4f}")
    print(f"  │ {'Veto次数':12} {vetos_17:<22} {vetos_r5:<22}")
    print(f"  │ {'胜/平/负':12} {wins['v1.7']}胜 {wins['tie']}平 {wins['r5']}负")
    print(f"  │ {'Δ (r5-v1.7)':12} {delta_mean:+.4f}")
    print(f"  └{'─'*55}")

    print(f"\n  ┌─ 维度对比 ─{'─'*45}")
    print(f"  │ {'维度':<10} {'v1.7':<10} {'r5':<10} {'Δ':>8}  {'趋势':>6}")
    print(f"  │ {'─'*10} {'─'*10} {'─'*10} {'─'*8}  {'─'*6}")
    for dk in DIM_LABELS.values():
        m1 = dim_agg["v1.7"][dk]["mean"]
        m2 = dim_agg["r5"][dk]["mean"]
        d = m2 - m1
        trend = "📈 r5↑" if d > 0.02 else ("📉 r5↓" if d < -0.02 else "≈ 持平")
        print(f"  │ {dk:<8}  {m1:.4f}    {m2:.4f}    {d:+.4f}   {trend}")
    print(f"  └{'─'*55}")

    # Score distribution
    print(f"\n  ┌─ 分数分布 ─{'─'*45}")
    for label, scores in [("v1.7", all_v17), ("r5", all_r5)]:
        buckets = {"[0.0-0.2)":0,"[0.2-0.4)":0,"[0.4-0.6)":0,"[0.6-0.8)":0,"[0.8-1.0)":0,"[1.0]":0}
        for s in scores:
            if s >= 1.0: buckets["[1.0]"] += 1
            elif s >= 0.8: buckets["[0.8-1.0)"] += 1
            elif s >= 0.6: buckets["[0.6-0.8)"] += 1
            elif s >= 0.4: buckets["[0.4-0.6)"] += 1
            elif s >= 0.2: buckets["[0.2-0.4)"] += 1
            else: buckets["[0.0-0.2)"] += 1
        bars = "  ".join(f"{k}: {'█'*v}{v}" for k,v in buckets.items())
        print(f"  │ {label}: {bars}")

    # Per-dialogue table
    print(f"\n  ┌─ 逐条对比 ─{'─'*45}")
    print(f"  │ {'#':<3} {'案例':<22} {'v1.7':>6} {'r5':>6} {'Δ':>8}  {'胜者'}")
    print(f"  │ {'─'*3} {'─'*22} {'─'*6} {'─'*6} {'─'*8}  {'─'*4}")
    for i, r in enumerate(results):
        title = r["case_title"][:20] if r["case_title"] else r["dialogue_preview"][:20]
        v17s = f"{r['v17']['score']:.4f}"
        r5s = f"{r['r5']['score']:.4f}"
        d = f"{r['delta']:+.4f}"
        w = r["winner"]
        veto_mark = ""
        if r["v17"]["veto"]: veto_mark += " ⚠v17"
        if r["r5"]["veto"]: veto_mark += " ⚠r5"
        print(f"  │ {i+1:<3} {title:<22} {v17s:>6} {r5s:>6} {d:>8}  {w}{veto_mark}")
    print(f"  └{'─'*55}")

    # Save
    out = {
        "phase": 1,
        "config": {
            "task_model": TASK_MODEL,
            "judge_model": CL_MODEL,
            "judge_version": "v2.0 (5-dim + veto + few-shot, thinking=disabled)",
            "test_source": "expert_dataset.json (独立样本, 排除merged_train重叠)",
            "n_dialogues": n,
            "seed": seed,
            "v17_chars": len(PROMPT_V17),
            "r5_chars": len(PROMPT_R5),
            "total_time_s": round(total_time, 1),
        },
        "summary": {
            "v17_mean": m17, "v17_std": s17,
            "r5_mean": mr5, "r5_std": sr5,
            "delta": delta_mean,
            "v17_vetos": vetos_17, "r5_vetos": vetos_r5,
            "wins": wins,
            "dim_comparison": dim_agg,
        },
        "results": results,
    }

    out_path = os.path.join(OUT_DIR, "h2h_r5_v17_expert_blind.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Phase 1 盲测结果 → {out_path}")
    print(f"   v1.7={m17:.4f}  r5={mr5:.4f}  Δ={delta_mean:+.4f}")
    return out_path


# ═══════════════════════════════════════════════════════════
# Phase 2: Expert Comparison
# ═══════════════════════════════════════════════════════════

def compare_to_expert(result_path):
    """Compare blind H2H results against expert ground truth."""
    data = json.load(open(result_path, "r", encoding="utf-8"))
    results = data["results"]

    print(f"\n{'=' * 80}")
    print(f"Phase 2 · 校标对标")
    print(f"  数据: {result_path}")
    print(f"  样本: {len(results)} 条")
    print(f"{'=' * 80}")

    # Expert scores are 1-10, normalize to 0-1
    # Claude judge scores are already 0-1

    comparisons = []
    for r in results:
        exp_score = r.get("_expert_score")
        if exp_score is None:
            continue

        exp_norm = exp_score / 10.0  # normalize 1-10 → 0-1
        v17_score = r["v17"]["score"]
        r5_score = r["r5"]["score"]

        # Which version is closer to expert?
        v17_err = abs(v17_score - exp_norm)
        r5_err = abs(r5_score - exp_norm)
        closer = "v1.7" if v17_err < r5_err else ("r5" if r5_err < v17_err else "tie")

        # Expert agreement: did Claude judge agree with expert on veto?
        exp_should_popup = r.get("_should_popup")
        v17_vetoed = bool(r["v17"]["veto"])
        r5_vetoed = bool(r["r5"]["veto"])

        comparisons.append({
            "expert_id": r["expert_id"],
            "case_title": r.get("case_title", ""),
            "expert_score": exp_score,
            "expert_norm": exp_norm,
            "v17_score": v17_score,
            "r5_score": r5_score,
            "v17_error": v17_err,
            "r5_error": r5_err,
            "closer_to_expert": closer,
            "expert_tone": r.get("_expert_tone", ""),
            "expert_should_popup": exp_should_popup,
            "v17_vetoed": v17_vetoed,
            "r5_vetoed": r5_vetoed,
        })

    # ── Aggregate ──
    v17_errors = [c["v17_error"] for c in comparisons]
    r5_errors = [c["r5_error"] for c in comparisons]
    v17_mae = mean(v17_errors)
    r5_mae = mean(r5_errors)

    closer_counts = {"v1.7": 0, "r5": 0, "tie": 0}
    for c in comparisons:
        closer_counts[c["closer_to_expert"]] += 1

    # Correlation: Claude judge score vs expert score
    v17_scores = [c["v17_score"] for c in comparisons]
    r5_scores = [c["r5_score"] for c in comparisons]
    exp_scores = [c["expert_norm"] for c in comparisons]

    def pearson_r(xs, ys):
        n = len(xs)
        if n < 3: return 0
        mx, my = mean(xs), mean(ys)
        num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
        dx = sum((x-mx)**2 for x in xs) ** 0.5
        dy = sum((y-my)**2 for y in ys) ** 0.5
        return num/(dx*dy) if dx*dy > 0 else 0

    r_v17 = pearson_r(v17_scores, exp_scores)
    r_r5 = pearson_r(r5_scores, exp_scores)

    # ── Display ──
    print(f"\n  ┌─ 校标对齐度 ─{'─'*45}")
    print(f"  │ {'':16} {'v1.7':<14} {'r5':<14}")
    print(f"  │ {'─'*16} {'─'*14} {'─'*14}")
    print(f"  │ {'MAE (vs 专家)':16} {v17_mae:.4f}        {r5_mae:.4f}")
    print(f"  │ {'更接近专家次数':16} {closer_counts['v1.7']:<14} {closer_counts['r5']:<14}")
    print(f"  │ {'与专家相关性 r':16} {r_v17:.4f}        {r_r5:.4f}")
    print(f"  └{'─'*55}")

    # Breakdown by expert tone
    print(f"\n  ┌─ 按专家语气拆分 ─{'─'*45}")
    tones = {}
    for c in comparisons:
        t = c["expert_tone"] or "未标注"
        if t not in tones: tones[t] = {"v17_err": [], "r5_err": [], "n": 0}
        tones[t]["v17_err"].append(c["v17_error"])
        tones[t]["r5_err"].append(c["r5_error"])
        tones[t]["n"] += 1

    print(f"  │ {'语气':<10} {'n':<4} {'v1.7 MAE':<12} {'r5 MAE':<12} {'更准':<6}")
    print(f"  │ {'─'*10} {'─'*4} {'─'*12} {'─'*12} {'─'*6}")
    for tone, stats in sorted(tones.items()):
        vmae = mean(stats["v17_err"])
        rmae = mean(stats["r5_err"])
        better = "r5" if rmae < vmae else ("v1.7" if vmae < rmae else "tie")
        print(f"  │ {tone:<10} {stats['n']:<4} {vmae:.4f}       {rmae:.4f}       {better}")

    # Top disagreements: where Claude judge and expert diverge most
    print(f"\n  ┌─ 最大分歧 (|v17-专家| + |r5-专家| 最大) ─{'─'*30}")
    comparisons_sorted = sorted(comparisons, key=lambda c: c["v17_error"] + c["r5_error"], reverse=True)
    for c in comparisons_sorted[:5]:
        print(f"  │ {c['case_title'][:30]}")
        print(f"  │   专家={c['expert_score']}/10  v1.7={c['v17_score']:.4f}  r5={c['r5_score']:.4f}  "
              f"更近={c['closer_to_expert']} (Δv17={c['v17_error']:.3f} Δr5={c['r5_error']:.3f})")

    # ── Per-item expert comparison table ──
    print(f"\n  ┌─ 逐条校标对比 ─{'─'*45}")
    print(f"  │ {'案例':<22} {'专家':>5} {'v1.7':>6} {'r5':>6} {'Δv17':>7} {'Δr5':>7} {'更近':>5}")
    print(f"  │ {'─'*22} {'─'*5} {'─'*6} {'─'*6} {'─'*7} {'─'*7} {'─'*5}")
    for c in comparisons:
        title = c["case_title"][:20]
        print(f"  │ {title:<22} {c['expert_score']:>5} {c['v17_score']:.4f} {c['r5_score']:.4f} "
              f"{c['v17_error']:+.4f} {c['r5_error']:+.4f} {c['closer_to_expert']:>5}")
    print(f"  └{'─'*55}")

    # Save
    out = {
        "phase": 2,
        "source": result_path,
        "summary": {
            "v17_mae": v17_mae, "r5_mae": r5_mae,
            "v17_expert_corr": r_v17, "r5_expert_corr": r_r5,
            "closer_to_expert": closer_counts,
            "by_tone": {t: {"n": s["n"], "v17_mae": mean(s["v17_err"]), "r5_mae": mean(s["r5_err"])}
                        for t, s in tones.items()},
        },
        "comparisons": comparisons,
    }

    out_path = result_path.replace(".json", "_expert_compare.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Phase 2 对标结果 → {out_path}")

    # ── Final verdict ──
    print(f"\n{'=' * 60}")
    print(f"🏆 最终判决")
    print(f"{'=' * 60}")
    print(f"  Claude Judge 盲评: v1.7={data['summary']['v17_mean']:.4f}  r5={data['summary']['r5_mean']:.4f}")
    print(f"  与专家一致性 MAE:   v1.7={v17_mae:.4f}  r5={r5_mae:.4f}")
    print(f"  与专家相关性 r:      v1.7={r_v17:.4f}  r5={r_r5:.4f}")
    print(f"  更接近专家次数:     v1.7×{closer_counts['v1.7']}  r5×{closer_counts['r5']}  tie×{closer_counts['tie']}")

    return out


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="r5 vs v1.7 H2H · 盲测 + 校标对标")
    parser.add_argument("-n", type=int, default=30, help="测试对话数量 (default: 30)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (default: 42)")
    parser.add_argument("--compare", nargs="?", const="AUTO", metavar="RESULT_PATH",
                        help="运行校标对标。不带参数=盲测后自动对标；带路径=仅对标已有结果")
    args = parser.parse_args()

    if args.compare == "AUTO":
        # Phase 1 + Phase 2
        result_path = run_blind_h2h(n=args.n, seed=args.seed)
        compare_to_expert(result_path)
    elif args.compare:
        # Phase 2 only
        compare_to_expert(args.compare)
    else:
        # Phase 1 only
        run_blind_h2h(n=args.n, seed=args.seed)
