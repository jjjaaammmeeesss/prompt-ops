"""v1.7 vs v2.2.1 vs v2.3 三方盲测 · N=30"""
import json, os, re, time, random, requests
from concurrent.futures import ThreadPoolExecutor
from statistics import mean, stdev

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "results", "auto_research_judge_v2")
os.makedirs(OUT, exist_ok=True)

# ── Prompts ──
v17  = open(os.path.join(BASE, "prompts_archive", "system_prompt_backup_v17.txt"), "r", encoding="utf-8").read().strip()
v221 = open(os.path.join(BASE, "prompts_archive", "system_prompt_v2.2.1.txt"), "r", encoding="utf-8").read().strip()
v23  = open(os.path.join(BASE, "prompts_archive", "system_prompt_v2.3.txt"), "r", encoding="utf-8").read().strip()

# ── API ──
DS_KEY = None
for line in open(os.path.join(BASE, ".env"), encoding="utf-8"):
    if line.startswith("DEEPSEEK_API_KEY="):
        DS_KEY = line.split("=", 1)[1].strip(); break
DS_URL = "https://api.deepseek.com/v1/chat/completions"
CL_URL = "https://s.lconai.com/v1/messages"
CL_KEY = "CLAUDE_API_KEY_PLACEHOLDER"

# ── Dataset ──
data = json.load(open(os.path.join(BASE, "data", "expert_dataset.json"), "r", encoding="utf-8"))
valid = [e for e in data if len(e.get("dialogue", "").strip()) > 50]
rng = random.Random(42)
cases = rng.sample(valid, 30)

# ── Judge ──
JUDGE_HEADER = """你是亲子沟通弹窗评估专家。给弹窗从五个维度打分（1-5整数）。

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
"""
JUDGE_FOOTER = """

请严格输出JSON（不要markdown包裹，不要额外文字）：
{"veto":null,"being_seen":1-5,"dialogue_fidelity":1-5,"core_insight":1-5,"natural_language":1-5,"warmth":1-5,"brief_reason":"一句话中文理由"}"""

DIM_WEIGHTS = [("being_seen", 0.25), ("dialogue_fidelity", 0.20), ("core_insight", 0.20),
               ("natural_language", 0.20), ("warmth", 0.15)]
DIM_LABELS = {"being_seen": "看见感", "dialogue_fidelity": "对话忠实度",
              "core_insight": "命中核心", "natural_language": "人话感", "warmth": "温度"}


def gen(sys_prompt, dialogue):
    headers = {"Authorization": f"Bearer {DS_KEY}", "Content-Type": "application/json"}
    for attempt in range(4):
        try:
            resp = requests.post(DS_URL, headers=headers,
                json={"model": "deepseek-v4-pro", "max_tokens": 800, "temperature": 0.7,
                      "messages": [{"role": "system", "content": sys_prompt},
                                   {"role": "user", "content": f"对话：\n{dialogue}\n\n请生成弹窗："}]},
                timeout=(30, 120))
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            if len(text) >= 20:
                return text
        except Exception:
            time.sleep(3)
    return "[GEN_FAILED]"


def judge_blind(dialogue, popup):
    prompt = JUDGE_HEADER + dialogue + "\n\n### AI弹窗\n" + popup + JUDGE_FOOTER
    headers = {"x-api-key": CL_KEY, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
    for attempt in range(3):
        try:
            resp = requests.post(CL_URL, headers=headers,
                json={"model": "claude-opus-4-8", "max_tokens": 1024, "temperature": 0.0,
                      "thinking": {"type": "disabled"},
                      "system": "你是严格的亲子沟通弹窗评估专家。只输出JSON，不要markdown包裹。",
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=(30, 90))
            resp.raise_for_status()
            for block in resp.json().get("content", []):
                if isinstance(block, dict) and block.get("type") == "text":
                    raw = block["text"].strip()
                    try:
                        return json.loads(raw)
                    except json.JSONDecodeError:
                        m = re.search(r"\{.*\}", raw, re.DOTALL)
                        if m:
                            return json.loads(m.group(0))
                    return {"error": "parse"}
        except Exception:
            if attempt < 2:
                time.sleep(3)
    return {"error": "api_failed"}


def compute(s):
    veto = s.get("veto")
    if veto and str(veto).strip() not in ("null", "none", ""):
        return 0.0, True, str(veto)
    ws, tw = 0.0, 0.0
    for dk, w in DIM_WEIGHTS:
        v = s.get(dk)
        if isinstance(v, (int, float)) and 1 <= v <= 5:
            ws += (v - 1) / 4 * w
            tw += w
    return ws / tw if tw > 0 else 0.0, False, None


# ── Run ──
print(f"N=30 | seed=42")
print(f"v1.7={len(v17)}c | v2.2.1={len(v221)}c | v2.3={len(v23)}c")
print()

results = []
all_s17, all_s22, all_s23 = [], [], []
wins = {"v1.7": 0, "v2.2.1": 0, "v2.3": 0}
veto_count = {"v1.7": 0, "v2.2.1": 0, "v2.3": 0}
word_ok = {"v1.7": 0, "v2.2.1": 0, "v2.3": 0}
t_start = time.time()

for i, case in enumerate(cases):
    dialogue = case["dialogue"]
    eid = case.get("id", f"case_{i}")[:40]
    title = case.get("case_title", "")[:30]

    with ThreadPoolExecutor(max_workers=3) as pool:
        f17 = pool.submit(gen, v17, dialogue)
        f22 = pool.submit(gen, v221, dialogue)
        f23 = pool.submit(gen, v23, dialogue)
        p17 = f17.result(); p22 = f22.result(); p23 = f23.result()

    with ThreadPoolExecutor(max_workers=3) as pool:
        j17 = pool.submit(judge_blind, dialogue, p17)
        j22 = pool.submit(judge_blind, dialogue, p22)
        j23 = pool.submit(judge_blind, dialogue, p23)
        s17r = j17.result(); s22r = j22.result(); s23r = j23.result()

    s17, v17f, v17r = compute(s17r)
    s22, v22f, v22r = compute(s22r)
    s23, v23f, v23r = compute(s23r)

    all_s17.append(s17); all_s22.append(s22); all_s23.append(s23)
    if v17f: veto_count["v1.7"] += 1
    if v22f: veto_count["v2.2.1"] += 1
    if v23f: veto_count["v2.3"] += 1

    # word count check (Chinese chars)
    for label, p in [("v1.7", p17), ("v2.2.1", p22), ("v2.3", p23)]:
        cc = len(re.sub(r'[\s\n\-—]', '', p))  # rough chinese char count
        if cc <= 200:
            word_ok[label] += 1

    ranked = sorted([("v1.7", s17), ("v2.2.1", s22), ("v2.3", s23)], key=lambda x: x[1], reverse=True)
    wins[ranked[0][0]] += 1

    eta = f"ETA {(time.time()-t_start)/max(i,1)*(30-i)/60:.0f}m" if i else ""
    print(f"[{i+1:2d}/30] {title[:25]:<25} | v1.7={s17:.4f} v2.2.1={s22:.4f} v2.3={s23:.4f} | "
          f"1st={ranked[0][0]} veto={'v17' if v17f else ''}{'v221' if v22f else ''}{'v23' if v23f else ''} "
          f"| {eta}", flush=True)

    results.append({
        "case_id": eid, "case_title": title, "dialogue": dialogue,
        "v17": {"popup": p17, "score": s17, "veto": v17r if v17f else None,
                "dims": {DIM_LABELS[k]: s17r.get(k) for k in DIM_LABELS},
                "brief": s17r.get("brief_reason", "")},
        "v221": {"popup": p22, "score": s22, "veto": v22r if v22f else None,
                 "dims": {DIM_LABELS[k]: s22r.get(k) for k in DIM_LABELS},
                 "brief": s22r.get("brief_reason", "")},
        "v23": {"popup": p23, "score": s23, "veto": v23r if v23f else None,
                "dims": {DIM_LABELS[k]: s23r.get(k) for k in DIM_LABELS},
                "brief": s23r.get("brief_reason", "")},
    })
    time.sleep(0.3)

total_t = time.time() - t_start

# ── Summary ──
n = len(results)
print()
print("=" * 80)
print(f"N=30 汇总 · {total_t/60:.1f}min")
print("=" * 80)

print(f"\n{'':15} {'v1.7':>10} {'v2.2.1':>10} {'v2.3':>10}")
print(f"{'─'*15} {'─'*10} {'─'*10} {'─'*10}")
print(f"{'均分':15} {mean(all_s17):10.4f} {mean(all_s22):10.4f} {mean(all_s23):10.4f}")
print(f"{'标准差':15} {stdev(all_s17):10.4f} {stdev(all_s22):10.4f} {stdev(all_s23):10.4f}")
print(f"{'Veto次数':15} {veto_count['v1.7']:>10} {veto_count['v2.2.1']:>10} {veto_count['v2.3']:>10}")
print(f"{'字数合规(≤200)':15} {word_ok['v1.7']:>9}/{n} {word_ok['v2.2.1']:>9}/{n} {word_ok['v2.3']:>9}/{n}")
print(f"{'第1次数':15} {wins['v1.7']:>10} {wins['v2.2.1']:>10} {wins['v2.3']:>10}")

# Dim comparison
print(f"\n{'维度':<12} {'v1.7':<10} {'v2.2.1':<10} {'v2.3':<10}")
print(f"{'─'*12} {'─'*10} {'─'*10} {'─'*10}")
for dk in DIM_LABELS.values():
    vals = {}
    for ver, key in [("v1.7", "v17"), ("v2.2.1", "v221"), ("v2.3", "v23")]:
        vs = []
        for r in results:
            v = r[key]["dims"].get(dk)
            if isinstance(v, (int, float)) and 1 <= v <= 5:
                vs.append((v - 1) / 4)
        vals[ver] = mean(vs) if vs else 0
    print(f"{dk:<12} {vals['v1.7']:<10.4f} {vals['v2.2.1']:<10.4f} {vals['v2.3']:<10.4f}")

# Save
out = {
    "config": {"n": n, "seed": 42, "task_model": "deepseek-v4-pro", "judge_model": "claude-opus-4-8",
               "v17_chars": len(v17), "v221_chars": len(v221), "v23_chars": len(v23)},
    "summary": {
        "v17_mean": mean(all_s17), "v17_std": stdev(all_s17),
        "v221_mean": mean(all_s22), "v221_std": stdev(all_s22),
        "v23_mean": mean(all_s23), "v23_std": stdev(all_s23),
        "vetos": veto_count, "wins": wins, "word_compliance": word_ok,
    },
    "results": results,
}
out_path = os.path.join(OUT, "h2h_v17_v221_v23_n30.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"\nSaved to {out_path}")
