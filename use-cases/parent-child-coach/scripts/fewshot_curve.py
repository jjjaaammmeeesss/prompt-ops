"""
Few-shot 边际收益曲线 — 并行版，两阶段二分逼近
Phase 1: [0, 8, 16, 24] → Phase 2: zoom in
"""
import json, os, time, re, requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)

DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_KEY:
    env_file = os.path.join(BASE_DIR, ".env")
    if os.path.exists(env_file):
        for line in open(env_file):
            if line.startswith("DEEPSEEK_API_KEY="):
                DEEPSEEK_KEY = line.split("=", 1)[1].strip()
                break
if not DEEPSEEK_KEY:
    raise RuntimeError("DEEPSEEK_API_KEY env var required")
TASK_MODEL = "deepseek-v4-pro"

CLAUDE_URL = "https://s.lconai.com/v1/messages"
CLAUDE_KEY = "CLAUDE_API_KEY_PLACEHOLDER"
CLAUDE_MODEL = "claude-opus-4-8"

v21 = open("system_prompt_v2.1.txt", "r", encoding="utf-8").read()

# === Build pool ===
mipro = json.load(open("results/config_20260714_120510.json", "r", encoding="utf-8"))
mipro_fs = mipro["few_shots"]
expert = json.load(open("data/expert_dataset.json", "r", encoding="utf-8"))

expert_pool = []
for e in expert:
    ref = e.get("reference_popup", "")
    dia = e.get("dialogue", "")
    if ref and len(ref.strip()) > 30 and dia and len(dia.strip()) > 50:
        ref = re.sub(r'\*\*（请专家手写弹窗正文）\*\*[：:]\s*', '', ref)
        ref = re.sub(r'^>\s*', '', ref).strip()
        if len(ref) > 20:
            expert_pool.append({"question": dia.strip(), "answer": ref})

dataset = json.load(open("data/dataset_merged_train.json", "r", encoding="utf-8"))
test_indices = np.linspace(0, len(dataset) - 1, 12, dtype=int)
test_set = [dataset[i] for i in test_indices]
test_dialogues = {s["question"][:100] for s in test_set}

mipro_dialogues = {fs["question"][:80] for fs in mipro_fs}
clean_pool = [e for e in expert_pool
              if e["question"][:100] not in test_dialogues
              and e["question"][:80] not in mipro_dialogues]

fewshot_pool = list(mipro_fs)
for ex in clean_pool:
    if len(fewshot_pool) >= 24:
        break
    fewshot_pool.append(ex)

print(f"Pool: {len(fewshot_pool)} | Test: {len(test_set)}")

# === Judge prompt ===
JUDGE_PROMPT = """你是亲子沟通教练评估专家。请从以下7个维度对弹窗评分(1-5整数):

1. 发心承认(0.20): 是否先看见家长的发心和难处?
2. 洞察准确性(0.20): 是否基于对话具体行为命中痛点?
3. 模式揭示(0.10): 是否把单次事件连成反复模式?
4. 邀请感(0.10): 是否用邀请/试探语气而非宣告?
5. 建议可操作性(0.15): 建议是否具体可执行?(纯诊断标N/A)
6. 措辞自然度(0.15): 是否口语化、不爹味?
7. 专一度(0.10): 是否聚焦一个主要矛盾讲透?

对话:
{dialogue}

弹窗:
{popup}

请输出JSON(只输出JSON):
{{"acknowledgment":1-5,"insight_accuracy":1-5,"pattern_revelation":1-5,"invitational_tone":1-5,"actionability":"1-5或N/A","naturalness":1-5,"focus":1-5}}"""

WEIGHTS = [("acknowledgment",0.20),("insight_accuracy",0.20),("pattern_revelation",0.10),
           ("invitational_tone",0.10),("actionability",0.15),("naturalness",0.15),("focus",0.10)]


def build_prompt(k):
    if k == 0:
        return v21
    examples = fewshot_pool[:k]
    fs_block = "\n\n---\n\n## 七、参考示范\n\n以下是系统在类似对话中的标准弹窗，请保持同样的诊断深度和语气：\n\n"
    for i, ex in enumerate(examples):
        fs_block += f"### 示范 {i+1}\n\n**对话：**\n{ex['question'].strip()}\n\n**弹窗：**\n{ex['answer'].strip()}\n\n"
    return v21 + fs_block


def generate_one(args):
    """(dialogue, system_prompt) → popup text"""
    dialogue, system_prompt = args
    headers = {"Authorization": f"Bearer {DEEPSEEK_KEY}", "Content-Type": "application/json"}
    payload = {"model": TASK_MODEL, "max_tokens": 800, "temperature": 0.7,
               "messages": [{"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"对话：\n{dialogue}\n\n请生成弹窗："}]}
    for attempt in range(3):
        try:
            resp = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except:
            if attempt < 2:
                time.sleep(2)
    return "[ERROR]"


def judge_one(args):
    """(dialogue, popup) → (score, dims_dict)"""
    dialogue, popup = args
    prompt = JUDGE_PROMPT.format(dialogue=dialogue, popup=popup)
    headers = {"x-api-key": CLAUDE_KEY, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
    payload = {"model": CLAUDE_MODEL, "max_tokens": 512, "temperature": 0.0,
               "system": "你是严格的评估专家，只输出JSON。",
               "messages": [{"role": "user", "content": prompt}]}
    for attempt in range(3):
        try:
            resp = requests.post(CLAUDE_URL, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            for block in data.get("content", []):
                if isinstance(block, dict) and block.get("type") == "text":
                    m = re.search(r'\{.*\}', block["text"], re.DOTALL)
                    if m:
                        scores = json.loads(m.group(0))
                        # Compute weighted score
                        ws, tw = 0.0, 0.0
                        dims = {}
                        for dim, w in WEIGHTS:
                            v = scores.get(dim)
                            if v == "N/A" or v is None:
                                if dim == "actionability":
                                    dims[dim] = "N/A"
                                    continue
                                continue
                            if isinstance(v, (int, float)) and 1 <= v <= 5:
                                dims[dim] = v
                                ws += ((v - 1) / 4) * w
                                tw += w
                        score = ws / tw if tw > 0 else 0.0
                        return (score, dims)
        except:
            if attempt < 2:
                time.sleep(2)
    return (None, {})


def evaluate_k(k):
    """Evaluate one K value with parallel API calls."""
    prompt = build_prompt(k)
    dialogues = [ex["question"] for ex in test_set]

    # Phase A: Generate all 12 popups in parallel
    gen_tasks = [(d, prompt) for d in dialogues]
    popups = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(generate_one, t): i for i, t in enumerate(gen_tasks)}
        results = [None] * len(gen_tasks)
        for f in as_completed(futures):
            idx = futures[f]
            results[idx] = f.result()
        popups = results

    # Phase B: Judge all 12 popups in parallel
    judge_tasks = [(dialogues[i], popups[i]) for i in range(len(test_set))]
    scores = []
    dims_list = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(judge_one, t): i for i, t in enumerate(judge_tasks)}
        results = [None] * len(judge_tasks)
        for f in as_completed(futures):
            idx = futures[f]
            results[idx] = f.result()
        for score, dims in results:
            if score is not None:
                scores.append(score)
                dims_list.append(dims)

    mean_s = np.mean(scores) if scores else 0
    std_s = np.std(scores) if scores else 0

    dim_avgs = {}
    for dim, _ in WEIGHTS:
        vals = [d[dim] for d in dims_list if d.get(dim) and d[dim] != "N/A"]
        dim_avgs[dim] = round(float(np.mean(vals)), 2) if vals else 0

    n_ok = len(scores)
    print(f"  K={k}: {mean_s:.4f} ± {std_s:.4f} | {n_ok}/{len(test_set)} ok | {len(prompt)} chars")
    return {"k": k, "mean": mean_s, "std": std_s, "prompt_len": len(prompt),
            "dim_avgs": dim_avgs, "n_ok": n_ok}


# ============================================================
# PHASE 1: Coarse [0, 8, 16, 24]
# ============================================================
print("\n" + "="*60)
print("PHASE 1 — 粗颗粒度: K = 0, 8, 16, 24")
print("="*60)

phase1 = {}
for k in [0, 8, 16, 24]:
    t0 = time.time()
    phase1[k] = evaluate_k(k)
    print(f"  (took {time.time()-t0:.0f}s)")

best_k = max(phase1, key=lambda k: phase1[k]["mean"])
print(f"\nPhase 1 best: K={best_k} ({phase1[best_k]['mean']:.4f})")

# ============================================================
# PHASE 2: Zoom in
# ============================================================
if best_k == 0:
    phase2_k = [3, 5]
elif best_k == 8:
    phase2_k = [5, 12]
elif best_k == 16:
    phase2_k = [12, 20]
else:
    phase2_k = [20]

print(f"\n{'='*60}")
print(f"PHASE 2 — 细颗粒度: K = {phase2_k}")
print("="*60)

phase2 = {}
for k in phase2_k:
    if k <= len(fewshot_pool):
        t0 = time.time()
        phase2[k] = evaluate_k(k)
        print(f"  (took {time.time()-t0:.0f}s)")

# ============================================================
# SUMMARY
# ============================================================
all_k = sorted(set(list(phase1.keys()) + list(phase2.keys())))
baseline = phase1[0]["mean"]

print(f"\n{'='*60}")
print("FEW-SHOT 边际收益曲线")
print("="*60)
print(f"{'K':<6} {'Score':<10} {'±Std':<10} {'Δ vs 0':<10} {'Chars':<8} {'Phase'}")
print("-"*55)
for k in all_k:
    r = phase1.get(k) or phase2.get(k)
    ph = "P1" if k in phase1 else "P2"
    print(f"{k:<6} {r['mean']:.4f}     ±{r['std']:.4f}   {r['mean']-baseline:+.4f}     {r['prompt_len']:<8} {ph}")

all_results = {**phase1, **phase2}
overall_best_k = max(all_results, key=lambda k: all_results[k]["mean"])
best = all_results[overall_best_k]

# Determine saturation point: first K where adding more doesn't beat by > 0.01
sorted_ks = sorted(all_results.keys())
saturation_k = sorted_ks[0]
for i in range(1, len(sorted_ks)):
    if all_results[sorted_ks[i]]["mean"] - all_results[sorted_ks[i-1]]["mean"] <= 0.01:
        saturation_k = sorted_ks[i-1]
        break
    saturation_k = sorted_ks[i]

print(f"\n最优 K = {overall_best_k} (score={best['mean']:.4f}, Δ={best['mean']-baseline:+.4f})")
print(f"饱和点 ≈ K = {saturation_k} (边际增益耗尽)")

# Save
out = {
    "config": {"task_model": TASK_MODEL, "judge_model": CLAUDE_MODEL, "n_test": len(test_set)},
    "phase1": {str(k): phase1[k] for k in phase1},
    "phase2": {str(k): phase2[k] for k in phase2},
    "best_k": overall_best_k,
    "best_score": best["mean"],
    "baseline_score": baseline,
    "saturation_k": saturation_k,
}
json.dump(out, open(os.path.join(BASE_DIR, "results", "fewshot_curve.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2, default=str)
print(f"Saved to: results/fewshot_curve.json")
