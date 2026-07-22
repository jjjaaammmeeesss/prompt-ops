"""
v2.0 (DeepSeek-judge优化) vs v2.1 (Claude-judge优化) 头对头对比
任务模型: DeepSeek v4 pro | Judge: Claude via 智创聚合
"""
import json, os, sys, time, re, requests
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)

# === API Config ===
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_KEY:
    raise RuntimeError("DEEPSEEK_API_KEY env var required")
TASK_MODEL = "deepseek-v4-pro"

CLAUDE_URL = "https://s.lconai.com/v1/messages"
CLAUDE_KEY = "CLAUDE_API_KEY_PLACEHOLDER"
CLAUDE_MODEL = "claude-opus-4-8"

# === Load prompts & test set ===
prompt_v20 = open("system_prompt_v2.0.txt", "r", encoding="utf-8").read()
prompt_v21 = open("system_prompt_v2.1.txt", "r", encoding="utf-8").read()

dataset = json.load(open("data/dataset_merged_train.json", "r", encoding="utf-8"))
indices = np.linspace(0, len(dataset) - 1, 12, dtype=int)
test_set = [dataset[i] for i in indices]

print(f"v2.0: {len(prompt_v20)} chars")
print(f"v2.1: {len(prompt_v21)} chars")
print(f"Test set: {len(test_set)} samples\n")

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

弹窗A:
{popup_a}

弹窗B:
{popup_b}

请输出JSON(只输出JSON):
{{"A": {{"acknowledgment":1-5,"insight_accuracy":1-5,"pattern_revelation":1-5,"invitational_tone":1-5,"actionability":"1-5或N/A","naturalness":1-5,"focus":1-5}}, "B": {{...(同上)}}, "winner": "A"或"B"或"tie", "reason": "简短理由"}}"""

WEIGHTS = [("acknowledgment",0.20),("insight_accuracy",0.20),("pattern_revelation",0.10),
           ("invitational_tone",0.10),("actionability",0.15),("naturalness",0.15),("focus",0.10)]


def generate(system_prompt, dialogue):
    headers = {"Authorization": f"Bearer {DEEPSEEK_KEY}", "Content-Type": "application/json"}
    payload = {"model": TASK_MODEL, "max_tokens": 800, "temperature": 0.7,
               "messages": [{"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"对话：\n{dialogue}\n\n请生成弹窗："}]}
    for attempt in range(3):
        try:
            resp = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < 2:
                time.sleep(3)
            else:
                return f"[ERROR: {e}]"


def judge_pair(dialogue, popup_a, popup_b):
    prompt = JUDGE_PROMPT.format(dialogue=dialogue, popup_a=popup_a, popup_b=popup_b)
    headers = {"x-api-key": CLAUDE_KEY, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
    payload = {"model": CLAUDE_MODEL, "max_tokens": 1024, "temperature": 0.0,
               "system": "你是严格的评估专家，只输出JSON。",
               "messages": [{"role": "user", "content": prompt}]}
    for attempt in range(3):
        try:
            resp = requests.post(CLAUDE_URL, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            content = data.get("content", [])
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    raw = block["text"]
                    # Parse JSON
                    m = re.search(r'\{.*\}', raw, re.DOTALL)
                    if m:
                        return json.loads(m.group(0))
            raise ValueError(f"No text in response: {str(content)[:200]}")
        except Exception as e:
            if attempt < 2:
                time.sleep(3)
            else:
                return {"error": str(e)}


def compute_score(scores):
    if not scores or "error" in str(scores):
        return 0.0
    ws, tw = 0.0, 0.0
    for dim, w in WEIGHTS:
        v = scores.get(dim)
        if v == "N/A" or v is None:
            if dim == "actionability":
                continue
            continue
        if isinstance(v, (int, float)) and 1 <= v <= 5:
            ws += ((v - 1) / 4) * w
            tw += w
    return ws / tw if tw > 0 else 0.0


# === Run H2H ===
results = []
wins = {"v20": 0, "v21": 0, "tie": 0}
all_scores = {"v20": [], "v21": []}

for i, example in enumerate(test_set):
    dialogue = example["question"]
    d_short = dialogue[:60].replace("\n", " ")
    print(f"[{i+1}/{len(test_set)}] {d_short}...")

    # Generate
    popup_a = generate(prompt_v20, dialogue)  # A = v2.0
    time.sleep(0.5)
    popup_b = generate(prompt_v21, dialogue)  # B = v2.1
    time.sleep(0.5)

    # Judge
    verdict = judge_pair(dialogue, popup_a, popup_b)
    time.sleep(1)

    if "error" in verdict:
        print(f"  Judge error: {verdict['error']}")
        continue

    score_a = compute_score(verdict.get("A", {}))
    score_b = compute_score(verdict.get("B", {}))
    winner = verdict.get("winner", "tie")

    all_scores["v20"].append(score_a)
    all_scores["v21"].append(score_b)
    wins["v20" if winner == "A" else "v21" if winner == "B" else "tie"] += 1

    results.append({"dialogue": dialogue[:200], "v20_popup": popup_a, "v21_popup": popup_b,
                    "v20_score": score_a, "v21_score": score_b, "winner": winner,
                    "reason": verdict.get("reason", "")})
    print(f"  v2.0={score_a:.4f} | v2.1={score_b:.4f} | winner={winner} | {verdict.get('reason','')[:60]}")

# === Summary ===
n = len(results)
v20_mean = np.mean(all_scores["v20"]) if all_scores["v20"] else 0
v21_mean = np.mean(all_scores["v21"]) if all_scores["v21"] else 0
print(f"\n{'='*60}")
print(f"v2.0 (DeepSeek-judge): {v20_mean:.4f} ± {np.std(all_scores['v20']):.4f}")
print(f"v2.1 (Claude-judge):   {v21_mean:.4f} ± {np.std(all_scores['v21']):.4f}")
print(f"Δ: {v21_mean - v20_mean:+.4f}")
print(f"Wins: v2.0={wins['v20']}, v2.1={wins['v21']}, tie={wins['tie']}")

# Per-dimension analysis
dim_avgs = {"v20": {}, "v21": {}}
for dim, _ in WEIGHTS:
    v20_vals = [r["v20_score"] for r in results if dim in r.get("v20_score", {})]
    dim_avgs["v20"][dim] = np.mean([1.0]) if not v20_vals else np.mean(v20_vals)

out_path = os.path.join(BASE_DIR, "results", "h2h_v20_v21.json")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
json.dump({"config": {"task_model": TASK_MODEL, "judge_model": CLAUDE_MODEL, "n": n},
           "summary": {"v20_mean": v20_mean, "v21_mean": v21_mean, "delta": v21_mean - v20_mean,
                        "v20_wins": wins["v20"], "v21_wins": wins["v21"], "ties": wins["tie"]},
           "results": results}, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"\nSaved to: {out_path}")
