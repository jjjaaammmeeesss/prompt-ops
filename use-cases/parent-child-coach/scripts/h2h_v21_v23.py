"""
v2.1 (baseline) vs v2.3 (K=12 few-shots) 头对头对比
任务模型: DeepSeek v4 pro | Judge: Claude via 智创聚合
"""
import json, os, time, re, requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)

DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_KEY:
    for line in open(".env"):
        if line.startswith("DEEPSEEK_API_KEY="):
            DEEPSEEK_KEY = line.split("=", 1)[1].strip()
            break
TASK_MODEL = "deepseek-v4-pro"

CLAUDE_URL = "https://s.lconai.com/v1/messages"
CLAUDE_KEY = "CLAUDE_API_KEY_PLACEHOLDER"
CLAUDE_MODEL = "claude-opus-4-8"

WEIGHTS = [("acknowledgment",0.20),("insight_accuracy",0.20),("pattern_revelation",0.10),
           ("invitational_tone",0.10),("actionability",0.15),("naturalness",0.15),("focus",0.10)]

JUDGE_PROMPT = """你是亲子沟通教练评估专家。请从以下7个维度对两个弹窗分别评分(1-5整数):

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

v21 = open("system_prompt_v2.1.txt", "r", encoding="utf-8").read()
v23 = open("system_prompt_v2.3.txt", "r", encoding="utf-8").read()
print(f"v2.1: {len(v21)} chars | v2.3: {len(v23)} chars")

dataset = json.load(open("data/dataset_merged_train.json", "r", encoding="utf-8"))
indices = np.linspace(0, len(dataset) - 1, 12, dtype=int)
test_set = [dataset[i] for i in indices]
print(f"Test: {len(test_set)} samples\n")


def generate(sys_prompt, dialogue):
    headers = {"Authorization": f"Bearer {DEEPSEEK_KEY}", "Content-Type": "application/json"}
    payload = {"model": TASK_MODEL, "max_tokens": 800, "temperature": 0.7,
               "messages": [{"role": "system", "content": sys_prompt},
                            {"role": "user", "content": f"对话：\n{dialogue}\n\n请生成弹窗："}]}
    for attempt in range(3):
        try:
            resp = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=90)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except:
            if attempt < 2:
                time.sleep(2)
    return "[ERROR]"


def compute_score(scores):
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


# Run H2H
results = []
wins = {"v21": 0, "v23": 0, "tie": 0}
all_s21, all_s23 = [], []

for i, ex in enumerate(test_set):
    dialogue = ex["question"]
    d_short = dialogue[:60].replace("\n", " ")
    print(f"[{i+1}/{len(test_set)}] {d_short}...")

    # Generate in parallel
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_a = pool.submit(generate, v21, dialogue)
        f_b = pool.submit(generate, v23, dialogue)
        popup_a = f_a.result()
        popup_b = f_b.result()

    # Judge
    prompt = JUDGE_PROMPT.format(dialogue=dialogue, popup_a=popup_a, popup_b=popup_b)
    headers = {"x-api-key": CLAUDE_KEY, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
    payload = {"model": CLAUDE_MODEL, "max_tokens": 1024, "temperature": 0.0,
               "system": "你是严格的评估专家，只输出JSON。",
               "messages": [{"role": "user", "content": prompt}]}
    verdict = {}
    for attempt in range(3):
        try:
            resp = requests.post(CLAUDE_URL, headers=headers, json=payload, timeout=90)
            resp.raise_for_status()
            data = resp.json()
            for block in data.get("content", []):
                if isinstance(block, dict) and block.get("type") == "text":
                    m = re.search(r'\{.*\}', block["text"], re.DOTALL)
                    if m:
                        verdict = json.loads(m.group(0))
            if verdict:
                break
        except:
            if attempt < 2:
                time.sleep(2)

    if "error" in verdict or not verdict:
        print(f"  Judge error: {verdict.get('error', 'no response')}")
        continue

    sa = compute_score(verdict.get("A", {}))
    sb = compute_score(verdict.get("B", {}))
    winner = verdict.get("winner", "tie")
    wins["v21" if winner == "A" else "v23" if winner == "B" else "tie"] += 1
    all_s21.append(sa)
    all_s23.append(sb)
    results.append({"dialogue": dialogue[:200], "v21_popup": popup_a, "v23_popup": popup_b,
                    "v21_score": sa, "v23_score": sb, "winner": winner,
                    "reason": verdict.get("reason", "")})
    print(f"  v2.1={sa:.4f} | v2.3={sb:.4f} | winner={winner} | {verdict.get('reason','')[:80]}")

# Summary
n = len(results)
m21 = np.mean(all_s21) if all_s21 else 0
m23 = np.mean(all_s23) if all_s23 else 0
print(f"\n{'='*60}")
print(f"v2.1 (baseline): {m21:.4f} ± {np.std(all_s21):.4f}")
print(f"v2.3 (K=12):     {m23:.4f} ± {np.std(all_s23):.4f}")
print(f"Δ: {m23 - m21:+.4f}")
print(f"Wins: v2.1={wins['v21']}, v2.3={wins['v23']}, tie={wins['tie']}")

# Per-dimension analysis
for version, scores_list in [("v2.1", all_s21), ("v2.3", all_s23)]:
    print(f"\n{version} per-dimension (from judge raw scores):")
    dim_map = {}
    for r in results:
        key = "A" if version == "v2.1" else "B"
        # We don't have raw dims from judge in this version — we only have composite scores
        pass
    print("  (composite scores only in this H2H format)")

out = {"config": {"task_model": TASK_MODEL, "judge_model": CLAUDE_MODEL, "n": n},
       "summary": {"v21_mean": m21, "v23_mean": m23, "delta": m23 - m21,
                    "v21_wins": wins["v21"], "v23_wins": wins["v23"], "ties": wins["tie"]},
       "results": results}
json.dump(out, open("results/h2h_v21_v23.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2, default=str)
print(f"\nSaved to: results/h2h_v21_v23.json")
