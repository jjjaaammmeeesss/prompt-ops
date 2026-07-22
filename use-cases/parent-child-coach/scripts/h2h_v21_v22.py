"""v2.1 (裸 prompt) vs v2.2 (12 expert-curated few-shots) 头对头"""
import json, os, time, re, requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)

DS_URL = "https://api.deepseek.com/v1/chat/completions"
CL_URL = "https://s.lconai.com/v1/messages"
CL_KEY = "CLAUDE_API_KEY_PLACEHOLDER"
for line in open(".env"):
    if line.startswith("DEEPSEEK_API_KEY="):
        DS_KEY = line.split("=", 1)[1].strip()
        break

WEIGHTS = [("ack",0.20),("insight",0.20),("pattern",0.10),("invite",0.10),
           ("action",0.15),("natural",0.15),("focus",0.10)]

JUDGE = """7维度评分(1-5), A=v2.1裸, B=v2.2(12专家few-shot):
1.发心承认 2.洞察准确性 3.模式揭示 4.邀请感 5.建议可操作性(纯诊断N/A) 6.措辞自然度 7.专一度

对话:
{d}

弹窗A:
{a}

弹窗B:
{b}

输出JSON: {{"A":{{...}},"B":{{...}},"winner":"A/B/tie","reason":"简短理由"}}"""

v21 = open("system_prompt_v2.1.txt", encoding="utf-8").read()
v22 = open("system_prompt_v2.2.txt", encoding="utf-8").read()
dataset = json.load(open("data/dataset_merged_train.json", encoding="utf-8"))
indices = np.linspace(0, len(dataset) - 1, 12, dtype=int)
test_set = [dataset[i] for i in indices]
print(f"v2.1: {len(v21)}c | v2.2: {len(v22)}c | test: {len(test_set)}\n")


def generate(sp, dia):
    for _ in range(3):
        try:
            r = requests.post(DS_URL,
                headers={"Authorization": f"Bearer {DS_KEY}", "Content-Type": "application/json"},
                json={"model": "deepseek-v4-pro", "max_tokens": 800, "temperature": 0.7,
                      "messages": [{"role": "system", "content": sp},
                                   {"role": "user", "content": f"对话：\n{dia}\n\n请生成弹窗："}]},
                timeout=90)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except:
            time.sleep(2)
    return "[ERROR]"


def calc_score(s):
    ws, tw = 0.0, 0.0
    for dim, w in WEIGHTS:
        v = s.get(dim)
        if v == "N/A" or v is None:
            if dim == "action":
                continue
            continue
        if isinstance(v, (int, float)) and 1 <= v <= 5:
            ws += ((v - 1) / 4) * w
            tw += w
    return ws / tw if tw > 0 else 0.0


results = []
w21 = w22 = tie = 0
s21, s22 = [], []

for i, ex in enumerate(test_set):
    dia = ex["question"]
    short = dia[:60].replace("\n", " ")
    print(f"[{i+1}/12] {short}...")

    with ThreadPoolExecutor(2) as pool:
        fa = pool.submit(generate, v21, dia)
        fb = pool.submit(generate, v22, dia)
        popup_a = fa.result()
        popup_b = fb.result()

    prompt = JUDGE.format(d=dia, a=popup_a, b=popup_b)
    verdict = {}
    for _ in range(3):
        try:
            r = requests.post(CL_URL,
                headers={"x-api-key": CL_KEY, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
                json={"model": "claude-opus-4-8", "max_tokens": 1024, "temperature": 0.0,
                      "system": "只输出JSON。",
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=90)
            r.raise_for_status()
            for block in r.json().get("content", []):
                if isinstance(block, dict) and block.get("type") == "text":
                    m = re.search(r"\{.*\}", block["text"], re.DOTALL)
                    if m:
                        verdict = json.loads(m.group(0))
            if verdict:
                break
        except:
            time.sleep(2)

    if not verdict:
        print("  judge fail")
        continue

    sa = calc_score(verdict.get("A", {}))
    sb = calc_score(verdict.get("B", {}))
    winner = verdict.get("winner", "tie")
    if winner == "A":
        w21 += 1
    elif winner == "B":
        w22 += 1
    else:
        tie += 1
    s21.append(sa)
    s22.append(sb)
    results.append({"dia": dia[:200], "v21_popup": popup_a, "v22_popup": popup_b,
                    "v21_score": sa, "v22_score": sb, "winner": winner,
                    "reason": verdict.get("reason", "")})
    print(f"  v2.1={sa:.4f} | v2.2={sb:.4f} | w={winner} | {verdict.get('reason','')[:80]}")

n = len(results)
m21 = np.mean(s21)
m22 = np.mean(s22)
print()
print("=" * 60)
print(f"v2.1 (裸):  {m21:.4f} +- {np.std(s21):.4f}")
print(f"v2.2 (12专家): {m22:.4f} +- {np.std(s22):.4f}")
print(f"Delta:        {m22 - m21:+.4f}")
print(f"Wins: v2.1={w21}, v2.2={w22}, tie={tie}")

out = {
    "config": {"task_model": "deepseek-v4-pro", "judge_model": "claude-opus-4-8", "n": n},
    "summary": {"v21_mean": m21, "v22_mean": m22, "delta": m22 - m21,
                "v21_wins": w21, "v22_wins": w22, "ties": tie},
    "results": results,
}
json.dump(out, open("results/h2h_v21_v22.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2, default=str)
print("Saved: results/h2h_v21_v22.json")
