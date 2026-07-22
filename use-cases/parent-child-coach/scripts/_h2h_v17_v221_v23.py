"""v1.7 vs v2.2.1 vs v2.3 三方盲测 · 系统采集新用例"""
import json, os, re, time, requests
from concurrent.futures import ThreadPoolExecutor

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Load prompts ──
v17  = open(os.path.join(BASE, "system_prompt_backup_v17.txt"), "r", encoding="utf-8").read().strip()
v221 = open(os.path.join(BASE, "system_prompt_v2.2.1.txt"), "r", encoding="utf-8").read().strip()
v23  = open(os.path.join(BASE, "system_prompt_v2.3.txt"), "r", encoding="utf-8").read().strip()

# ── API ──
DS_KEY = None
for line in open(os.path.join(BASE, ".env"), encoding="utf-8"):
    if line.startswith("DEEPSEEK_API_KEY="):
        DS_KEY = line.split("=", 1)[1].strip(); break
DS_URL = "https://api.deepseek.com/v1/chat/completions"
CL_URL = "https://s.lconai.com/v1/messages"
CL_KEY = "CLAUDE_API_KEY_PLACEHOLDER"

# ── Load test cases ──
test_data = json.load(open(os.path.join(BASE, "data", "expert_test.json"), "r", encoding="utf-8"))
# [2]=被骂三次, [4]=窗帘有人, [5]=搭乐高
cases = [test_data[2], test_data[4], test_data[5]]

print(f"v1.7: {len(v17)}c | v2.2.1: {len(v221)}c | v2.3: {len(v23)}c")
print(f"Cases: {len(cases)}")
print()

# ── Judge prompt ──
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

DIM_WEIGHTS = [
    ("being_seen", 0.25), ("dialogue_fidelity", 0.20), ("core_insight", 0.20),
    ("natural_language", 0.20), ("warmth", 0.15),
]
DIM_LABELS = {
    "being_seen": "看见感", "dialogue_fidelity": "对话忠实度",
    "core_insight": "命中核心", "natural_language": "人话感", "warmth": "温度",
}


def deepseek(sys_prompt, dialogue):
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
all_results = []

for ci, case in enumerate(cases):
    dialogue = case["dialogue"]
    title = case.get("case_title", "?")
    ref_popup = case.get("reference_popup", "")
    expert_score = case.get("expert_score", "?")

    # Generate 3 versions in parallel
    with ThreadPoolExecutor(max_workers=3) as pool:
        f17 = pool.submit(deepseek, v17, dialogue)
        f22 = pool.submit(deepseek, v221, dialogue)
        f23 = pool.submit(deepseek, v23, dialogue)
        p17 = f17.result()
        p22 = f22.result()
        p23 = f23.result()

    # Judge 3 versions in parallel
    with ThreadPoolExecutor(max_workers=3) as pool:
        j17 = pool.submit(judge_blind, dialogue, p17)
        j22 = pool.submit(judge_blind, dialogue, p22)
        j23 = pool.submit(judge_blind, dialogue, p23)
        s17_raw = j17.result()
        s22_raw = j22.result()
        s23_raw = j23.result()

    s17, _, _ = compute(s17_raw)
    s22, _, _ = compute(s22_raw)
    s23, _, _ = compute(s23_raw)

    all_results.append((ci, title, dialogue, ref_popup, expert_score,
                        p17, p22, p23, s17_raw, s22_raw, s23_raw, s17, s22, s23))
    time.sleep(0.5)

# ── Print all ──
for ci, title, dialogue, ref, escore, p17, p22, p23, s17r, s22r, s23r, s17, s22, s23 in all_results:
    print("#" * 70)
    print(f"## CASE {ci + 1}: {title}")
    print(f"## 专家评分: {escore}")
    print("#" * 70)
    print()
    print("-" * 70)
    print("【对话原文】")
    print("-" * 70)
    print(dialogue.strip())
    print()
    print("-" * 70)
    print("【校标 · 专家参考弹窗】")
    print("-" * 70)
    print(ref.strip() if ref.strip() else "(无校标)")
    print()

    for label, p, sr, sc in [("v1.7", p17, s17r, s17), ("v2.2.1", p22, s22r, s22), ("v2.3", p23, s23r, s23)]:
        print("-" * 70)
        print(f"【{label} 弹窗】({len(p)} chars)  --  score = {sc:.4f}")
        dim_parts = []
        for dk, dl in DIM_LABELS.items():
            dim_parts.append(f"{dl}: {sr.get(dk, '?')}/5")
        print("  " + "  ".join(dim_parts))
        print(f"  |  {sr.get('brief_reason', '')}")
        print("-" * 70)
        print(p.strip())
        print()

    ranked = sorted([("v1.7", s17), ("v2.2.1", s22), ("v2.3", s23)], key=lambda x: x[1], reverse=True)
    rank_str = " > ".join(f"{n}({sc:.4f})" for n, sc in ranked)
    print(f">>> 排名: {rank_str}")
    print()

# ── Summary ──
print("#" * 70)
print("## 三轮汇总 (v1.7 vs v2.2.1 vs v2.3)")
print("#" * 70)
print(f"{'#':<3} {'案例':<24} {'v1.7':>7} {'v2.2.1':>7} {'v2.3':>7}  {'第1':<8} {'第2':<8} {'第3':<8}")
print("-" * 78)
wins = {"v1.7": 0, "v2.2.1": 0, "v2.3": 0}
for ci, title, dialogue, ref, escore, p17, p22, p23, s17r, s22r, s23r, s17, s22, s23 in all_results:
    t = title[:22]
    ranked = sorted([("v1.7", s17), ("v2.2.1", s22), ("v2.3", s23)], key=lambda x: x[1], reverse=True)
    first, second, third = ranked[0][0], ranked[1][0], ranked[2][0]
    wins[first] += 1
    print(f"{ci + 1:<3} {t:<24} {s17:7.4f} {s22:7.4f} {s23:7.4f}  {first:<8} {second:<8} {third:<8}")
print("-" * 78)
print(f"  第1次数: v1.7={wins['v1.7']}  v2.2.1={wins['v2.2.1']}  v2.3={wins['v2.3']}")
