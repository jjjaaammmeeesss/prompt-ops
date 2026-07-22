"""v1.5 (soul) vs v1.7 (baseline) vs v2.2 (current) 三版本盲评对比"""
import json, os, time, re, requests, random
import numpy as np
from concurrent.futures import ThreadPoolExecutor

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)

DS_URL = "https://api.deepseek.com/v1/chat/completions"
CL_URL = "https://s.lconai.com/v1/messages"
CL_KEY = "CLAUDE_API_KEY_PLACEHOLDER"
for line in open(".env"):
    if line.startswith("DEEPSEEK_API_KEY="):
        DS_KEY = line.split("=", 1)[1].strip(); break

# 绕过可能崩溃的代理
os.environ["no_proxy"] = "api.deepseek.com,s.lconai.com"
DS_KWARGS = {
    "headers": {"Authorization": f"Bearer {DS_KEY}", "Content-Type": "application/json"},
    "timeout": 120,
}
CL_KWARGS = {
    "headers": {"x-api-key": CL_KEY, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
    "timeout": 120,
}

# ── 加载三个版本 ──

# v1.5: 解析 markdown, 提取 System Prompt + User Prompt
v15_md = open("D:/星灵-soul-手搓/亲子沟通洞见/路线A_自上而下法_鼓励版/prompts/prompt_A轨_v1.5_鼓励版.md", encoding="utf-8").read()

# 提取 System Prompt (## 2. System Prompt 下的 ``` 代码块)
m_sp = re.search(r'## 2\. System Prompt\n+```\n(.*?)\n```', v15_md, re.DOTALL)
v15_system = m_sp.group(1).strip() if m_sp else ""

# 提取 User Prompt: 从 ## 3. User Prompt 后的第一道 fence 到 ## 4. 版本历史
# 文件中的 fence 不匹配(开4反引号, 无匹配闭合), 用行号定位
v15_lines = open("D:/星灵-soul-手搓/亲子沟通洞见/路线A_自上而下法_鼓励版/prompts/prompt_A轨_v1.5_鼓励版.md", encoding="utf-8").readlines()

# 找到 ## 3. User Prompt 后的第一个 fence 行
up_start = None
up_end = None
in_section = False
for i, line in enumerate(v15_lines):
    if line.startswith("## 3. User Prompt"):
        in_section = True
        continue
    if in_section and line.strip() in ("```", "````") and up_start is None:
        up_start = i + 1  # fence 下一行开始
        continue
    if up_start is not None and line.startswith("## 4. 版本历史"):
        up_end = i  # 版本历史前一行结束
        break

if up_start and up_end:
    # 提取内容, 并去除末尾的 fence 行
    raw = "".join(v15_lines[up_start:up_end]).strip()
    # 去掉末尾的 fence (```)
    if raw.endswith("```"):
        raw = raw[:-3].strip()
    v15_user_template = raw
else:
    v15_user_template = ""

# v1.7
v17 = open("system_prompt_backup_v17.txt", encoding="utf-8").read().strip()

# v2.2
v22 = open("system_prompt_v2.2.txt", encoding="utf-8").read().strip()

print(f"v1.5 system: {len(v15_system)}c | user_template: {len(v15_user_template)}c")
print(f"v1.7: {len(v17)}c")
print(f"v2.2: {len(v22)}c")

# ── 测试集 ──
dataset = json.load(open("data/dataset_merged_train.json", encoding="utf-8"))
# 取 12 个均匀分布的样本
indices = np.linspace(0, len(dataset) - 1, 12, dtype=int)
test_set = [dataset[i] for i in indices]
print(f"测试样本: {len(test_set)}\n")

# ── Judge Prompt (盲评: X/Y/Z 随机对应三个版本) ──
JUDGE = """你是亲子沟通教练评估专家。以下是同一个对话由三个不同版本 prompt 生成的弹窗（已随机打乱，标为 X/Y/Z）。

请对每个弹窗从以下维度评分（1-5 整数），然后综合排名（第1/2/3名）：

1. 发心承认: 是否先看见家长的发心、情绪和难处？
2. 洞察准确性: 是否基于对话具体行为命中痛点？
3. 模式揭示: 是否把单次事件连成反复模式？
4. 邀请感: 是否用邀请/试探语气而非宣告/说教？
5. 可操作性: 建议或洞察是否具体可落地？（纯诊断无建议标 N/A）
6. 措辞自然度: 是否口语化、生活化、不爹味？
7. 专一度: 是否聚焦一个主要矛盾讲透？

对话:
{dialogue}

弹窗 X:
{popup_x}

弹窗 Y:
{popup_y}

弹窗 Z:
{popup_z}

输出 JSON（只输出 JSON，键名必须完全一致）:
{{"X":{{"发心承认":1-5,"洞察准确性":1-5,"模式揭示":1-5,"邀请感":1-5,"可操作性":"1-5或N/A","措辞自然度":1-5,"专一度":1-5}},"Y":{{...(同上)}},"Z":{{...(同上)}},"ranking":["第1名","第2名","第3名"],"reason":"简短说明排名理由"}}"""

# 两版本 judge
JUDGE2 = """你是亲子沟通教练评估专家。以下是同一个对话由两个不同版本 prompt 生成的弹窗（已随机打乱，标为 X/Y）。

请对每个弹窗从以下维度评分（1-5 整数），然后选择胜者：

1. 发心承认: 是否先看见家长的发心、情绪和难处？
2. 洞察准确性: 是否基于对话具体行为命中痛点？
3. 模式揭示: 是否把单次事件连成反复模式？
4. 邀请感: 是否用邀请/试探语气而非宣告/说教？
5. 可操作性: 建议或洞察是否具体可落地？（纯诊断无建议标 N/A）
6. 措辞自然度: 是否口语化、生活化、不爹味？
7. 专一度: 是否聚焦一个主要矛盾讲透？

对话:
{dialogue}

弹窗 X ({version_x}):
{popup_x}

弹窗 Y ({version_y}):
{popup_y}

输出 JSON（只输出 JSON）:
{{"X":{{"发心承认":1-5,"洞察准确性":1-5,"模式揭示":1-5,"邀请感":1-5,"可操作性":"1-5或N/A","措辞自然度":1-5,"专一度":1-5}},"Y":{{...(同上)}},"winner":"X"或"Y"或"tie","reason":"简短理由"}}"""

# ── 维度权重（与历史 H2H 一致）──
DIMS = [
    ("发心承认", 0.20), ("洞察准确性", 0.20),
    ("模式揭示", 0.10), ("邀请感", 0.10),
    ("可操作性", 0.15), ("措辞自然度", 0.15), ("专一度", 0.10),
]


def calc_score(scores):
    """从维度评分计算综合分数 (0-1)"""
    ws, tw = 0.0, 0.0
    for dim, w in DIMS:
        v = scores.get(dim)
        if v == "N/A" or v is None:
            if dim == "可操作性": continue
            continue
        if isinstance(v, (int, float)) and 1 <= v <= 5:
            ws += ((v - 1) / 4) * w
            tw += w
    return ws / tw if tw > 0 else 0.0


def generate_v15(dia):
    """v1.5: System Prompt + User Prompt(with {user_input}) → 解析 JSON → popup_text"""
    user_prompt = v15_user_template.replace("{user_input}", dia)
    user_prompt = user_prompt.replace("{profile_context}", "")
    user_prompt = user_prompt.replace("{context_block}", "")
    for attempt in range(3):
        try:
            r = requests.post(DS_URL,
                json={"model": "deepseek-v4-pro", "max_tokens": 4096, "temperature": 0.4,
                      "messages": [
                          {"role": "system", "content": v15_system},
                          {"role": "user", "content": user_prompt}
                      ]},
                **DS_KWARGS)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            # 尝试解析 JSON 提取 popup_text
            m = re.search(r'"popup_text":\s*"((?:[^"\\]|\\.)*)"', content, re.DOTALL)
            if m:
                popup = m.group(1)
                popup = popup.replace('\\"', '"').replace('\\n', '\n')
                if popup and popup != "null" and len(popup) >= 5:
                    return popup
            # 回退：尝试通用 JSON 解析
            mj = re.search(r'\{.*\}', content, re.DOTALL)
            if mj:
                try:
                    obj = json.loads(mj.group(0))
                    pt = obj.get("popup_text", "")
                    if pt and pt != "null" and len(pt) >= 5:
                        return pt
                except:
                    pass
            if content and len(content) > 5:
                return content[:300]
            return "[ERROR]"
        except Exception as e:
            print(f"  v1.5 attempt {attempt+1}: {str(e)[:80]}")
            time.sleep(3 + attempt * 3)
    return "[ERROR]"


def generate_simple(sp, dia):
    """v1.7 / v2.2: 简单 system prompt → 弹窗文本"""
    for attempt in range(3):
        try:
            r = requests.post(DS_URL,
                json={"model": "deepseek-v4-pro", "max_tokens": 800, "temperature": 0.7,
                      "messages": [{"role": "system", "content": sp},
                                   {"role": "user", "content": f"对话：\n{dia}\n\n请生成弹窗："}]},
                **DS_KWARGS)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"  gen attempt {attempt+1}: {str(e)[:80]}")
            time.sleep(3 + attempt * 3)
    return "[ERROR]"


# ── 主循环 ──
results = []
scores_by_version = {"v1.5": [], "v1.7": [], "v2.2": []}
rank_counts = {"v1.5": [0,0,0], "v1.7": [0,0,0], "v2.2": [0,0,0]}  # [1st, 2nd, 3rd]

for i, ex in enumerate(test_set):
    dia = ex["question"]
    short = dia[:50].replace("\n", " ")
    print(f"[{i+1}/12] {short}...")

    # 并行生成三个版本
    with ThreadPoolExecutor(3) as pool:
        f15 = pool.submit(generate_v15, dia)
        f17 = pool.submit(generate_simple, v17, dia)
        f22 = pool.submit(generate_simple, v22, dia)
        popup_15 = f15.result()
        popup_17 = f17.result()
        popup_22 = f22.result()

    # 检查是否有足够的有效弹窗
    popups = {"v1.5": popup_15, "v1.7": popup_17, "v2.2": popup_22}
    valid = {k: v for k, v in popups.items() if v != "[ERROR]" and len(v) >= 3}
    if len(valid) < 2:
        print(f"  Too many errors (valid={len(valid)}), skip")
        continue

    # 随机打乱 → X/Y/Z (或 X/Y，如果只有 2 个有效)
    keys = list(valid.keys())
    random.shuffle(keys)
    labels = ["X", "Y", "Z"][:len(keys)]
    mapping = {keys[i]: labels[i] for i in range(len(keys))}
    reverse_map = {labels[i]: keys[i] for i in range(len(keys))}

    # 构建 judge prompt
    label_popups = {labels[i]: valid[keys[i]] for i in range(len(keys))}
    if len(labels) == 3:
        prompt = JUDGE.format(
            dialogue=dia,
            popup_x=label_popups["X"],
            popup_y=label_popups["Y"],
            popup_z=label_popups["Z"],
        )
    else:
        # 只有 2 个版本，用两版评判
        prompt = JUDGE2.format(
            dialogue=dia,
            popup_x=label_popups["X"],
            popup_y=label_popups["Y"],
            version_x=reverse_map["X"],
            version_y=reverse_map["Y"],
        )

    # Claude 评判
    verdict = {}
    for attempt in range(3):
        try:
            r = requests.post(CL_URL,
                json={"model": "claude-opus-4-8", "max_tokens": 2048, "temperature": 0.0,
                      "system": "你是严格的评估专家，只输出JSON。",
                      "messages": [{"role": "user", "content": prompt}]},
                **CL_KWARGS)
            r.raise_for_status()
            for block in r.json().get("content", []):
                if isinstance(block, dict) and block.get("type") == "text":
                    m = re.search(r"\{.*\}", block["text"], re.DOTALL)
                    if m:
                        verdict = json.loads(m.group(0))
            if verdict:
                break
        except Exception as e:
            print(f"  judge attempt {attempt+1}: {str(e)[:80]}")
            time.sleep(3)

    if not verdict:
        print("  judge fail")
        continue

    # 计算每个版本得分
    scores_by_label = {}
    for label in labels:
        if label in verdict:
            scores_by_label[label] = calc_score(verdict[label])
        else:
            scores_by_label[label] = 0.0

    # 映射回真实版本
    for label, version in reverse_map.items():
        if label in scores_by_label:
            scores_by_version[version].append(scores_by_label[label])

    # 记录排名 (3版) 或 winner (2版)
    if len(labels) == 3:
        ranking = verdict.get("ranking", [])
        for rank_idx, label in enumerate(ranking):
            if label in reverse_map:
                version = reverse_map[label]
                rank_counts[version][rank_idx] += 1
        print(f"  X({reverse_map['X']})={scores_by_label.get('X',0):.4f} | "
              f"Y({reverse_map['Y']})={scores_by_label.get('Y',0):.4f} | "
              f"Z({reverse_map['Z']})={scores_by_label.get('Z',0):.4f} | "
              f"rank: {ranking}")
    else:
        winner = verdict.get("winner", "tie")
        if winner == "X":
            rank_counts[reverse_map["X"]][0] += 1
            rank_counts[reverse_map["Y"]][1] += 1
        elif winner == "Y":
            rank_counts[reverse_map["Y"]][0] += 1
            rank_counts[reverse_map["X"]][1] += 1
        else:
            rank_counts[reverse_map["X"]][0] += 1
            rank_counts[reverse_map["Y"]][0] += 1
        print(f"  X({reverse_map['X']})={scores_by_label.get('X',0):.4f} | "
              f"Y({reverse_map['Y']})={scores_by_label.get('Y',0):.4f} | "
              f"winner={winner}")

    results.append({
        "dia": dia[:200],
        "popup_v15": popup_15,
        "popup_v17": popup_17,
        "popup_v22": popup_22,
        "mapping": mapping,
        "judge": verdict,
    })

# ── 汇总 ──
print(f"\n{'='*60}")
print(f"有效样本: {len(results)}")

means = {}
for v in ["v1.5", "v1.7", "v2.2"]:
    ss = scores_by_version[v]
    if ss:
        means[v] = (np.mean(ss), np.std(ss))
        print(f"{v}: {np.mean(ss):.4f} ± {np.std(ss):.4f} (n={len(ss)})")
    else:
        means[v] = (0, 0)
        print(f"{v}: no valid scores")

# 排名汇总
print(f"\n排名统计:")
for v in ["v1.5", "v1.7", "v2.2"]:
    rc = rank_counts[v]
    print(f"  {v}: 第1={rc[0]}, 第2={rc[1]}, 第3={rc[2]}")

# Cohen's d
if len(results) >= 4:
    print(f"\n效应量:")
    pairs = [("v1.5", "v1.7"), ("v1.5", "v2.2"), ("v1.7", "v2.2")]
    for a, b in pairs:
        sa = scores_by_version[a]
        sb = scores_by_version[b]
        if len(sa) >= 3 and len(sb) >= 3:
            pooled_std = (np.std(sa) + np.std(sb)) / 2
            if pooled_std > 0:
                d = (np.mean(sb) - np.mean(sa)) / pooled_std
                print(f"  {a} vs {b}: d={d:.2f}")

out = {
    "config": {"task_model": "deepseek-v4-pro", "judge_model": "claude-opus-4-8", "n": len(results)},
    "summary": {
        v: {"mean": float(means[v][0]), "std": float(means[v][1]), "n": len(scores_by_version[v]),
            "rank_counts": {"1st": rank_counts[v][0], "2nd": rank_counts[v][1], "3rd": rank_counts[v][2]}}
        for v in ["v1.5", "v1.7", "v2.2"]
    },
    "results": results,
}
json.dump(out, open("results/h2h_v15_v17_v22.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2, default=str)
print(f"\nSaved: results/h2h_v15_v17_v22.json")
print("Done!")
