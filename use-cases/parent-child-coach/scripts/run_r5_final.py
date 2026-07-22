"""r5 Final Prompt Runner — 快/慢双通道评估

快慢通道 = 弹窗触发机制的两种策略：
  快通道（句级正则）: 逐句扫描，匹配到冲突关键词立即触发弹窗
  慢通道（300字轮询）: 对话按300字分窗口，每个窗口调 r5 生成弹窗

每条对话同时跑两条通道，Claude v2.0 评分，对比两条通道的表现。
"""

import argparse, json, os, re, sys, time
from statistics import mean, stdev
from typing import Dict, List, Tuple
import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "results", "auto_research_judge_v2")
os.makedirs(OUT_DIR, exist_ok=True)

# ── API ──────────────────────────────────────────────────
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

# ── r5 prompt ────────────────────────────────────────────
with open(os.path.join(OUT_DIR, "final_best_prompt.txt"), "r", encoding="utf-8") as f:
    PROMPT_R5 = f.read().strip()

# ── 快通道: 句级关键词正则 ──────────────────────────────────
# 这些是亲子冲突对话中常见的"该弹窗了"的信号句 — 家长情绪升级、孩子受伤沉默、权力对抗
FAST_TRIGGERS = [
    # 威胁/命令（家长情绪升级到威胁阶段）
    (re.compile(r"你再\s*\S\s*[我就]"), "威胁式管教"),
    (re.compile(r"凭[什啥]么"), "权力对抗"),
    (re.compile(r"我养你|供你吃|供你穿"), "养育付出绑架"),
    # 否定/贴标签
    (re.compile(r"你(怎么|又|老是|总是|每次都)"), "负面标签"),
    (re.compile(r"笨死|没用|废物|不争气|丢脸"), "人格攻击"),
    (re.compile(r"我看你就是|你[就都]是[不会敢]"), "武断结论"),
    # 情感拒绝
    (re.compile(r"别(说|吵|烦|哭|闹)[了我]"), "情绪压制"),
    (re.compile(r"[我都你]?不管你了?|随便你|爱[怎咋].*[怎咋]"), "情感撤回"),
    (re.compile(r"我[很就]?失望|太让[我人].*失望"), "失望表达"),
    # 孩子受伤信号
    (re.compile(r"(你不[爱喜].*[了我]|[你我].*不爱.*[我了你])"), "安全感破裂"),
    (re.compile(r"[你我].*凭什么|[你]?是不是.*[讨厌不喜]"), "被抛弃恐惧"),
    # 沉默/回避（需要干预）
    (re.compile(r"^\s*[\(（]?[^。！？\n]+[\)）]?\s*$"), None),  # 太泛，跳过
    # 家长自我暴露脆弱
    (re.compile(r"我也?不[知会想懂]|我能怎么|我不知道.*怎么"), "家长无力"),
    (re.compile(r"我.*[压累急]力.*大|头疼|快?崩溃"), "家长压力信号"),
    # 认知窄化
    (re.compile(r"(只[有能会]|除非|不然就|一定[得要])"), "认知窄化"),
    (re.compile(r"[^不]?应该|[必须]须[得要]"), "绝对化要求"),
    # 情感转折点（孩子从抗拒转向沟通 / 家长从指责转向倾听）
    (re.compile(r"你说[得对]|[好好]吧.*我[以后会]*"), "态度软化"),
]

# 这些是真正需要弹窗的核心信号 — 更精确的匹配
CORE_TRIGGERS = [
    (re.compile(r"(你再|我就|给我|必须|不准|不许|不要|别想)"), "家长命令/威胁"),
    (re.compile(r"(你凭什么|你不爱我|你讨厌我|是不是.*不爱)"), "孩子情感危机"),
    (re.compile(r"(你每次都这样|你老是|你总是|你怎么又)"), "家长指责模式"),
    (re.compile(r"(我说了|跟你说了|讲了多少遍|说三遍了)"), "家长不耐烦"),
    (re.compile(r"(沉默|不说话|不回答|低头|不理)"), "沟通断裂"),
    (re.compile(r"(你根本|你从来|你一点都)"), "家长绝对化否定"),
]

# ── Judge v2.0 ───────────────────────────────────────────
SCORING_PROMPT = """你是亲子沟通弹窗评估专家。给弹窗从五个维度打分（1-5）。

### 一级否决
事实性错误（编造对话中不存在的内容）或语气严重误判 → 总分=0

### 维度
1. 看见感: 家长读完觉得"你懂我"？
2. 对话忠实度: 每个判断都能在对话中找到依据？
3. 命中核心: 抓住了最该被看见的点？
4. 人话感: 像真人在说话？无术语/模板/框架标签？
5. 温度: 姿态是盟友还是教师？

Few-shot校准: 翻日记对话中，弹窗A"信任的门不是撞开的"=全5分; 弹窗B含"多极"+"你正戴着X的眼镜"=人话感1分。

对话：
{dialogue}

AI回应：
{response}

输出JSON:
{{"veto":null或"事实性错误"或"语气严重误判","being_seen":1-5,"dialogue_fidelity":1-5,"core_insight":1-5,"natural_language":1-5,"warmth":1-5,"brief_reason":"一句话"}}"""

DIM_WEIGHTS = [("being_seen",0.25),("dialogue_fidelity",0.20),("core_insight",0.20),
               ("natural_language",0.20),("warmth",0.15)]
DIM_LABELS = {"being_seen":"看见感","dialogue_fidelity":"对话忠实度","core_insight":"命中核心",
              "natural_language":"人话感","warmth":"温度"}

# ═══════════════════════════════════════════════════════════
# API Helpers
# ═══════════════════════════════════════════════════════════

def deepseek(messages, max_tokens=2048, temp=0.7):
    headers = {"Authorization": f"Bearer {DS_KEY}", "Content-Type": "application/json"}
    for attempt in range(4):
        try:
            resp = requests.post(DS_URL, headers=headers,
                json={"model":"deepseek-v4-pro","max_tokens":max_tokens,"temperature":temp,"messages":messages},
                timeout=(30,120))
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"    ⚠ DS error (attempt {attempt+1}/4): {type(e).__name__}", flush=True)
            time.sleep(2**attempt*3)
    raise RuntimeError("DeepSeek failed")

def claude(text):
    headers = {"x-api-key":CL_KEY,"anthropic-version":"2023-06-01","Content-Type":"application/json"}
    for attempt in range(4):
        try:
            resp = requests.post(CL_URL, headers=headers,
                json={"model":"claude-opus-4-8","max_tokens":1024,"temperature":0.0,
                      "thinking":{"type":"disabled"},
                      "system":"你是严格的亲子沟通弹窗评估专家。只输出JSON。",
                      "messages":[{"role":"user","content":text}]}, timeout=(30,90))
            resp.raise_for_status()
            for block in resp.json().get("content",[]):
                if isinstance(block,dict) and block.get("type")=="text":
                    return block["text"]
        except Exception as e:
            print(f"    ⚠ CL error (attempt {attempt+1}/4): {type(e).__name__}", flush=True)
            time.sleep(2**attempt*3)

def parse_json(raw):
    raw = raw.strip()
    try: return json.loads(raw)
    except: pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try: return json.loads(m.group(0))
        except: pass
    raise ValueError(f"Cannot parse: {raw[:200]}")

def judge_one(dialogue, popup):
    try:
        scores = parse_json(claude(SCORING_PROMPT.format(dialogue=dialogue, response=popup)))
        veto = scores.get("veto")
        if veto and str(veto).strip() not in ("null","none",""):
            return 0.0, scores
        ws, tw = 0.0, 0.0
        for dk, w in DIM_WEIGHTS:
            v = scores.get(dk)
            if isinstance(v,(int,float)) and 1<=v<=5:
                ws += (v-1)/4*w; tw += w
        return ws/tw if tw else 0.0, scores
    except Exception as e:
        print(f"    ❌ Judge failed: {type(e).__name__}", flush=True)
        return 0.0, {}

# ═══════════════════════════════════════════════════════════
# 快通道: 句级正则触发
# ═══════════════════════════════════════════════════════════

def split_sentences(dialogue):
    """Split dialogue into sentences (by newlines and Chinese punctuation)."""
    # First split by newlines (each line is typically one person's utterance)
    lines = [l.strip() for l in dialogue.split("\n") if l.strip()]
    sentences = []
    for line in lines:
        # Further split long lines by Chinese sentence-ending punctuation
        parts = re.split(r"(?<=[。！？\.!\?])", line)
        for p in parts:
            p = p.strip()
            if p:
                sentences.append(p)
    return sentences


def fast_channel_trigger(sentences: List[str], context_before: str = "") -> List[Tuple[int, str, str]]:
    """Scan sentences, return list of (sentence_index, matched_pattern, sentence_text).

    Stops after first 2 triggers to avoid over-triggering.
    """
    triggers_fired = []
    cumulative = context_before
    for i, sent in enumerate(sentences):
        cumulative += sent
        for pattern, label in CORE_TRIGGERS:
            if pattern.search(sent):
                # Avoid duplicate triggers on same sentence
                if not any(t[0] == i for t in triggers_fired):
                    triggers_fired.append((i, label, sent, cumulative[:300]))
                    break
        if len(triggers_fired) >= 2:
            break
    return triggers_fired


# ═══════════════════════════════════════════════════════════
# 慢通道: 300字轮询
# ═══════════════════════════════════════════════════════════

def slow_channel_windows(dialogue: str, window_size: int = 300) -> List[Tuple[int, str]]:
    """Split dialogue into ~300-char windows. Returns (window_index, window_text)."""
    windows = []
    # Accumulate sentences until we hit ~300 chars
    sentences = split_sentences(dialogue)
    current = ""
    win_idx = 0
    for sent in sentences:
        current += sent
        if len(current) >= window_size:
            windows.append((win_idx, current[:600]))  # cap at 600 chars for context
            current = ""
            win_idx += 1
    # Don't forget trailing content
    if current.strip() and len(current) >= 100:  # min 100 chars for a window
        windows.append((win_idx, current[:600]))
    return windows


# ═══════════════════════════════════════════════════════════
# Main Runner
# ═══════════════════════════════════════════════════════════

def run(n_dialogues=12):
    # Load test set
    path = os.path.join(BASE_DIR, "data", "dataset_merged_train.json")
    data = json.load(open(path, "r", encoding="utf-8"))
    step = max(1, len(data) // n_dialogues)
    test_set = [data[i] for i in range(0, min(len(data), step * n_dialogues), step)][:n_dialogues]
    test_set = [{"dialogue": item["question"], "id": f"d{i}"}
                for i, item in enumerate(test_set) if item.get("question")][:n_dialogues]

    print("=" * 80, flush=True)
    print(f"R5 FINAL · 快慢双通道 Runner", flush=True)
    print(f"  测试对话: {len(test_set)}条", flush=True)
    print(f"  r5 prompt: {len(PROMPT_R5)} chars", flush=True)
    print(f"", flush=True)
    print(f"  ⚡ 快通道: 句级正则匹配 → 命中立即触发弹窗", flush=True)
    print(f"  🔍 慢通道: 300字窗口轮询 → 每窗口生成弹窗", flush=True)
    print(f"  Judge:    claude-opus-4-8 v2.0 (thinking=disabled)", flush=True)
    print("=" * 80, flush=True)

    fast_results = []
    slow_results = []
    t_start = time.time()

    for di, item in enumerate(test_set):
        dialogue = item["dialogue"]
        sentences = split_sentences(dialogue)
        d_short = dialogue[:70].replace("\n", " ")
        print(f"\n{'─'*80}", flush=True)
        print(f"[对话 {di+1}/{len(test_set)}] {d_short}...", flush=True)
        print(f"  总{len(dialogue)}字, {len(sentences)}句", flush=True)

        # ── 快通道 ──
        fast_triggers = fast_channel_trigger(sentences)
        if fast_triggers:
            print(f"  ⚡ 快通道: {len(fast_triggers)}次触发", flush=True)
            for si, label, sent, context in fast_triggers:
                print(f"    句{si}: [{label}] \"{sent[:60]}...\"", flush=True)
                gen_start = time.time()
                popup = deepseek([
                    {"role": "system", "content": PROMPT_R5},
                    {"role": "user", "content": f"对话：\n{context}\n\n请生成弹窗："},
                ])
                gen_time = time.time() - gen_start
                time.sleep(0.2)

                score, dims = judge_one(dialogue, popup)
                time.sleep(0.2)
                veto = dims.get("veto","")
                reason = dims.get("brief_reason","")
                veto_str = f" [VETO:{veto}]" if veto and str(veto).strip() not in ("null","none","") else ""
                print(f"    → Score: {score:.4f}{veto_str} (gen {gen_time:.0f}s) | {reason[:80]}", flush=True)

                fast_results.append({
                    "dialogue_id": item["id"], "channel": "fast",
                    "trigger_sentence": si, "trigger_label": label,
                    "triggered_sent": sent, "context": context[:300],
                    "popup": popup, "overall_score": score,
                    "per_dim": {k: dims.get(k) for k in DIM_LABELS},
                    "veto": veto, "brief_reason": reason,
                })
        else:
            print(f"  ⚡ 快通道: 无触发", flush=True)

        # ── 慢通道 ──
        slow_windows = slow_channel_windows(dialogue)
        print(f"  🔍 慢通道: {len(slow_windows)}个窗口", flush=True)
        for wi, win_text in slow_windows:
            win_short = win_text[:60].replace("\n"," ")
            print(f"    窗口{wi} ({len(win_text)}字): \"{win_short}...\"", flush=True)
            gen_start = time.time()
            popup = deepseek([
                {"role": "system", "content": PROMPT_R5},
                {"role": "user", "content": f"对话：\n{win_text}\n\n请生成弹窗："},
            ])
            gen_time = time.time() - gen_start
            time.sleep(0.2)

            score, dims = judge_one(dialogue, popup)
            time.sleep(0.2)
            veto = dims.get("veto","")
            reason = dims.get("brief_reason","")
            veto_str = f" [VETO:{veto}]" if veto and str(veto).strip() not in ("null","none","") else ""
            print(f"    → Score: {score:.4f}{veto_str} (gen {gen_time:.0f}s) | {reason[:80]}", flush=True)

            slow_results.append({
                "dialogue_id": item["id"], "channel": "slow",
                "window_index": wi, "window_chars": len(win_text),
                "popup": popup, "overall_score": score,
                "per_dim": {k: dims.get(k) for k in DIM_LABELS},
                "veto": veto, "brief_reason": reason,
            })

    # ── Summary ─────────────────────────────────────────
    total_time = time.time() - t_start
    print(f"\n{'=' * 80}", flush=True)
    print(f"📊 快慢通道 · 对比结果", flush=True)
    print(f"  总耗时: {total_time/60:.1f}m", flush=True)
    print(f"{'=' * 80}", flush=True)

    def summarize(label, results):
        scores = [r["overall_score"] for r in results]
        vetos = sum(1 for r in results if r.get("veto") and str(r["veto"]).strip() not in ("null","none",""))
        if not scores: return None
        dims_coll = {dk: [] for dk,_ in DIM_WEIGHTS}
        for r in results:
            for dk in DIM_LABELS:
                v = r["per_dim"].get(dk)
                if isinstance(v,(int,float)): dims_coll[dk].append((v-1)/4)
        return {
            "n": len(scores), "mean": mean(scores), "std": stdev(scores) if len(scores)>1 else 0,
            "vetos": vetos, "dim_means": {dk: (mean(vals) if vals else 0) for dk,vals in dims_coll.items()},
        }

    fs = summarize("快通道", fast_results)
    ss = summarize("慢通道", slow_results)

    print(f"\n  ┌─ 总体对比 ─{'─'*45}", flush=True)
    print(f"  │ {'指标':<12} {'⚡ 快通道(句级正则)':<25} {'🔍 慢通道(300字轮询)':<25}", flush=True)
    print(f"  │ {'─'*12} {'─'*25} {'─'*25}", flush=True)
    if fs:
        print(f"  │ {'弹窗数':<12} {fs['n']:<25} {ss['n'] if ss else 'N/A':<25}", flush=True)
        print(f"  │ {'均分':<12} {fs['mean']:.4f} ± {fs['std']:.4f}   {ss['mean']:.4f} ± {ss['std']:.4f}" if ss else "", flush=True)
        print(f"  │ {'Veto次数':<12} {fs['vetos']:<25} {ss['vetos'] if ss else 'N/A':<25}", flush=True)
    print(f"  └{'─'*55}", flush=True)

    # Dimension comparison
    if fs and ss:
        print(f"\n  ┌─ 维度对比 ─{'─'*45}", flush=True)
        print(f"  │ {'维度':<10} {'快通道':<12} {'慢通道':<12} {'差值':>8}", flush=True)
        print(f"  │ {'─'*10} {'─'*12} {'─'*12} {'─'*8}", flush=True)
        for dk,_ in DIM_WEIGHTS:
            fm = fs["dim_means"].get(dk,0)
            sm = ss["dim_means"].get(dk,0)
            diff = fm - sm
            sign = "+" if diff > 0 else ""
            print(f"  │ {DIM_LABELS[dk]:8s}  {fm:.4f}       {sm:.4f}       {sign}{diff:+.4f}", flush=True)
        print(f"  └{'─'*55}", flush=True)

    # Per-dialogue breakdown
    print(f"\n  ┌─ 按对话拆分 ─{'─'*45}", flush=True)
    for di, item in enumerate(test_set):
        f_dialogue = [r for r in fast_results if r["dialogue_id"] == item["id"]]
        s_dialogue = [r for r in slow_results if r["dialogue_id"] == item["id"]]
        f_mean = mean([r["overall_score"] for r in f_dialogue]) if f_dialogue else None
        s_mean = mean([r["overall_score"] for r in s_dialogue]) if s_dialogue else None
        f_str = f"{f_mean:.4f} ({len(f_dialogue)}次)" if f_mean is not None else "无触发"
        s_str = f"{s_mean:.4f} ({len(s_dialogue)}窗)" if s_mean is not None else "无窗口"
        d_short = item["dialogue"][:50].replace("\n"," ")
        print(f"  │ d{di}: {d_short}...", flush=True)
        print(f"  │   ⚡{f_str}  🔍{s_str}", flush=True)

    # Save
    out_file = os.path.join(OUT_DIR, "r5_eval_channels.json")
    summary = {
        "config": {"prompt":"r5_final","prompt_chars":len(PROMPT_R5),
                   "task_model":"deepseek-v4-pro","judge_model":"claude-opus-4-8",
                   "judge_version":"v2.0 (thinking disabled)",
                   "fast_triggers": [(p.pattern,l) for p,l in CORE_TRIGGERS],
                   "slow_window_chars": 300,
                   "n_dialogues": len(test_set), "total_time_s": round(total_time,1)},
        "fast_channel": fs,
        "slow_channel": ss,
        "fast_results": fast_results,
        "slow_results": slow_results,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved → {out_file}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="r5 Final · 快慢双通道 Runner")
    parser.add_argument("-n", type=int, default=12, help="测试对话数量 (default: 12)")
    args = parser.parse_args()
    run(n_dialogues=args.n)
