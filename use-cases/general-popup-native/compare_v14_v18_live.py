"""Compare v1.4 vs v1.7 vs v1.8 on business/negotiation cases."""
import json, os, re, sys

def load_key(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    m = re.search(r"sk-[A-Za-z0-9]+", text)
    if not m:
        raise RuntimeError("key not found")
    return m.group(0)

def call(prompt, dialogue, api_key, label):
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
    try:
        resp = client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[{"role":"system","content":prompt},{"role":"user","content":dialogue}],
            temperature=0.3, max_tokens=4096,
        )
        content = resp.choices[0].message.content or ""
        if not content:
            print(f"  [{label}] EMPTY. finish={resp.choices[0].finish_reason}", file=sys.stderr)
        return content
    except Exception as e:
        print(f"  [{label}] ERROR: {e}", file=sys.stderr)
        return f"[ERROR: {e}]"

def main():
    base = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(base, "system_prompt_v1.4.txt"), encoding="utf-8") as f:
        p14 = f.read()
    with open(os.path.join(base, "results/self_evolve_v7/best_prompt_r2.txt"), encoding="utf-8") as f:
        p17 = f.read()
    with open(os.path.join(base, "results/self_evolve_v7/best_prompt_r8.txt"), encoding="utf-8") as f:
        p18 = f.read()

    with open(os.path.join(base, "dataset.json"), encoding="utf-8") as f:
        dataset = json.load(f)

    key = load_key("D:/ob-new202603/钥匙库/DeepSeek公司key.md")

    # Known eval scores
    scores_map = {
        "manager_anger":       {"v1.4": 0.9667, "v1.7": 0.7467, "v1.8": 0.9},
        "partner_unspoken_need": {"v1.4": 0.9,    "v1.7": 0.7644, "v1.8": 0.9333},
    }

    prompts = {"v1.4": p14, "v1.7": p17, "v1.8": p18}

    for case_id in ["manager_anger", "partner_unspoken_need"]:
        case = next(d for d in dataset if d["id"] == case_id)
        print("=" * 70)
        print(f"用例: {case['id']} — {case['title']}")
        print(f"关系: {case['relation']}  |  场景: {case['scene']}")
        scores = scores_map[case_id]
        print(f"eval 分数: v1.4={scores['v1.4']}  v1.7={scores['v1.7']}  v1.8={scores['v1.8']}")
        print("=" * 70)
        print("\n【对话原文】")
        print(case["dialogue"])
        print()

        for ver in ["v1.4", "v1.7", "v1.8"]:
            print("-" * 70)
            print(f"【{ver} 弹窗】")
            popup = call(prompts[ver], case["dialogue"], key, ver)
            print(popup if popup.strip() else "（空输出）")
            print()
        print()

if __name__ == "__main__":
    main()
