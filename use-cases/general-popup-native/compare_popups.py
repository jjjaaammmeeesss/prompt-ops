"""Generate popups for one sample dialogue using baseline vs optimized prompt."""
import json
import os
import sys


def load_key(path: str) -> str:
    import re
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    m = re.search(r"sk-[A-Za-z0-9]+", text)
    if not m:
        raise RuntimeError("key not found")
    return m.group(0)


def call_deepseek(prompt: str, dialogue: str, api_key: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": dialogue},
        ],
        temperature=0.3,
        max_tokens=512,
    )
    return resp.choices[0].message.content


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(base_dir, "system_prompt.txt"), "r", encoding="utf-8") as f:
        baseline_prompt = f.read()

    # latest optimized result (any config_*.json in results/)
    result_dir = os.path.join(base_dir, "results")
    result_files = sorted(
        [p for p in os.listdir(result_dir) if p.endswith(".json") and p.startswith("config")],
        reverse=True,
    )
    if not result_files:
        print("No optimized result found", file=sys.stderr)
        sys.exit(1)
    with open(os.path.join(result_dir, result_files[0]), "r", encoding="utf-8") as f:
        optimized_prompt = json.load(f)["prompt"]

    with open(os.path.join(base_dir, "dataset.json"), "r", encoding="utf-8") as f:
        dataset = json.load(f)

    key = load_key("D:/ob-new202603/钥匙库/DeepSeek公司key.md")

    sample = next(d for d in dataset if d["id"] == "manager_anger")
    print("=" * 60)
    print("DIALOGUE:", sample["id"])
    print(sample["dialogue"])
    print()

    print("=" * 60)
    print("BASELINE PROMPT POPUP:")
    print(call_deepseek(baseline_prompt, sample["dialogue"], key))
    print()

    print("=" * 60)
    print("OPTIMIZED PROMPT POPUP:")
    print(call_deepseek(optimized_prompt, sample["dialogue"], key))
    print()


if __name__ == "__main__":
    main()
