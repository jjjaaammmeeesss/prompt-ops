#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compare merged v2.2 vs PDO-optimized v2.3."""
import json
import os
import re


def load_key(path: str) -> str:
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
    key = load_key("D:/ob-new202603/钥匙库/DeepSeek公司key.md")

    with open(os.path.join(base_dir, "system_prompt_v2.2.txt"), "r", encoding="utf-8") as f:
        v22 = f.read()
    with open(os.path.join(base_dir, "system_prompt_v2.3_pdo.txt"), "r", encoding="utf-8") as f:
        v23 = f.read()

    with open(os.path.join(base_dir, "dataset.json"), "r", encoding="utf-8") as f:
        dataset = json.load(f)

    for sample_id in ["manager_anger", "couple_chase_dodge", "friend_soft_boundary"]:
        sample = next(d for d in dataset if d["id"] == sample_id)
        print("\n" + "=" * 70)
        print(f"SCENE: {sample['id']} | {sample.get('title', '')}")
        print(sample["dialogue"])
        for name, prompt in [("v2.2 MERGED", v22), ("v2.3 PDO", v23)]:
            print("-" * 70)
            print(f"[{name}]:")
            print(call_deepseek(prompt, sample["dialogue"], key))
        print("=" * 70)


if __name__ == "__main__":
    main()
