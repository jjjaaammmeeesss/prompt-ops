#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compare baseline, PDO-optimized, and MIPROv2-optimized popups."""
import json
import os
import re
import sys


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


def load_latest_result(base_dir: str, prefix: str) -> str:
    result_dir = os.path.join(base_dir, "results")
    files = sorted(
        [p for p in os.listdir(result_dir) if p.endswith(".json") and p.startswith(prefix)],
        reverse=True,
    )
    if not files:
        raise RuntimeError(f"No result found for prefix {prefix}")
    with open(os.path.join(result_dir, files[0]), "r", encoding="utf-8") as f:
        return json.load(f)["prompt"]


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    key = load_key("D:/ob-new202603/钥匙库/DeepSeek公司key.md")

    with open(os.path.join(base_dir, "system_prompt.txt"), "r", encoding="utf-8") as f:
        baseline = f.read()
    pdo = load_latest_result(base_dir, "config_pdo_")
    mipro = load_latest_result(base_dir, "config_mipro_")
    with open(os.path.join(base_dir, "system_prompt_v2.2.txt"), "r", encoding="utf-8") as f:
        merged = f.read()

    with open(os.path.join(base_dir, "dataset.json"), "r", encoding="utf-8") as f:
        dataset = json.load(f)

    for sample_id in ["manager_anger", "couple_chase_dodge", "friend_soft_boundary"]:
        sample = next(d for d in dataset if d["id"] == sample_id)
        print("\n" + "=" * 70)
        print(f"SCENE: {sample['id']} | {sample.get('title', '')}")
        print(sample["dialogue"])
        for name, prompt in [("BASELINE", baseline), ("PDO v2.1", pdo), ("MIPROv2", mipro), ("MERGED v2.2", merged)]:
            print("-" * 70)
            print(f"[{name}]:")
            print(call_deepseek(prompt, sample["dialogue"], key))
        print("=" * 70)


if __name__ == "__main__":
    main()
