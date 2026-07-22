"""
Phase C: 数据集合并与配置更新

1. 从 expert_dataset.json 中提取可训练记录（有 dialogue）
2. 按 case 分组，80/20 随机拆分
3. 转换格式与 dataset_50_questions.json 合并
4. 创建 expert_test.json（带专家评分，用于 Phase E 验证）
5. 输出合并后的训练集和更新后的 config

用法: python scripts/merge_datasets.py
输出:
  - data/expert_train.json       (对话 + 空答案，MIPROv2 训练)
  - data/expert_test.json        (对话 + 专家评分，Phase E 验证)
  - data/dataset_merged_train.json (50条旧 + expert_train 合并)
"""

import json
import os
import random
import hashlib
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERT_PATH = os.path.join(BASE_DIR, "data", "expert_dataset.json")
OLD_DATASET_PATH = os.path.join(BASE_DIR, "dataset_50_questions.json")

OUTPUT_DIR = os.path.join(BASE_DIR, "data")


def case_key(record: dict) -> str:
    """Extract a stable case-level grouping key.

    Multi-window records from the same case share the same dialogue.
    We hash the dialogue to group them.
    """
    dialogue = record.get("dialogue", "")
    if not dialogue:
        return record.get("id", "unknown")
    # Use first 200 chars as grouping key (same dialogue = same case)
    return hashlib.md5(dialogue[:200].encode()).hexdigest()


def main():
    random.seed(42)

    # 1. Load expert dataset
    with open(EXPERT_PATH, "r", encoding="utf-8") as f:
        expert_data = json.load(f)

    # 2. Filter trainable records (have dialogue)
    trainable = [r for r in expert_data if r.get("dialogue")]
    print(f"可训练记录（有 dialogue）: {len(trainable)}/{len(expert_data)}")

    # 3. Group by case
    case_groups = defaultdict(list)
    for r in trainable:
        ck = case_key(r)
        case_groups[ck].append(r)

    unique_cases = list(case_groups.keys())
    print(f"唯一条目（按对话去重）: {len(unique_cases)}")

    # 4. 80/20 split by case
    random.shuffle(unique_cases)
    split_idx = int(len(unique_cases) * 0.8)
    train_cases = set(unique_cases[:split_idx])
    test_cases = set(unique_cases[split_idx:])

    train_records = []
    test_records = []
    for ck, records in case_groups.items():
        if ck in train_cases:
            train_records.extend(records)
        else:
            test_records.extend(records)

    print(f"Train: {len(train_records)} 条窗口级记录 ({len(train_cases)} cases)")
    print(f"Test:  {len(test_records)} 条窗口级记录 ({len(test_cases)} cases)")

    # 5. Convert to MIPROv2 format (question = dialogue, answer = "")
    def to_question_format(records):
        """Convert expert records to {question, answer} format, deduplicating by dialogue."""
        seen = set()
        result = []
        for r in records:
            dhash = hashlib.md5(r["dialogue"].encode()).hexdigest()
            if dhash in seen:
                continue
            seen.add(dhash)
            result.append({
                "question": r["dialogue"],
                "answer": "",  # No golden answer for training
            })
        return result

    expert_train_q = to_question_format(train_records)
    print(f"Expert train (去重后): {len(expert_train_q)} 条")

    # 6. Load old dataset
    with open(OLD_DATASET_PATH, "r", encoding="utf-8") as f:
        old_dataset = json.load(f)
    print(f"旧数据集: {len(old_dataset)} 条")

    # 7. Merge (deduplicate against old dataset by dialogue hash)
    old_hashes = {hashlib.md5(item["question"].encode()).hexdigest() for item in old_dataset}
    new_from_expert = [q for q in expert_train_q
                       if hashlib.md5(q["question"].encode()).hexdigest() not in old_hashes]
    print(f"专家训练集中新增（去重旧数据后）: {len(new_from_expert)} 条")

    merged_train = old_dataset + new_from_expert
    print(f"合并训练集: {len(merged_train)} 条")

    # 8. Save outputs
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # expert_train.json (expert-only training records)
    with open(os.path.join(OUTPUT_DIR, "expert_train.json"), "w", encoding="utf-8") as f:
        json.dump(expert_train_q, f, ensure_ascii=False, indent=2)
    print(f"✓ 已保存: data/expert_train.json ({len(expert_train_q)} 条)")

    # expert_test.json (with expert scores for Phase E)
    # Filter test records to those with expert scores
    test_with_scores = [r for r in test_records if r.get("expert_score") is not None]
    print(f"Expert test (有专家评分): {len(test_with_scores)} 条")

    # Also include test records without scores but with dialogue+popup (for judge eval)
    test_for_eval = []
    seen_test = set()
    for r in test_records:
        dhash = hashlib.md5(r["dialogue"].encode()).hexdigest()
        if dhash in seen_test:
            continue
        seen_test.add(dhash)
        test_for_eval.append({
            "id": r["id"],
            "dialogue": r["dialogue"],
            "system_popup": r.get("system_popup", ""),
            "expert_score": r.get("expert_score"),
            "expert_name": r.get("expert_name"),
            "reference_popup": r.get("reference_popup", ""),
            "case_title": r.get("case_title", ""),
            "hit_checklist": r.get("hit_checklist", []),
            "forbidden_list": r.get("forbidden_list", []),
        })

    with open(os.path.join(OUTPUT_DIR, "expert_test.json"), "w", encoding="utf-8") as f:
        json.dump(test_for_eval, f, ensure_ascii=False, indent=2)
    print(f"✓ 已保存: data/expert_test.json ({len(test_for_eval)} 条)")

    # dataset_merged_train.json (50 old + expert new)
    merged_path = os.path.join(OUTPUT_DIR, "dataset_merged_train.json")
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(merged_train, f, ensure_ascii=False, indent=2)
    print(f"✓ 已保存: data/dataset_merged_train.json ({len(merged_train)} 条)")

    # 9. Summary
    print(f"\n{'='*50}")
    print(f"Phase C 合并完成")
    print(f"{'='*50}")
    print(f"  旧数据集:         {len(old_dataset)} 条")
    print(f"  专家训练集:       {len(expert_train_q)} 条（去重后）")
    print(f"  合并训练集:       {len(merged_train)} 条")
    print(f"  专家测试集:       {len(test_for_eval)} 条")
    print(f"  其中含专家评分:   {len(test_with_scores)} 条")
    print(f"\n  测试集评分分布:   {sorted([r['expert_score'] for r in test_for_eval if r['expert_score'] is not None])}")


if __name__ == "__main__":
    main()
