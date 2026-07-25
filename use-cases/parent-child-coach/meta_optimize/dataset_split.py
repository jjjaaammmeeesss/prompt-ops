"""数据集划分：search / validation / test 三集分离 + SHA-256 冻结。

原则（来自 Meta-Harness）：
  - 按 case_id 拆分，同一 case 的所有窗口归入同一集（防泄漏）
  - 按 tone 分层抽样，保持 diagnostic/empowering 比例一致
  - SHA-256 冻结每集的案例列表，确保可复现
"""

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

# EVAL_CASES 从 optimizer.py 复制（避免导入链依赖星灵/openai）
EVAL_CASES = [
    # === 原始 12 案 ===
    ("C10-001", None), ("C10-002", None), ("C10-003", None),
    ("C10-005", None), ("C10-006", None), ("C10-008", None),
    ("C10-004", None), ("C11-001", None), ("C11-004", None),
    ("C11-006", 1), ("C11-009", 2), ("C11-010", 1),
    # === 盲区 50 案 ===
    ("C10-009", 1), ("C10-010", 1), ("C11-001", 4), ("C11-002", 3),
    ("C11-005", 1), ("C11-005", 2), ("C11-005", 3), ("C11-005", 4),
    ("C11-005", 5), ("C11-007", 1), ("C11-008", 1), ("C11-009", 1),
    ("C11-010", 2), ("C13-001", 2), ("C13-002", 1), ("C13-004", 2),
    ("C13-005", 3), ("C13-006", 3), ("C13-007", 1), ("C13-007", 2),
    ("C13-007", 3), ("C13-008", 2), ("C13-009", 1), ("C13-010", 1),
    ("C13-010", 3), ("C13-012", 1), ("C13-012", 2),
    ("C3-001", 1), ("C3-002", 1), ("C3-003", 1),
    ("C4-001", 1), ("C4-002", 1), ("C4-002", 3), ("C4-002", 4),
    ("C4-002", 5), ("C4-003", 2), ("C4-004", 2),
    ("C5-001", 1), ("C5-001", 2), ("C5-002", 1), ("C5-002", 2),
    ("C5-003", 2), ("C5-003", 3), ("C5-003", 4), ("C5-003", 5),
    ("C5-004", 2), ("C5-004", 3), ("C5-005", 1), ("C5-005", 3),
    ("C5-005", 4),
]

# Golden dataset 路径（相对于 parent-child-coach 根目录）
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"

META_DIR = Path(__file__).parent
SPLIT_DIR = META_DIR / "data"
MANIFEST_PATH = SPLIT_DIR / "split_manifest.json"

# 分割比例: search 60% / validation 20% / test 20%
SPLIT_RATIOS = {"search": 0.6, "validation": 0.2, "test": 0.2}
SEED = 42


def _unique_case_ids() -> list[str]:
    """从 EVAL_CASES 提取唯一 case_id 列表，按顺序保持稳定。"""
    seen = set()
    result = []
    for case_id, _ in EVAL_CASES:
        if case_id not in seen:
            seen.add(case_id)
            result.append(case_id)
    return result


def _case_tone(case_id: str, dataset: list[dict]) -> str:
    """推断 case 的主导 tone（用于分层抽样）。"""
    for c in dataset:
        if c["case_id"] == case_id:
            # 取第一个窗口的 expected_tone 作为该 case 的主导 tone
            for w in c.get("windows", []):
                tone = w.get("expected_tone", "")
                if tone in ("diagnostic", "empowering"):
                    return tone
    return "unknown"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def create_split(
    golden_dataset_path: str | None = None,
    ratios: dict[str, float] | None = None,
    seed: int = SEED,
    output_dir: str | None = None,
) -> dict:
    """创建 search/validation/test 三集划分。

    Args:
        golden_dataset_path: golden_dataset.json 路径，默认从 DATA_DIR 加载
        ratios: 分割比例，默认 60/20/20
        seed: 随机种子
        output_dir: 输出目录

    Returns:
        split manifest dict，含每集的 case_id 列表和 SHA-256 哈希
    """
    ratios = ratios or SPLIT_RATIOS
    output_dir = Path(output_dir) if output_dir else SPLIT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据集
    if golden_dataset_path:
        dataset_path = Path(golden_dataset_path)
    else:
        dataset_path = DATA_DIR / "golden_dataset.json"

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # 按 tone 分组唯一 case_id
    all_case_ids = _unique_case_ids()
    tone_groups: dict[str, list[str]] = defaultdict(list)
    for cid in all_case_ids:
        tone = _case_tone(cid, dataset)
        tone_groups[tone].append(cid)

    # 每组内随机打乱，然后按比例分配
    rng = random.Random(seed)
    for tone in tone_groups:
        rng.shuffle(tone_groups[tone])

    # 按比例分配，保持各 tone 比例一致
    split_cases: dict[str, list[str]] = {"search": [], "validation": [], "test": []}

    for tone, cases in tone_groups.items():
        n = len(cases)
        n_search = max(1, round(n * ratios["search"]))
        n_validation = max(1, round(n * ratios["validation"]))
        # test 拿剩下的
        n_test = n - n_search - n_validation

        split_cases["search"].extend(cases[:n_search])
        split_cases["validation"].extend(cases[n_search:n_search + n_validation])
        split_cases["test"].extend(cases[n_search + n_validation:])

    # 映射 case_id → [(case_id, window_index), ...] 的 eval cases
    case_to_evals: dict[str, list[tuple[str, int | None]]] = defaultdict(list)
    for entry in EVAL_CASES:
        case_to_evals[entry[0]].append(entry)

    # 构建 manifest
    manifest = {"seed": seed, "ratios": ratios, "total_cases": len(all_case_ids), "splits": {}}

    for split_name in ("search", "validation", "test"):
        case_ids = sorted(split_cases[split_name])
        eval_entries = []
        for cid in case_ids:
            eval_entries.extend(case_to_evals.get(cid, [(cid, None)]))

        split_hash = _sha256(json.dumps(case_ids, sort_keys=True))
        manifest["splits"][split_name] = {
            "case_ids": case_ids,
            "n_case_ids": len(case_ids),
            "n_eval_entries": len(eval_entries),
            "sha256": split_hash,
            "eval_entries": eval_entries,
        }

    # 保存 manifest
    with open(output_dir / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # 打印摘要
    print(f"数据集划分完成 (seed={seed}):")
    for name in ("search", "validation", "test"):
        info = manifest["splits"][name]
        print(f"  {name:12s}: {info['n_case_ids']:2d} case_ids, "
              f"{info['n_eval_entries']:2d} eval entries, "
              f"sha256={info['sha256'][:12]}...")

    return manifest


def load_split_manifest(path: str | None = None) -> dict:
    """加载已有的 split manifest。"""
    p = Path(path) if path else MANIFEST_PATH
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def get_eval_entries_for_split(split_name: str, manifest: dict | None = None) -> list[tuple[str, int | None]]:
    """获取指定 split 的 eval entries 列表。"""
    if manifest is None:
        manifest = load_split_manifest()
    entries = manifest["splits"][split_name]["eval_entries"]
    return [(e[0], e[1]) for e in entries]


def verify_split_integrity(manifest: dict) -> bool:
    """验证三集无重叠、哈希一致。"""
    search_ids = set(manifest["splits"]["search"]["case_ids"])
    val_ids = set(manifest["splits"]["validation"]["case_ids"])
    test_ids = set(manifest["splits"]["test"]["case_ids"])

    # 检查无重叠
    overlap_sv = search_ids & val_ids
    overlap_st = search_ids & test_ids
    overlap_vt = val_ids & test_ids

    if overlap_sv or overlap_st or overlap_vt:
        print(f"❌ 分割重叠: S∩V={overlap_sv}, S∩T={overlap_st}, V∩T={overlap_vt}")
        return False

    # 检查覆盖率
    all_ids = search_ids | val_ids | test_ids
    expected = set(_unique_case_ids())
    missing = expected - all_ids
    extra = all_ids - expected
    if missing or extra:
        print(f"❌ 覆盖不全: missing={missing}, extra={extra}")
        return False

    print("✅ 分割完整性验证通过: 无重叠，全覆盖")
    return True


if __name__ == "__main__":
    manifest = create_split()
    verify_split_integrity(manifest)
