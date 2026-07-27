"""对比 system_prompt_v1.4 与 v1.5 在 20 个案例上的表现。

输出：results/compare_v14_v15.json
"""

from __future__ import annotations

import json
from pathlib import Path

from runner_v10 import V10Runner

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_cases() -> list[dict]:
    cases = json.loads((HERE / "data" / "test_dialogues.json").read_text(encoding="utf-8"))
    business = json.loads((HERE / "data" / "business_dialogues.json").read_text(encoding="utf-8"))
    # 合并，取前 20 个
    merged = cases + business
    return merged[:20]


def run_version(version: str, cases: list[dict]) -> list[dict]:
    runner = V10Runner(
        prompt_path=f"system_prompt_{version}.txt",
        model="anthropic/claude-opus-4-7",
        api_base="https://luanapi.xingluan.cn",
        api_key="06131c3816c4483c8a4da408102d52e3",
        window_size=5000,
    )
    results = []
    for c in cases:
        out = runner.run(c["dialogue"])
        popups = out["popups"]
        popup_texts = [p["text"] for p in popups]
        has_anhao = any("安好" in t for t in popup_texts)
        # 简单硬规则检查
        violations = []
        for t in popup_texts:
            n = len(t.replace(" ", "").replace("\n", ""))
            if n > 200:
                violations.append(f"字数{n}>200")
            elif n > 180:
                violations.append(f"字数{n}>180")
            elif n < 60 and n > 2:
                violations.append(f"字数{n}<60")
            if "——" not in t:
                violations.append("缺少——")
        results.append({
            "case_id": c["id"],
            "expect": c.get("expect", ""),
            "n_popups": len(popups),
            "popup_texts": popup_texts,
            "has_anhao": has_anhao,
            "violations": violations,
        })
    return results


def main():
    cases = load_cases()
    print(f"加载 {len(cases)} 个案例")

    report = {
        "cases": [{"id": c["id"], "expect": c.get("expect", ""), "title": c.get("title", "")} for c in cases],
        "v1.4": run_version("v1.4", cases),
        "v1.5": run_version("v1.5", cases),
    }

    # 统计
    for version in ["v1.4", "v1.5"]:
        rows = report[version]
        n_popups = sum(r["n_popups"] for r in rows)
        quiet = [r for r in rows if r["expect"] == "安静"]
        quiet_misfires = [r for r in quiet if r["n_popups"] > 0]
        missing = [r for r in rows if r["expect"] != "安静" and r["n_popups"] == 0]
        anhao_in_popup = [r for r in rows if r["has_anhao"]]
        hard_violations = sum(len(r["violations"]) for r in rows)
        print(f"\n=== {version} ===")
        print(f"总弹窗数: {n_popups}")
        print(f"安静案例误弹: {len(quiet_misfires)} / {len(quiet)}")
        print(f"应弹未弹: {len(missing)}")
        print(f"弹窗含'安好': {len(anhao_in_popup)}")
        print(f"硬规则违规数: {hard_violations}")

    (RESULTS_DIR / "compare_v14_v15.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n报告已保存: {RESULTS_DIR / 'compare_v14_v15.json'}")


if __name__ == "__main__":
    main()
