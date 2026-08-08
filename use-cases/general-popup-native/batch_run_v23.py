"""ty2.4 全量 60 题批量执行。

读取 multica 工作目录下的 60 个对话文件，用 system_prompt_ty2.4.txt
逐条弹窗，输出 checkpoint + 报告。

Usage:
    python batch_run_v23.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results" / "batch"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CASES_DIR = Path(
    r"C:\Users\h\multica_workspaces_desktop-api.multica.ai"
    r"\a3d95521-baf9-4931-a927-08db47f604b5"
    r"\d9234763\workdir\cases\职场办公第二批-7.31-50条"
)

GEN_MODEL = os.environ.get("GEN_MODEL", "deepseek/deepseek-chat")
GEN_API_BASE = os.environ.get("GEN_API_BASE", "https://api.deepseek.com/v1")
GEN_API_KEY = os.environ.get("GEN_API_KEY", os.environ.get("DEEPSEEK_API_KEY", ""))
_TEMPERATURE = float(os.environ.get("GEN_TEMPERATURE", "0.3"))

CHECKPOINT_PATH = RESULTS_DIR / "checkpoint_ty24.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("batch_v23")


def load_cases() -> list[dict]:
    """加载所有 60 个对话文件"""
    cases = []
    for scene_dir in sorted(CASES_DIR.iterdir()):
        if not scene_dir.is_dir():
            continue
        for txt_file in sorted(scene_dir.glob("*.txt")):
            raw = txt_file.read_text(encoding="utf-8")
            lines = raw.strip().split("\n")
            # 去掉 # 开头的标题行
            dialogue_lines = [l for l in lines if not l.startswith("#")]
            dialogue = "\n".join(dialogue_lines).strip()
            if not dialogue:
                continue
            case_id = f"{scene_dir.name}/{txt_file.stem}"
            cases.append({
                "case_id": case_id,
                "scene": scene_dir.name,
                "source_file": txt_file.name,
                "dialogue": dialogue,
            })
    return cases


def load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        return json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    return {"results": {}}


def save_checkpoint(data: dict):
    CHECKPOINT_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_popup(system_prompt: str, dialogue: str) -> str | None:
    import litellm

    user_msg = f"对话：\n{dialogue}"
    kwargs: dict = dict(
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=_TEMPERATURE,
        max_tokens=4096,
        timeout=180,
    )
    if GEN_API_BASE:
        kwargs["api_base"] = GEN_API_BASE
    if GEN_API_KEY:
        kwargs["api_key"] = GEN_API_KEY

    for attempt in range(3):
        try:
            resp = litellm.completion(**kwargs)
            text = (resp.choices[0].message.content or "").strip()
            break
        except Exception as e:
            logger.warning("生成尝试 %d/3 失败: %s", attempt + 1, e)
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)

    if not text or len(text) <= 10:
        return None
    if text.strip() == "安好":
        return None
    for prefix in ["弹窗：", "弹窗:", "【弹窗】", "输出："]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return text


def main():
    if not GEN_API_KEY:
        logger.error("缺少生成模型 API key（设置 KIMI_API_KEY 或 GEN_API_KEY）")
        sys.exit(1)

    prompt_path = HERE / "system_prompt_ty2.4.txt"
    if not prompt_path.exists():
        logger.error("找不到 ty2.4 prompt: %s", prompt_path)
        sys.exit(1)
    system_prompt = prompt_path.read_text(encoding="utf-8")

    cases = load_cases()
    logger.info("加载 %d 个对话用例 (%s)", len(cases), CASES_DIR)

    checkpoint = load_checkpoint()
    completed = set(checkpoint["results"].keys())

    logger.info("已缓存 %d 条，需执行 %d 条", len(completed), len(cases) - len(completed))

    for i, case in enumerate(cases):
        case_id = case["case_id"]
        if case_id in completed:
            continue

        logger.info("[%d/%d] %s", i + 1, len(cases), case_id)

        try:
            popup = generate_popup(system_prompt, case["dialogue"])
        except Exception as e:
            logger.error("  生成失败: %s", e)
            popup = None
            error = str(e)
        else:
            error = None

        wc = len(popup) if popup else 0
        has_ni = "你" in popup if popup else False
        over_140 = wc > 140
        over_180 = wc > 180

        checkpoint["results"][case_id] = {
            "case_id": case_id,
            "scene": case["scene"],
            "source_file": case["source_file"],
            "prompt_version": "ty2.4",
            "dialogue": case["dialogue"],
            "popup": popup,
            "wc": wc,
            "has_ni": has_ni,
            "over_140": over_140,
            "over_180": over_180,
            "error": error,
            "executed_at": datetime.now(timezone.utc).isoformat(),
        }
        save_checkpoint(checkpoint)

        status = "安好" if popup is None else f"{wc}字"
        flags = []
        if has_ni: flags.append("含'你'")
        if over_180: flags.append(f">{180}")
        elif over_140: flags.append(f">{140}")
        flag_str = f"  ⚠ {', '.join(flags)}" if flags else ""
        logger.info("  → %s%s", status, flag_str)

    # ── 汇总 ──
    results = checkpoint["results"]
    total = len(results)
    popups = [r for r in results.values() if r.get("popup")]
    silent = total - len(popups)
    ni_count = sum(1 for r in results.values() if r.get("has_ni"))
    over_180_count = sum(1 for r in results.values() if r.get("over_180"))
    over_140_count = sum(1 for r in results.values() if r.get("over_140"))
    wcs = [r["wc"] for r in results.values() if r.get("wc") and r["wc"] > 0]

    summary = {
        "prompt_version": "ty2.4",
        "gen_model": GEN_MODEL,
        "total": total,
        "popups": len(popups),
        "silent": silent,
        "has_ni": ni_count,
        "ni_rate": round(ni_count / total * 100, 1) if total else 0,
        "over_140": over_140_count,
        "over_140_rate": round(over_140_count / total * 100, 1) if total else 0,
        "over_180": over_180_count,
        "over_180_rate": round(over_180_count / total * 100, 1) if total else 0,
        "wc_min": min(wcs) if wcs else 0,
        "wc_max": max(wcs) if wcs else 0,
        "wc_avg": round(sum(wcs) / len(wcs), 1) if wcs else 0,
    }

    print("\n" + "=" * 60)
    print("  ty2.4 全量 60 题结果")
    print("=" * 60)
    print(f"  弹窗: {summary['popups']}/{total}  |  沉默(安好): {summary['silent']}/{total}")
    print(f"  含'你': {ni_count}/{total} ({summary['ni_rate']}%)")
    print(f"  超 140 字: {over_140_count}/{total} ({summary['over_140_rate']}%)")
    print(f"  超 180 字: {over_180_count}/{total} ({summary['over_180_rate']}%)")
    print(f"  字数: min={summary['wc_min']}  max={summary['wc_max']}  avg={summary['wc_avg']}")
    print(f"\n  checkpoint: {CHECKPOINT_PATH}")
    print("=" * 60)

    # 保存汇总报告
    report_path = RESULTS_DIR / f"batch_v23_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("汇总报告: %s", report_path)


if __name__ == "__main__":
    main()
