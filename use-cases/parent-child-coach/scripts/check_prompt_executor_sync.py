#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""prompt ↔ 执行器 双向同步一致性检查器。

背景（见 CLAUDE.md「双向同步元规则」）：更新 prompt 必须同步检查执行器，
反之亦然。本脚本把这条元规则变成可自动验证的闸门：

  A. 生产 prompt 版本（权威源 = realtime/config.yaml 的 generator.system_prompt_path）
  B. 执行器版本号声明：__version__ + PROMPT_VERSION（适配的 prompt 版本）
  C. 生产链单一引用（config / 脚本默认 / 硬编码）必须指向生产版本
  D. 多版本能力（run_v418 的 PROMPT_MAP、popup_generator 的回退链）必须含生产版本
  E. 所有被引用的 prompt 文件必须真实存在

用法:
  python scripts/check_prompt_executor_sync.py            # 报告 + 有 FAIL 时 exit 1
  python scripts/check_prompt_executor_sync.py --strict   # 任一 WARN 也 exit 1

配套闸门（每次跑完执行器后必跑，互补）:
  # 输入构造奇偶 + 版本三方对齐（执行器 __version__ / PROMPT_VERSION / config.yaml）
  python scripts/check_executor_input_parity.py --json <pipeline_output.json>

  # 本脚本查"prompt↔执行器版本号同步"；parity 脚本查"执行器喂给生产的输入是否
  # 与生产 TextBuffer 相对滑动奇偶一致"（防绝对前900从0累积导致的输入趋同）。
  # 两者独立、互补，回归应同时跑。

对比脚本（compare_v4012_v4019.py 等）设计上就引用多个历史版本做对比，
不属于生产链，不在本检查范围内。
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent

PROMPT_VERSION_RE = __import__("re").compile(r"system_prompt_v([\d.]+)\.txt")
EXECUTOR_VERSION_RE = __import__("re").compile(r'__version__\s*=\s*["\']([^"\']+)["\']')
PROMPT_VER_CONST_RE = __import__("re").compile(r'PROMPT_VERSION\s*=\s*["\']([^"\']+)["\']')
_MAIN_DEFAULT_PROMPT_RE = __import__("re").compile(
    r'parser\.add_argument\("--prompt".*?default="(v[\d.]+)"',
    __import__("re").DOTALL)


def _versions(text: str) -> list[str]:
    seen, out = set(), []
    for m in PROMPT_VERSION_RE.finditer(text):
        v = m.group(1)
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _yaml_version(rel: str) -> str | None:
    """从 yaml 提取 prompt 文件引用的版本号（支持 system_prompt.file 与
    generator.system_prompt_path 两种结构）。"""
    p = PROJECT / rel
    if not p.exists():
        return None
    try:
        import yaml
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    cand = None
    for section, key in (("system_prompt", "file"),
                         ("generator", "system_prompt_path"),
                         ("popup", "system_prompt_path")):
        sec = data.get(section) if isinstance(data, dict) else None
        if isinstance(sec, dict) and sec.get(key):
            cand = sec[key]
            break
    if not cand:
        return None
    m = PROMPT_VERSION_RE.search(str(cand))
    return m.group(1) if m else None


def main() -> int:
    strict = "--strict" in sys.argv
    fails, warns, passes = [], [], []

    # A) 生产 prompt 版本（权威源）
    prod = _yaml_version("realtime/config.yaml")
    if not prod:
        fails.append("无法从 realtime/config.yaml 提取生产 prompt 版本")
        prod = ""
    else:
        passes.append(f"生产 prompt 版本（权威源）: v{prod}")
        if not (PROJECT / f"system_prompt_v{prod}.txt").exists():
            fails.append(f"生产 prompt 文件不存在: system_prompt_v{prod}.txt")

    # B) 执行器版本号声明
    for rel, label in (("realtime/popup_generator.py", "生产执行器 popup_generator"),
                       ("scripts/run_v4019_pipeline.py", "测试执行器 run_v4019")):
        p = PROJECT / rel
        if not p.exists():
            fails.append(f"执行器缺失: {rel}")
            continue
        text = p.read_text(encoding="utf-8")
        ver = EXECUTOR_VERSION_RE.search(text)
        pv = PROMPT_VER_CONST_RE.search(text)
        if ver:
            passes.append(f"{label}: __version__ = {ver.group(1)}")
        else:
            fails.append(f"{label}: 缺 __version__（{rel}）")
        if pv:
            ok = pv.group(1).lstrip("v") == prod
            (passes if ok else fails).append(
                f"{label}: PROMPT_VERSION = {pv.group(1)}"
                + ("" if ok else f" ≠ 生产 v{prod}")
            )
        else:
            fails.append(f"{label}: 缺 PROMPT_VERSION 声明（{rel}）")
        # 测试执行器 main() 的 --prompt 默认值也必须 == 生产版本（防不带参运行加载旧版）
        if "run_v4019" in rel:
            m = _MAIN_DEFAULT_PROMPT_RE.search(text)
            if m:
                ok = m.group(1).lstrip("v") == prod
                (passes if ok else fails).append(
                    f"{label}: main --prompt 默认值 = {m.group(1)}"
                    + ("" if ok else f" ≠ 生产 v{prod}（不带参运行会加载旧版 prompt）")
                )
            else:
                fails.append(f"{label}: 未找到 main() 的 --prompt 默认值（无法校验）")

    # C) 生产链单一引用：必须仅指向生产版本
    single_refs = [
        ("config.yaml", "use-case 主配置"),
        ("realtime/config.yaml", "生产配置"),
        ("scripts/blind_test_50.py", "盲测脚本默认"),
        ("scripts/adversarial_leakage.py", "对抗测试脚本"),
    ]
    for rel, label in single_refs:
        p = PROJECT / rel
        if not p.exists():
            continue
        vs = _versions(p.read_text(encoding="utf-8", errors="ignore"))
        if not vs:
            continue
        for v in vs:
            (passes if v == prod else fails).append(
                f"{label}: 引用 v{v}" + (" ✓" if v == prod else f"（生产 v{prod}）@ {rel}")
            )

    # D) 多版本能力：必须含生产版本（允许列历史版做回退/对比）
    multi_refs = [
        ("scripts/run_v4019_pipeline.py", "测试执行器 PROMPT_MAP"),
        ("realtime/popup_generator.py", "生产执行器回退链"),
    ]
    for rel, label in multi_refs:
        p = PROJECT / rel
        if not p.exists():
            continue
        vs = _versions(p.read_text(encoding="utf-8", errors="ignore"))
        if not vs:
            fails.append(f"{label}: 未引用任何 prompt 版本（{rel}）")
            continue
        if prod in vs:
            passes.append(f"{label}: 含生产 v{prod} ✓（另含 {[f'v{x}' for x in vs if x != prod]}）")
        else:
            fails.append(f"{label}: 缺少生产 v{prod}（当前引用 {[f'v{x}' for x in vs]}）@ {rel}")

    # E) 存在性：所有被引用文件必须真实存在
    all_refs = set()
    for p in PROJECT.rglob("*"):
        if p.suffix in (".py", ".yaml", ".md", ".txt") and "prompts_archive" not in str(p):
            try:
                all_refs |= set(_versions(p.read_text(encoding="utf-8", errors="ignore")))
            except Exception:
                pass
    for v in sorted(all_refs):
        if not (PROJECT / f"system_prompt_v{v}.txt").exists():
            warns.append(f"引用了不存在的 prompt 文件: system_prompt_v{v}.txt")

    # 报告
    print("=" * 62)
    print("prompt ↔ 执行器 双向同步一致性检查")
    print("=" * 62)
    for item in passes:
        print(f"  [PASS] {item}")
    for item in warns:
        print(f"  [WARN] {item}")
    for item in fails:
        print(f"  [FAIL] {item}")
    print("-" * 62)
    print(f"PASS {len(passes)} | WARN {len(warns)} | FAIL {len(fails)}")
    print("环境一致 ✓ prompt 与执行器同步" if not fails else
          "存在不一致，请按 CLAUDE.md「双向同步元规则」修复后重跑。")

    if fails:
        return 1
    if strict and warns:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
