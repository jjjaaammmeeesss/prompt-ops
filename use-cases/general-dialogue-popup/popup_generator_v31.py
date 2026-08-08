# @persistent — 一般场景弹窗 v3.1 专用生成器（预分析解析）
"""一般场景弹窗 v3.1 生成器：解析 LLM 输出的预分析块（`==========` 为界），剥离元信息，返回纯弹窗正文。

适配 system_prompt_v3.1.txt 的预分析流程（元信息 / 关键句归属 / 错别字，无类型项）。

从 v2.0 执行器 popup_generator.py 复制剥离逻辑起步，作为 v3.1 独立演进载体：
- v3.1 与 v2.0 的预分析段当前完全一致，剥离逻辑完全相同
- 之所以单独建文件：v3.1 未来若预分析格式或规则分化和 v2.0 不同，可在本文件独立演进，不污染 v2.0 执行器
- 剥离逻辑：`==========` partition + 按行剥离预分析元信息行（兜底防泄露）+ 前缀清理 + 安静判定

核心接口：
    parse_popup_output(text) -> str | None    # 剥离预分析，返回纯正文（安静返回 None）
    generate_popup(system_prompt, dialogue)   # 调 LLM 生成 → 解析 → 返回纯正文（安静返回 None）

__version__ 自增记录本执行器迭代；PROMPT_VERSION 声明它适配的 prompt 版本，
与 use-cases/general-dialogue-popup/system_prompt_v3.1.txt 保持双向对齐。
"""

from __future__ import annotations

__version__ = "1.0"           # v3.1 执行器独立版本号（起点）
PROMPT_VERSION = "v3.1"       # 适配的 prompt 版本（= system_prompt_v3.1.txt）

import os
import re

# 预分析元信息行（1.元信息 / 2.关键句归属 / 3.错别字）。逐行匹配，
# 用于即使 LLM 漏掉 `==========` 分隔符也兜底剥离，防止元信息泄露进弹窗正文。
# （亲子版含"类型"，v3.1 去掉类型项，与 v2.0 相同。）
_PRE_ANALYSIS_LINE_RE = re.compile(
    r"^\s*\d\.\s*(元信息|关键句归属|错别字)\s*[:：]"
)
_META_SEPARATOR_LINES = {"==========", "---", "--"}


def _strip_meta_lines(text: str) -> str:
    """按行删除预分析元信息行与分隔符行，返回纯弹窗正文。

    与 `==========` split 配合构成双重保险：即使 LLM 漏输出分隔符，
    元信息也不会残留进弹窗正文。
    """
    if not text:
        return ""
    kept = []
    for line in text.splitlines():
        if _PRE_ANALYSIS_LINE_RE.match(line):
            continue
        if line.strip() in _META_SEPARATOR_LINES:
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def _strip_label(text: str, label: str) -> str:
    """去掉正文行首的「弹窗：」等标签，保留内容。"""
    t = text.strip()
    for sep in (f"{label}：", f"{label}:"):
        if t.startswith(sep):
            return t[len(sep):].strip()
    return t


# 前缀清理：去掉可能的 "弹窗：" / "输出：" 等前缀
_PREFIXES = ["弹窗：", "弹窗:", "【弹窗】", "输出：", "正文："]


def parse_popup_output(text: str) -> str | None:
    """剥离预分析块，返回纯弹窗正文；判断安静时返回 None。

    安静信号（v3.1 规则）：输出"安好"两个字时表示决定不弹窗。
    """
    if not text:
        return None
    text = text.strip()

    # 安静：整体就是"安好"
    if text == "安好":
        return None

    # 以 ========== 为界，前段为预分析、后段为正文
    if "==========" in text:
        _, _, body = text.partition("==========")
    else:
        body = text  # 无分隔符：整段当正文，交给 _strip_meta_lines 兜底

    # 兜底剥离残留的预分析元信息行
    body = _strip_meta_lines(body)

    # 清理前缀标签
    body = _strip_label(body, "弹窗")
    for prefix in _PREFIXES:
        if body.startswith(prefix):
            body = body[len(prefix):].strip()

    body = body.strip()

    # 正文也是安静信号（LLM 在预分析后仍决定不弹）
    if not body or body == "安好" or len(body) <= 10:
        return None
    return body


def generate_popup(
    system_prompt: str,
    dialogue: str,
    model: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> str | None:
    """调用 LLM 生成弹窗 → 解析预分析 → 返回纯正文（安静返回 None）。

    模型/凭据默认从环境变量读：GEN_MODEL / GEN_API_KEY(DEEPSEEK_API_KEY) / GEN_API_BASE。
    """
    import litellm
    import time

    model = model or os.environ.get("GEN_MODEL", "deepseek/deepseek-v4-pro")
    api_key = api_key or os.environ.get("GEN_API_KEY") or os.environ.get("DEEPSEEK_API_KEY", "")
    api_base = api_base or os.environ.get("GEN_API_BASE", "https://api.deepseek.com/v1")

    kwargs: dict = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"当前对话：\n{dialogue}"},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=180,
    )
    if api_base:
        kwargs["api_base"] = api_base
    if api_key:
        kwargs["api_key"] = api_key

    for attempt in range(3):
        try:
            resp = litellm.completion(**kwargs)
            raw = (resp.choices[0].message.content or "").strip()
            return parse_popup_output(raw)
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    return None
