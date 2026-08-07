"""v2.3 单 prompt 运行器 —— 无需 MultiAgentOrchestrator。

v2.3 是路线 A 单智能体架构：一个 system_prompt 文件驱动完整弹窗生成。
与路线 B（多智能体 v3.1）不同，这里不需要感知层/总控层/生产层的链式调用，
也不需要 tone_rules.py 规则引擎。一次 LLM 调用即可。

v2.3 输出的是自然语言弹窗文本（不是 JSON）。为了让 co-evolution loop
能提取结构化字段（tone/popup_text/contradiction），在 user message 中
追加了 JSON 输出指令。prompt 核心逻辑不做任何修改。

v2.3 的 tone 采用中文标签：
    - "诊断式弹窗"  → diagnostic
    - "鼓励式弹窗"  → empowering
"""

import json
from openai import OpenAI

# v2.3 中文 tone → 英文标准化
_TONE_MAP = {
    "诊断式弹窗": "diagnostic",
    "诊断式": "diagnostic",
    "诊断": "diagnostic",
    "鼓励式弹窗": "empowering",
    "鼓励式": "empowering",
    "鼓励": "empowering",
}

# 追加到 user message 的 JSON 输出指令（不修改 prompt 本身）
_JSON_OUTPUT_INSTRUCTION = """
请输出严格 JSON（不要 markdown 包裹，不要额外解释）：
{"type": "诊断式弹窗" 或 "鼓励式弹窗", "popup_text": "弹窗完整内容", "contradiction": "此场景的核心矛盾（一句话）"}
"""

# v2.6+ 简化输出指令（不要求 type/contradiction，由 judge 从文本中自行判断）
_JSON_OUTPUT_INSTRUCTION_V26 = """
请输出严格 JSON（不要 markdown 包裹，不要额外解释）：
{"popup_text": "弹窗完整内容"}
"""


def _normalize_tone(raw_tone: str) -> str:
    """将 v2.3 的中文 tone 标签标准化为英文。"""
    tone = raw_tone.strip()
    # 精确匹配
    if tone in _TONE_MAP:
        return _TONE_MAP[tone]
    # 模糊匹配（去空格、去"弹窗"后缀）
    tone_lower = tone.lower().replace(" ", "")
    for cn, en in _TONE_MAP.items():
        if cn.replace(" ", "") in tone_lower or tone_lower in cn.replace(" ", ""):
            return en
    # 直接英文匹配
    if tone_lower in ("diagnostic", "empowering"):
        return tone_lower
    return tone  # 保留原始值供调试


def run_v23_once(
    client: OpenAI,
    model: str,
    system_prompt: str,
    dialogue: str,
    temperature: float = 0.3,
    max_tokens: int = 640,
    json_output_instruction: str | None = None,
) -> dict:
    """用 v2.3 system_prompt 生成一次弹窗。

    发送方式与 evaluate_v23.py 的 deepseek() 一致：prompt 作为 system message，
    对话 + JSON 输出指令作为 user message。

    Args:
        json_output_instruction: 覆盖默认的 JSON 输出格式要求。
                                 传 None 使用 v2.3 默认（type+popup_text+contradiction）。
                                 传 _JSON_OUTPUT_INSTRUCTION_V26 用于 v2.6+。

    Returns:
        {
            "tone": "diagnostic" | "empowering" | "",
            "popup_text": "...",
            "contradiction": "...",
            "should_popup": True | False,
            "raw_json": {...},   # 完整 LLM 输出
            "error": ""          # 非空 = 失败
        }
    """
    result = {
        "tone": "",
        "popup_text": "",
        "contradiction": "",
        "should_popup": True,
        "raw_json": {},
        "error": "",
    }

    instruction = json_output_instruction if json_output_instruction is not None else _JSON_OUTPUT_INSTRUCTION
    user_message = dialogue + "\n" + instruction

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=120,
        )
        raw = resp.choices[0].message.content or "{}"
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        return result

    # 解析 JSON（v2.3 输出偶尔包含 markdown 包裹）
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            cleaned = "\n".join(lines)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            result["error"] = f"JSON parse failed: {raw[:200]}"
            return result

    result["raw_json"] = data

    # 提取字段（v2.3 用 "type" 字段而非 "tone"，值是中文标签）
    tone = data.get("type", "") or data.get("tone", "")
    result["tone"] = _normalize_tone(tone)

    result["popup_text"] = (
        data.get("popup_text", "")
        or data.get("popup", "")
        or data.get("content", "")
        or ""
    )
    result["contradiction"] = (
        data.get("contradiction", "")
        or data.get("main_contradiction", "")
        or data.get("core_contradiction", "")
        or ""
    )

    # should_popup: v2.3 通常不显式输出此字段，有 popup_text 即视为应该弹窗
    sp = data.get("should_popup")
    if sp is not None:
        result["should_popup"] = bool(sp)
    else:
        result["should_popup"] = bool(result["popup_text"].strip())

    return result


def read_v23_prompt(path: str | None = None) -> str:
    """读取 v2.3 system_prompt 文件内容。"""
    from pathlib import Path

    if path is None:
        path = Path(__file__).parent.parent / "prompts_archive" / "system_prompt_v2.3.txt"

    return Path(path).read_text(encoding="utf-8")
