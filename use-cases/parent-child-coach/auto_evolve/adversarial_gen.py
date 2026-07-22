"""对抗测试用例生成器 —— 针对 v2.3 已知失败模式自动生成测试对话。

用法:
    from auto_evolve.adversarial_gen import generate_adversarial_cases

    cases = generate_adversarial_cases(
        judge_client=claude_client,
        judge_model="claude-opus-4-7",
        failure_profile=[{"mode": "tone_blindspot", ...}],
        existing_dialogues=["已有对话1", ...],
        n_per_mode=3,
    )
"""

import json
import hashlib
from pathlib import Path
from anthropic import Anthropic

# ═══════════════════════════════════════════════════════════════
# 对抗用例生成 Prompt
# ═══════════════════════════════════════════════════════════════

_GEN_SYSTEM = """你是亲子沟通领域的测试专家，精通儿童发展心理学和家庭教育。
你的任务是生成能触发 AI 弹窗系统**特定失败模式**的对抗测试对话。

## 核心原则
1. **自然真实**：对话要像真实家庭发生的互动，不要教科书化
2. **精准触发**：对话模式必须能触发目标失败模式
3. **场景多样**：年龄、性别、家庭结构、互动类型都需要变化
4. **边界清晰**：对话 8-15 轮，含关键触发时刻

## 输出格式
严格 JSON：
{
  "cases": [
    {
      "dialogue": "完整对话文本",
      "expected_tone": "diagnostic 或 empowering",
      "target_failure": "失败模式名称",
      "difficulty": "easy 或 medium 或 hard",
      "reference_popup": "简短参考弹窗（30-100字）",
      "why_it_triggers": "为什么这个对话会触发目标失败（一句话）"
    }
  ]
}"""


def _build_gen_prompt(
    failure_mode: str,
    failure_description: str,
    existing_examples: list[str],
    n: int,
) -> str:
    """构建对抗用例生成的 user prompt。"""
    examples_text = ""
    if existing_examples:
        examples_text = "\n".join(f"- {ex[:120]}..." for ex in existing_examples[:5])

    return f"""## 目标失败模式
- 模式名称: {failure_mode}
- 失败描述: {failure_description}

## 已有案例（不要重复类似场景）
{examples_text or "（无已有案例）"}

## 任务
请生成 {n} 个能触发「{failure_mode}」失败模式的新亲子对话。"""


# ═══════════════════════════════════════════════════════════════
# 失败模式定义（来自 v2.3 审计文档）
# ═══════════════════════════════════════════════════════════════

FAILURE_MODES = {
    "tone_blindspot_diagnostic_bias": {
        "name": "Tone 盲区：系统性偏向 Diagnostic",
        "description": """v2.3 的 M5 tone 匹配率仅 40%。系统倾向于在应该鼓励时错误输出 diagnostic。
这通常发生在：家长已经做了值得肯定的互动，但对话中同时存在问题行为时，
系统聚焦问题而忽略了家长已经做对的部分。""",
    },
    "tone_blindspot_empowering_bias": {
        "name": "Tone 盲区：该诊断时错误鼓励",
        "description": """当家长行为中存在需要指出的盲区时，系统错误地输出了 empowering。
这通常发生在：家长表面上处理得很好，但实际上在回避关键问题，
或对话语气温和但底层信念需要被挑战。""",
    },
    "generalization_gap": {
        "name": "泛化断裂：新场景模式",
        "description": """v2.3 在 70 个熟悉 case 上表现 0.883，但在 12 个新 case 上骤降到 0.625。
这说明 prompt 对训练集中未出现的对话模式泛化不足。
需要生成与现有 70 题明显不同的对话类型（隔代养育、单亲家庭、特殊需求儿童、
跨文化冲突、青少年早期等）。""",
    },
    "being_seen_weak": {
        "name": "看见感偏弱",
        "description": """v2.3 的 being_seen 维度均分 4.66（五维最低）。
系统有时没有充分 acknowledge 家长已有的努力和正确做法。
需要生成"家长已经很努力、做对了很多，但系统可能只看问题"的对话。""",
    },
    "insight_depth_insufficient": {
        "name": "洞察深度不足",
        "description": """v2.3 的 M6 洞察质量仅 3.8/5。系统有时停在表面行为分析，
没有触及深层信念/情绪模式。需要生成"表面行为相同但底层信念不同"的迷惑性对话。""",
    },
    "edge_case_low_score": {
        "name": "已知低分边缘 Case 模式",
        "description": """C5-001（摆碗碎碗, ~0.463）和 DS_001（收盘子, ~0.513）是已知最低分。
需要生成类似模式的对话：家长在培养责任感，但互动中涉及"意外/失误/犯错"的场景，
系统需要在"肯定动机"和"指出问题"之间精准平衡。""",
    },
}


# ═══════════════════════════════════════════════════════════════
# 主要接口
# ═══════════════════════════════════════════════════════════════

def generate_adversarial_cases(
    judge_client: Anthropic,
    judge_model: str,
    failure_profile: list[dict],
    existing_dialogues: list[str],
    n_per_mode: int = 3,
    verbose: bool = False,
) -> list[dict]:
    """根据 failure_profile 生成对抗测试用例。

    Args:
        judge_client: Claude client（星鸾）
        judge_model: Claude model name
        failure_profile: [{"mode": "tone_blindspot_diagnostic_bias", "examples": [...]}, ...]
        existing_dialogues: 已有对话文本列表（用于去重参考）
        n_per_mode: 每个失败模式生成的用例数

    Returns:
        [{"dialogue": "...", "expected_tone": "...", "target_failure": "...",
          "difficulty": "...", "reference_popup": "...", "case_id": "adv_xxx"}]
    """
    all_cases = []

    for fp in failure_profile:
        mode_key = fp.get("mode", "")
        mode_def = FAILURE_MODES.get(mode_key)
        if not mode_def:
            if verbose:
                print(f"  ⚠️ 未知失败模式: {mode_key}，跳过")
            continue

        prompt = _build_gen_prompt(
            failure_mode=mode_def["name"],
            failure_description=mode_def["description"],
            existing_examples=fp.get("examples", []) + existing_dialogues,
            n=n_per_mode,
        )

        if verbose:
            print(f"  🎯 生成对抗用例: {mode_def['name']} (目标 {n_per_mode} 个)")

        try:
            resp = judge_client.messages.create(
                model=judge_model,
                max_tokens=3000,
                temperature=0.7,  # 对抗生成需要一定创造性
                system=_GEN_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = ""
            for block in resp.content:
                if block.type == "text":
                    raw = block.text
                    break
        except Exception as e:
            if verbose:
                print(f"  ❌ 生成失败: {e}")
            continue

        # 解析 JSON
        cases = _parse_gen_response(raw)
        if verbose:
            print(f"  ✅ 生成 {len(cases)} 个用例")

        # 添加 case_id + mode 元数据
        for c in cases:
            c["mode"] = mode_key
            c["case_id"] = _make_case_id(c["dialogue"])
        all_cases.extend(cases)

    return all_cases


def _parse_gen_response(raw: str) -> list[dict]:
    """解析对抗生成器的 JSON 响应。"""
    import re

    cleaned = raw.strip()
    # 去掉可能的 thinking
    cleaned = re.sub(r'<thinking>.*?</thinking>', '', cleaned, flags=re.DOTALL)
    # 提取 JSON（可能被 markdown 包裹）
    m = re.search(r'```(?:json)?\s*(\{.*\})\s*```', cleaned, re.DOTALL)
    if m:
        cleaned = m.group(1)
    # 取第一个完整 JSON 对象
    m = re.search(r'\{.*"cases"\s*:\s*\[.*\]\s*\}', cleaned, re.DOTALL)
    if m:
        cleaned = m.group(0)

    try:
        data = json.loads(cleaned)
        return data.get("cases", [])
    except json.JSONDecodeError:
        # 尝试提取数组
        m = re.search(r'\[.*\]', cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return []


def _make_case_id(dialogue: str) -> str:
    """根据对话内容生成唯一 case_id。"""
    h = hashlib.sha256(dialogue.encode()).hexdigest()[:10]
    return f"adv_{h}"


# ═══════════════════════════════════════════════════════════════
# 预验证：检查对抗用例是否真能触发目标失败
# ═══════════════════════════════════════════════════════════════

def pre_validate_cases(
    task_client,
    task_model: str,
    judge_client: Anthropic,
    judge_model: str,
    system_prompt: str,
    cases: list[dict],
    verbose: bool = False,
) -> list[dict]:
    """预验证对抗用例：用当前 prompt 跑一遍，确认真的能触发失败。

    只保留确实触发了目标失败的用例（M5=0 或 M6<3.5）。
    """
    from auto_evolve.v23_runner import run_v23_once
    from auto_evolve.evaluator import (
        compute_m5_tone, build_m6_prompt_claude, parse_claude_judge_response,
    )
    from auto_evolve.dual_client import call_judge_claude

    validated = []
    for c in cases:
        dialogue = c.get("dialogue", "")
        expected_tone = c.get("expected_tone", "")

        # 用当前 prompt 生成弹窗
        result = run_v23_once(task_client, task_model, system_prompt, dialogue)
        if result["error"]:
            if verbose:
                print(f"  ⚠️ {c['case_id']}: 生成失败 — {result['error'][:60]}")
            continue

        # 检查是否能触发失败
        triggered = False
        reason = ""

        # 检查 tone 匹配
        m5 = compute_m5_tone(result["tone"], expected_tone)
        if m5 is not None and m5 == 0.0:
            triggered = True
            reason = f"tone_mismatch: sys={result['tone']} gold={expected_tone}"

        # 检查 M6（如果有 reference_popup）
        if not triggered and c.get("reference_popup", "").strip():
            m6_prompt = build_m6_prompt_claude(
                dialogue=dialogue,
                reference_popup=c["reference_popup"],
                sys_popup=result["popup_text"],
                sys_direction=result["tone"],
                sys_contradiction=result["contradiction"],
            )
            m6_raw = call_judge_claude(m6_prompt)
            m6_score, _ = parse_claude_judge_response(m6_raw, "m6")
            if m6_score is not None and m6_score < 3.5:
                triggered = True
                reason = f"M6_low={m6_score}"

        if triggered:
            c["_validated"] = True
            c["_trigger_reason"] = reason
            validated.append(c)
            if verbose:
                print(f"  ✅ {c['case_id']}: 触发 — {reason}")
        elif verbose:
            print(f"  ⏭️  {c['case_id']}: 未触发失败，跳过 (tone={result['tone']})")

    return validated


def load_adversarial_pool(pool_dir: Path) -> list[dict]:
    """加载已积累的对抗用例池。"""
    cases = []
    if not pool_dir.exists():
        return cases
    for f in sorted(pool_dir.glob("round_*_cases.json")):
        try:
            cases.extend(json.loads(f.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, KeyError):
            continue
    return cases


def save_adversarial_round(cases: list[dict], round_num: int, pool_dir: Path) -> Path:
    """保存本轮对抗用例。"""
    pool_dir.mkdir(parents=True, exist_ok=True)
    path = pool_dir / f"round_{round_num:03d}_cases.json"
    path.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
