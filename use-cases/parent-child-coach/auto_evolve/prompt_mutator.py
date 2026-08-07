"""提示词变异引擎 —— 分析失败 → LLM 提议修改 → 产出新 prompt。

策略：
  1. 输入：当前 prompt + 评估失败分析
  2. LLM 分析根因，提出 3-5 条具体的 prompt 修改
  3. 应用修改，产出新 prompt 文件

设计原则：
  - 每次只改一个 prompt（避免混淆因果）
  - 修改必须具体可追溯（精确到行/段落的增删改）
  - 保留修改历史（版本号递增）
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from json_repair import loads as _repair_loads
except ImportError:
    _repair_loads = None

# ═══════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════


@dataclass
class FailureCase:
    """单个失败案例。"""
    case_id: str = ""
    gold_tone: str = ""
    sys_tone: str = ""
    tone_match: bool = True
    m6_score: Optional[float] = None
    m7_score: Optional[float] = None
    sys_popup: str = ""
    sys_contradiction: str = ""
    gold_reference: str = ""
    failure_summary: str = ""


@dataclass
class FailureReport:
    """结构化失败分析报告。"""
    baseline_scores: dict = field(default_factory=dict)
    failures: list[FailureCase] = field(default_factory=list)
    success_cases: list[FailureCase] = field(default_factory=list)
    top_patterns: list[str] = field(default_factory=list)


@dataclass
class PromptMutation:
    """一次提示词修改。"""
    version_from: str = ""
    version_to: str = ""
    target_prompt: str = ""       # "master" | "perception" | "production"
    edit_description: list[str] = field(default_factory=list)
    modified_text: str = ""
    rationale: str = ""


# ═══════════════════════════════════════════════════════════════
# 失败分析
# ═══════════════════════════════════════════════════════════════

def build_failure_report(baseline_path: str) -> FailureReport:
    """从基线评估结果构建失败分析报告。"""
    with open(baseline_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    report = FailureReport(baseline_scores=data.get("aggregate", {}))

    for case in data.get("per_case", []):
        fc = FailureCase(
            case_id=case["case_id"],
            gold_tone=case["gold_tone"],
            sys_tone=case["sys_tone"],
            tone_match=case["m5_tone_match"] == 1.0,
            m6_score=case.get("m6_insight_score"),
            m7_score=case.get("m7_safety_score"),
            sys_popup=case.get("sys_popup_text", ""),
            sys_contradiction=case.get("sys_main_contradiction", ""),
            gold_reference=case.get("gold_reference_popup", ""),
        )

        # 判断是否失败
        is_failure = (
            not fc.tone_match or
            (fc.m6_score is not None and fc.m6_score < 3.5) or
            (fc.m7_score is not None and fc.m7_score < 3.5)
        )
        if is_failure:
            # 生成失败摘要
            reasons = []
            if not fc.tone_match:
                reasons.append(f"tone mismatch: expected {fc.gold_tone}, got {fc.sys_tone}")
            if fc.m6_score is not None and fc.m6_score < 3.5:
                reasons.append(f"insight quality low ({fc.m6_score}/5)")
            if fc.m7_score is not None and fc.m7_score < 3.5:
                reasons.append(f"safety concern ({fc.m7_score}/5)")
            fc.failure_summary = "; ".join(reasons)
            report.failures.append(fc)
        else:
            report.success_cases.append(fc)

    # 识别失败模式
    patterns = _detect_patterns(report.failures)
    report.top_patterns = patterns

    return report


def _detect_patterns(failures: list[FailureCase]) -> list[str]:
    """从失败案例中识别重复模式。"""
    patterns = []
    tone_fails = [f for f in failures if not f.tone_match]
    if len(tone_fails) >= 2:
        # 检查是否都是 diagnostic vs empowering 的混淆
        diag_to_emp = [
            f for f in tone_fails
            if f.gold_tone == "empowering" and f.sys_tone in ("diagnostic", "mixed")
        ]
        if len(diag_to_emp) >= 2:
            patterns.append(
                f"系统倾向输出 diagnostic/mixed，但 {len(diag_to_emp)}/{len(tone_fails)} 个案例专家期望 empowering。"
                f"根因：总控把'发现问题'当成默认方向，遗漏了家长的积极时刻。"
            )

        other_mismatches = [f for f in tone_fails if f not in diag_to_emp]
        for f in other_mismatches:
            patterns.append(
                f"{f.case_id}: expected {f.gold_tone}, got {f.sys_tone}. "
                f"Contradiction: {f.sys_contradiction[:100]}"
            )

    low_m6 = [f for f in failures if f.m6_score is not None and f.m6_score < 3.5]
    if low_m6:
        patterns.append(
            f"{len(low_m6)} 个案例洞察质量偏低 (M6 < 3.5): "
            + ", ".join(f"{f.case_id}({f.m6_score})" for f in low_m6)
        )

    low_m7 = [f for f in failures if f.m7_score is not None and f.m7_score < 3.5]
    if low_m7:
        patterns.append(
            f"{len(low_m7)} 个案例存在安全隐患 (M7 < 3.5): "
            + ", ".join(f"{f.case_id}({f.m7_score})" for f in low_m7)
        )

    return patterns


# ═══════════════════════════════════════════════════════════════
# LLM 驱动的 Prompt 变异
# ═══════════════════════════════════════════════════════════════

MUTATOR_SYSTEM_PROMPT = """你是提示词工程专家，专门优化 LLM prompt 以提高下游任务表现。

工作方式（两步法）：
1. **根因诊断**：不要只看表面失败模式。深层思考：prompt 的哪个结构性缺陷导致了系统反复犯同类错误？
   - 是缺少强制检查步骤？
   - 是选项排序造成了默认偏差？
   - 是缺少具体判断标准导致模型瞎猜？
   - 是缺少反例/边界案例导致模型过度泛化？

2. **精准修改**：提出 1-3 条关键修改（不是 5 条散弹），每条必须：
   - 直接针对根因
   - 给出精确的 before/after 文本（确保 before 能在原 prompt 中找到）
   - 解释这条修改如何阻断失败路径

修改约束：
- 优先做结构性修改（加检查步骤、改顺序、加判断标准），而非措辞微调
- 考虑加 1-2 个简洁的 few-shot 示例，如果示例能比规则更有效地消除歧义
- 保持 prompt 的语言风格和角色设定不变
- 不要删除已有的关键检查机制

输出严格 JSON：
{
  "root_cause": "<一句话，导致失败的最深层结构性缺陷>",
  "edits": [
    {
      "location": "<原文关键词，用于定位>",
      "before": "<精确的原文片段，确保一字不差>",
      "after": "<修改后的文本>",
      "reason": "<如何阻断失败路径>"
    }
  ],
  "expected_improvement": "<预期改善>"
}"""


def build_mutator_prompt(
    current_prompt: str,
    failure_report: FailureReport,
    prompt_name: str,
    previous_attempts: Optional[list[dict]] = None,
) -> str:
    """构建变异器 prompt。previous_attempts 为之前 discard 的尝试列表。"""
    failures_text = "\n".join(
        f"### {f.case_id}\n"
        f"- 失败: {f.failure_summary}\n"
        f"- 系统判断的矛盾: {f.sys_contradiction[:200]}\n"
        f"- 系统输出的弹窗(前150字): {f.sys_popup[:150]}\n"
        f"- 专家期望口吻: {f.gold_tone}\n"
        f"- 专家参考弹窗(前150字): {f.gold_reference[:150]}"
        for f in failure_report.failures
    ) if failure_report.failures else "（无失败案例）"

    success_text = "\n".join(
        f"- {s.case_id}: tone={s.sys_tone} ✓, M6={s.m6_score}, M7={s.m7_score}"
        for s in failure_report.success_cases
    ) if failure_report.success_cases else "（无成功案例）"

    patterns_text = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(failure_report.top_patterns))

    # 之前失败的尝试 —— 避免重复
    if previous_attempts:
        attempts_text = "\n".join(
            f"  - 尝试 {i+1} ({a.get('version', '?')}, 综合={a.get('overall', '?'):.3f}, discard 原因: {a.get('reason', '?')[:80]}):\n"
            f"    修改方向: {' | '.join(a.get('edit_descriptions', []))[:200]}"
            for i, a in enumerate(previous_attempts)
        )
        attempts_section = f"""
=== 之前的失败尝试（不要重复这些方向）===
{attempts_text}

⚠️ 上面这些修改方向已经被验证无效或有害。请提出**不同方向**的修改。"""
    else:
        attempts_section = ""

    return f"""当前 prompt 名称：{prompt_name}

=== 基线评估结果 ===
综合得分: {failure_report.baseline_scores}
M1触发: {failure_report.baseline_scores.get('m1_trigger_accuracy', '?')}
M5口吻: {failure_report.baseline_scores.get('m5_tone_match', '?')}
M6洞察: {failure_report.baseline_scores.get('m6_insight_quality', '?')}
M7安全: {failure_report.baseline_scores.get('m7_safety_score', '?')}

=== 失败模式 ===
{patterns_text}

=== 失败案例详情 ===
{failures_text}

=== 成功案例（保持不退化）===
{success_text}
{attempts_section}

=== 当前 Prompt ===
{current_prompt}

请分析当前 prompt 的哪部分导致了上述失败模式，并提出精确的修改。

⚠️ 重要：所有 JSON 字符串值中的双引号必须用 \\" 转义，换行用 \\n 转义。不要在字符串值中使用未转义的特殊字符。"""


# ═══════════════════════════════════════════════════════════════
# Prompt 读写
# ═══════════════════════════════════════════════════════════════

XINGLING_PROMPTS = Path(
    "D:/星灵-soul-手搓/亲子沟通洞见/路线A_自上而下法_鼓励版/prompts"
)

PROMPT_FILES = {
    "master": "prompt_总控_v3.1.md",
    "perception": "prompt_感知层_v3.1.md",
    "production": "prompt_生产层_v3.1.md",
    "ideal": "prompt_理想模式_v3.0.md",
}


def read_prompt(name: str, path: Optional[str] = None) -> str:
    """读取 prompt 文件内容。path 为自定义路径时优先使用。"""
    if path:
        return Path(path).read_text(encoding="utf-8")
    fname = PROMPT_FILES.get(name)
    if not fname:
        raise ValueError(f"Unknown prompt: {name}. Options: {list(PROMPT_FILES)}")
    return (XINGLING_PROMPTS / fname).read_text(encoding="utf-8")


def extract_version(text: str) -> str:
    """从 prompt 文本中提取版本号。"""
    m = re.search(r"v(\d+\.\d+)", text)
    return m.group(0) if m else "v0.0"


def bump_version(version: str) -> str:
    """递增版本号。v3.0 → v3.1"""
    parts = version.lstrip("v").split(".")
    minor = int(parts[1]) if len(parts) > 1 else 0
    return f"v{parts[0]}.{minor + 1}"


def apply_mutation(prompt_text: str, edits: list[dict], new_version: str) -> str:
    """将 LLM 提议的修改应用到 prompt 文本。

    对于每条修改，尝试定位并替换。如果精确匹配失败，在末尾追加。
    """
    modified = prompt_text
    applied = 0

    for edit in edits:
        before = edit.get("before", "")
        after = edit.get("after", "")
        if before and before in modified:
            modified = modified.replace(before, after, 1)
            applied += 1
        else:
            # 记录未应用的修改
            reason = edit.get("reason", "")
            print(f"  ⚠️  无法定位: {edit.get('location', '?')} —— {reason[:80]}")

    # 更新内部标题版本号
    old_ver = extract_version(prompt_text)
    # 通用版本替换：v3.0 → v3.1 等
    modified = re.sub(
        r"v\d+\.\d+",
        new_version,
        modified,
        count=1,
    )

    # 如果标题没被替换（可能格式特殊），尝试第二处
    if new_version not in modified.split("\n")[0]:
        modified = re.sub(rf"v{re.escape(old_ver.lstrip('v'))}", new_version, modified, count=1)

    print(f"  应用 {applied}/{len(edits)} 条修改 → {new_version}")
    return modified


def write_prompt_variant(name: str, text: str, version: str) -> Path:
    """写入变体 prompt 文件。返回文件路径。"""
    base = PROMPT_FILES.get(name)
    if not base:
        raise ValueError(f"Unknown prompt: {name}")
    # 生成变体文件名: prompt_总控_v3.1.md → prompt_总控_v3.2.md
    stem = Path(base).stem
    # 去掉旧版本号后缀，替换为新版本
    import re
    stem_clean = re.sub(r"_v\d+\.\d+$", "", stem)
    variant_name = f"{stem_clean}_{version}.md"
    path = XINGLING_PROMPTS / variant_name
    path.write_text(text, encoding="utf-8")
    return path


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def propose_mutation(
    client,
    model: str,
    prompt_name: str,
    failure_report: FailureReport,
    current_text: Optional[str] = None,
    current_version: Optional[str] = None,
    previous_attempts: Optional[list[dict]] = None,
    new_version_override: Optional[str] = None,
) -> PromptMutation:
    """分析失败 → 提议 prompt 修改 → 返回突变。

    Args:
        current_text: 当前 prompt 文本。None 时从原始文件读取。
        current_version: 当前版本号。None 时从 prompt 文本中提取。
        previous_attempts: 之前 discard 的尝试列表，用于避免重复方向。
        new_version_override: 自定义新版本号，避免版本碰撞。
    """
    if current_text:
        current = current_text
        old_version = current_version or extract_version(current)
    else:
        current = read_prompt(prompt_name)
        old_version = extract_version(current)
    new_version = new_version_override or bump_version(old_version)

    user_prompt = build_mutator_prompt(current, failure_report, prompt_name, previous_attempts)

    raw = ""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": MUTATOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.4,
            timeout=90,
        )
        raw = resp.choices[0].message.content or ""
        # 优先用标准 json.loads
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # 降级到 json_repair（容忍未转义字符、尾随逗号等）
            if _repair_loads is None:
                raise
            data = _repair_loads(raw)
    except Exception as e:
        return PromptMutation(
            version_from=old_version,
            version_to=new_version,
            target_prompt=prompt_name,
            rationale=f"LLM call failed: {e}",
        )

    edits = data.get("edits", [])
    analysis = data.get("analysis", "")
    expected = data.get("expected_improvement", "")

    # 应用修改
    modified_text = apply_mutation(current, edits, new_version)

    # 写入文件
    variant_path = write_prompt_variant(prompt_name, modified_text, new_version)

    return PromptMutation(
        version_from=old_version,
        version_to=new_version,
        target_prompt=prompt_name,
        edit_description=[f"{e.get('reason', '')}" for e in edits],
        modified_text=modified_text,
        rationale=f"Analysis: {analysis}\nExpected: {expected}",
    )
