---
title: P2 Quotable Phrase — Inverted Tone Mapping
date: 2026-08-05
category: logic-errors
module: parent-child-coach
problem_type: logic_error
component: service_object
severity: medium
symptoms:
  - "诊断式弹窗指出问题但不给可引用话术——诊断了毛病却没给药方"
  - "鼓励式弹窗因缺话术被重试或拒绝——短肯定语被强加了说教模板"
  - "P2 相关代码注释和日志全部引用 encouraging，强化了错误的心理模型"
root_cause: logic_error
resolution_type: code_fix
tags:
  - p2
  - quotable-phrase
  - popup-tone
  - diagnostic
  - encouraging
  - popup-generator
  - tone-mapping
  - v4.0.18
related_components:
  - testing_framework
---

# P2 Quotable Phrase — Inverted Tone Mapping

## Problem

自 v4.0.14 起，P2 可引用话术检查（quotable phrase gate）被挂到了鼓励式弹窗而非诊断式弹窗上。鼓励式弹窗（仅肯定家长的积极瞬间）被要求包含家长可直接复用的话术，而诊断式弹窗（指出问题并需要教家长怎么说话）反而不做检查——逻辑与产品需求完全相反。

## Symptoms

- 诊断式弹窗指出沟通问题后不提供"你可以这样说"的具体话术——这正是诊断式弹窗的核心价值所在
- 鼓励式弹窗（30-60字短肯定语）因缺少话术被重试甚至拒绝，本质上就不该带说教模板
- 代码注释和日志中到处是 "Encouraging popup missing quotable repair phrase"，强化了错误的维护心智模型

## What Didn't Work

- v4.0.14 原始实现将 P2 检查放在 `popup.tone == PopupTone.ENCOURAGING` 下。当时团队认为鼓励式弹窗应该为家长示范好的语言。重试逻辑和日志全部引用 encouraging。
- 测试中从未捕获此问题，因为测试对话大多触发诊断式弹窗（直接通过，不检查），而鼓励式路径很少跑 P2 门控。

## Solution

共 6 处修改，跨 2 个文件。所有修改是同一逻辑操作：将话术要求从 encouraging 移到 diagnostic。

### 生产代码：`realtime/popup_generator.py`（3 处）

**修改 1 — `_build_messages()` type_instruction**

Before:
```python
if tone == PopupTone.DIAGNOSTIC:
    type_instruction = (
        f"请生成**诊断式弹窗**（{DIAGNOSTIC_MIN_CHARS}-{DIAGNOSTIC_MAX_CHARS}字）。"
        "必须：先承认发心 → 揭示具体模式 → 给出一个微小可做的尝试。"
    )
else:
    type_instruction = (
        f"请生成**鼓励式弹窗**（{ENCOURAGING_MIN_CHARS}-{ENCOURAGING_MAX_CHARS}字）。"
        "必须：具体点出家长刚展现的积极模式 → 简短有力。"
        "必须包含至少一句家长可直接引用的话术"
        "（以「你可以这样说：\"……\"」形式给出，引号内为实际措辞）。"
    )
```

After:
```python
if tone == PopupTone.DIAGNOSTIC:
    type_instruction = (
        f"请生成**诊断式弹窗**（{DIAGNOSTIC_MIN_CHARS}-{DIAGNOSTIC_MAX_CHARS}字）。"
        "必须：先承认发心 → 揭示具体模式 → 给出一个微小可做的尝试。"
        "必须包含至少一句家长可直接引用的话术"
        "（以「你可以这样说：\"……\"」形式给出，引号内为实际措辞）。"
    )
else:
    type_instruction = (
        f"请生成**鼓励式弹窗**（{ENCOURAGING_MIN_CHARS}-{ENCOURAGING_MAX_CHARS}字）。"
        "必须：具体点出家长刚展现的积极模式 → 简短有力。"
    )
```

**修改 2 — `generate()` P2 门控条件**

Before:
```python
if (
    popup.tone == PopupTone.ENCOURAGING
    and not has_quotable_phrase(popup.full_text)
):
    logger.warning("Encouraging popup missing quotable repair phrase; retrying once")
```

After:
```python
if (
    popup.tone == PopupTone.DIAGNOSTIC
    and not has_quotable_phrase(popup.full_text)
):
    logger.warning("Diagnostic popup missing quotable repair phrase; retrying once")
```

**修改 3 — 重试失败日志**

`"Encouraging popup still missing..."` → `"Diagnostic popup still missing..."`

### 测试代码：`scripts/run_v418_pipeline.py`（3 处）

**修改 4 — `generate_popup()` type_instruction**：话术要求从 `else` 分支（encouraging）移到 `if tone == "diagnostic"` 分支，与生产代码完全对齐。

**修改 5 — P2 检查门控**：`tone == "encouraging"` → `tone == "diagnostic"`

**修改 6 — 版本号**：`__version__ = "1.2"`，changelog 记录 "P2 修正：话术检查从鼓励式搬到诊断式"

## Why This Works

根因是两种弹窗 tone 的语义被颠倒了：

- **诊断式弹窗**（80-200字）指出家长沟通中的问题。它的职责是揭示盲区**并教家长怎么说**。没有可引用话术，弹窗只诊断不给药——家长知道有问题但不知道怎么办。"你可以这样说：'……'"正是从诊断到行动的桥梁。

- **鼓励式弹窗**（20-80字）肯定好的瞬间。它的职责是命名家长刚展现的积极模式，简短有力。给鼓励式弹窗加话术是本末倒置——家长已经说对了，弹窗要帮他们看见**为什么对**，而不是给他们已经用过的词。

FC_TONE_OFF 机制（家长行为 override 关键词强制 encouraging→diagnostic）现在形成完整链条：问题行为→强制诊断式→诊断式需要话术→P2 执行检查。修复前这条链在第二节断裂。

## Prevention

- 编写按语义类别（tone、type、mode、channel）分支的条件逻辑时，加一行注释说明**为什么**这个类别需要这个处理。原始代码没有解释为什么 encouraging 需要话术——一行注释就能在设计阶段暴露颠倒的逻辑。
- 引入新门控/检查的代码与产品 spec 结对 review。用户的表述（"鼓励式弹窗不用建议话术，而恰恰是诊断式弹窗需要"）当场就能说清楚——这本该是出发点，而非事后纠正。
- 添加回归测试：含 parent-override 关键词的对话（触发诊断式）验证 P2 话术检查生效，不含关键词的对话（鼓励式）验证 P2 跳过。

## Related Issues

- 主源文档：`specs/fix-pipeline-gaps-spec.md` §8（Codex 审计 #6 缺口）
- Commit: `a81f182` on `feat/compare-scripts-and-v30`
- multica: [REN-76](https://multica.app/ren/76) — v4.0.18 管线修复
- 12 题全量测试：avg=5.875，Codex judge，确认修复无回归
- 已知残留：`CLAUDE.md` L57 的已知问题行（生产代码 P2 修正待验收）
- 可能过时：`docs/v4.0.17_simple_guide.md` 第三道闸门描述仍将 P2 挂在鼓励式上
