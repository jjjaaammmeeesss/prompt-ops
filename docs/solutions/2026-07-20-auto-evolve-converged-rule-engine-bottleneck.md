---
problem_type: auto_evolve_convergence
tags: [auto-evolve, rule-engine, tone-mismatch, diagnostic-bias, prompt-ops]
applies_when: auto-evolve 连续多次 discard，且失败模式集中在 tone_mismatch
created: 2026-07-20
status: resolved-v3.1
---

# 多智能体 v3.1 — 规则引擎修复打开新 prompt 空间

## v3.1 改动（2026-07-20）

采用方案 A+B：
1. **移除规则 8**（泛化单独 → diagnostic）— "你每次"/"每次都" 高频词过度触发
2. **新增规则 7**（genuine_transformation → 保留 LLM 判定）— 正面信号覆写

## v3.1 评估结果（62 案 × n=3）

| 指标 | v3.0 | v3.1 | Δ |
|------|------|------|---|
| overall | 0.793 | 0.797 | +0.004 ✅ |
| M5 (tone match) | 55.0% | 61.0% | **+6.0pp** |
| M1 | 85.5% | 83.9% | -1.6pp |
| M6 | 3.70 | 3.66 | -0.04 |
| M7 | 4.75 | 4.78 | +0.03 |

核心成果：M5 tone 匹配率 +6pp，正是规则引擎改动 targeting 的指标。

## 问题回顾

62 案 × n=3 降噪基线 overall=0.793。8 轮自动迭代（perception/master/production 三目标轮换），7/7 全部 discard，最佳变体 Δ=-0.018。

## 根因

**规则引擎 `tone_rules.py` 决定 tone，prompt 变异无法影响 tone 匹配率。**

变异器反复尝试"让系统输出更多 empowering"（因为 25/29 失败案例 gold=empowering），但：
1. 改 perception prompt → 信号字段不变 → tone 不变
2. 改 master prompt 强制 empowering → C10-008/C13-008（gold=diagnostic）被破坏
3. 改 production prompt → tone 由规则层决定，production 只是执行

## 诊断证据

抽样 8 个 tone_mismatch 案例，跑感知层 + 规则引擎，发现 7/8 是规则误触发：

| 案例 | gold | sys | 触发规则 | 误触原因 |
|------|------|-----|---------|---------|
| C10-008 | diagnostic | empowering | rule3 (safety) | "车"/"窗台" 只是提及，非紧急 |
| C11-010 | empowering | diagnostic | rule5 (conflict+need) | "我不要" 是常见冲突词，但对话有转变 |
| C10-009 | empowering | diagnostic | rule8 (gen alone) | "你每次"/"每次都" 过于常见 |
| C10-010 | empowering | diagnostic | rule8 (gen alone) | 同上 |
| C11-005 w1 | empowering | diagnostic | rule8 (gen alone) | perception 信号 has_generalization=True 但有转变 |
| C11-005 w2 | empowering | diagnostic | rule6 (gen+need) | 同上 + need_unmet |
| C11-005 w4 | empowering | diagnostic | rule6 (gen+need) | 同上 |
| C11-009 | empowering | diagnostic | rule5 (conflict+need) | "不给" 触发冲突词 |

## 核心问题

1. **规则 8（泛化单独 → diagnostic）过于激进** — "你每次"/"每次都"/"老是" 在任何亲子冲突中都会出现，无法区分"卡在盲区"和"有转变但仍有批评"
2. **关键词兜底太宽** — `_GENERALIZATION_KEYWORDS` 包含高频日常词
3. **缺少正面信号覆写** — 即使 perception 检测到 `positive_moment_category="genuine_transformation"`，规则仍会触发 diagnostic

## 建议修复方向（需用户确认）

### 方案 A：移除规则 8（最小改动）
- 删除 "规则 8: 泛化/贴标签（无其他信号）→ diagnostic"
- 让 LLM 在仅有泛化信号时自行判断
- 预期影响：C10-009, C10-010, C11-005 w1 从 diagnostic → LLM 判定（可能 empowering）

### 方案 B：增加正面信号覆写（更精准）
- 在规则 7/8/9 之前增加检查：若 `positive_moment_category == "genuine_transformation"`，保留 LLM 判定
- 需要感知层准确识别 genuine_transformation（当前 v2.5 已有此字段）

### 方案 C：收紧关键词（保守）
- 从 `_GENERALIZATION_KEYWORDS` 移除 "每次都", "老是"（保留 "你总是", "你永远", "从来"）
- 从 `_CONFLICT_KEYWORDS` 移除 "我不要", "不给"（保留更激烈的 "你出去", "别吃饭"）

## 教训

- **auto-evolve 的边界**：当 tone 由确定性代码决定时，prompt 变异无法改善 tone 匹配率。需要区分"prompt 可优化"和"代码需优化"的失败模式。
- **关键词兜底的陷阱**：高频词会过度触发规则，应该用更严格的短语或要求多信号并发。
- **降噪阈值**：`MAX_CASE_REGRESSION=0.15` 对 5 分制 M6 太严（n=3 judge 噪声 ±0.3），应设为 0.5。
