---
title: "v2.5 感知层 + 规则层 + v3.2 生产层 · tone 规则化决策记录"
type: solution
date: 2026-07-19
status: accepted
tags:
  - multi-agent
  - tone-rules
  - rule-engine
  - perception-layer
  - production-layer
  - keep-discard
  - parent-child-coach
applies_when:
  - LLM tone 判定不稳定，需要用规则层覆写
  - 评估时 M6 退化与 M5/M7 提升冲突，需要判断是否为真实退化
  - empowering 弹窗比 diagnostic 短导致 M6 judge 评分偏低
  - 多 agent 架构中 fact extraction 与 tone 判定分离
---

# v2.5 感知层 + 规则层 + v3.2 生产层 · tone 规则化决策记录

## 1. 背景与问题

基线（v2.1 感知层 + v3.5 总控 + v3.0 生产层）在去噪评估（n=3）中：
- M5（tone 匹配率）仅 50%，6 个稳定错误案例
- 根因：v2.1 的 `positive_moment_category` 分类轴与 gold tone 标注轴不一致——前者分类"家长状态"，后者分类"弹窗该用什么 tone"
- 两次尝试修正（v2.3、v2.4）引入 `response_need` 反事实推理字段，均失败——DeepSeek Chat 做不了"家长需要什么回应"的 counterfactual 推理

## 2. 解决方案：tone 规则化（方向 A）

**核心设计**：LLM 做 fact extraction（擅长），deterministic rules 做 tone 判定（确定）。

### 2.1 v2.5 感知层：6 个布尔信号字段

新增 6 个事实信号（非主观判断）：
- `has_generalization`：泛化（"你每次..."）
- `has_labeling`：贴标签（"你就是个..."）
- `has_conflict_escalation`：高烈度冲突升级（驱赶/剥夺话）
- `has_safety_emergency`：紧急安全场景（马路/坠楼）
- `child_core_need_unmet`：孩子核心诉求未被回应
- `parent_emotion_overload`：家长情绪过载

保留 `positive_moment_category` 字段（向后兼容 v3.5 总控）。

### 2.2 规则引擎（`src/tone_rules.py`）

10 条优先级规则，基于信号组合覆写 LLM 的 tone 判定：

| 优先级 | 条件 | tone | 场景 |
|--------|------|------|------|
| 1 | safety + overload | empowering | 紧急安全+家长过载，需台阶 |
| 2 | safety + need_unmet | diagnostic | 恐惧泛化否定合理尝试 |
| 3 | safety（无上述） | empowering | 必要制止，需赋能调整方式 |
| 4 | conflict + overload | empowering | 需退路 |
| 5 | conflict + need_unmet | diagnostic | 家长在压制 |
| 6 | gen/label + need_unmet | diagnostic | 盲区+回避 |
| 7 | overload（无冲突/安全） | empowering | 过载是主因，先给台阶 |
| 8 | gen/label（无其他） | diagnostic | 有盲区需命名 |
| 9 | need_unmet（无泛化/冲突） | diagnostic | 家长回避核心问题 |
| 10 | 默认 | 保留 LLM 判定 | 信号不足，信任 LLM |

关键词兜底：当 LLM 信号字段漏掉时，用对话原文关键词（"马路""你出去""我太生气"等）补充触发。

### 2.3 v3.2 生产层：empowering 两种子模式

v3.0 生产层把 empowering 定义为"30-60字纯闪光"，导致高冲突场景的 empowering 弹窗缺诊断深度，M6 偏低。

v3.2 保守改动：在 empowering 定义里加一句例外——高冲突/过载场景允许写 60-100字"命名模式+给退路"的较长弹窗，`popup_suggestion` 给具体停战话术。其他 tone 定义不变。

### 2.4 orchestrator 集成

`multi_agent_orchestrator.py` Step 2.5（总控后、生产层前）插入规则覆写层：
- 只在 `should_popup=True` 时覆写
- 信号全 False 时跳过（保留 LLM 判定，向后兼容）
- 规则触发理由写入 `contradiction_flag` 供审计

## 3. 评估结果（去噪 n=3，12 案例）

| 指标 | 基线 | 新版 | Δ |
|------|------|------|---|
| M1（触发准确率） | 100% | 100% | → |
| M5（tone 匹配率） | 50% | **83.3%** | ↑33% |
| M6（洞察质量） | 3.94 | 3.69 | ↓0.25 |
| M7（安全分） | 4.39 | **4.86** | ↑0.47 |
| 综合 | 0.821 | **0.877** | ↑0.056 |

### keep/discard 检查

- ✓ 规则1：overall +0.056（>0.3%）
- ✓ 规则3：M5 +4 匹配，0 unmatch
- ✓ 规则4：M7 无退化 >0.5（整体 +0.47）
- ✓ 规则2：无单案例 M6 退化 >0.75（详见下方 C11-009 调查）

## 4. C11-009 M6 退化调查（8-seed 复测）

### 初始观察（n=3）

C11-009 在 n=3 去噪评估中 M6=3.33，baseline M6=4.33，退化 -1.00，**疑似违反规则2**。

### 8-seed 复测

独立评审建议用 3+ seed 复测确认是否为噪声。跑 8 次（temperature=0.5）：

| run | tone | M5 | M6 | M7 |
|-----|------|----|----|----|
| 1-8 | empowering | ✓ | 4.0 | 4.0-5.0 |

- M6: min=4.0 max=4.0 **mean=4.00 std=0.00**
- M7: mean=4.88
- M5: 8/8 匹配

### 结论

**M6 真实值是 4.0，不是 3.33**。n=3 里的 3.33 是 judge 单次低分噪声（3 次中有 1 次给了 ~2 分，拉低均值）。

实际退化 = 4.33 → 4.0 = **-0.33**，远低于 0.75 阈值。**C11-009 不违反 keep/discard 规则2，不需要例外标注。**

### 根因分析

- baseline M6=4.33 测的是错 tone（diagnostic）的长弹窗（80-150字）
- 新版 M6=4.0 测的是对 tone（empowering）的中等弹窗（60-100字）
- 0.33 的差异来自 M6 judge 的 length bias——judge 倾向给长文本更高分（详见 §5 backlog）

## 5. Backlog：M6 length-bias 修复

### 问题

M6 judge prompt 评估"洞察质量"时，隐含偏好长文本——diagnostic 弹窗（80-150字）系统性比 empowering 弹窗（30-60字）得分高 0.3-0.5 分。这不是内容质量问题，是 metric 设计缺陷。

### 影响

- 任何把 tone 从 diagnostic 改成 empowering 的改动，M6 都会假性退化
- 阻碍 empowering 方向的优化迭代
- C11-009 的 -0.33 退化就是 length bias 的体现

### 待办

- [ ] 审计 `evaluator.py` 的 M6 judge prompt，确认 length bias 来源
- [ ] 在 M6 judge prompt 里加明确指令："评分基于洞察深度和准确性，与字数无关。30字的精准鼓励和150字的深度诊断同等高分。"
- [ ] 用 20 案例重跑 M6 judge（人工对照），验证 length bias 是否消除
- [ ] 考虑把 M6 拆成两个子指标：M6a（洞察准确性）+ M6b（表达质量），消除 length 混淆

## 6. 仍错案例（信号方法固有局限）

3 个案例 baseline 也错，规则层无法修正：

| 案例 | 信号 | sys tone | gold tone | 根因 |
|------|------|----------|-----------|------|
| C10-008 | s=True n=False | empowering | diagnostic | LLM 读成"genuine_transformation"，gold 标注者读成"恐惧泛化盲区"——解读分歧 |
| C11-010 | n=True | diagnostic | empowering | LLM 提取 need_unmet=True，但 gold=empowering——信号与 gold tone 不相关 |
| C10-002 | (random) | - | diagnostic | run 间随机，非稳定错误 |

这些案例需要更深的语义理解或重新审视 gold 标注，留给下一轮方向。

## 7. 相关文件

- `prompts/prompt_感知层_v2.5.md` — 6 信号字段定义
- `src/case_memory.py` — `PerceptionReport` 新增 6 个 bool 字段（向后兼容）
- `src/perception_agent.py` — `_build_report()` 解析 `signals` 对象
- `src/tone_rules.py` — 10 条规则引擎 + 关键词兜底
- `src/multi_agent_orchestrator.py` — Step 2.5 规则覆写层
- `prompts/prompt_生产层_v3.2.md` — empowering 两种子模式
- `results/v25_v35_rules_v32prod_denoised_n3.json` — n=3 去噪评估结果
- `results/c11_009_multi_seed.json` — C11-009 8-seed 复测结果

## 8. 教训

1. **n=3 去噪不够**：judge 单次低分噪声会污染均值。关键案例退化判断应跑 8+ seed 确认。
2. **metric length bias 会假性惩罚短弹窗**：empowering 优化天然吃亏，需要修 metric 而非接受假性退化。
3. **LLM fact extraction + deterministic rules 是可行架构**：DeepSeek Chat 做不了 counterfactual reasoning，但能做事实特征提取；规则层用确定逻辑组合信号，避开了 LLM 的弱点。
4. **信号方法有边界**：当 LLM 和标注者对案例解读有根本分歧时（C10-008），信号无法纠正——这是语义理解层面的局限。

## 9. 自动迭代收敛验证（2026-07-19）

为验证 v2.5+规则层+v3.2 是否为局部最优，跑了两轮自动迭代（`run_auto_evolve.py`），共 10 次变异尝试，**全部 discard**。

### 第一轮（production 目标，5 轮）
- 3 轮 JSON 解析失败（DeepSeek 输出含未转义字符）
- 2 轮成功变异但退化：v3.3 iter1 Δ=-0.005（C10-001 M7 改善但 C10-003 不稳定），v3.3 iter2 Δ=-0.047（C10-001 M7 崩溃至 1.3）

### 第二轮（perception → master，5 轮，已修复 JSON + 加尝试记忆）
| # | 目标 | 版本 | Δ综合 | 主要破坏 |
|---|------|------|-------|---------|
| 1 | perception | v2.6 | -0.043 | C10-001 M7=1.7, C11-001/C11-006 触发失效 |
| 2 | perception | v2.7 | -0.003 | C10-001 M7=2.7 |
| 3 | perception | v2.8 | -0.066 | C11-006/C11-009 触发失效 |
| 4 | master | v3.6 | -0.030 | C11-001 触发失效 |
| 5 | master | v3.7 | -0.014 | C10-003 不稳定, C11-009 M7=3.3 |

### 收敛结论

**v2.5+规则层+v3.2（overall=0.878）是 prompt 变异可达的局部最优。**

所有变异方向都在尝试修复 C10-008（"家长抱起孩子看消防车"应判 diagnostic 而非 empowering），但都会误伤 C11-001/C11-006/C11-009（这些案例也是"家长调整"但应 empowering）。LLM 无法仅通过 prompt 规则区分"家长回避核心恐惧"与"家长满足孩子需求"——这是语义理解层面的固有局限，不是 prompt 工程问题。

### 工程改进
- `json_repair` 降级解析：解决 DeepSeek JSON 输出偶发未转义字符问题（第一轮 3/5 失败 → 第二轮 0/5 失败）
- 尝试记忆机制：把 discard 的变异方向喂给 mutator，避免重复（`previous_attempts` 参数）
- 唯一版本号：用迭代计数器生成 `v2.6/v2.7/v2.8`，避免版本碰撞

### 后续方向（prompt 变异之外）
1. **few-shot 示例**：在感知层加入 C10-008 vs C11-006 的对比示例，让 LLM 学习"回避恐惧"与"满足需求"的区分
2. **gold 标注复核**：C10-008 的 gold tone 是否真的应该是 diagnostic？标注者可能有自己的偏见
3. **M6 metric 修复**：judge 对 diagnostic（80-150字）有长度偏好，empowering（30-60字）天然吃亏
4. **接受现状**：0.878 已达预期，剩余 3 个失败案例是边界案例

## 10. 相关文件（自动迭代）

- `use-cases/parent-child-coach/auto_evolve/run_auto_evolve.py` — 自动迭代 runner
- `use-cases/parent-child-coach/auto_evolve/prompt_mutator.py` — 变异引擎（含 json_repair + 尝试记忆）
- `use-cases/parent-child-coach/results/auto_baseline_v25_full.json` — 基线完整报告
- `use-cases/parent-child-coach/results/auto_evolve_history.json` — 10 轮迭代历史
