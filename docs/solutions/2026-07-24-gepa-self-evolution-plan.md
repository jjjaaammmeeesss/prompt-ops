# GEPA + BetterTogether 自进化计划 · 通用弹窗 Prompt 优化

**日期**: 2026-07-24
**状态**: 计划阶段
**上下文**: [MIPROv2 vs PDO 对比分析](../../use-cases/general-popup-native/README.md)、[4 种 DSPy 原生进化策略调研](#)

---

## 1. 问题陈述

### 1.1 当前状态

| 系统 | 策略 | 效果 |
|------|------|------|
| `general-dialogue-popup/evolve.py` | 自定义进化（LLM 变异 + GLM judge） | v1.0→v1.4: 4.08→4.17，3 轮后停滞 |
| `general-popup-native/` PDO | 对决 bandit + LLM judge (8 维) | 产出 v2.1，过度精简 (~200 字)，丢领域约束 |
| `general-popup-native/` MIPROv2 | 贝叶斯优化 + LLM judge (6 维) | 产出 v2.2，情感好但同样过度精简 |
| **人工合并 v2.2** | 恢复完整领域结构 + 融入优化洞察 | **表现最优** |

### 1.2 核心问题（不是优化器不够强）

**Judge 系统性偏差**：所有依赖标量 metric 的优化器（PDO、MIPROv2）都在往「短 = 好」的方向收敛，因为 Judge（DeepSeek-chat）过度奖励简洁、格式正确的输出，惩罚长 prompt 的复杂约束。

**人工 v2.2 为什么赢了**：人知道哪些约束不能丢（三种误判场景、生命立场、安好静音逻辑），自动化优化器不知道。

```
问题链: Judge 偏差 → 标量 metric 被污染 → 优化器梯度指向错误方向 → 丢领域知识
解决链: GEPA 文本反思 → 绕过标量偏差 → 保留约束的前提下精准编辑
```

### 1.3 为什么 GEPA 是突破口

GEPA 是 DSPy 唯一一个**不只看标量分数的优化器**。它的反思机制用**文本反馈**（执行轨迹、失败案例、评语），不是裸分数。

| 优化器 | 反馈信号 | 能感知领域约束丢失吗？ |
|--------|---------|---------------------|
| MIPROv2 | 标量 metric | ❌ "分数更高 = 更好"，不知道约束丢了 |
| PDO | pairwise 对比 | ❌ "A > B"，不知道为什么 |
| **GEPA** | **文本反馈** | ✅ "这个弹窗丢失了'安好'静音逻辑" → 反思 → 修复 |
| COPRO | 标量 metric | ❌ 同 MIPROv2 |

---

## 2. 方案设计

### 2.1 双层闭环架构

```
┌──────────────────────────────────────────────────────────────┐
│ 外层循环 · 元控制器 (SelfEvolutionRunner，max 5 轮)          │
│                                                              │
│  Round 0: 基线评估（冻结 origin baseline）                    │
│                                                              │
│  Round 1..N:                                                 │
│    ┌─────────────────────────────────────────────────────┐   │
│    │ ① 失败模式分类                                       │   │
│    │   取上一轮最低分案例 → 按维度归类失败类型              │   │
│    │   （tone 判定 / 脑补幻觉 / 约束丢失 / 字数 / 结构）   │   │
│    │                                                     │   │
│    │ ② 策略调度                                           │   │
│    │   根据失败模式选择本轮内层策略                         │   │
│    │   （见 §2.3 策略目录）                                │   │
│    │                                                     │   │
│    │ ③ 执行内层策略 ─ 可插拔 ──────────────────────→      │   │
│    │   ┌───────────────────────────────────────────┐     │   │
│    │   │ 内层循环（策略自选，示例：GEPA→MIPROv2）    │     │   │
│    │   │                                           │     │   │
│    │   │ GEPA 内部种群进化（预算驱动，~35 代）        │     │   │
│    │   │   种群评估 → LLM 反思失败轨迹              │     │   │
│    │   │   → 变异指令 (1-3 处精准编辑)              │     │   │
│    │   │   → Pareto 选择 → 下一代                   │     │   │
│    │   │   预算耗尽 → 输出种群最优个体               │     │   │
│    │   │                                           │     │   │
│    │   │   (可选) → MIPROv2 措辞精调 + few-shot     │     │   │
│    │   └───────────────────────────────────────────┘     │   │
│    │                                                     │   │
│    │ ④ 评估 & 基线守护                                    │   │
│    │   候选 vs 冻结 origin baseline（防漂移）              │   │
│    │                                                     │   │
│    │ ⑤ 决策                                               │   │
│    │   keep: overall↑>0.02 且 领域约束保留率>80%         │   │
│    │   discard: 否则，记录本轮失败模式 → 回到 ② 换策略     │   │
│    │                                                     │   │
│    │ ⑥ 收敛检查                                           │   │
│    │   连续 2 轮无提升 → 停止                              │   │
│    │   或达到 max_rounds → 停止                            │   │
│    └─────────────────────────────────────────────────────┘   │
│                                                              │
│  每轮持久化: prompt 版本 + 评估分数 + 失败模式 → 可追溯       │
│  输出: 最优 prompt + 进化轨迹                                  │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 外层功能（元控制器）

外层只做 4 件事，其他都留给内层或省略：

| # | 功能 | 说明 |
|----|------|------|
| **①** | **失败模式分类** | 从 metric 返回的 6 维分数中取最低维度作为失败类型。不调额外 LLM，直接复用 metric 输出。是策略调度的依据 |
| **②** | **策略调度** | 根据失败模式从策略目录中选择下一轮内层策略（见 §2.3）。首轮默认 GEPA→MIPROv2 |
| **③** | **基线守护** | 每轮候选 vs 冻结的 origin baseline（防漂移）。keep 条件：overall↑>0.02 且领域约束保留率>80% |
| **④** | **版本 & 错误收集** | 每轮持久化：prompt 文本、6 维分数、失败模式标记、keep/discard 决策。做到任意一轮可复盘 |

不做的：
- ~~跨轮趋势追踪~~ — `patience=2` 就够，不逐 case 画曲线
- ~~策略效能画像~~ — 需要多次运行积累，Phase 2 的事
- ~~预算感知早停~~ — 固定 `max_rounds=5` 已够
- ~~中断恢复~~ — 工程细节，后期补
- ~~人工闸门~~ — print 结果，人 Ctrl+C
- ~~候选池管理~~ — 只保留 winner + 日志记 discard

失败模式 → 策略的调度映射（初始）：

| 主要失败维度 | 调度策略 |
|-------------|---------|
| 约束保真度 / 硬规则违规 | GEPA only（文本反思能感知约束丢失） |
| 措辞 / 字数 | GEPA → MIPROv2（MIPROv2 擅长精炼） |
| tone 判定 | PDO（pairwise 对比对 tone 更敏感） |
| 综合偏低无突出弱项 | 换起点：从上一轮 discard 方向反向尝试 |

映射表随实验迭代调整，不作为代码常量写死。

### 2.3 内层策略目录

内层策略可插拔，外层通过策略名调用。初始支持 4 种：

| 策略名 | 构成 | 适用场景 |
|--------|------|---------|
| `gepa` | GEPA only，`auto="medium"` | 需要文本反思恢复领域约束 |
| `gepa_mipro` | GEPA → MIPROv2，GEPA `auto="light"` + MIPROv2 `auto="light"` | 默认首轮策略，兼顾骨架保持与措辞精炼 |
| `mipro` | MIPROv2 only，`auto="light"` | 仅需措辞微调 |
| `pdo` | PDO only | 反面对照 / pairwise 对比场景 |

每种策略各自传入 compatible 的 metric（GEPA 用文本反馈 metric，MIPROv2/PDO 用标量 metric）。

示例 — 默认首轮用的 `gepa_mipro`：

```python
BetterTogether(
    metric=metric_with_textual_feedback,
    gepa=GEPA(
        metric=metric_with_textual_feedback,
        auto="light",
        reflection_minibatch_size=3,
        candidate_selection_strategy="pareto",
        use_merge=True,
    ),
    mipro=MIPROv2(
        metric=scalar_metric,
        auto="light",
        max_bootstrapped_demos=2,
        max_labeled_demos=2,
    ),
)
```

### 2.4 Metric 设计

GEPA 需要**同时返回标量分数和文本反馈**的 metric；MIPROv2/PDO 只需标量分数。

```python
def gepa_metric(example, prediction, trace=None):
    """
    返回: (score, feedback_text)
    - score: 0-1 标量（用于 Pareto 排序，同时用于失败模式分类）
    - feedback_text: 结构化文本（用于 GEPA LLM 反思）
    """
    dialogue = example.dialogue
    popup = prediction.answer

    # 6 维 LLM judge 评分
    scores = judge.evaluate(dialogue, popup)  # accuracy, stance, length, ...

    # 硬规则检查
    violations = check_hard_rules(popup)  # 字数/结构/立场

    # 领域约束保真度
    fidelity = check_fidelity(popup)  # 是否保留了原始约束？

    score = weighted_average(scores, violations, fidelity)

    # 文本反馈（这是 GEPA 的关键输入）
    feedback = f"""
    [弹窗]: {popup}
    [6维评分]: {scores}
    [硬规则违规]: {violations if violations else '无'}
    [领域保真度]: {fidelity}
    [评语]: {scores.get('comment', '')}
    [改进方向]: {suggest_improvement(scores, violations)}
    """

    return score, feedback
```

**失败模式分类**直接复用 metric 返回的 6 维分数——取最低维度作为该 case 的失败类型，不做额外 LLM 调用。

### 2.5 实验设计

| 组 | 策略 | prompt 输入 | 预期 |
|----|------|-----------|------|
| **A (基线)** | 无优化 | `system_prompt.txt` (v1.0 完整领域版) | 当前最优人工基线 |
| **B** | GEPA only | v1.0 完整领域版 | 保持骨架 + 反思改进 |
| **C** | 双层自进化（外层调度 + 内层插拔） | v1.0 完整领域版 | **主角**：骨架保持 + 策略自适应 |
| **D (对照)** | MIPROv2 only | v1.0 完整领域版 | 已知会精简过度（反面对照） |
| **E (对照)** | PDO only | v1.0 完整领域版 | 已知会精简过度（反面对照） |

> 注：C 组取代了之前写死的 BetterTogether(GEPA→MIPROv2)，升级为外层可调度不同内层策略。

**评估方式**：
- 自动评估：6 维 LLM judge 评分（定量）
- 人工审查：3 个场景（manager_anger / couple_chase_dodge / friend_soft_boundary）并排对比
- **核心判据**：优化后的 prompt 是否**保留了原始领域约束**（三种误判场景、生命立场、安好静音等）

---

## 3. 实现路线图

### Phase 1: 核心脚本 `self_evolve.py`（今天）

参照 `evolve.py` 的闭环模式，写一个独立的自进化脚本，而非嵌入 prompt-ops 框架。

**文件**: `use-cases/general-popup-native/self_evolve.py`

```
self_evolve.py 架构:
  class SelfEvolutionRunner:
    # 元控制器 4 功能
    - baseline: 冻结的 origin prompt + 基线评估结果     (③ 基线守护)
    - history: [(round, prompt, scores, strategy, decision), ...]  (④ 版本 & 错误收集)
    - strategy_catalog: {name → compile_fn, metric_fn}   (② 策略调度)

    run():
      round 0 → 基线评估

      round 1..N:
        # ① 失败模式分类：取上一轮最低维度
        failure_mode = classify(scores)  # → "tone" | "fidelity" | "length" | ...

        # ② 策略调度：失败模式 → 策略
        strategy = schedule(failure_mode)

        # ③ 执行内层策略
        candidate = strategy.compile(current_prompt)

        # ④ 评估 & 基线守护
        scores = evaluate(candidate)
        if should_keep(scores, baseline):   # overall↑>0.02 & 约束保留>80%
          current_prompt = candidate
          history.record(round, candidate, scores, strategy.name, "keep")
        else:
          history.record(round, candidate, scores, strategy.name, "discard")
          # 下一轮 schedule() 会读到上次 discard 的失败模式，换策略

        # 收敛: patience=2 或 max_rounds=5
        if converged(): break
```

**关键设计决策**（需确认）：

1. **失败模式分类粒度？**
   - 粗粒度：直接用 metric 6 维中最低维度命名（不调额外 LLM）
   - 细粒度：调 LLM 对失败 case 写一句诊断
   - 推荐粗粒度（最简，够用）

2. **首轮默认策略？**
   - 推荐 `gepa_mipro`（GEPA→MIPROv2），因为尚无失败信号，选覆盖最广的策略

3. **Budget 分配**：
   - 外层: `max_rounds=5`, `patience=2`
   - GEPA 内层: `max_full_evals=5`（每轮 ~60 metric calls）
   - 总计 ~5 × 60 × 2（task+judge）= ~600 API 调用
   - 对比：MIPROv2 `num_trials=10` 用了 ~156s

### Phase 2: Metric 增强

- [ ] 2.1 写 `metric_gepa.py` — 同时返回标量分数 + 文本反馈
  - 在现有 `PopupLLMJudgeMetric` 基础上增加 text feedback 输出
  - 6 维评分 + 硬规则检查 + 评语 → 结构化文本
- [ ] 2.2 写 `metric_scalar.py` — MIPROv2 用的纯标量 metric
  - 复用现有 `metric_judge.py`

### Phase 3: 实验运行

- [ ] 3.1 烟雾测试：`max_rounds=1`, `max_full_evals=1`
- [ ] 3.2 正式运行：`max_rounds=5`, `max_full_evals=5`
- [ ] 3.3 并排对比：基线 vs GEPA only vs self_evolve vs MIPROv2 vs PDO

### Phase 4: 沉淀

- [ ] 4.1 更新 README.md 记录实验结果
- [ ] 4.2 如果 self_evolve 显著优于现有方案，提炼为 `SelfEvolutionStrategy` 加入框架
- [ ] 4.3 提取 Judge 偏差修复方案

---

## 4. 配置模板

### config_gepa.yaml

```yaml
system_prompt:
  file: "system_prompt.txt"
  inputs: ["dialogue"]
  outputs: ["answer"]

dataset:
  adapter_class: "prompt_ops.core.datasets.ConfigurableJSONAdapter"
  path: "dataset.json"
  input_field: "dialogue"
  golden_output_field: "reference_popup"
  train_size: 0.6
  validation_size: 0.2

model:
  name: "deepseek/deepseek-chat"
  task_model: "deepseek/deepseek-chat"
  proposer_model: "deepseek/deepseek-chat"
  adapter_type: "dspy"
  temperature: 0.0
  max_tokens: 4096

metric:
  class: "use-cases/general-popup-native/metric_gepa.py"

optimization:
  strategy: "gepa"
  auto: "medium"                  # light/medium/heavy
  # 或手动预算:
  # max_full_evals: 3
  reflection_minibatch_size: 3
  candidate_selection_strategy: "pareto"
  use_merge: true
  num_threads: 2
  seed: 42
```

### config_bettertogether.yaml

```yaml
# ... (同上 system_prompt, dataset, model, metric)

optimization:
  strategy: "bettertogether"
  pipeline: "gepa -> mipro"       # 阶段顺序

  gepa:
    auto: "light"
    reflection_minibatch_size: 3
    candidate_selection_strategy: "pareto"

  mipro:
    auto: "light"
    max_bootstrapped_demos: 2
    max_labeled_demos: 2
    num_threads: 2
```

---

## 5. 风险与缓解

| 风险 | 概率 | 缓解 |
|------|------|------|
| GEPA 在小数据集（15 条）上反思质量不足 | 中 | 增大 `reflection_minibatch_size`，或从 general-dialogue-popup 借更多测试数据 |
| GEPA 同样受 Judge 偏差影响（反思方向被误导） | 中 | 在反馈文本中显式加入「领域约束保留检查」，引导反思关注保真度 |
| GEPA DSPy 3.2 首次实战，有 bug | 高 | 先用 `auto="light"` 快速验证流程，确认能跑通再加预算 |
| BetterTogether 两个阶段的 metric 不兼容 | 低 | GEPA 用文本反馈 metric，MIPROv2 用标量 metric，在 compile 时各自传入 |
| API 费用过高 | 低 | GEPA `auto="light"` ≈ 几次 full eval，15 条数据成本可控 |

---

## 6. 成功标准

1. **领域约束保留率 > 80%**：优化后的 prompt 保留了生命立场、三种误判场景、安好静音等关键约束
2. **自动评分 ≥ 基线**：6 维 LLM judge 均分不低于基线 prompt
3. **人工审查 ≥ 基线**：3 个场景的并排对比中，至少 2/3 场景优于或持平基线
4. **不再过度精简**：优化后 prompt 长度不低于原始 prompt 的 60%

---

## 附录：参考

- [GEPA 论文: Reflective Prompt Evolution Can Outperform Reinforcement Learning](https://arxiv.org/abs/2505.05250)
- [DSPy GEPA API](https://dspy.ai/api/optimizers/GEPA/overview)
- [DSPy BetterTogether API](https://dspy.ai/api/optimizers/BetterTogether)
- [general-popup-native README](../../use-cases/general-popup-native/README.md) — PDO vs MIPROv2 实测数据
- [general-dialogue-popup evolve.py](../../use-cases/general-dialogue-popup/evolve.py) — 自定义进化循环参考
