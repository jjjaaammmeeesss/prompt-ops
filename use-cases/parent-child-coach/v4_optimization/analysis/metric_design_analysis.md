# 评估指标设计方案分析

> 分析对象：`parent-child-coach` v4 MIPROv2 运行链路。结论基于仓库源码、2026-07-16 运行日志，以及临时解包读取的 DSPy 3.2.1 官方 wheel 源码；未运行优化器，未调用任何 LLM。
>
> 版本边界：仓库只声明 `dspy>=2.6.0`，没有 lock file（`D:\prompt-ops\pyproject.toml:18-29`），当前默认 Python 环境也未安装 DSPy。因此无法证明 2026-07-16 那次运行的精确 DSPy patch 版本。运行日志中的模块名、auto 参数和弃用警告与 DSPy 3.2.1 源码一致，本文用 3.2.1 做内部机制核查；与本结论有关的调用链同时由本仓库调用点和运行日志交叉验证。DSPy 源码原始临时位置为 `C:\Users\h\AppData\Local\Temp\dspy-source-3.2.1\dspy\...`，不在仓库内，故无法伪装成 `D:\prompt-ops` 下的文件；项目侧所有引用均使用 `D:\prompt-ops` 绝对路径。

## 1. DSPy MIPROv2 Metric 使用机制分析

### 1.1 先纠正一个框架混淆：本次运行不是 GEPA

本项目明确实例化 `dspy.MIPROv2`，并把自定义 metric 传入构造器（`D:\prompt-ops\src\prompt_ops\core\prompt_strategies.py:390-413`），随后把 trainset/valset 传给 `optimizer.compile(...)`（`D:\prompt-ops\src\prompt_ops\core\prompt_strategies.py:617-636`）。运行日志也明确进入 `dspy.teleprompt.mipro_optimizer_v2` 的 Step 1（`D:\prompt-ops\use-cases\parent-child-coach\v4_optimization\results\run_log.txt:28-41`）。

因此，当前 pipeline 的真实循环是：

1. 生成多组 few-shot candidate；
2. 基于数据、程序和 few-shot candidate 提议 instruction candidate；
3. 用 Optuna/TPE 在“instruction × demo set”离散组合上最大化验证集平均 metric。

GEPA 是 DSPy 的另一个独立 optimizer，不是 MIPROv2 内部阶段。DSPy 3.2.1 的 GEPA wrapper 会捕获 predictor trace，调用支持 `score + feedback` 的 metric，并把反馈交给反思模型提出新 instruction（官方 wheel `dspy\teleprompt\gepa\gepa.py:27-54, 521-566`）。当前 `BasicOptimizationStrategy` 没有实例化 GEPA，`LLMJudgeMetric` 也只接受 `(gold, pred, trace=False)` 并返回 float（`D:\prompt-ops\use-cases\parent-child-coach\scripts\llm_judge_metric.py:190-268`），所以不存在可供本次运行分析的 GEPA Generate-Evaluate-Propose-Apply 回路。

### 1.2 Bootstrapping 中 metric 如何被调用

MIPROv2 把 `self.metric` 和 `metric_threshold` 传给 few-shot demo 生成器（DSPy 3.2.1 `dspy\teleprompt\mipro_optimizer_v2.py:403-448`）。BootstrapFewShot 对每个训练样本先仅用 `example.inputs()` 调 teacher，得到 prediction；随后执行：

```python
prediction = teacher(**example.inputs())
metric_val = self.metric(example, prediction, trace)
success = metric_val >= self.metric_threshold  # 配置了阈值时
```

对应 DSPy 3.2.1 `dspy\teleprompt\bootstrap.py:199-223`。只有 `success` 的 trace 才被转成 augmented demos。因此 metric 在这里不是给候选 instruction 排名，而是一个**是否接纳 teacher 轨迹为 bootstrapped demonstration 的 gate**；返回值在配置阈值时按原始量纲直接比较，不会自动乘 10 或乘 100。

本项目存在一个比“未比较 golden answer”更直接的阻断问题：metric 明确返回 `[0,1]`（`D:\prompt-ops\use-cases\parent-child-coach\scripts\llm_judge_metric.py:193-196,243-268`），配置却设置 `metric_threshold: 7.0`（`D:\prompt-ops\use-cases\parent-child-coach\v4_optimization\config_v4.yaml:26-35`），并原样传给 MIPROv2（`D:\prompt-ops\src\prompt_ops\core\prompt_strategies.py:394-413`）。所以 `metric_val >= 7.0` 永远为假。运行日志与此完全吻合：44 次尝试后 `Bootstrapped 0 full traces`（`D:\prompt-ops\use-cases\parent-child-coach\v4_optimization\results\run_log.txt:94-138`）。正确阈值若意图表达“7/10”应为 `0.70`。

### 1.3 Instruction proposal 与 candidate evaluation 中 metric 如何被调用

需要把“提议”和“评价”分开：

- **提议 instruction 时不直接调用 metric。** MIPROv2 将 trainset、program、demo candidates 交给 GroundedProposer（DSPy 3.2.1 `dspy\teleprompt\mipro_optimizer_v2.py:456-506`）。项目本身也只对 proposer 做 wrapper 并透传这些参数（`D:\prompt-ops\src\prompt_ops\core\prompt_strategies.py:438-516`）。
- **提议之后的搜索评价才调用 metric。** MIPROv2 用 metric 构建 `Evaluate(devset=valset, metric=self.metric, ...)`，再为每个 trial 选择 instruction 和 demo set，运行候选 program，并在每个 val example 上执行 `score = metric(example, prediction)`（DSPy 3.2.1 `dspy\teleprompt\mipro_optimizer_v2.py:220-270, 509-661, 765-782`；`dspy\evaluate\evaluate.py:165-180`）。
- Evaluate 将单样本分数求和并以 `100 * mean(metric)` 形成候选 program 的 score（DSPy 3.2.1 `dspy\evaluate\evaluate.py:180-224`）；Optuna study 的方向是 `maximize`（`dspy\teleprompt\mipro_optimizer_v2.py:661`）。因此 `[0,1]` metric 在候选搜索阶段显示为 `[0,100]` program score，但这与 bootstrap 的原始阈值比较是两套尺度，不能拿 `7.0` 混用。

### 1.4 metric 返回值在各阶段的真实含义

| 阶段 | metric 返回值的用途 | 当前含义 |
|---|---|---|
| Bootstrap | 与 `metric_threshold` 直接比较，决定是否保存 teacher trace | “该输出是否满足五维 rubric”，不表示是否像本条专家答案；且 7.0 阈值使所有 trace 被拒绝 |
| Instruction proposal | 不直接调用 metric；候选由 trainset、程序摘要、demo candidates 等生成 | metric 只能通过上一步 demo 质量间接影响 proposal；当前 bootstrapped demos 为 0，影响被切断 |
| Candidate search | 对 valset 每条 prediction 打分，Evaluate 聚合成 `100 × 平均单条分`，Optuna 最大化 | 最大化 rubric judge 认为的通用质量，而非最大化对本条专家策略的复现 |

## 2. 当前 LLMJudgeMetric 实现诊断

### 2.1 实现事实

当前类的五维及权重是看见感 0.25、对话忠实度 0.20、命中核心 0.20、人话感 0.20、温度 0.15（`D:\prompt-ops\use-cases\parent-child-coach\scripts\llm_judge_metric.py:44-51`）。Judge prompt 只提供 `{dialogue}` 与 `{response}`（`D:\prompt-ops\use-cases\parent-child-coach\scripts\llm_judge_metric.py:138-147`）。调用点只取：

```python
dialogue = self._extract_text(gold, "question")
response = self._extract_text(pred, "answer")
prompt = SCORING_PROMPT.format(dialogue=dialogue, response=response)
```

证据位于 `D:\prompt-ops\use-cases\parent-child-coach\scripts\llm_judge_metric.py:190-206`。代码没有读取 `gold.answer`。Judge 返回的 1–5 分先映射到 0–1，再按有效维度权重归一；veto 或失败返回 0（`D:\prompt-ops\use-cases\parent-child-coach\scripts\llm_judge_metric.py:224-268`）。所以“当前返回 0–10”的描述不符合实际源码；实际是 0–1。

共享 metric 设计恰好展示了相反模式：内置 similarity/correctness template 都同时接收 prediction 与 expected ground truth（`D:\prompt-ops\src\prompt_ops\core\metrics.py:109-143`），默认映射也是 `pred -> output`、`gold -> ground_truth`（`D:\prompt-ops\src\prompt_ops\core\metrics.py:169-180`）。FacilityMetric 也显式从 gold 的 output field 取 ground truth 再比较 prediction（`D:\prompt-ops\src\prompt_ops\core\metrics.py:550-578`）。

### 2.2 已识别问题与证据

1. **目标错位：专家答案不是评价目标。** `gold` 只贡献 question；gold.answer 对本条 prediction 的 score 没有任何影响（`llm_judge_metric.py:198-206`）。若保持 dialogue 不变，把专家答案替换成任意文本，metric 数值分布不变。
2. **“命中核心”仍由 judge 自己定义，而不是由专家标注定义。** Prompt 要 judge 判断“最该被看见的点”（`D:\prompt-ops\use-cases\parent-child-coach\scripts\llm_judge_metric.py:81-86`），但没有告诉 judge 专家认为本条案例的核心是什么。它优化的是 judge 的隐含偏好。
3. **静态校准样本不等于逐例 golden supervision。** Prompt 中 A/B/C 三个固定弹窗只校准评分风格（`D:\prompt-ops\use-cases\parent-child-coach\scripts\llm_judge_metric.py:104-134`），不能让第 i 条 prediction 对齐第 i 条 expert answer。
4. **Bootstrap 阈值量纲错误导致 bootstrapped few-shot 全灭。** 证据链见 1.2；这使“metric 是否比较 gold 会怎样改变 bootstrapped demo”在本次配置下先被更基础的错误遮蔽。
5. **配置与真实数据/切分不一致。** YAML 注释声称 71 条并配置 80/20（`D:\prompt-ops\use-cases\parent-child-coach\v4_optimization\config_v4.yaml:1,10-16,38-46`），实际 JSON 是 56 条；运行日志显示 44 train、11 val、1 test，而不是题设中的 56 train + 14 test（`D:\prompt-ops\use-cases\parent-child-coach\v4_optimization\results\run_log.txt:15-24`）。这不改变 metric 缺陷，但会改变评估方差与优化预算，必须先冻结真实数据版本。
6. **可复现性不足。** `dspy>=2.6.0` 无上限、无 lock（`D:\prompt-ops\pyproject.toml:18-29`）；日志还记录 baseline 调用使用了当前 Evaluate 已不支持的 `return_outputs`（`D:\prompt-ops\use-cases\parent-child-coach\v4_optimization\results\run_log.txt:18`）。在修 metric 前应固定 DSPy 版本，否则内部行为可能漂移。

### 2.3 有效目标函数

忽略调用失败时，当前单样本目标可写为：

\[
m(q, \hat y)=\sum_d w_d\,\frac{s_d(q,\hat y)-1}{4}
\]

其中 \(d\) 是五个 rubric 维度，\(s_d\) 完全由 judge 根据 dialogue \(q\) 与 prediction \(\hat y\) 给出；专家答案 \(y^*\) 不在函数中。MIPROv2 搜索阶段最大化：

\[
\arg\max_{instruction,demos}\;\frac{1}{|V|}\sum_{(q,y^*)\in V}m(q, f_{instruction,demos}(q))
\]

由于 \(m\) 与 \(y^*\) 无关，实际目标是“让 Claude rubric judge 喜欢”，不是“复现专家的 intent/tone/strategy”。高分输出可能是专家也认可的好答案，但这是 rubric 与专家偏好相关所带来的间接结果，不是监督链路保证。

## 3. Golden Answer 在 Pipeline 中的使用追踪

### 3.1 数据进入 pipeline

配置把 JSON 的 `answer` 声明为 `golden_output_field`（`D:\prompt-ops\use-cases\parent-child-coach\v4_optimization\config_v4.yaml:10-16`）。ConfigurableJSONAdapter 会从该字段取 outputs，并放进标准样本的 `outputs`（`D:\prompt-ops\src\prompt_ops\core\datasets.py:174-205,431-458`）。所以 golden answer **确实被加载进 DSPy Example**，不是在数据入口丢失。

### 3.2 各阶段是否有效使用

| 位置 | 是否接触 golden answer | 是否用于决定“预测好不好” |
|---|---|---|
| 数据适配 | 是，成为 example output | 否，只是装载 |
| Labeled demos | 是；`max_labeled_demos: 5` 允许专家问答作为 few-shot 候选（`D:\prompt-ops\use-cases\parent-child-coach\v4_optimization\config_v4.yaml:30-31`） | 间接影响生成，但没有由当前 metric 判断它与预测的对齐程度 |
| Bootstrap teacher 推理 | teacher 只接收 `example.inputs()`；gold 对象随后传给 metric | 当前 metric 只读 gold.question，完全忽略 gold.answer |
| Instruction proposal | GroundedProposer 接收 trainset 与 demo candidates（项目 wrapper：`D:\prompt-ops\src\prompt_ops\core\prompt_strategies.py:460-516`） | 可能把标注答案作为数据/示例上下文，属于生成条件，不属于评价目标 |
| Validation candidate evaluation | Evaluate 将完整 example 传给 metric | metric 仍只读 question，因此 gold.answer 不影响排名 |
| Baseline/test evaluation | 同上 | 同上；且本次 baseline 还因 API 参数不兼容失败（`run_log.txt:18`） |

### 3.3 结论

Golden answers **在结构上被加载，也可能作为 labeled few-shot 或 proposer 上下文被看见；但在决定 bootstrap trace 是否合格、候选 instruction/demo 组合谁胜出、最终 program 谁最佳的目标函数中没有被使用**。这正是“有标签数据但没有 label-sensitive loss”的状态。

若 metric 改为比较 gold.answer，在阈值修为 0.70 后，bootstrapping 会发生实质变化：相同 teacher prediction 在当前 metric 下只需满足通用 rubric；新 metric 下还必须接近本条专家意图/策略，接纳集合通常会改变，生成的 bootstrapped demonstrations 也会不同。无法保证一定更多或更少，但可以确定 selection boundary 改变。按当前错误的 7.0 阈值，无论 metric 是否比较 golden answer，仍会是 0 条 bootstrapped trace，所以必须先修尺度。

## 4. 改进方案对比

| 方案 | 实现工作量 | 单次 eval 成本 | 语义敏感度 | 专家对齐 | MIPROv2 兼容性 |
|---|---:|---:|---|---|---|
| A. N-gram/序列重叠 | 低 | 近零、确定性 | 低；中文分词会显著影响 BLEU/ROUGE | 中低：直接追文字，但可能错罚等价表达 | 高；直接返回 0–1 float，适合 bootstrap 与 Evaluate |
| B. 带 golden 的 LLM judge | 中 | 每例 1 次 judge 调用；因多输入 expert_popup，token 成本明显增加，但不必然“翻倍” | 高 | 高：可直接评价意图、语气、策略是否一致 | 高；返回 0–1 float 即可 |
| C. ROUGE-L gate + golden LLM judge | 中高 | 低重叠样本零 LLM 成本，高重叠样本同 B | 中高，但 gate 可能误杀语义等价改写 | 高于 A；受 gate recall 限制 | 高；注意分段函数和阈值尺度 |

### 4.1 Option A：N-gram overlap

建议中文优先用**字符级 ROUGE-L 或 chrF**，不要直接采用依赖英文式 tokenization 的 BLEU。返回 `rouge_l_f1(pred.answer, gold.answer)`，严格归一到 `[0,1]`。

- **Bootstrapping：** threshold 0.70 会只保留与专家措辞/结构高度接近的 teacher trace；demonstrations 更“像专家”，但可能把高质量改写误判为失败，导致 bootstrapped demo 偏少。
- **Instruction proposal evaluation：** proposal 本身仍不调用 metric；其候选在后续验证中会被奖励复制专家常用词序和篇章结构。
- **优化方向：** 最清晰地“追文本”，未必追到“专家策略”。模型可能通过复用高频句式提高分数，却没有真正命中个案核心。
- **实现注意：** 先固定规范化（Unicode、空白、标点）；否则标点与换行造成无意义波动。不要在 56 条数据上边看 val 结果边反复调分词规则。

### 4.2 Option B：带 golden answer 的 LLM-as-judge

把 judge 输入改为 `(dialogue, predicted_popup, expert_popup)`，明确要求：专家答案不是唯一合法措辞，而是该案例的**意图、核心洞察、介入策略、温度与表达边界的参考标准**。建议维度拆成：专家核心/策略对齐、对话忠实度、语气与温度对齐、自然表达、有效差异说明；仍保留事实错误 veto。

- **Bootstrapping：** threshold 0.70 会接纳“语义和策略与专家一致”的 teacher trace，允许自然改写；比 A 更可能得到质量足够且多样的 demonstrations。
- **Instruction proposal evaluation：** 每个 candidate 在 valset 上都被按专家标注排序，Optuna 才真正获得“靠近专家”的反馈信号。
- **优化方向：** 从“judge 自行判断什么是好亲子弹窗”转为“judge 判断 prediction 是否复现本条专家的 intent/tone/strategy”。
- **风险控制：** temperature=0 不能消除 judge 偏差。需先用独立人工对比集检查 judge-expert rank correlation、重复评分稳定性和位置偏差；专家答案始终放固定位置，并要求 judge 先抽取 expert intent 再比较。

### 4.3 Option C：ROUGE-L gate + golden LLM judge

按题设规则定义：若字符级 ROUGE-L `<0.30`，直接返回 ROUGE-L；否则调用 Option B judge。它节省明显偏离样本的 LLM 成本，并能挡住空输出、跑题和格式垃圾。

- **Bootstrapping：** 低于 0.30 的 trace 不能通过 0.70 bootstrap threshold；高于 gate 后再由语义 judge 决定，能形成较干净 demo 集。
- **Instruction proposal evaluation：** 大量差候选成本更低；接近专家的候选仍由语义判断排序。
- **优化方向：** 先要求最低文本锚定，再追语义策略。但 0.299 与 0.301 会走完全不同路径，且优秀等价改写可能因低字面重叠被永久压低。
- **实现注意：** 不能让 gate 下分数与 judge 分数尺度断裂。题设“低于 0.3 返回 ROUGE-L”可执行，但应先在固定校准集上验证 0.30 的 false-reject rate；未校准前不要把 0.30 当领域事实。

## 5. 推荐方案与实施路径

### 5.1 推荐

**主推荐 Option B；把 Option A 作为离线诊断指标，而不是主目标。**

原因：需求是让 MIPROv2 “chase the expert”，核心是专家的意图、语气和介入策略，不是逐字复刻。Option B 唯一同时做到“逐例使用 golden answer”和“允许有效改写”。当前已有 LLMJudgeMetric、重试、解析、veto、五维加权框架，改动集中在取 `gold.answer` 与 judge prompt，工程增量可控。Option C 应在获得真实成本和低 ROUGE 样本的人工 false-reject 数据后再启用；现在直接上 gate 会把未经验证的 0.30 变成新的隐藏目标。

另一个必须先完成的前置项是：**把 `metric_threshold` 从 7.0 修为与 metric 同量纲的 0.70**。否则无论 A/B/C，bootstrap 都不会工作。

### 5.2 分步实施路径

1. **冻结运行基线。** 固定 DSPy 版本；确认本轮数据究竟是 56、70 还是 71 条，并固定 train/val/test split manifest。当前可核验事实是文件 56 条、日志 44/11/1。
2. **先修尺度，不改语义。** 将 bootstrap threshold 改为 0.70；增加单测：metric=0.69 不接纳、metric=0.70 接纳。避免把 Evaluate 的 0–100 聚合分误当单样本 metric 尺度。
3. **扩展 metric 输入。** 在 `LLMJudgeMetric.__call__` 中读取 `expert_response = _extract_text(gold, "answer")`；任一 dialogue/prediction/expert 缺失时返回 0 并记录可定位错误。
4. **重写 judge contract。** Prompt 显式给出三段内容，先概括 expert 的核心意图与策略，再评 prediction 是否实现相同目标；不要求同词同句。输出保持结构化 JSON，最终仍严格归一 `[0,1]`。
5. **做 judge 校准，不直接跑 MIPROv2。** 建一个不会参与优化的小型人工 pairwise 集，包含：专家原文、优秀改写、表面相似但策略错、低字面重叠但语义对、术语泄漏、事实编造。检查 expert 原文得分上界、优秀改写召回、错误样本区分度、重复评分稳定性。
6. **把 ROUGE-L/chrF 作为旁路观测。** 记录而不参与主分，观察其与 Option B、人工偏好的相关性。只有数据证明 `<0.30` 几乎都是坏输出，才升级为 Option C gate。
7. **小预算 dry evaluation 后再优化。** 不先跑完整 MIPROv2；先对 baseline 与少量固定 prediction 离线计算新 metric，确认 gold.answer 改变会改变 score，确认 bootstrap 能产生非零 traces，再启动正式搜索。
8. **验收优化方向。** 最终不仅报告 MIPRO program score，还报告 held-out 人工 pairwise 胜率、judge-human 相关性、事实错误率、bootstrapped demo 数量与 token/调用成本。测试集在搜索期间必须完全封存。

### 5.3 预期影响

- Bootstrap 从“0 条可用轨迹”恢复为由 0.70 gate 选择的有效轨迹；若采用 Option B，入选依据从通用 rubric 质量转为逐例专家策略对齐。
- Instruction/demo 组合的 Optuna 排名会对 gold.answer 敏感，优化目标才与“更接近专家手写改写”一致。
- 相比纯 ROUGE，能保留专家风格下的多种自然表达；相比当前 judge，减少 judge 自行发明“核心洞察”的自由度。
- 代价是 judge 输入 token 增加、校准工作增加，以及仍需持续监控 judge bias。该代价换来的是目标函数从“看起来像高质量亲子文案”变成“在本条案例上复现专家意图、语气和策略”，这是本次优化要解决的主要矛盾。
