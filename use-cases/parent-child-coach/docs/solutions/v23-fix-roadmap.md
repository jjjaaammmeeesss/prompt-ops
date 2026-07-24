# v2.3 修复路线图 (FIX_ROADMAP)

---

## 一、速效修复（1 天内，立即做）

### 1.1 DECIDE Bug 修复

**问题**：`should_keep(baseline, candidate)` 的语义被三个调用方同时破坏——`baseline` 参数在第二轮起传入的是"上一轮被保留的 candidate"，而非 v2.3 原点。v2.4 R2 (score=0.684) 与 v2.4 R1 (score=0.666) 比较被判 KEEP，但与 v2.3 (score=0.686) 比较应判 DISCARD。一个退化版本被错误保留。

**修复**（涉及 `optimizer.py`、`v23_evolve.py`、`run_auto_evolve.py` 三个文件，相同模式）：

在 `optimizer.py` L637 附近，将：

```python
baseline = candidate  # L650: 语义污染，baseline 不再是原点
```

改为引入独立变量：

```python
origin_baseline = baseline  # 冻结原点，永不被覆盖
# ...
should_keep(origin_baseline, candidate)  # L637: 始终与原点比
```

`v23_evolve.py` L568-581 的 `prev_path` 逻辑同理：Round 2+ 不应加载 `round_{N-1}_eval.json` 作为 `prev_report`，而应始终使用 `baseline_round_000.json`。

**预期效果**：消除退化版本被错误保留的系统性风险。修正后 v2.3→v2.4→v2.5 三轮全部 DISCARD，最终 prompt 保持在 v2.3 (0.686)。

---

### 1.2 M1 崩溃修复（错误处理 + 输出格式加固）

**问题 A**：模型在 borderline case 上以 ~20-40% 概率输出 `{"type": "诊断式弹窗", "popup_text": ""}`——合法 JSON、无 error 字段、但空内容。当前代码对此零感知，直接计入 M1=0。

**修复 A** — `optimizer.py` `_evaluate_case_once`（约 L218 之后），在 JSON 解析成功后增加空输出检测：

```python
# 在 sys_popup_text 提取之后，M1 判定之前
if not sys_popup_text or not sys_popup_text.strip():
    # 空输出视为"模型未完成生成"，触发重试（最多 2 次）
    if retry_count < MAX_EMPTY_RETRIES:
        continue  # 重新调用 API
```

**问题 B**：降噪逻辑 `_denoise_case_runs`（L282）对 M1 使用严格多数投票——3 次 run 中 2 次空输出即判 M1=0。但空输出是模型非确定性噪声，不是"模型认为不该弹窗"。

**修复 B** — `_denoise_case_runs` 中，对 `should_popup` 的聚合改为"至少一次有效输出"策略：

```python
# 对于 should_popup 判定：只要有 >=1 次 run 产出了非空 popup_text
# 就取那些有效 run 的 should_popup 值，而非对所有 run（含空输出）做多数投票
valid_runs = [r for r in case_runs if r.get("sys_popup_text", "").strip()]
if valid_runs:
    # 在有效 run 中做多数投票
    should_popup = majority_vote([r["sys_should_popup"] for r in valid_runs])
else:
    should_popup = False  # 所有 run 都空，才判不弹窗
```

**问题 C**：提示词层面，模型在"不知道该说什么"时选择了沉默。

**修复 C** — 在 prompt 末尾（所有规则之后）加一句 fallback 指令：

```
如果你看完整个对话，一时不知道该说什么——就说出你看到的最简单的事实。
比如"你刚才停顿了一秒"或"孩子看了你一眼"。不要什么都不说。
```

**预期效果**：M1 从当前 50-52% 恢复到 65-70%（v2.3 n=1 水平）。空输出 case 数从 26 降到接近 0。

---

## 二、中效重构（3-5 天）

### 2.1 Tone 问题的 Prompt 人肉重写

**诊断回顾**：v2.3 的 diagnostic bias 不是"缺少鼓励规则"造成的，而是三个结构特征锁死的：(a) 角色定义隐含"发现盲区"为默认姿态；(b) 输出格式要求先命名 `contradiction`（矛盾点），priming 模型走向诊断；(c) 内部推理三步（情绪→动机→决策）是分析框架，天然导向"我发现了什么问题"。

Mutator 的"加规则"策略在三轮迭代中反复失败（0.686→0.684→0.646），因为它在症状层打补丁而非动架构。

**重写大纲**：

```
# Prompt · v2.6 — 结构重写

## 一、你信什么（价值观，移到最前面）
- 人先被看见，才可能松动。看见包括：看见他在努力什么、他在乎什么、
  他被什么困住了。
- 你不是纠错器。你是镜子——先照见他已有的光，再照见他的盲区。
- 语气不是你选出来的，是你真的看见了之后自然流出来的。

## 二、你默默地做什么（内心操作，简化为两步）
1. 读对话：这个家长此刻在努力做什么？他卡在什么地方？
2. 决定说什么：先说出你看见的努力（哪怕只有一点点），
   再说出他可能没看见的视角。

## 三、你怎么说（说话框架，替代 type 分类）
- 不区分"诊断式"和"鼓励式"。你只说一句话，这句话里同时有"看见"
  和"新的视角"。
- 格式：{"popup_text": "..."}
  popup_text 是一句 15-40 字的中文，面向家长，像朋友在旁边轻轻说。
- 删除 contradiction 字段。judge 从 popup_text 中自行判断洞察质量。

## 四、不能做的事（硬规则，精简为 3 条）
1. 不能否定孩子感受（如"别哭了""这有什么好怕的"）
2. 不能笼统表扬（如"你真棒""继续加油"）
3. 不能什么都不说——哪怕只说"你刚才看了孩子一眼"也可以

## 五、范例（替代禁令，2 对对话-弹窗）
示例 1：
对话：妈妈："你怎么又把衣服扔地上了！我说了多少次！"
弹窗：「你说过很多次，你累了。不是衣服的问题——是你一直在做，却好像
没人看见。先停一下，你已经在尽力了。」

示例 2：
对话：爸爸："他数学考了 98，为什么不是 100？那 2 分丢哪了？"
弹窗：「98 分已经很好了。你在意的不是那 2 分——你怕他不够好。
但他够好了，你也是。」
```

**关键改动**：

| 改动 | 旧 (v2.3) | 新 (v2.6) | 理由 |
|------|----------|----------|------|
| 顺序 | 角色→格式→价值观→内心操作→规则 | 价值观→内心操作→说话框架→规则→范例 | 先确立姿态再做分析 |
| type 分类 | `诊断式弹窗` / `鼓励式弹窗` 二选一 | 删除，不要求模型自选类型 | 消除"选哪个"的认知负担 |
| contradiction 字段 | 要求输出矛盾点 | 删除 | 消除"先找矛盾"的 diagnostic priming |
| 决策流程 | 串行门控（先找对的→有则鼓励） | 并行（同时看见努力和盲区） | 消除规则冲突导致的决策瘫痪 |
| 范例 | 无 | 2 对对话-弹窗 | 模型从范例学习比从禁令学习更稳定 |
| 长度 | ~1707 chars | 预期 ~1200 chars | 更短=中后段指令遵循度更高 |

**预期效果**：M5 从 0.34-0.40 提升到 0.55-0.65（tone 分布从 70% diagnostic 转为更均衡）。M1 不再因规则冲突而崩溃。

---

### 2.2 Mutator 策略切换：从"加规则"到"多策略探索"

**诊断回顾**：当前 Mutator 的策略空间是单维的——它只有"加规则"这一个动作。FailureReport 正确统计了失败模式（"expected empowering, got diagnostic"），但错误引导 Mutator 每次都做出"加更多鼓励规则"的回应。缺少一个关键的元分析层：判断失败是由"规则缺失"还是"框架矛盾"造成的。

**重构方案**：

每轮变异生成 N 个异构候选（必须来自不同策略族）：

| 策略族 | 触发条件 | 动作 |
|--------|---------|------|
| **A — 删减** | M1 下降 + prompt 长度增长 | 删除 contradiction 字段 / 删除 type 分类 / 合并重复规则 |
| **B — 重排** | tone 偏差持续 2 轮未改善 | 价值观移到格式前面 / 范例移到规则前面 |
| **C — 范例化** | 某条规则在 failure report 中被反复提及但未改善 | 选失败率最高的 1 个 case，将其 gold 对话-弹窗作为范例嵌入 |
| **D — 参数调整** | M1 波动大（run-to-run 一致性低） | temperature 0.3→0.1 或 0 |
| **E — pass** | 连续 2 轮无显著改善 | 保持当前 prompt，改变评估策略（n_runs 增加、对抗案例池刷新） |

**每轮选择逻辑**：
1. 读取 FailureReport，提取失败模式
2. 对每个策略族，计算"匹配度分数"（该族是否针对当前失败模式）
3. 选匹配度最高的 3 个族，各生成 1 个候选
4. 每个候选独立评估，should_keep 与原点对比
5. 如果多候选同时通过，选 overall score 最高的

**预期效果**：突破"加规则→变长→M1 崩溃→加更多规则→更崩溃"的死循环。策略空间从 1 维扩展到 5 维。

---

### 2.3 对抗用例生成策略重构

**诊断回顾**：(a) `tone_blindspot_diagnostic_bias` 模式三轮零产出——因为 LLM 无法将"diagnostic bias 的触发条件"与"普通 empowering 案例"区分开；(b) `tone_blindspot_empowering_bias` 模式的预验证逻辑与 v2.3 真实行为反向——v2.3 几乎总是输出 diagnostic，导致 `expected_tone=diagnostic` 的案例在预验证中因 tone 匹配而全部通过，无法捕获 empowering bias；(c) 产出的 8 个案例高度同质化，6 个共享同一模板。

**重构方案**：

**A. 预验证过滤器模式感知化**

```python
# 当前（模式无关）
if sys_tone != expected_tone:
    keep()  # tone mismatch → 保留

# 改为（模式感知）
if mode == "tone_blindspot_empowering_bias":
    # 该模式目标：系统应 diagnostic 但错误 empowering
    # v2.3 下几乎不可能触发——用 M6 作为代理信号
    if sys_tone == "empowering" and m6_score < 3.0:
        keep()
elif mode == "tone_blindspot_diagnostic_bias":
    # 该模式目标：系统应 empowering 但错误 diagnostic
    if sys_tone == "diagnostic" and expected_tone == "empowering":
        keep()
```

**B. 为 LLM 生成器提供具体锚点**

```markdown
# 旧描述（泛化，LLM 无法操作）
"系统倾向于在应该鼓励时错误输出 diagnostic"

# 新描述（具体锚点，LLM 可模仿变异）
"以下 2 个案例中系统犯了同样的错误——家长已主动调整，系统仍输出诊断：
案例 A：妈妈道歉并接纳孩子感受，系统说'你还没真正理解他'
案例 B：孩子自我协商成功且家长全程未升级，系统说'你错过了关键信号'
请生成 3 个同一失败模式的新案例，变化：孩子年龄段 / 冲突类型 / 家长性别。"
```

**C. 多样性强制约束**

在生成 prompt 中加：
```
已产出的案例场景：[列出已有案例的 1 句话摘要]
新案例必须避开上述场景。至少覆盖以下维度中尚未出现的组合：
- 孩子年龄：3-6岁 / 7-12岁 / 13-18岁
- 冲突类型：学业 / 生活习惯 / 社交 / 情绪
- 家长性别：爸爸 / 妈妈
```

**预期效果**：对抗案例池从 8 个同质案例扩充到 15-20 个多样化案例，覆盖至少 3 种不同失败模式，在 should_keep 决策中提供有效区分度。

---

## 三、长效架构改进（需要重新设计）

### 3.1 Co-evolution 机制是否适用单 Prompt 优化？

**当前问题**：Co-evolution（prompt + 对抗案例交替进化）的理论假设是 prompt 和测试集在军备竞赛中互相提升。但 v2.3-v2.5 三轮实际运行暴露了两个根本矛盾：

1. **对抗案例生成器与被测 prompt 共享同一个 LLM**。当被测 prompt 有系统性偏差（diagnostic bias）时，同一个 LLM 驱动的生成器也无法识别该偏差的边界——因为 LLM 自身的认知框架与被测 prompt 同源。"tone_blindspot_diagnostic_bias 三轮零产出"就是证据。

2. **62 个 golden case 已占总池的 ~89%**，对抗案例的信号被淹没。在 should_keep 决策中，8 个对抗案例无法扭转由 62 个 golden case 主导的整体分数方向。

**建议**：
- **短期**：保留 co-evolution 框架，但对抗案例的预验证改用独立 judge model（如百度 GLM），与被测模型（DeepSeek）解耦。
- **长期**：如果连续 3 轮对抗案例的增量区分力不显著（定义为：移除对抗案例前后 should_keep 判定不变），则冻结对抗生成，改为人工标注新的 golden case。Co-evolution 退化为"定期注入人工标注的对抗案例 + LLM 自动变异 prompt"的半自动流水线。

---

### 3.2 是否需要引入 Human-in-the-Loop？

**当前痛点**：
- Mutator 的策略空间是单维的（只会加规则），因为 FailureReport 只给统计不给归因。
- Judge 的 M6/M7 评分噪声高达 0.057（同 prompt 两次评估的分数差），超过了 should_keep 的 MIN_OVERALL_IMPROVEMENT 阈值 (0.003)。
- v2.6 的人肉重写是需要的——它基于对 prompt 结构的因果理解，而非统计信号。

**建议的分层人机协作模型**：

```
Layer 1 — 全自动（当前层）：对抗生成 + prompt 变异 + 评估 + keep/discard
  适用：微调（改 1-2 句话、调参数）、回归检测（新 prompt 不退化）

Layer 2 — 人审核（新增）：当连续 2 轮 DISCARD 或 M1/M5 波动 >15% 时触发
  动作：人工阅读 FailureReport + 当前 prompt，判断根因是"规则缺失"还是"框架矛盾"
  输出：策略方向指令（"删 contradiction 字段" / "加重排" / "加范例"）
  工具：一个简单的 YAML 配置界面

Layer 3 — 人主导（新增）：每 5 轮或每 2 周触发一次
  动作：人工重写 prompt 架构，作为新的 origin baseline
  输出：新的 baseline_round_000.json
  触发条件：连续 5 轮无改善 OR 生产环境用户反馈出现新模式
```

**不引入 HITL 的风险**：Mutator 在单维策略空间中做梯度下降，最终收敛到局部最优——一个臃肿、矛盾、M1 崩盘的 prompt。v2.5 (0.646) 就是这个局部最优的实例。

---

### 3.3 Judge Model 一致性（切换到百度 GLM 后的注意事项）

**需要验证的项**：
1. **评分分布偏移**：GLM 和 DeepSeek 在 M6/M7 的 1-5 分制上的均值和中位数是否有系统性差异。用 62 个 golden case 做双 judge 对比，计算 Cohen's Kappa。
2. **Tone 分类一致性**：GLM 对 `diagnostic` vs `empowering` 的分类标准是否与 DeepSeek 一致。尤其关注 borderline case（如中性陈述）。
3. **噪声水平**：GLM judge 的 test-retest 可靠性（同 case 同 prompt 重复评分的一致性）。如果噪声 > DeepSeek 的 0.057，should_keep 的阈值需要重新校准。
4. **对空输出的评分行为**：GLM judge 如何处理 `popup_text=""` 的 case？是给 M6=1 还是跳过？

**建议在正式切换前跑一个校准脚本**：

```python
# calibration.py 伪代码
for case in golden_62:
    run_deepseek_judge(case)  # 现有 judge
    run_glm_judge(case)       # 新 judge
report:
    - M5 agreement (Cohen's Kappa)
    - M6 Pearson r, MAE
    - M7 Pearson r, MAE
    - 空输出 case 的处理差异
```

---

## 四、执行优先级矩阵

```
                    预期收益
              低          中          高
          ┌───────────┬───────────┬───────────┐
    低    │ 1.2-C     │ 1.2-A     │ 1.1       │
          │ fallback   │ 空输出    │ DECIDE    │
          │ 指令      │ 重试      │ bug 修复   │
          ├───────────┼───────────┼───────────┤
实  中    │ 2.3-A     │ 2.2       │ 1.2-B     │
现        │ 预验证    │ Mutator   │ 降噪宽松   │
难        │ 模式感知  │ 多策略    │ 策略      │
度   ├───────────┼───────────┼───────────┤
          │ 2.3-C     │ 3.3       │ 2.1       │
    高    │ 多样性    │ Judge     │ Prompt    │
          │ 约束      │ 校准      │ 人肉重写   │
          ├───────────┼───────────┼───────────┤
          │ 3.1       │ 3.2       │           │
          │ Co-evol   │ HITL      │           │
          │ 退化为    │ 分层模型  │           │
          │ 半自动    │           │           │
          └───────────┴───────────┴───────────┘
```

### 执行顺序建议

**第 1 天（立即）**：
1. 修 DECIDE bug（1.1）—— 改 3 个文件，<30 行代码，消除系统性风险
2. 空输出重试（1.2-A）—— `_evaluate_case_once` 加 5 行
3. 降噪宽松策略（1.2-B）—— `_denoise_case_runs` 改 10 行
4. fallback 指令（1.2-C）—— prompt 末尾加 3 行

**第 2-3 天**：
5. Prompt 人肉重写（2.1）—— 产出 v2.6 初稿，在 62 个 golden case 上跑 baseline
6. 如果 v2.6 M5 > 0.50 且 M1 > 0.65，进入下一轮自动变异

**第 4-5 天**：
7. Mutator 多策略重构（2.2）—— 给 Mutator 加删减/重排/范例化三个新策略族
8. 对抗用例生成重构（2.3）—— 预验证模式感知 + LLM 锚点 + 多样性约束

**第 2 周**：
9. Judge 校准（3.3）—— GLM vs DeepSeek 双 judge 对比
10. HITL 分层模型（3.2）—— 先实现 Layer 2（失败时人工审核），Layer 3 按需触发

**第 3-4 周**：
11. Co-evolution 评估（3.1）—— 基于 v2.6+ 的 3 轮运行数据，判断是否退化为半自动
