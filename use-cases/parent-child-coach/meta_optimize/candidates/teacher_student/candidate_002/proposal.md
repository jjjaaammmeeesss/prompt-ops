# Mutation Proposal · teacher_student · candidate_002

## 失败源分析

### prompt (1 cases)
症状：prompt 措辞/结构导致 tone 误判。症状：system prompt 中对 'diagnostic vs empowering' 的判定标准模糊、边界 case 覆盖不足、或者 prompt 指令自相矛盾。可修——proposer 可以改 prompt 文本。
涉及：C11-005

### judge (3 cases)
症状：M5 gold label 本身可能有问题。症状：系统输出的 tone 在语义上合理，但 gold label 标记为不匹配。例如窗口文本本身很模糊、专家标注可能有分歧。不可修——标记为 'judge noise'，不计入优化目标。
涉及：C10-005, C10-008, C11-006

### dataset (0 cases)
症状：该 case 的窗口文本本身模糊，任何模型都难以判断 tone。症状：窗口文本截断了关键上下文、对话处于 tone 切换的临界点、或 case 本身存在标注歧义。不可修——标记为 'hard case'，降低权重。
涉及：

### search (0 cases)
症状：proposer 陷入局部最优。症状：连续多次修改方向相同但无改善，或反复在 2-3 个版本间振荡。可修——切换可变参数、加大变异幅度、或从不同角度切入。
涉及：

### model (0 cases)
症状：DeepSeek 对该类语义存在稳定的错误边界。症状：同一 case 在不同 prompt 版本下都稳定输出错误 tone。这是模型能力天花板——prompt 修不了，需要架构级改动（如引入 multi-agent）。不可修在 L1 层面。
涉及：

## 整体策略
2 条修改: 多条 empowering 失败（C11-005 w1/w3/w4/w5）是因为弹窗在鼓励中混入了诊断性分析（“但...; 强化对 empowering 弹窗中转折分析症状的识别，确保 Validator 将带“但”的混合内容明确定性为不一致，

## 修改列表
### Edit 1: harness.md
**原因**：多条 empowering 失败（C11-005 w1/w3/w4/w5）是因为弹窗在鼓励中混入了诊断性分析（“但...”），导致系统判为 diagnostic 或 mixed，与 gold empowering 不匹配。加入此强制规则后，Validator 会将此类弹窗判定为不一致，从而触发重试生成纯鼓励内容。
**影响 case**：C11-005, C11-005, C11-005, C11-005

**Before**:
```
- empowering（鼓励式）弹窗必须：
  1. 指出家长的具体做法好在哪里（不是空洞的「你真棒」）
  2. 说明为什么这种做法对孩子有积极影响
  3. 鼓励家长继续强化这种行为
```

**After**:
```
- empowering（鼓励式）弹窗必须：
  1. 指出家长的具体做法好在哪里（不是空洞的「你真棒」）
  2. 说明为什么这种做法对孩子有积极影响
  3. 鼓励家长继续强化这种行为
  4. 全文不得出现用“但”“然而”“不过”等转折引出对家长做法问题或盲区的分析；若出现，视为mixed，判为不一致
```

### Edit 2: harness.md
**原因**：强化对 empowering 弹窗中转折分析症状的识别，确保 Validator 将带“但”的混合内容明确定性为不一致，与上述 empowering 定义修改配套，使重试环节能生成纯 empowering 文本。
**影响 case**：C11-005, C11-005, C11-005, C11-005

**Before**:
```
不一致的典型症状：
- 弹窗标签=empowering，但内容实际上是分析家长的盲区 → 这是假 empowering
- 弹窗标签=diagnostic，但内容实际上是肯定家长的做法 → 这是假 diagnostic
- 弹窗标签和内容方向一致但程度不够 → 仍然算一致（不因"不够好"而判不一致）
```

**After**:
```
不一致的典型症状：
- 弹窗标签=empowering，但内容实际上是分析家长的盲区 → 这是假 empowering
- 弹窗标签=empowering，但使用了“但”“然而”“不过”等转折词并随后指出家长的问题或盲区 → 这同样是假 empowering（实质为 mixed）
- 弹窗标签=diagnostic，但内容实际上是肯定家长的做法 → 这是假 diagnostic
- 弹窗标签和内容方向一致但程度不够 → 仍然算一致（不因"不够好"而判不一致）
```

## 风险
