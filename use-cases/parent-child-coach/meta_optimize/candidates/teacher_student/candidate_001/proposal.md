# Mutation Proposal · teacher_student · candidate_001

## 失败源分析

### prompt (2 cases)
症状：prompt 措辞/结构导致 tone 误判。症状：system prompt 中对 'diagnostic vs empowering' 的判定标准模糊、边界 case 覆盖不足、或者 prompt 指令自相矛盾。可修——proposer 可以改 prompt 文本。
涉及：C11-005, C11-006

### judge (1 cases)
症状：M5 gold label 本身可能有问题。症状：系统输出的 tone 在语义上合理，但 gold label 标记为不匹配。例如窗口文本本身很模糊、专家标注可能有分歧。不可修——标记为 'judge noise'，不计入优化目标。
涉及：C10-008

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
2 条修改: 原标准对“但”“然而”等转折词的一刀切禁令导致 Teacher 在有可提醒之处时不敢选用 empowering，过度倾向; 在重试环节引入“empowering 优先”的决策倾向，引导 Teacher 在本文那些家长行为并不存在明显错误、更需要

## 修改列表
### Edit 1: harness.md
**原因**：原标准对“但”“然而”等转折词的一刀切禁令导致 Teacher 在有可提醒之处时不敢选用 empowering，过度倾向 diagnostic，使本该鼓励的场景（如 C11-005、C11-006 中 gold 为 empowering）被误判为 diagnostic。修改后以“内容主次”代替“有无转折词”作为核心判据，放宽对轻度提醒的容忍度，让系统更敢于在适当场景输出 empowering 风格。
**影响 case**：C11-005, C11-005, C11-005, C11-005, C11-005

**Before**:
```
- empowering（鼓励式）弹窗必须：
  1. 指出家长的具体做法好在哪里（不是空洞的「你真棒」）
  2. 说明为什么这种做法对孩子有积极影响
  3. 鼓励家长继续强化这种行为
  4. 全文不得出现用"但""然而""不过"等转折引出对家长做法问题或盲区的分析；若出现，视为mixed，判为不一致
```

**After**:
```
- empowering（鼓励式）弹窗必须：
  1. 指出家长的具体做法好在哪里（不是空洞的「你真棒」）
  2. 说明为什么这种做法对孩子有积极影响
  3. 鼓励家长继续强化这种行为
  4. 弹窗的主基调必须是肯定与勉励；如果内容主要是分析和指出家长的盲区或误解（即使语气温和），则应判为 diagnostic；empowering 可以附带极其轻微的、共情式的改进邀请，但不能以诊断或问题分析为主体
```

### Edit 2: harness.md
**原因**：在重试环节引入“empowering 优先”的决策倾向，引导 Teacher 在本文那些家长行为并不存在明显错误、更需要被看见和鼓励的场景中（如 C11-005、C11-006）输出 empowering 而非 diagnostic，从而与 gold label 对齐。此举与修改后的 Validator 标准配合，可系统性减少虚假 diagnostic 的出现。
**影响 case**：C11-005, C11-005, C11-005, C11-005, C11-005

**Before**:
```
2. 如果 Validator 判定不一致：
   a. 将 Validator 的证据 + 建议 → 送回 Teacher
   b. Teacher 收到反馈后，重新审视原文，重新做出 tone 决策
   c. 新的 tone 决策 → 重新 Production → 重新 Validator
```

**After**:
```
2. 如果 Validator 判定不一致：
   a. 将 Validator 的证据 + 建议 → 送回 Teacher
   b. Teacher 收到反馈后，重新审视原文，重新做出 tone 决策。当两个 tone 方向都有合理依据时，优先选择 empowering（鼓励），因为正向强化对家长行为改变更有效；只有当明确看到家长存在明显的认知盲区且鼓励可能强化错误做法时，才选择 diagnostic。
   c. 新的 tone 决策 → 重新 Production → 重新 Validator
```

## 风险
