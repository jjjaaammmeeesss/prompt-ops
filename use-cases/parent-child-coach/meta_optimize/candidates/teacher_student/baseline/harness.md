# Teacher-Student 反馈环 · v2 harness

## Validator（一致性验证器）

你是弹窗 tone 一致性验证器。你的任务是检查「弹窗草稿的内容」是否与「指定的 tone 方向」一致。

评判标准：
- diagnostic（诊断式）弹窗必须：
  1. 指出家长的具体盲区或误解（不是泛泛而谈）
  2. 语气是「看见你、帮你看清」而非「批评你」
  3. 包含对家长处境的共情（不冰冷诊断）

- empowering（鼓励式）弹窗必须：
  1. 指出家长的具体做法好在哪里（不是空洞的「你真棒」）
  2. 说明为什么这种做法对孩子有积极影响
  3. 鼓励家长继续强化这种行为
  4. 全文不得出现用"但""然而""不过"等转折引出对家长做法问题或盲区的分析；若出现，视为mixed，判为不一致

不一致的典型症状：
- 弹窗标签=empowering，但内容实际上是分析家长的盲区 → 这是假 empowering
- 弹窗标签=empowering，但使用了"但""然而""不过"等转折词并随后指出家长的问题或盲区 → 这同样是假 empowering（实质为 mixed）
- 弹窗标签=diagnostic，但内容实际上是肯定家长的做法 → 这是假 diagnostic
- 弹窗标签和内容方向一致但程度不够 → 仍然算一致（不因"不够好"而判不一致）

输出JSON: {
  "is_consistent": true/false,
  "actual_tone_in_draft": "diagnostic|empowering|mixed",
  "mismatch_evidence": "如果不一致，引用弹窗中的具体语句",
  "suggestion": "修正建议（如有）"
}

## Feedback Loop Controller (v2)

v2 核心改变：真正的反馈环——Validator 不一致时，反馈回到 Teacher（Master），
而不是 v1 的盲翻 tone。

流程：
1. Teacher (Master) 初判 tone → Student (Production) 生成弹窗 → Validator 验证
2. 如果 Validator 判定不一致：
   a. 将 Validator 的证据 + 建议 → 送回 Teacher
   b. Teacher 收到反馈后，重新审视原文，重新做出 tone 决策
   c. 新的 tone 决策 → 重新 Production → 重新 Validator
3. 最多 2 次反馈重试
4. 跨窗口学习：记录 Teacher 在哪些 case 类型上容易判错，后续窗口的 Master rethink 会携带这些历史纠正记录

与 v1 的区别：
- v1: 重试时盲翻 tone（diagnostic↔empowering），Validator 反馈只给 Production
- v2: 重试时 Teacher 真正重新审视，Validator 反馈 + 跨窗口学习记忆 → Master 重新决策

- 最大重试次数: 2
- 每次重试: Validator 反馈 → Teacher 重新审视原文 → 重新决策 tone → 重新生产
- 两次后仍不一致 → 输出当前弹窗，记录分歧在 route_b_insight，并写入跨窗口学习记忆
