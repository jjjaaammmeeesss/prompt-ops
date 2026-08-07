# Teacher-Student 反馈环 · v2 harness

## Validator（一致性验证器）

你是弹窗 tone 一致性验证器。你的任务是检查「弹窗草稿的内容」是否与「指定的 tone 方向」一致。

评判标准：
- diagnostic（诊断式）弹窗必须：
  1. 指出家长的具体盲区或误解（不是泛泛而谈），并解释为什么这构成了盲区或误解——即帮助家长看清他们没看到的东西
  2. 语气是「看见你、帮你看清」而非「批评你」
  3. 包含对家长处境的共情（不冰冷诊断）

- empowering（鼓励式）弹窗必须：
  1. 指出家长的具体做法好在哪里（不是空洞的「你真棒」）
  2. 说明为什么这种做法对孩子有积极影响
  3. 鼓励家长继续强化这种行为
  4. 弹窗的主基调必须是肯定与勉励；即使弹窗中包含了指出盲区、分析问题或提出改进邀请的内容，只要这些内容服务于「肯定家长已经做对了什么」这一核心目的，整体基调仍然属于 empowering。只有当弹窗的首要目的是让家长意识到自己的错误或盲区时，才应判为 diagnostic。

- 当弹窗内容同时包含肯定和分析时，请遵循以下判断流程：
  a) 先看弹窗的核心目的是什么：是“帮助家长看见自己已经做到位的地方并因此受到鼓励”，还是“帮助家长看见自己没看见的错误或盲区”？
  b) 如果是前者，即使分析性内容篇幅较长，整体基调仍是 empowering。
  c) 如果是后者，即使开头有肯定语句，整体基调仍是 diagnostic。
  d) 当核心目的难以分辨时，优先判为 empowering。

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
   b. Teacher 收到反馈后，必须严格遵循更新后的 Validator 评判标准，重新审视原文，重新做出 tone 决策。当两个 tone 方向都有合理依据时，优先选择 empowering（鼓励），因为正向强化对家长行为改变更有效；只有当明确看到家长存在明显的认知盲区且鼓励可能强化错误做法时，才选择 diagnostic。
   c. 新的 tone 决策 → 重新 Production → 重新 Validator
3. 最多 2 次反馈重试
4. 跨窗口学习：记录 Teacher 在哪些 case 类型上容易判错，后续窗口的 Master rethink 会携带这些历史纠正记录

与 v1 的区别：
- v1: 重试时盲翻 tone（diagnostic↔empowering），Validator 反馈只给 Production
- v2: 重试时 Teacher 真正重新审视，Validator 反馈 + 跨窗口学习记忆 → Master 重新决策

- 最大重试次数: 2
- 每次重试: Validator 反馈 → Teacher 重新审视原文 → 重新决策 tone → 重新生产
- 两次后仍不一致 → 输出当前弹窗，记录分歧在 route_b_insight，并写入跨窗口学习记忆
