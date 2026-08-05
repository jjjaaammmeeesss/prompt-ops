# 修复 v4.0.18 管线测试器缺口 · Spec

## 1. 结果

将 `run_v418_pipeline.py` 从"快慢窗口模拟 + 裸调 Stage 2"升级为"快慢窗口模拟 + 生产级 Stage 2 调用"（含周易上下文注入、去抖、tone override、语义去重、P2 话术检查），使其测试结果接近生产管线行为。

## 2. 范围边界

**做**：
- 为每个窗口注入默认周易上下文（不跑真正的 Stage 1 LLM，用中性默认状态）
- 对齐 user_msg 格式（含 type_instruction、输出格式要求）
- 添加简化版去抖门控（同窗口同类型抑制）
- 添加 FC_TONE_OFF 关键词扫描 + tone 强制
- 添加 FC_STALE difflib 语义去重（与历史弹窗比对）
- 添加 P2 鼓励式话术检查（引号内 ≥4 字）
- 对齐窗口参数（慢通道触发 120 字、接线 --window-size）
- 修复 primary_popup 时间顺序（按 popup 对象原始顺序而非分组后）
- 修复 strip_pre_analysis 缺分隔符时的处理（返回 raw_text 并警告）

**不做**：
- 不跑真正的 ZhouYiAnalyzer LLM 调用（增加 1 倍 API 调用量，超出测试管线范围）
- 不改生产代码（realtime/ 只读）
- 不改 channel_spec.py（它定义的是"统一规格"，不是"测试参数"）

## 3. 约束

- 保持 Python import 兼容（litellm + pydantic + difflib，不新增依赖）
- 不改变现有 CLI 接口
- 不改变输出 JSON schema（向后兼容）
- 周易上下文用默认/中性值，明确标注为 "test-default"

## 4. 既有决策

- 默认周易上下文：坤卦（全掌控/稳态），risk=低，container=不适用，tone=auto
  - 这是最中性的状态，不会误导弹窗生成方向
  - 明确标注 `_zhouyi_source: "test-default"` 区别于真实分析
- 去抖门控：只做同窗口类型连续抑制（简化版，不做时间冷却，因为测试是批量非实时）
- 所有新增逻辑放在 `run_v418_pipeline.py` 中，不创建新文件

## 5. 任务拆解

### T1: 添加默认周易上下文 + 对齐 user_msg 格式（#1, #2）
- 创建 `_build_zhouyi_context()` 生成默认卦象上下文
- 修改 `generate_popup()` 的 system message = zhouyi_context + system_prompt
- 修改 user_msg 包含 type_instruction（诊断式/鼓励式引导）
- 相关文件：`scripts/run_v418_pipeline.py`

### T2: 添加 FC_TONE_OFF + P2 话术检查（#4, #6）
- 从 `popup_generator.py` 复制 PARENT_OVERRIDE_KEYWORDS + detect_parent_override
- 添加 `has_quotable_phrase()` 正则检查
- 在 `generate_popup()` 中集成
- 相关文件：`scripts/run_v418_pipeline.py`

### T3: 添加去抖门控 + FC_STALE 语义去重（#3, #5）
- 添加 `_debounce_check()` 简化版去抖
- 添加 `semantic_similarity()` difflib 比对
- 在 `run_case()` 中集成去重逻辑
- 相关文件：`scripts/run_v418_pipeline.py`

### T4: 对齐窗口参数 + 接线 --window-size（#7）
- 慢通道触发改为 120 字（匹配生产 char_trigger）
- 修复 `--window-size` 未传入 `run_case()` 的 bug
- 相关文件：`scripts/run_v418_pipeline.py`

### T5: 修复 primary_popup + strip 处理（#8, #9）
- primary_popup 按 popup 原始时间顺序取最后一个
- strip_pre_analysis 缺分隔符时返回 raw_text 并标记 warning
- 相关文件：`scripts/run_v418_pipeline.py`

### T6: 验证 — 跑 12 题测试
- `python scripts/run_v418_pipeline.py --no-judge` 确认不崩溃
- 检查输出 JSON 中新增字段是否正确

### T7: Codex 审计
- 用 Codex 审计修改后的代码，确认缺口已修复

## 6. 验收标准

- [x] `python scripts/run_v418_pipeline.py --no-judge` 全部 12 题不崩溃
- [x] 每个弹窗的 `raw_response` 中包含 zhouyi_context 注入痕迹
- [x] user_msg 包含 type_instruction
- [x] 输出 JSON 中 `_zhouyi_source: "test-default"` 字段存在
- [x] `--window-size 200` 实际生效（窗口变小，窗口数增多）
- [x] primary_popup 时间顺序正确（fast/slow 混合按触发顺序）
- [x] 同质弹窗被去抖/去重拦截时有日志输出
- [~] Codex 审计确认 #1-#9 全部修复或已标注为有意简化

## 7. Codex 审计结果 (2026-08-05)

| # | 判定 | 说明 |
|---|------|------|
| 1 | PARTIAL | 周易上下文已注入，但默认坤卦+低风险在生产会被 P0 拦截。**有意简化：spec 明确不跑真实 Stage 1** |
| 2 | FIXED | user_msg 含 type_instruction + 输出格式要求 |
| 3 | FIXED | 去抖门控已接入，同通道同 trigger 连续时跳过 |
| 4 | PARTIAL→FIXED | 关键词扫描+override 已接入；**修复：默认 tone 改为 encouraging，使 FC_TONE_OFF 产生真实 encouraging→diagnostic 切换** |
| 5 | FIXED | difflib 语义去重已接入，最近 5 条比对 |
| 6 | NOT_FIXED→FIXED | **修复：默认 tone 改为 encouraging**，P2 条件可达；**修复：重试失败时标记 p2_retry_failed** |
| 7 | FIXED | slow_threshold 参数全链路接线 |
| 8 | FIXED | separator_missing 字段标记缺分隔符情况 |
| 9 | FIXED | popup_order 排序取时间线最后非空弹窗 |

**最终结论**: 9/9 已修复或标注为有意简化。核心修复（默认 tone=encouraging）解开了 FC_TONE_OFF + P2 的阻塞链。
