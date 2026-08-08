# v4.0.19 预处理实现位置决策 · Spec

> 决策链：测试执行器跟随生产 1:1 → 定位预处理 → 提示词为主、代码兜底为修复手段。

## 1. 结果

v4.0.19 的对话预处理（错字修正 + 说话人归属）由**生产提示词**实现（`system_prompt_v4.0.19.txt` 预分析三步），测试执行器不再自建复刻逻辑。代码侧死代码（`attribute_speakers` / `generate_popup` / 孤儿辅助函数）已从 `scripts/run_v4019_pipeline.py` 删除。

## 2. 范围边界

**做**：
- 删除测试执行器 `run_v4019_pipeline.py` 中未启用的代码侧预处理（`attribute_speakers`、`generate_popup` 及其孤儿辅助函数）
- 删除独立死函数 `_resolve_declared_type` + `_DECLARED_TYPE_MAP`（零调用点，用户 2026-08-06 批准）
- 记录"提示词为主"的决策依据

**不做**：
- 不改生产 `realtime/popup_generator.py`
- 不改生产提示词 `system_prompt_v4.0.19.txt`
- 不引入新的代码层预处理

## 3. 约束

- 测试执行器 = 生产执行器 1:1（`PopupGenerator.generate()` 内部闭环，测试不自建）
- 铁律 #1 只测不改：不修改被测系统代码
- 窗口规格单一真源 `channel_spec.py`：300 字本片段 + 往前 900 字滑动窗口
- prompt↔执行器同步闸门 0 FAIL

## 4. 既有决策

- 测试执行器直接 import 生产 `realtime/popup_generator.py`（PROMPT_VERSION=v4.0.19）
- 窗口切分：fast 通道 250~300 字本片段 + 900 字前文；slow 通道 300 字段 + 900 字前文；两段相邻连续、无重叠
- 评估模型 ≠ 被测模型（铁律 #6）：glm-5.2 评估 deepseek

## 5. 任务拆解

- [x] 定位预处理实现位置：确认在提示词（`system_prompt_v4.0.19.txt:129-130`）而非执行器代码
- [x] 论证提示词 vs 代码哪个更好 → 提示词为主、代码兜底
- [x] 确认 `compare_v4012_v4019.py` 不 import 待删函数
- [x] 删除 `attribute_speakers` + `generate_popup` 及孤儿辅助函数
- [x] 编译校验 + 同步闸门 0 FAIL
- [x] 删 `_resolve_declared_type` 独立死代码（用户 2026-08-06 批准）

## 6. 验收标准

- [x] `run_v4019_pipeline.py` 中 `attribute_speakers` / `generate_popup` / `_resolve_declared_type` / `_DECLARED_TYPE_MAP` 零残留
- [x] `py_compile` 通过，`compare_v4012_v4019.py` 可解析
- [x] `check_prompt_executor_sync.py` 0 FAIL

---

## 决策依据（为何提示词为主）

| 维度 | 提示词实现 | 代码实现（attribute_speakers） |
|------|-----------|------------------------------|
| 项目哲学 | ✅ 契合「告诉 LLM 怎么思考」 | ❌ 复刻「替 LLM 做决定」的规则引擎 |
| 与生产一致 | ✅ 生产本来就这么干（预分析三步在 prompt） | ❌ 测试独有，偏离生产，造成"测试≠生产"漂移 |
| 维护成本 | ✅ 只改 prompt 一处，prompt↔执行器同步闸门兜底 | ❌ 双份实现，改一处漏一处（正是 v4.0.19 低分根因） |
| 出错兜底 | 需在 prompt 内强调（错别字/归属规则） | 代码层可强制，但污染生产一致性 |

**结论**：预处理必须在提示词内实现（生产现状），代码仅作为修复手段保留。若提示词修复不彻底（如 C5-004 张冠李戴），**修复方向是改提示词**，而非在测试执行器里补代码层预处理——否则再次制造"测试≠生产"漂移。

---

## 后续决策：测试执行器完整模拟真实调用链（2026-08-06）

### 决策
`run_v4019_pipeline.py` v1.3：从 mock 默认卦象升级为**逐窗口真实执行生产完整链路**——
`真实 ZhouYiAnalyzer(Stage1) → P0 硬拦截 → 真实 DebounceGate → 生产 PopupGenerator(Stage2)`。
窗口切分**保持 300+900 死规定不碰**（绝不回 TextBuffer 3000+500）。`analyzer=None` 时 mock 回退，供历史对比脚本。

### 关键修复
**负时间去抖 bug**：`DebounceGate` 跨 case 共享但 `sim_now` 每 case 从 0 重置，导致 `elapsed = 20 - 上个case的sim_now` 变负 → 全部被"绝对最小间隔未到"误拦。修复：每 case `debounce.reset()`（对齐生产 `StreamOrchestrator.reset`）。

### 全量结果（12 题）
- 有弹窗 **8 题**（C10-004/005/006/008、C5-003/004/005、C3-002）
- 0 弹窗 **4 题** = 全部 P0 硬拦截（判定坤卦+低风险+不适用）：
  - C10-002（睡不着）/C10-003（过马路）→ **Stage1 误判**（明显该弹被判稳态）
  - C5-001（摆碗）/DS_001（收盘子）→ **P0 过度保守**（和谐但有教育契机，标注 `should_popup:true`）

### 暴露的生产问题
1. **Stage1 识别力**：C10-002/C10-003 冲突场景被判"坤/低"→ 误拦，需改善 ZhouYiAnalyzer 对非激烈冲突但存在情绪/安全风险场景的判定。
2. **P0 过度保守**：把"表面和谐但有教育契机"的对话当纯日常拦，可能与"教育契机也是弹窗价值"的测试口径冲突。
3. **测试口径分叉**：真实调用链拦掉一半样本 → V1-V6 弹窗质量评估样本不足。**待用户定夺**：忠实端到端 / 强制生成测质量 / 两轨并存。

### 版本
`run_v4019_pipeline.py` __version__ 1.2 → 1.3。
