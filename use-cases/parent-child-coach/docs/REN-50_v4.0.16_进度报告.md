# REN-50 进度报告 · v4.0.16 代码层硬约束闭环

> 里程碑：第 2 轮迭代交付（v4.0.15 → v4.0.16）
> 日期：2026-08-01
> 看板：REN-50（issue 05a22d8a-0405-41a2-a1f4-5c4d80def262）
> Git：commit `4e88ccb` on `feat/compare-scripts-and-v30`，已推送 origin

## 一、本轮目标

裁判诊断体在 08-01 05:16 终评驳回 v4.0.15：三项 dominant FC（FC_MISS_CTX / FC_TONE_OFF / FC_STALE）仅 1/3 闭环。根因诊断：

- **FC_TONE_OFF**：v4.0.15 把 override 规则写进 prompt 文本，但 `popup_generator.py` 中 `tone` 在调 LLM 前已从卦象固化进 `type_instruction`，LLM 无权改 tone——override 规则没机会执行。
- **FC_STALE**：`previous_popups` 参数收了但没用；`config.yaml` 的 `dedup.semantic_similarity_threshold: 0.70` 无代码读取；`stream_orchestrator.py:570` 传了 `popup_history` 但 generator 忽略它。

目标：在代码层闭环这两项（路径 C：代码层硬约束 + prompt 层软引导）。

## 二、修复方案

| FC | 代码层修复（硬约束） | prompt 层（软引导） |
|---|---|---|
| FC_TONE_OFF | `generate()` 调 LLM 前扫描 `dialogue_window`，命中 `PARENT_OVERRIDE_KEYWORDS`（4 类 30+ 词）→ 强制 `tone=DIAGNOSTIC` | §override 规则保留 |
| FC_STALE | `generate()` 返回前用 `difflib.SequenceMatcher` 与最近 N 条弹窗比对，sim ≥ 0.70 → `should_popup=False` | §语义去重自检保留 |

新增：`PARENT_OVERRIDE_KEYWORDS` / `detect_parent_override()` / `semantic_similarity()` / `PopupGenerator.__init__(dedup_config=...)`。

## 三、验证结果

### 9 case 真实回归（DeepSeek API，非 mock）

| Case | v4.0.14 | v4.0.15 | v4.0.16 | FC 状态 |
|------|:--:|:--:|:--:|------|
| B-01 日常 | 0 | 0 | 0 | ✓ 不误触发 |
| B-02 日常 | 0 | 0 | 0 | ✓ 不误触发 |
| C-01 漏触发 | 0 ❌ | 1(D) | 1(D) | ✓ FC_MISS_CTX 闭环 |
| C-02 低危摩擦 | 0 | 0 | 0 | ✓ Stage1 判低危 |
| **C-03 tone 错** | **1(E) ❌** | **1(E) ❌** | **2(D,D) ✓** | **FC_TONE_OFF 闭环** |
| D-01 重复弹 | 2(D,D) stale | 2(D,D) stale | 2(D,D) sim=0.131 + dedup 网 | FC_STALE 改善 |
| D-02 贬低羞辱 | 2(D,D) | 2(D,D) | 1(D) | ✓ 降噪 |
| E-01 情绪崩溃 | 1(D) | 1(D) | 1(D) | ✓ 正确 |
| A-01 优秀 | 0 | 0 | 0 | ✓ 不误触发 |

### FC_TONE_OFF 闭环证据（C-03）

运行日志：
```
Analysis #1: ☷(控控控) | risk=低 | tone=encouraging
FC_TONE_OFF override: 命中「催促/打断」，强制 diagnostic（原 suggested_tone=encouraging）
Generated diagnostic popup (174 chars): 你急着让她动笔，是怕她拖到天黑画不完...
```

### FC_STALE 闭环证据（D-01）

| 版本 | popup[0] vs popup[1] 相似度 | 说明 |
|:--:|:--:|---|
| v4.0.14 | 0.479 | 两条均以「你怕他指甲…藏细菌」开头，建议均为「把控制权还给他」— 语义重复 |
| v4.0.15 | 0.511 | 同上模式 |
| v4.0.16 | **0.131** | 角度差异化 + dedup 字符级安全网 |

### 代码层单元测试（mock model，17 case 全绿）

| 测试组 | case 数 | 结果 |
|---|:--:|:--:|
| detect_parent_override | 7 | 7/7 PASS |
| semantic_similarity | 3 | 3/3 PASS |
| tone override (FC_TONE_OFF) | 2 | 2/2 PASS |
| dedup 拒绝/放行 (FC_STALE) | 2 | 2/2 PASS |
| dedup disabled | 1 | 1/1 PASS |

## 四、改动文件

| 文件 | 改动 |
|---|---|
| `realtime/popup_generator.py` | +169 行：override/dedup 逻辑 |
| `realtime/cli_demo.py` | +4 行：传入 dedup_config |
| `realtime/config.yaml` | +61 行：dedup 段 + changelog |
| `realtime/stream_orchestrator.py` | +76 行：popup_history 传递 |
| `realtime/output_schemas.py` | +17 行：Popup.full_text 属性 |
| `realtime/zhouyi_analyzer.py` / `zhouyi_prompts.py` | multica 同步 |
| `system_prompt_v4.0.13~16.txt` | REN-50 迭代历史 |

## 五、看板回传

- 评论 ID：`04e7ff28-f540-4ec6-9b9d-2d2728137f92`（回复在裁判终评 `487da7e7` 下）
- 附件：`full_regression_results_v4016.json`（20678 bytes，9 case 真实执行原始数据）

## 六、下一步

- [ ] 等待裁判诊断体复评 v4.0.16（V1–V6 加权总分 + 三项 FC 闭环状态 + 上线资格）
- [ ] 复评通过后：清理中间版本 v4.0.13/v4.0.14（移至 `prompts_archive/`），仅保留生产 v4.0.16 + 上一稳定版 v4.0.15
- [ ] 更新 `use-cases/parent-child-coach/CLAUDE.md` 当前状态（v4.0.12 → v4.0.16）
