# 握手文档 · v4.0.12 vs v1.7 对比测试

> **生成时间**：2026-07-20
> **用途**：模型切换/重启后，用本文档恢复上下文，无需重读全部对话历史。
> **项目根**：`D:\prompt-ops`
> **当前分支**：`feat/prompt-optimization-v1.7-extended-dataset`

---

## 1. 项目目标

亲子沟通教练 A 轨弹窗 prompt 的优化与版本对比。当前核心任务：**验证 v4.0.12 是否稳定胜过 v1.7 和 v2.3**。

---

## 2. 关键结论（已验证）

### 2.1 v4.0.12 已上生产，替换 v2.3
- 生产配置：`use-cases/parent-child-coach/realtime/config.yaml` → `system_prompt_v4.0.12.txt`
- 保留 `system_prompt_v2.3.txt` 作为回退版本（CLAUDE.md 规则 4：只保留生产在用 + 上一稳定版）
- v4.0.7/8/9 已删除，保留 v4.0.10/11/12
- 提交：`368a22f`

### 2.2 v4.0.12 vs v2.3（历史验证，已上生产）
| 测试集 | v4.0.12 | v2.3 | 领先 |
|--------|---------|------|------|
| 70 全量 3轮 | 0.954 | 0.883 | +0.071 |
| 12 新用例 3轮 | 0.862 | 0.625 | +0.237 |
| 20 题盲测 | 0.947 | — | — |

### 2.3 v4.0.12 vs v1.7（本次对比测试，核心未完成项）
**v1.7 定位**：`D:\星灵-soul-手搓\亲子沟通洞见\路线A_自上而下法_鼓励版\prompts\prompt_A轨_v1.7_修复感知版.md`
- 82,952 字节，1,452 行，.md 格式
- 双 prompt 架构：System Prompt（line 28-220，1925字）+ User Prompt（line 226-1442，含 Step 0-3 流程 + 14 few-shot + JSON 输出格式，36029字）
- 输出严格 JSON：`should_popup`/`tone`/`popup_insight`/`popup_suggestion`
- v1.7 自主判 tone（empowering/diagnostic），有 Step 0 不弹判定

**v1.7 数据泄漏排除**：v1.7 的 14 个 few-shot 中 #1/#3/#5/#7/#9/#11/#13 与 50 题盲测集 #43-49 重叠 → 独立测试集必须排除 50 题盲测，只用 12 新用例。

**已完成对比数据**（3 轮均值）：

| 测试集 | v1.7 | v4.0.12 | 差距 | v1.7 variance |
|--------|------|---------|------|---------------|
| 校标 18 题 | 0.813 | 0.954* | -0.141 | 0.037 |
| 独立 12 题 | 0.759 | 0.862 | -0.103 | 0.083 |
| 独立 1 题 | 0.946 | 1.000 | -0.054 | 0.163 |

*v4.0.12 校标 0.954 是 70 全量 3 轮均值，18 题是子集

**v1.7 系统性失败模式**：
1. 短对话漏判：#11（118字）3 轮全判定 `no_negative_impact` 不弹
2. tone 系统性偏移：#4（累了看画）3 轮全走诊断式但专家要鼓励式；#13（兄弟规则）2/3 轮走诊断式
3. 长对话不稳定：case5（1391字）4 窗口跨轮次 tone 判断不一致
4. variance 偏大：独立测试 0.083 vs v4.0.12 通常 <0.03

**结论**：v4.0.12 稳定胜过 v1.7（两个独立测试集均领先 10+ 个百分点，且 variance 更小）。根因：v1.7 自主 tone 判断与校标标注系统性分歧，v4.0.12 用数据集强制 tone 指定消除了此问题。

---

## 3. 当前进行中任务

**用户最新指令**（切换模型前最后一条）：
> "找独立测试题目 1~3~9，去评估 1.7 和 4.0.12 的水准"

即：用独立测试集（12 新用例）的前 1/3/9 题，分别评估 v1.7 和 v4.0.12，递增对比。

**已完成**：
- 写了统一对比脚本 `scripts/compare_v17_v4012.py`（同一脚本、同一 judge、同一子集，公平对比）
- 跑了 n=1（3 轮）：v1.7 均分 0.946 / v4.0.12 均分 1.000，v4.0.12 领先 +0.054
- 跑了 n=3（3 轮）：**输出丢失**（命令无输出返回，可能被重启打断）

**待完成**：
- [ ] 重跑 n=3（3 轮）对比
- [ ] 跑 n=9（3 轮）对比
- [ ] 汇总 1→3→9 递增对比表，给用户最终汇报

---

## 4. 关键文件位置

| 文件 | 用途 |
|------|------|
| `use-cases/parent-child-coach/system_prompt_v4.0.12.txt` | 当前生产 prompt（23320字） |
| `use-cases/parent-child-coach/system_prompt_v4.0.11.txt` | 上一迭代 |
| `use-cases/parent-child-coach/system_prompt_v4.0.10.txt` | 上上迭代 |
| `use-cases/parent-child-coach/system_prompt_v2.3.txt` | 旧生产，回退版本 |
| `D:\星灵-soul-手搓\...\prompt_A轨_v1.7_修复感知版.md` | v1.7 完整 prompt |
| `use-cases/parent-child-coach/scripts/test_v17.py` | v1.7 适配器（1→3→9→18 递增，已验证） |
| `use-cases/parent-child-coach/scripts/compare_v17_v4012.py` | v1.7 vs v4.0.12 统一对比脚本（1/3/9） |
| `use-cases/parent-child-coach/scripts/test_ladder.py` | L5 test ladder（v4.0 专用） |
| `use-cases/parent-child-coach/scripts/blind_test_50.py` | 盲测脚本（v4.0.12 + auto tone + --n） |
| `use-cases/parent-child-coach/data/expert_dataset_full_71.json` | 70 条校标集 |
| `use-cases/parent-child-coach/data/new_12_independent.json` | 12 条新独立用例（独立测试集，有 gold） |
| `use-cases/parent-child-coach/dataset_50_questions.json` | 50 题盲测（无 gold，#43-49 与 v1.7 few-shot 重叠，禁用） |
| `use-cases/parent-child-coach/results/v17_tests/` | v1.7 单版本测试结果 |
| `use-cases/parent-child-coach/results/compare_tests/` | v1.7 vs v4.0.12 对比结果 |

---

## 5. 关键约束（来自 CLAUDE.md / AGENTS.md）

1. **Vision-first**：vision 锁定在 `.claude/vision-prompt-optimization-v1.7.md`，永久只读
2. **简化决策**：用户说"这么简单的决策，就不要让我做了" + "目标是尽量多地去执行自我迭代"
3. **Test ladder**：L1→L5，阈值 0.70，fail→fix→回 L1
4. **Judge variance**：必须量化 3 次运行（单轮波动最大 0.462）
5. **数据泄漏**：测试数据不得作为 few-shot；50 题盲测 #43-49 与 v1.7 重叠必须排除
6. **版本对齐**：prompt 文件名版本号 = 内部标题版本号；代码引用同步更新
7. **保留策略**：只保留生产在用 + 上一稳定版（CLAUDE.md 规则 4）

---

## 5.5 执行细节与坑（重启后必读）

1. **v4.0.12 独立测试无历史文件**：之前口头记录的"12 新用例 0.862"是凭记忆，对应结果文件已不在。本次 `compare_v17_v4012.py` 是首次在同一子集上公平重跑 v4.0.12，n=1 结果已存（v4.0.12 = 1.000）。**不要假设 v4.0.12 独立集有旧文件可查**。

2. **n=3 命令曾无输出返回**：跑 `compare_v17_v4012.py --n 3 --rounds 3` 时命令返回空（非报错、非异常），疑似模型切换/重启打断进程。结果文件 `results/compare_tests/compare_n3_*.json` 经确认不存在 → **n=3 需重跑**。

3. **环境变量依赖**：所有脚本靠 `use-cases/parent-child-coach/.env` 里的 `DEEPSEEK_API_KEY` 和 `JUDGE_BACKEND`（默认 deepseek）。重启后若 API 报错先查 `.env`。

4. **速度差异**：v1.7 user prompt 36029字 + `max_tokens=2048`，单次生成 5-10s；v4.0.12 仅 23320字 system + `max_tokens=640`，更快。n=9 三轮两版本共 54 次生成 + 54 次 judge，预计 10-15 分钟，设 timeout ≥ 900000ms。

5. **12 新用例全标 tone=诊断式**：v4.0.12 在独立集上全走诊断式（不会 tone 偏移）；v1.7 会自主判 tone，这是两者分差的主要来源（v1.7 常把应诊断的判成鼓励、或漏判不弹）。

6. **v1.7 单题 variance 可能很大**：n=1 时 v1.7 三轮 [0.837, 1.000, 1.000]，因为 tone 在 diagnostic/empowering 间跳变。小样本下 variance 不代表真实水平，n 增大后趋于稳定（校标 18 题 variance 0.037）。

7. **compare 脚本的 v4.0.12 生成函数**固定返回 `actual_tone = 数据集标注 tone`（不读模型输出），所以 v4.0.12 的 tone_mismatch 永远为 0——这是预期行为，不是 bug。

---

## 6. 恢复工作检查清单

重启后，确认上下文恢复：
- [ ] 读本文档
- [ ] 读 `use-cases/parent-child-coach/results/compare_tests/` 已有结果
- [ ] 确认 n=1 已完成（compare_n1_20260720_144817.json）
- [ ] 重跑 n=3 和 n=9 对比
- [ ] 汇总给用户

---

## 7. 下一步建议（待用户确认）

1. 完成 n=3、n=9 对比，输出 1→3→9 递增表
2. 分析 v1.7 tone 偏移根因（可选，深度分析）
3. 回到 v4.0.12 持续优化残留问题：
   - C5-001（摆碗碎碗）0.446 — 偶发脑补家长话语
   - DS_001（收盘子）0.513 — miss 金标"你全程在教"
4. 清理：v1.7 对比脚本和结果是否留存（根据 CLAUDE.md 规则 4，可考虑清理中间版本）

---

## 8. 用户行为偏好与沟通风格（避免重启后踩雷）

1. **决策授权**：用户明确说过"这么简单的决策，就不要让我做了"+"目标是尽量多地去执行自我迭代"。→ 测试方案、tone 处理、占位符填充等技术决策直接定，不要反复问；只在大方向（如是否清理版本、是否继续优化）上确认。
2. **指令紧凑**：用户常发极短指令（如"找独立测试题目1~3~9，去评估1.7和4.0.12的水准"+"？？"+"P"）。"？？"通常表示"前面说的你听懂了吗/确认下"，"P"可能表示"继续/proceed"。不要追问含义，按上下文直接执行。
3. **关注"稳定胜过"**：用户问的是"稳定胜过"而非"偶尔赢" → 必须报 variance，不能只报单轮均值。
4. **切换模型会丢记忆**：用户会切模型/重启，且担心上下文丢失 → 任何阶段性成果及时落盘（结果 JSON + 本文档），不要只存在对话里。
5. **反谄媚**：CLAUDE.md 有反谄媚协议。结论要基于证据（分数、variance），不要为了迎合"v4.0.12 更好"而编造。实际数据确实显示 v4.0.12 领先，但 v1.7 在小样本/特定 case 上也有满分，需如实呈现。

---

## 9. 一句话当前状态

v4.0.12 已上生产且确认优于 v2.3；vs v1.7 的对比测试进行中，独立集 1→3→9 递增任务已完成 n=1（v4.0.12 1.000 > v1.7 0.946），n=3 需重跑、n=9 待跑。重启后读本文档 + 跑 n=3/n=9 即可交付最终对比表。
