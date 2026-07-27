# 握手文档 — 周易 v4.0.12 亲子沟通实时弹窗

> 最后更新：2026-07-27
> 当前分支：`main`

---

## 一、项目概述

为亲子教练 App 构建**实时弹窗干预系统**。基于师父的「周易三爻八卦亲子沟通模型」，家长在对话过程中获得实时反馈——看见自己、看见孩子、看见关系。

当前阶段：**prompt 质量优化（已收敛冻结 v4.0.12）**。核心 prompt 版本为 v4.0.12（62KB，集成周易三爻八卦策略速查表 + tone 灵活覆盖 + 全面禁止脑补）。

---

## 二、核心文件索引

### Prompt（核心资产）
| 文件 | 说明 |
|------|------|
| `system_prompt_v4.0.12.txt` | **当前生产 prompt**（62,104 字符）— tone 灵活覆盖强化 + 全面禁止脑补 + 双盲区修复 |
| `system_prompt_v4.0.11.txt` | 上一稳定版（57,963 字符） |
| `system_prompt_v4.0.10.txt` | 回退保留（55,636 字符） |
| ~~`system_prompt_v4.0.9.txt`~~ | 中间版本，已按 CLAUDE.md 规则 4 清理 |
| ~~`system_prompt_v4.0.8.txt`~~ | 中间版本，已按 CLAUDE.md 规则 4 清理 |
| ~~`system_prompt_v4.0.7.txt`~~ | 中间版本，已按 CLAUDE.md 规则 4 清理 |
| ~~`system_prompt_v4.0.6.txt`~~ | 中间版本，已按 CLAUDE.md 规则 4 清理 |
| ~~`system_prompt_v4.0.5.txt`~~ | 中间版本，已按 CLAUDE.md 规则 4 清理 |
| `prompts_archive/system_prompt_v2.3.txt` | 专家手工终版基线（62KB），v4.0.x 进化源头 |
| `prompts_archive/system_prompt_v2.4.txt` | v2.3 自动迭代产物 |
| `prompts_archive/system_prompt_v2.5.txt` | v2.3 自动迭代产物（当前 v23_evolve 最新保留版） |
| `prompts_archive/system_prompt_v2.6.txt` | **v2.3 结构重写实验**（15–40 字、无 type/contradiction、JSON 仅 popup_text） |

v4.0 迭代链：v4.0（基线）→ v4.0.1（去模板）→ v4.0.3（精准化）→ v4.0.4（去临床术语）→ v4.0.5（结构 > 情绪）→ ~~v4.0.6（责任交还，方向对但执行偏）~~ → v4.0.7（责任交还精化，0 稳定失败）→ ~~v4.0.8（去术语滥用 + 禁止脑补动机，50/50 通过）~~ → ~~v4.0.9（tone 灵活覆盖 + 对象错位修复）~~ → v4.0.10（禁止脑补建议方案）→ v4.0.11（强化 tone 灵活覆盖）→ **v4.0.12（扩展禁止脑补到弹窗所有部分）**

### 测试脚本
| 文件 | 说明 |
|------|------|
| `scripts/test_ladder.py` | **测试梯子**：L1(1)→L2(3)→L3(9)→L4(27)→L5(全量)，失败即修 prompt 回 L1 |
| `scripts/blind_test_50.py` | 50 题盲测（H2H 5 维度盲评 + deepseek Judge），支持 `--tone-mode {forced-diag,auto}` |
| `scripts/llm_judge_metric.py` | LLM Judge 5 维度评分（strategy_alignment, dialogue_fidelity, tone_alignment, natural_language, core_insight） |
| `auto_evolve/v23_evolve.py` | v2.3 基线的自动进化循环（生成→Claude judge→变异→择优） |
| `auto_evolve/optimizer.py` | 多智能体 v3.x 自动优化器 |

### 数据集
| 文件 | 条目 | 说明 |
|------|------|------|
| `v4_optimization/data/expert_train_v4_clean.json` | **31 条** | ✅ 当前测试集，已清洗 tone 标注错误 |
| `v4_optimization/data/expert_train_v4.json` | 56 条 | 原始版，25 条被过滤（24 条金标是下划线占位符） |
| `data/expert_dataset.json` | 85 条 | 最完整的标注数据（含 reference_popup, expert_score 等），56 条与 v4 重叠，29 条独有（其中仅 9 条有实质参考弹窗） |
| `data/expert_test.json` | 7 条 | 小型 holdout 集，4-5 条有实质参考弹窗 |
| `data/expert_train.json` | 28 条 | ❌ 所有 answer 为空，不可用 |
| `dataset_50_questions.json` | 50 条 | ❌ 无金标答案，仅可用于盲测 |
| `data/dataset_merged_train.json` | 78 条 | ❌ 同上 |

### 标注源文件（markdown）
| 文件 | 标注人 |
|------|--------|
| `worksheet_13cases_专家打标_20260710_114609-子阳手动改写13用例.md` | 子阳 |
| `worksheet_快速通道_20260712_222129OK-廖老师打标.md` | 廖老师 |
| `worksheet_10cases-晓浩已手写改完.md` | 晓浩 |
| `worksheet_快速通道_v2_20260713_184046-子阳打标4用例.md` | 子阳 |

解析脚本：`scripts/archive/parse_expert_annotations.py`

### 配置
| 文件 | 说明 |
|------|------|
| `config.yaml` | 主配置（2026-07-24 已更新，指向 `system_prompt_v4.0.12.txt`） |
| `realtime/config.yaml` | **生产配置**，指向 `../system_prompt_v4.0.12.txt` |
| `v4_optimization/config_v4_quick.yaml` | MIPROv2 快速验证配置 |
| `.env` | API Keys（DEEPSEEK_API_KEY、ANTHROPIC_AUTH_TOKEN 等） |

---

## 三、架构决策

### 3.1 测试策略：测试梯子
```
L1 (1条) → 全过 → L2 (3条) → 全过 → L3 (9条) → 全过 → L4 (27条) → 全过 → L5 (31条全量)
 ↓ 有失败                                      ↓ 有失败
 修 prompt ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←← 回到 L1
```

通过标准：`weighted_score >= 0.70`，无 VETO（0.0 分）。

### 3.2 窗口分组策略（Plan A）
**问题**：31 条数据 = 31 个 300 字窗口，来自 ~22 个独特对话。同一对话的多个窗口（不同句子范围，可能不同 tone）共享同一段对话文本，AI 只生成一次弹窗，但被评估多次。

**解决方案**：按 `(dialogue, tone)` 分组 → 取组内最佳窗口分。同一对话+同一 tone → 缓存复用弹窗；不同 tone → 分别生成。

测试结果报告两个维度：
- **窗口级**（31窗）：原始分数
- **对话+tone 级**（~25组）：聚合后分数（这才是真实质量）

### 3.3 w甲/w乙/w丙/w丁 含义
**不是不同专家！** 是同一对话的不同 300 字滑动窗口标签（如 w甲=句1-11, w乙=句5-15）。解析正则：
```python
WINDOW_HEADER_RE = re.compile(r"###\s*(甲|乙|丙|丁|戊|己)\s*·\s*300字窗口\s*·\s*句\s*(\d+)\s*[-–]\s*(\d+)")
Record ID: f"{filename}_case{case_num}_w{label}"
```
真正标注人只有 3 位：**子阳、廖老师、晓浩**。

### 3.4 MIPROv2 优化结论
v4.0 上 MIPROv2 **无效**。原因：Proposer（deepseek-chat）无法有效修改 44.8KB 专家级 prompt，所有增益来自与测试数据重叠的 few-shot 示例（数据泄漏）。结论：这个 prompt 的复杂度超出了 MIPROv2 的优化能力边界。

### 3.5 Judge 模型
- **首选**：Claude（s.lconai.com / xingluan），但频繁 403 rate-limit
- **当前使用**：deepseek-chat（`JUDGE_BACKEND=deepseek`）
- 5 维度加权评分：strategy_alignment, dialogue_fidelity, tone_alignment, natural_language, core_insight

---

## 四、当前成绩（v4.0.12）

**v4.0.12 — 2026-07-20 生产交接对比 + 盲测：**

| 测试集 | v4.0.12 | v2.3（上代生产） | Δ |
|--------|---------|-----------------|-----|
| 70 全量校标 n=3 | **0.954** | 0.883 | **+0.071** |
| 12 独立新用例 n=3 | **0.867** | 0.625 | **+0.242** |
| 20 题盲测 auto | **0.947** | — | — |

**残留低分 case**：C5-001（摆碗碎碗, ~0.463）、DS_001（收盘子, ~0.513）。

### v4.0.9 → v4.0.12 关键修复

| 版本 | 修复点 |
|------|--------|
| v4.0.9 | tone 灵活覆盖规则：系统 tone=诊断式，但家长有 ≥2 个积极行为时自动切换鼓励式；修复对象错位（弹窗永远对家长说话） |
| v4.0.10 | 禁止在"下次"部分编造对话中未出现的具体道具/方法（C5-001 摆碗碎碗退化） |
| v4.0.11 | 强化 tone 灵活覆盖：30–60 字肯定必须是完整段落，禁止套话式短肯定混过关 |
| v4.0.12 | 扩展禁止脑补到弹窗所有部分；禁止编造家长未说出口的话/未做出的行为；进一步强化 tone 灵活覆盖触发判定（C5-005 妈妈哭场景） |

### 历史参考：v4.0.8 成绩

v4.0.8 在 L5（0 稳定失败）+ 50 题盲测（100% 通过、0 低分、人话感满分）双维度全面达成。后续 v4.0.9–v4.0.12 在此基础上继续修复全量校标暴露的 tone 覆盖和脑补问题。

---

## 五、v2.3 / v2.6 上下文（另一条进化线）

### 5.1 v2.3 是什么
`prompts_archive/system_prompt_v2.3.txt` 是 **2025 年三位专家手工标注的单 prompt 基线**（62KB），v4.0.x 的进化源头。它本身是一个完整可用的弹窗系统，2026-07-20 被 v4.0.12 替换为生产版，现作为回退版本保留。

**v2.3 特点**：
- 单 prompt 架构，~300 字切窗
- LLM 自主判断 diagnostic/empowering tone
- 输出稳定性极高（盲测 50 标准差 0.038，0 VETO）
- **短板**：tone 匹配率 M5=0.400，新场景泛化不足（12 新用例 0.625 vs 70 校标 0.883，差 0.258）

详细审计报告：`docs/solutions/2026-07-21-v23-test-audit-and-improvement-plan.md`

### 5.2 v2.4 / v2.5：自动进化尝试
`auto_evolve/v23_evolve.py` 以 v2.3 为原点，尝试自动迭代：
- v2.4 R2：0.684（被错误保留，后发现 baseline 漂移 bug）
- v2.5：0.646
- 当前状态：`results/v23_evolve/state.json` 显示第 4 轮、版本 v2.5、综合分 0.646

**已修复的 auto-evolve bug**（2026-07-24 已 commit）：
1. `should_keep` 始终与原点 v2.3 比较，防止退化版本被保留
2. 空 popup_text 触发重试，降噪时只在有效输出 run 中投票
3. v23_evolve 评估前持久化状态，支持 `--resume`
4. v23_runner 支持 v2.6+ 简化 JSON 输出格式

### 5.3 v2.6：结构重写实验
`prompts_archive/system_prompt_v2.6.txt` 是一次激进的结构重写尝试：
- 弹窗压缩到 **15–40 字**
- **取消诊断式/鼓励式分类**，改为"看清自己 / 看见对方 / 看见模式"三选一
- 输出格式简化为 `{"popup_text": "..."}`
- 强调"先照见已有的光，再照见盲区"

**目的**：探索 v4.0.x 是否因为 prompt 太长、结构太复杂而被 LLM 执行偏。v2.6 目前尚未接入完整评估，仅作为实验版本保留。

### 5.4 两条线的关系
```
v2.3（专家手工基线）
  ├──→ v4.0 → v4.0.12（当前生产，长 prompt 迭代路径）
  └──→ v2.4 → v2.5 → v2.6（短 prompt 结构重写实验路径）
```

v4.0.12 已证明在长 prompt 路径上收敛；v2.6 是探索短 prompt 路径是否更可控。两者目前不冲突，v4.0.12 继续服务生产，v2.6 待评估。

---

## 六、扩展测试集计划（已核验，ROI 低，搁置）

**目标（原计划）**：从 ~31 条扩展到 ~45 条，验证泛化能力。

**核验结果（2026-07-17，已证伪原判断）**：
- ~~`data/expert_dataset.json` 独有 29 条 → 其中 **9 条** 有实质 reference_popup~~ → 实测仅 **3 条** 有实质弹窗（其余 26 条是占位符或空），格式转换后可新增 ~2-3 条到 33-34
- ~~`data/expert_test.json` 7 条 → 其中 **4-5 条** 有实质参考弹窗~~ → 实测 4 条全部与 v4_clean 重复，0 条可新增

**结论**：扩展集实际只能 +2~3 条，ROI 崩溃。改用"攻 2 个稳定失败 case"路径（已完成，见 v4.0.7）。50 题盲测泛化用 `dataset_50_questions.json`（无金标，仅盲测）。

**格式转换**：`dialogue→question`, `reference_popup→answer`, `expert_tone→tone`（⚠️ 不是 `system_tone`——`system_tone` 多为空，必须用 `expert_tone`；空 tone 默认为诊断式）

**执行脚本**：需新写一个 `scripts/build_extended_test_set.py`，或手动用 Python 命令行。

**注意**：`expert_train_v4_clean.json` 中已有 answer 质量高（专家手写），而 `data/expert_dataset.json` 中 `reference_popup` 可能带有 `**（请专家手写弹窗正文）**：` 等模板前缀，需要清洗。

---

## 七、常用命令

```bash
# 跑测试梯子 L5（全量）
cd D:/prompt-ops/use-cases/parent-child-coach
$env:JUDGE_BACKEND="deepseek"
python scripts/test_ladder.py --level 5 --prompt system_prompt_v4.0.12.txt --output-dir results/ladder_tests

# 跑 50 题盲测（auto 模式 = 让 prompt 按信念维度自判 tone，推荐生产用）
python scripts/blind_test_50.py --prompt system_prompt_v4.0.12.txt --tone-mode auto

# 50 题盲测（forced-diag 模式 = 强制诊断式，与首跑一致，作 baseline 对比）
python scripts/blind_test_50.py --prompt system_prompt_v4.0.12.txt --tone-mode forced-diag

# v2.3 自动进化
python auto_evolve/v23_evolve.py --rounds 5
# 若中断恢复
python auto_evolve/v23_evolve.py --resume

# 查看最新测试结果
python -c "
import json, glob, os
files = glob.glob('results/ladder_tests/ladder_L5_*.json')
latest = max(files, key=os.path.getmtime)
with open(latest, 'r') as f:
    d = json.load(f)
print(f\"窗口级: {d['window_passed']}/{d['total_cases']} 均分 {d['window_avg_score']:.3f}\")
print(f\"对话级: {d['passed']}/{d['total_groups']} 均分 {d['avg_score']:.3f}\")
"

# 列出所有 prompt 版本
ls system_prompt_v4*.txt prompts_archive/system_prompt_v2*.txt
```

---

## 八、踩坑记录

1. **tone 标注错误**：`expert_train_v4_clean.json` 中曾有 5 条 tone 标注错误（标注为诊断式但金标内容是鼓励式），已手动修正。根因：parse 脚本的 `expert_tone` 提取可能不准。
2. **缓存 key 导致 VETO**：初版缓存用 `dialogue` 做 key，同一对话不同 tone 窗口共享弹窗，导致鼓励式弹窗被诊断为诊断式 VETO。修复：缓存 key 改为 `(dialogue, tone)`。
3. **MIPROv2 数据泄漏**：最优程序的 5 个 few-shot 全部与测试数据重叠，不能用于生产。
4. **Claude API 不稳定**：s.lconai.com 频繁 403，测试时用 `JUDGE_BACKEND=deepseek` 回退。xingluan endpoint（`api.xingluan.vip` / `luanapi.xingluan.cn`）可作为替代。
5. **版本漂移风险**：prompt 文件名与内部标题必须一致，所有代码引用必须同步更新。参见全局 CLAUDE.md 中的「提示词版本与代码版本对齐规则」。
6. **Judge 方差陷阱**（v4.0.7 迭代中发现）：单次 L5 运行的"失败"约 50% 是 Judge+生成方差（最大波动 0.462），不是 prompt 系统缺陷。判定稳定失败必须 3 轮量化取均值。
7. **tone 强制错配陷阱**（v4.0.7 50 题盲测发现）：50 题数据集无 tone 标注，盲测若强制 `tone="诊断式"` 会让鼓励场景被诊断式处理，导致看见感维度假性偏低。生产环境用 auto 模式。
8. **CBT 术语滥用陷阱**（v4.0.7 auto 模式发现，v4.0.8 已修复）：prompt 正例和自检规则诱导 AI 套术语（"牺牲叙事"、"读心术的眼镜"），人话感被扣分。v4.0.8 已修复为正例去术语前缀、自检以动态描述为主。
9. **脑补动机陷阱**（v4.0.7 发现，v4.0.8 已修复）：AI 输出"你心疼丈夫"、"你想控制孩子"等脑补句式。v4.0.8 加"只能描述对话中实际说出口的话和可观察的行为"硬规则。
10. **过度迭代风险**（v4.0.8 收敛结论）：v4.0.8 后边际效益接近零，但全量校标仍暴露 tone 覆盖和对象错位问题，因此继续到 v4.0.12。v4.0.12 后需观察生产数据再决定是否迭代。
11. **auto-evolve baseline 漂移**（v2.3/v2.4/v2.5 发现，2026-07-24 修复）：`should_keep(baseline, candidate)` 传入的是上一轮保留的 candidate，导致退化版本被错误保留。修复：冻结 `origin_baseline`，始终与 v2.3 原点比较。
12. **空输出误判 M1=0**（v2.3 auto-evolve 发现，2026-07-24 修复）：模型概率返回 `{"type": ..., "popup_text": ""}`，原代码直接计 M1=0。修复：空输出触发重试 + 降噪时只在有效输出 run 中投票。
