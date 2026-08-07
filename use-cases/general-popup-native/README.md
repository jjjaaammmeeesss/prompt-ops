# general-popup-native — 通用弹窗提示词（PDO vs MIPROv2 对比）

本项目内 use-case 用于演示如何用 prompt-ops 原生优化器循环优化一段通用对话弹窗 system prompt，并横向对比 **PDO** 与 **MIPROv2** 的效果。

## 文件说明

| 文件 | 说明 |
|------|------|
| `system_prompt.txt` | 基线 prompt（v1.0，复制自 `general-dialogue-popup`） |
| `system_prompt_v2.0.txt` | PDO 小参数优化结果（3 rounds / 3 duels） |
| `system_prompt_v2.1_pdo.txt` | PDO 大参数优化结果（10 rounds / 5 duels） |
| `system_prompt_mipro.txt` | MIPROv2 优化结果（含 2 个 few-shot demos） |
| `system_prompt_v2.2.txt` | **三版合并后的最终推荐版** |
| `system_prompt_v2.3_pdo.txt` | 以 v2.2 为基线再跑 PDO 的结果（过度简化，未采用） |
| `dataset.json` | 15 段通用对话场景，覆盖上下级/同事/朋友/情侣/家人等 |
| `config.yaml` | PDO 小参数配置 |
| `config_pdo.yaml` | PDO 大参数配置 |
| `config_mipro.yaml` | MIPROv2 配置 |
| `metric.py` | PDO 占位 metric |
| `metric_judge.py` | MIPROv2 用的 LLM-as-judge metric |
| `compare_popups.py` | 基线 vs 优化后 popup 单场景对比 |
| `compare_all.py` | 基线 / PDO / MIPROv2 三场景对比 |
| `results/` | 优化结果 JSON/YAML + 日志 |

## 运行方式

```bash
# PDO 大参数
prompt-ops migrate --config use-cases/general-popup-native/config_pdo.yaml \
                   --output-dir use-cases/general-popup-native/results

# MIPROv2
prompt-ops migrate --config use-cases/general-popup-native/config_mipro.yaml \
                   --output-dir use-cases/general-popup-native/results
```

当前配置使用 `deepseek/deepseek-chat` 作为 task/judge 模型，需要 `DEEPSEEK_API_KEY` 环境变量。

## 关键配置

### PDO
- `optimization.strategy: pdo`
- `optimization.task_type: open_ended`
- `optimization.use_labels: false`
- 大参数：`total_rounds: 10`，`num_duels_per_round: 5`，`num_eval_examples_per_duel: 1`，`num_initial_instructions: 10`
- 评判：pairwise LLM judge，维度包括准确、立场、字数、结构、语气、建议可操作性、禁止 JSON、保留原始价值观

### MIPROv2
- `optimization.strategy: basic`
- `auto: null`，`num_trials: 10`，`num_candidates: 6`
- `max_bootstrapped_demos: 2`，`max_labeled_demos: 2`
- `minibatch: false`（验证集只有 3 条，默认 minibatch_size=35 会报错）
- metric：`PopupLLMJudgeMetric`，用同一个 deepseek 模型对弹窗 6 维度打分（1-5），取平均归一化

## 运行数据

| 指标 | PDO 大参数 | MIPROv2 |
|------|------------|---------|
| 总耗时 | ~261 秒 | ~156 秒 |
| 候选 prompt 数 | 11 个 | 6 instructions × few-shot 组合 |
| duel/trial 次数 | 25 场 duel | 11 trials |
| 最终胜出方式 | Copeland 排名 | 最高 val score（100%） |

## 三版本定性对比

测试了三个场景：`manager_anger`（上下级）、`couple_chase_dodge`（情侣追逃）、`friend_soft_boundary`（朋友边界）。

### 共同优点
三个版本都能：
- 识别对话张力
- 只对「你」说话
- 给出 `——` 分隔的洞察 + 建议结构
- 保持中立、朋友语气

### 版本差异

| 版本 | 风格 | 优点 | 缺点 |
|------|------|------|------|
| **Baseline v1.0** | 详细、价值观完整 | 有三条信念公理、三个误判场景、安静信号“安好” | 有时过长、分段多、部分场景略显啰嗦 |
| **PDO v2.1** | 紧凑、结构清晰 | 直接点出循环，建议非常具体、可执行；明确“最近3轮”约束，避免泛泛而谈 | 丢失了原版部分价值观细节（如“两人都是完整独立的人”） |
| **MIPROv2** | 温和、情感共鸣强 | 对情绪命名准确（如“被冷落的委屈”），建议柔和 | 有时洞察偏长，接近 180 字上限；携带了 few-shot，但其中一个 few-shot 的 answer 为空 |

### 单场景观察

**manager_anger**
- Baseline：先描述“你”在退，再指出经理要的是“接住压力”
- PDO：直接点出“他追你逃”循环，建议最具体（今晚前发补救计划）
- MIPROv2：情绪描述最细腻（“失控的焦虑”“被误解的无力感”），但篇幅最长

**couple_chase_dodge**
- Baseline：简洁有力，抓住“他需要空间，你在安检”
- PDO：直接命名“你追他躲”，建议带具体请求
- MIPROv2：情感共鸣最强，但“两种需求在打架”略有术语感

**friend_soft_boundary**
- Baseline：指出“退让维持关系”，建议同时处理搬家和借钱
- PDO：建议温和，先肯定帮忙再谈钱
- MIPROv2：建议最硬（只帮两小时 + 定具体还款日），边界最清晰

## 合并版 v2.2

已按“结论”里的建议完成合并：

- 以 **PDO v2.1** 的紧凑结构为骨架
- 吸收 **Baseline** 的价值观细节：两人都是独立生命、镜子使命、三个误判场景、安静信号“安好”
- 吸收 **MIPROv2** 的情绪共情表达：点出“他真正在意的不是……而是……”
- 保留 PDO 的“最近 3 轮聚焦”和“建议贴场景、引用对方原话”约束
- 在硬规则里明确：`——` 之后**紧跟**建议句、**不要空行**、**不要输出 JSON/数组/编号**

合并后的产物：`system_prompt_v2.2.txt`。

### v2.2 样例输出观察

- `manager_anger`：既接住经理情绪，又主动承认责任，建议具体
- `couple_chase_dodge`：点出追逃循环，建议温柔且明确
- `friend_soft_boundary`：替自己开口要钱，边界清晰但不伤关系

## 以 v2.2 为基线再跑 PDO（v2.3）

按“结论”里的建议，又以 v2.2 为基线跑了一轮 PDO（10 rounds / 5 duels / 10 初始候选），产物为 `system_prompt_v2.3_pdo.txt`。

### v2.3 的问题

v2.3 的 prompt 被 PDO 简化成了一道“任务指令”：

> 分析以下对话，找出其中一方（即“你”）尚未察觉的关键盲点……

它丢失了 v2.2 中的价值观、朋友语气、镜子使命、安静信号等核心内容。虽然结构更清晰、建议更具体，但输出变得**像分析报告，不像弹窗**：

- 开头固定句式：“你尚未察觉的关键盲点是……”
- 大量引用“最近3轮”做证据
- 整体篇幅偏长，接近 180 字上限
- 语气中性、偏冷，缺少朋友轻声提醒的感觉

### 结论更新

- **v2.2 仍是最终推荐版**：综合了三个版本的优点，结构稳、价值观完整、建议具体、情绪不过度。
- **v2.3 是 PDO 过度简化的结果**：说明在没有更强 judge/metric 约束“产品体感”时，PDO 会为了赢 duel 而牺牲温度和价值观。
- 如果还要继续自动迭代，需要把“语气像朋友”“不套模板”“保留价值观”等维度写进 `judge_requirement`，否则容易退化成 v2.3。

## 修复的项目 bug

本次任务顺带修了 prompt-ops 框架的两个问题：

1. `src/prompt_ops/interfaces/cli.py:961`
   - 原：`open(prompt_file, "r")`（Windows 默认 GBK，中文 prompt 报错）
   - 改：`open(prompt_file, "r", encoding="utf-8")`

2. `src/prompt_ops/core/prompt_strategies.py:409-412`
   - 原：手动模式时省略 `auto`，但 DSPy `MIPROv2` 默认 `auto='light'`，导致和 `num_candidates` 冲突
   - 改：手动模式显式传 `auto=None`

## 已知问题

1. `dataset.json` 目前以 ASCII-escaped Unicode 保存，以绕过 `json.load` 的默认编码问题。
2. `config.yaml` 的 `judge_requirement` 不能包含中文特殊标点（如 `——`），否则 CLI 读 YAML 会 GBK 报错；已用 `---` 代替。
3. PDO/MIPROv2 的 `--output-dir` 目前似乎被忽略，结果先写到项目根目录 `results/`，已手动移到本目录 `results/`。
