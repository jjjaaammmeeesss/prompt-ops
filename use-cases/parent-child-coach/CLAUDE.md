# 亲子沟通弹窗 · Prompt 优化项目

> 为亲子教练 App 构建实时弹窗干预系统，基于周易三爻八卦模型。

## 怎么跑

```bash
cd D:/prompt-ops/use-cases/parent-child-coach

# 测试梯子 L5（全量 31 条）
$env:JUDGE_BACKEND="deepseek"
python scripts/test_ladder.py --level 5 --prompt system_prompt_v4.0.12.txt

# 50 题盲测
python scripts/blind_test_50.py --prompt system_prompt_v4.0.12.txt --tone-mode auto

# 四版本对比
python scripts/compare_four_versions.py --n 12 --rounds 3
```

## 技术栈

- 生成模型：DeepSeek V4（`deepseek-chat`），测试/生成一律用国产模型
- Judge 模型：Claude Opus 4.7（xingluan 代理），仅做裁判
- 配置：`realtime/config.yaml` 指向当前生产 prompt
- 生产系统：Stage 1（周爻分析器）→ Stage 2（v4.0.19 弹窗生成器，三类型：diagnostic / encouraging / child_insight）
- 快速通道：`keyword_trigger.py`（基础设施层，prompt 无关）

## 目录与约定

- `system_prompt_v4.0.19.txt` — **当前生产 prompt**（三弹窗类型）
- `system_prompt_v4.0.12.txt` — 上一稳定版（两弹窗类型），可回退
- `realtime/` — 生产实时弹窗系统
- `scripts/` — 测试脚本（test_ladder.py, blind_test_50.py, compare_four_versions.py 等）
- `results/` — 测试结果（按子目录分类）
- `meta_optimize/` — 三策略自进化对比系统
- `auto_evolve/` — v2.3 自动进化实验
- `prompts_archive/` — 退役/回退版本
- `docs/solutions/` — 过往问题解决方案
- `.claude/` — AI 会话元文件（observation-log.md, vision）

**版本号规则**：文件名 = 内部标题版本号，代码引用必须同步更新。只保留"生产在用 + 上一稳定版"，中间版本删除（淘汰版本不删除文件，保留历史在 `prompts_archive/` 或 `_candidates/` 中）。

## 双向同步元规则（prompt ↔ 执行器）

> **核心**：prompt 与执行器是耦合的。改任何一边，必须检查另一边是否需要同步更新。这条规则由 `scripts/check_prompt_executor_sync.py` 自动校验，不可绕过。

### 执行器清单
- **生产执行器**：`realtime/popup_generator.py`（Stage 2 弹窗生成）
- **测试执行器**：`scripts/run_v418_pipeline.py`（管线测试器，`@transient`）
- `realtime/stream_orchestrator.py` / `zhouyi_analyzer.py` / `zhouyi_prompts.py` 为 realtime 子系统，随生产执行器一同维护

### 执行器版本号规范
每个执行器文件顶部声明两个版本号，**两者必须与生产 `realtime/config.yaml` 一致**（检查脚本校验）：
- `__version__` — 执行器自身迭代版本号（自增，如 `"1.0"`）
- `PROMPT_VERSION` — 它适配的 prompt 版本（如 `"v4.0.19"`）

生产 prompt 版本的**权威源**是 `realtime/config.yaml` 的 `generator.system_prompt_path`。

### 双向同步契约
1. **更新 prompt**（升级版本号 / 改内容）→ 必须同步检查执行器是否需更新：字段解析、字数门（`DIAGNOSTIC/ENCOURAGING/CHILD_INSIGHT_*_CHARS`）、弹窗类型解析、P2 话术检查、回退链。若有对应改动，同步更新执行器 + 两个版本号。
2. **更新执行器**（改逻辑）→ 必须同步检查 prompt 是否需更新：新增逻辑是否有对应 prompt 指令（如新的弹窗类型、新的约束）。若有，同步更新 prompt + `PROMPT_VERSION`。
3. **commit message 必须写明两端版本**：`prompt v4.0.19 ↔ executor v1.0` 这种格式，便于追溯某版本执行器的生产行为。
4. **任何一边更新后**，跑 `python scripts/check_prompt_executor_sync.py`，确认 0 FAIL 才能提交。

### 一致性检查用法
```bash
cd D:/prompt-ops/use-cases/parent-child-coach
python scripts/check_prompt_executor_sync.py            # 有 FAIL 即不一致（历史文档引用为 WARN）
python scripts/check_prompt_executor_sync.py --strict   # 任一 WARN 也拦截（连 prompts_archive 历史引用一起管）
```

对比脚本（`compare_v4012_v4019.py` 等）设计上引用多历史版本做对比，不属于生产链，不受此规则约束。

## 当前状态

- **生产版本**：v4.0.19（2026-08-05 三弹窗类型：diagnostic / encouraging / child_insight）
- **v1.x 族最新**：v1.13（2026-07-29 对抗性测试通过，66.7% 决胜率）
  - 文件：`D:/星灵-soul-手搓/亲子沟通洞见/路线A_自上而下法_鼓励版/prompts/_candidates/prompt_A轨_v1.13_候选版.md`
  - 四项修复：F4 前置分流、F1 逐句原文锚定（写后自审）、F2 反话强制响应、F3 深度三步扫描
  - v1.11 保留作为 few-shot 策略参考
- **已停用（保留文件不删）**：v1.7、v1.12、v2.3（均确认劣于 v4.0.12）
- **残留硬骨头**：C10-004、C5-001（四版全败，系统性难题）
- **快速通道**：KeywordTrigger 是基础设施层，所有版本共享
- **管线测试器**：`scripts/run_v418_pipeline.py`（匹配 prompt v4.0.19）
  - 9 缺口已修复：生产级 Stage 2 等价调用（ZhouYi/debounce/FC_TONE_OFF/FC_STALE/P2 + FC_CHILD_INSIGHT）
  - P2 话术检查：诊断式和 child_insight 均需 quotable phrase
  - 三弹窗类型：diagnostic（100-200字）/ encouraging（30-60字）/ child_insight（50-100字）
- **v4.0.19 child_insight**：从 v3.x 架构迁入第三种弹窗"看见孩子"（`popup_generator.py` commit 待提交）
  - 触发：孩子展现特征表达 + 家长无显著负面行为
  - 优先级：安全路由 > FC_TONE_OFF > child_insight > 卦象 tone
  - 结构：「你的孩子可能是[特征]」+「ta可能更适合用[教育方式]来引导」
  - 见 `system_prompt_v4.0.19.txt`

## 下一步

- 运行 `run_v418_pipeline.py --prompt v4.0.19` 12 题全量测试，确认 child_insight 触发率
- 观察 v4.0.19 生产数据，确认三弹窗类型的分布合理性
- 考虑将 LLM 输出前缀（元信息/关键句归属）从弹窗正文字数中分离，解决字数虚高问题
