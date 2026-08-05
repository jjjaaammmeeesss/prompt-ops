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
- 生产系统：Stage 1（周爻分析器）→ Stage 2（v4.0.12 弹窗生成器）
- 快速通道：`keyword_trigger.py`（基础设施层，prompt 无关）

## 目录与约定

- `system_prompt_v4.0.12.txt` — **当前生产 prompt**（62KB）
- `realtime/` — 生产实时弹窗系统
- `scripts/` — 测试脚本（test_ladder.py, blind_test_50.py, compare_four_versions.py 等）
- `results/` — 测试结果（按子目录分类）
- `meta_optimize/` — 三策略自进化对比系统
- `auto_evolve/` — v2.3 自动进化实验
- `prompts_archive/` — 退役/回退版本
- `docs/solutions/` — 过往问题解决方案
- `.claude/` — AI 会话元文件（observation-log.md, vision）

**版本号规则**：文件名 = 内部标题版本号，代码引用必须同步更新。只保留"生产在用 + 上一稳定版"，中间版本删除（淘汰版本不删除文件，保留历史在 `prompts_archive/` 或 `_candidates/` 中）。

## 当前状态

- **生产版本**：v4.0.12（2026-07-28 四版本对比确认最优）
- **v1.x 族最新**：v1.13（2026-07-29 对抗性测试通过，66.7% 决胜率）
  - 文件：`D:/星灵-soul-手搓/亲子沟通洞见/路线A_自上而下法_鼓励版/prompts/_candidates/prompt_A轨_v1.13_候选版.md`
  - 四项修复：F4 前置分流、F1 逐句原文锚定（写后自审）、F2 反话强制响应、F3 深度三步扫描
  - v1.11 保留作为 few-shot 策略参考
- **已停用（保留文件不删）**：v1.7、v1.12、v2.3（均确认劣于 v4.0.12）
- **残留硬骨头**：C10-004、C5-001（四版全败，系统性难题）
- **快速通道**：KeywordTrigger 是基础设施层，所有版本共享
- **管线测试器**：`scripts/run_v418_pipeline.py` v1.2（匹配 prompt v4.0.18）
  - 9 缺口已修复：生产级 Stage 2 等价调用（ZhouYi/debounce/FC_TONE_OFF/FC_STALE/P2）
  - P2 话术检查已修正为诊断式（v1.2）
  - 12 题 Codex 裁判 avg=5.875，multica REN-76
- **v4.0.18 已知问题**：P2 话术检查生产代码同样需要修正（`realtime/popup_generator.py`）

## 下一步

- v4.0.18 生产代码 P2 修正验收
- 观察 v4.0.12 生产数据，确认是否需要 v4.0.13
- meta_optimize 三策略自进化（Senate/SAGA/TS）持续迭代
- v1.13 的「写后逐句锚定」方法论可考虑融入 v4.0.x 路线
