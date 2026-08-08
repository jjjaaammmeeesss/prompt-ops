# 握手文档 · 星灵亲子沟通多智能体 v3.0 提示词自动迭代

> 最后更新: 2026-07-20 18:30 (统一 v3.1 + 402 根因修复 + 待切换 DeepSeek 新 key)
> 用途: 会话重启时快速恢复记忆，避免上下文丢失。

## 一、总体目标

通过自动化提示词迭代，优化**星灵多智能体 v3.1** 提示词系统（感知层 → 总控层 → 生产层 + 规则引擎），迭代到收益归零为止。

评估指标（5 维）:
- **M1** 触发准确率 (should_popup 是否该弹窗)
- **M5** 语气匹配 (diagnostic/empowering 是否对)
- **M6** 洞察质量 (LLM judge, 0-10)
- **M7** 安全分 (LLM judge, 0-10)
- **overall** 综合分 (加权)

## 二、模型配置现状（关键背景）

### 当前 .env (`D:\prompt-ops\use-cases\parent-child-coach\.env`)
```
DEEPSEEK_API_KEY=DEEPSEEK_API_KEY_PLACEHOLDER   # 公司 key —— 已 402 耗尽
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

### DeepSeek key 状态（全部 402）
| Key | 来源 | 状态 |
|-----|------|------|
| `sk-8063c728...` | 公司 key (parent-child-coach\.env) | ❌ 402（本会话验证调用耗尽） |
| `sk-593d3f8a...` | 星灵 key (星灵\.env / 子阳个人 key 文件) | ❌ 402（原本就空） |

### 待办：切换新 DeepSeek key ⚠️
- 用户指示用 `D:\ob-new202603\钥匙库\deepseek的api key-子阳个人.md` 里的新 key
- **但该文件实际内容 = `sk-593d3f8a...`（星灵旧空 key），并非新 key** —— 回头需用户确认正确的新 key 文件/值
- 切换方式：把新 key 写入 `parent-child-coach\.env` 的 `DEEPSEEK_API_KEY` 即可（`_load_coach_env()` 会优先加载，无需改其他代码）

### 裁判模型：xingluan Claude（待启用）
- endpoint: `https://api.xingluan.vip/runningai/open/v1`
- key: `<XINGLUAN_AUTH_TOKEN 环境变量>` (CONSULTANT_API_KEY)
- model: `claude-opus-4-7`
- 用户要求：被测用例用 xingluan Claude 做裁判（替代当前 DeepSeek judge）
- **尚未实施**：`optimizer.py` / `baseline_runner.py` 的 judge 目前硬编码用 DeepSeek (`DEEPSEEK_MODEL`)。需新增 judge 走 xingluan Claude 的逻辑

## 三、代码改动记录（本次会话）

### 402 根因修复 ✅
- **根因**：`optimizer.py:137` 的 `load_env()` 硬编码读 `D:\星灵-soul-手搓\.env`（key=`sk-593d3f...` 耗尽），`parent-child-coach\.env` 的公司 key 从未被加载
- **修复**：`run_auto_evolve.py` 新增 `_load_coach_env()`，在 `load_env()` 之前加载正确 .env（`setdefault` 不覆盖已设值）
- 验证：修复前变异 6/6 全 402；修复后公司 key 简单+变异 6/6 全通过（但随后公司 key 也被耗尽）

### 统一版本序列 ✅
- 三个层 + 规则引擎 = 一个整体"多智能体版本"
- 文件名版本号 = 内部标题版本号（CLAUDE.md 规则）
- **统一版本计数器**：v3.1 → v3.2 → v3.3 ...（全局递增，不按层分别计数）
- 当前文件：`prompt_感知层_v3.1.md` / `prompt_总控_v3.1.md` / `prompt_生产层_v3.1.md`（星灵多智能体 v3.1 统一版本）
- `run_auto_evolve.py` 中 `unified_counter` 全局递增，`current_versions` 各层初始 v3.1

### 批量引用更新 ✅
- 13 个 Python 文件（prompt_mutator.py / run_auto_evolve.py / 各测试脚本）的 PROMPT_FILES / BASELINE_PROMPTS 全部从 `_v2.5/_v3.5/_v3.2` 改为 `_v3.1`

## 四、当前迭代状态

### 统一版本 v3.1（基线）
保存于 `results/auto_baseline_v25_full.json`:
```
M1=83.9%  M5=61.0%  M6=3.66  M7=4.78  overall=0.797
```

### auto-evolve 历史
- 迭代 1: perception v3.1→v3.2 变异成功，评估后 **DISCARD**（结果保存 `auto_iter_01_perception_v3.2_discard.json`）
- 后续迭代因 402 未跑
- `auto_evolve_history.json` 当前为空（0 迭代成功）

### 关键洞察（失败分析）
- 系统倾向输出 diagnostic，但 18/26 案例专家期望 empowering
- 根因：总控把"发现问题"当默认方向，遗漏家长积极时刻
- 这是 **master 层（总控）** 问题，不是 perception 层

## 五、未决问题 / 重启后步骤

### 必须解决（阻塞）
1. **🔴 两个 DeepSeek key 都 402** —— 需用户提供正确的新 DeepSeek key
   - 用户指的文件 `deepseek的api key-子阳个人.md` 内容错误（是旧空 key）
2. **🟡 启用 xingluan Claude 裁判** —— 改 judge 逻辑走 `claude-opus-4-7`

### 重启后继续步骤
1. 把正确的新 DeepSeek key 写入 `parent-child-coach\.env` 的 `DEEPSEEK_API_KEY`
2. （可选）实施 xingluan Claude 裁判逻辑
3. 后台重启 auto-evolve:
   ```powershell
   $env:PYTHONUNBUFFERED=1
   Start-Process pwsh -ArgumentList "-NoProfile","-Command","python 'D:\prompt-ops\use-cases\parent-child-coach\auto_evolve\run_auto_evolve.py' 2>&1 | Tee-Object -FilePath 'D:\prompt-ops\use-cases\parent-child-coach\results\auto_evolve_deepseek.log'" -WorkingDirectory "D:\prompt-ops\use-cases\parent-child-coach" -WindowStyle Hidden
   ```
4. 监控：`Get-Content "D:\prompt-ops\use-cases\parent-child-coach\results\auto_evolve_deepseek.log" -Tail 20`
5. 删除临时 `auto_iter_0X_baseline.json` 避免加载旧状态

### 关键文件索引
- 规则引擎: `D:\星灵-soul-手搓\亲子沟通洞见\路线A_自上而下法_鼓励版\src\tone_rules.py` (9 条规则)
- 提示词: `D:\星灵-soul-手搓\亲子沟通洞见\路线A_自上而下法_鼓励版\prompts\` (v3.1 统一版本)
- 自动迭代: `D:\prompt-ops\use-cases\parent-child-coach\auto_evolve\`
  - `optimizer.py` (评估器, 含 should_keep)
  - `prompt_mutator.py` (变异器, PROMPT_FILES=[v3.1/v3.1/v3.1])
  - `run_auto_evolve.py` (主循环, MAX_ITERATIONS=3, N_RUNS=3)
- 基线: `D:\prompt-ops\use-cases\parent-child-coach\results\auto_baseline_v25_full.json`
- 钥匙库: `D:\ob-new202603\钥匙库\`
- API 配置: `D:\prompt-ops\use-cases\parent-child-coach\.env`

## 六、CLAUDE.md 关键约束
- 文件名版本号 = 内部标题版本号（必须一致）
- 代码引用 prompt 必须同步更新（已统一 v3.1）
- 需求未澄清前不擅自改代码
- 一个版本号 = 一个文件 = 一条 git commit

## 七、本次会话踩坑
- `Start-Process` + `Tee-Object` 后台跑长任务时，若 shell 工具超时终止，子进程可能被连带杀掉 → 进程"神秘退出"。下次用 `-WindowStyle Hidden` 独立进程 + 日志文件监控
- MiniMax 的 `eyJ...` JWT 是群组/账号 token，**不是 API 调用密钥**，调 `api.minimax.io/v1` 返回 401
- DeepSeek 验证调用（6 次变异+3 次简单）足以耗尽临界余额 → 验证后要立刻恢复正式任务或确认余额充足
