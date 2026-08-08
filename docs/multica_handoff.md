---
title: Multica Task Handoff
category: Advanced
description: 从 multica runtime 接手任务到本机 opencode/cc 的五步流程
order: 90
icon: relay
---

# Multica 任务接手清单

当 multica 看板上的某个任务已经被 runtime agent（headless opencode）跑过、产出了成果，
你想在本机的 opencode 或 cc（Claude Code）里直接接手继续干时，按下面五步走。

## 核心认知

multica 的 task workdir（`~/multica_workspaces/<task>/`）是**一次性临时工位**：

- multica server **不存** working directory 的文件副本（官方文档原文：`does not automatically upload your entire working directory`）
- 任务完成后 24 小时内会被 GC 清理（`MULTICA_GC_TTL` 默认 24h）
- 本机直接去翻 workdir = 绕过隔离设计 + 拿到的文件随时可能消失

所以接手的本质不是"读 workdir"，而是**从前一个 agent 落进 multica 可见通道的成果里取**。
multica CLI 能读到的只有三处：**git 分支、issue attachment、issue comment/metadata**。
只躺在 workdir 里、没进这三处的成果，对 multica 而言等于不存在——接不到。

## 前置：本机 opencode / cc 装 multica-cli skill

第一次用要装，一劳永逸。skill 教本地 agent 怎么调已认证的 `multica` CLI。

- **opencode**：把 `multica-ai/multica-cli` 仓库的 `skills/multica-cli/SKILL.md` 复制到你某个 skill 加载目录（如 `C:\Users\h\.agents\skills\multica-cli\`）
- **cc (Claude Code)**：在 cc 里跑
  ```
  /plugin marketplace add multica-ai/multica-cli
  /plugin install multica-cli@multica-cli
  ```
- 两边都依赖本机已 `multica login`（或 `multica setup`）完成认证

## 五步接手流程

假设要接手的 issue key 是 `MUL-xx`，按顺序执行。

### 1. 读看板上下文

```
multica issue get MUL-xx --output json
multica issue comment list MUL-xx --recent 10 --output json
multica issue metadata list MUL-xx --output json
```

拿到：任务描述、讨论历史、已钉住的高信号状态（pr_url / pipeline_status / blocked_reason 等）。
长线程用 `--thread <comment-id> --tail 30` 精读，避免一次性灌进太多上下文。

### 2. 查代码成果（git 通道，主通道）

```
multica issue pull-requests MUL-xx --output json
```

拿到前一个 agent 开的 PR：repo、分支名、CI 状态、mergeability。
**代码内容的读取走 git，不走 multica**——multica 只存"哪个分支"这个元信息。

### 3. 拉代码到本地

两种等价方式，任选其一：

```
# 方式 A：用 multica repo checkout（自动找到 workspace 关联的 repo）
multica repo checkout <repo-id> --ref <branch>

# 方式 B：在本机已有 checkout 里直接 git 操作（D:\prompt-ops 已是 git repo 时优先）
git fetch origin
git checkout <branch>
```

### 4. 下载非代码附件（attachment 通道）

agent 如果把文件作为 attachment 贴到了 issue 或 comment 上，用：

```
multica attachment download <attachment-id> --output-dir .\handoff-in
```

attachment-id 从第 1 步的 `issue get --output json` 或 `comment list --output json` 里取。
代码改动走第 3 步，**不要**重复塞 attachment。

### 5. 兜底：查 agent 的实际操作记录

万一前一个 agent 改过文件但没 push 也没附 attachment，用 run-messages 看它到底动过什么：

```
multica issue runs MUL-xx --output json
multica issue run-messages <task-id> --issue MUL-xx --output json
```

能看到 agent 的工具调用、读写过的文件、跑过的命令。
**注意**：这只是"知道动过什么"的线索，不是"能拿到文件内容"——那些文件如果没进 git/attachment，依然读不到，
只能回 issue 里 @mention 那个 agent 让它补 push 或补附件。

## 接手后的回写

接手完成、本机继续干完后，回写同样走 multica 原生通道，**不要**建本地 handoff 文件旁路：

| 产物 | 回写通道 |
|---|---|
| 代码改动 | push 到同一分支（PR 自动 link 到 issue）；分支名或 PR body 带 `MUL-xx` |
| 进度/结论/决策 | `multica issue comment add MUL-xx --content-file ./reply.md`（评论必走文件，避免 shell 改写） |
| 高信号持久状态 | `multica issue metadata set MUL-xx --key <k> --value <v>`（只钉未来会重读的：pr_url / pipeline_status / blocked_reason） |
| 完成收尾 | 合并 PR 时 body 带 `Closes MUL-xx` → multica 自动转 Done；或 `multica issue status MUL-xx done` |

## 常见踩坑

- **钻进 `~/multica_workspaces/<task>/` 直接改文件**：绕过隔离、被 GC 清掉、directory lock 冲突、看板状态漂移。这就是被禁止的 handoff 旁路。
- **用 inline `--content` 写评论**：shell 会改写 backtick、`$()`、引号、换行。永远用 `--content-file`。
- **把运行日志/调查笔记钉进 metadata**：metadata 是高信号持久状态，不是日志。文档明确禁止钉 `attempts`、单次调查笔记、大日志、secrets。
- **@mention agent 只为道谢/确认**：每次 @mention agent 都会触发一次新 run，容易造成循环。
- **把 `backlog` 误当 todo**：`backlog` 不触发 run；`done`/`cancelled` 重新分配反而会立即触发 run。

## 何时不用这套

- 任务还没被任何 runtime agent 跑过 → 直接 `multica issue assign MUL-xx --to <你>` 让 multica 派给本机 runtime，走正常 dispatch。
- 你只是想看一眼任务状态 → `multica issue get MUL-xx` 就够，不用全套。
- 想让 multica 继续自动 retry / 跨 agent 转手 → 用看板 retry 按钮 或 `multica issue rerun`，本机不用接手。

## 参考

- multica CLI 命令参考：https://multica.ai/docs/cli
- Tasks 与 retry/session 续接语义：https://multica.ai/docs/tasks
- daemon 数据边界（为什么不存 workdir）：https://multica.ai/docs/daemon-runtimes
- multica-cli skill 源：https://github.com/multica-ai/multica-cli
