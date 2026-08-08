<h1 align="center">Prompt Ops</h1>

<p align="center">
  <a href="https://pypi.org/project/prompt-ops/"><img src="https://img.shields.io/pypi/v/prompt-ops.svg" /></a>
  <a href="https://www.llama.com/docs/overview/"><img alt="Llama Documentation" src="https://img.shields.io/badge/Llama_OSS-Documentation-4BA9FE?logo=meta" /></a>
  <a href="https://github.com/meta-llama/prompt-ops"><img alt="Llama Tools prompt-ops" src="https://img.shields.io/badge/Llama_Tools-prompt--ops-orange?logo=meta" /></a>
</p>

**prompt-ops** 是一个用于**提示词（prompt）自动迁移与优化的框架与工作区**。它包含两部分：

1. **底层框架**（继承自 [meta-llama/prompt-ops](https://github.com/meta-llama/prompt-ops)）：用 DSPy MIPROv2 / PDO 优化器把「能用的 prompt」自动改成「针对你场景更优的 prompt」，可配置、可度量、可复现。
2. **优化运营工作区**（本仓库 fork 后沉淀的实践）：一套「版本族管理 + 双版本对比脚本 + 国产模型当裁判」的工程化方法，用来长期迭代、回溯、验收生产 prompt 版本。

> 一句话：**框架负责优化，工作区负责让优化这件事变得可追溯、可验收。**

---

## 1. 底层框架

`src/prompt_ops/` 提供 `prompt-ops` 命令行工具：

```bash
prompt-ops create my-project   # 生成带样例 config + dataset 的项目骨架
prompt-ops migrate             # 用配置里的优化器把 prompt 优化一版，默认读 config.yaml
```

**核心能力：**

- **Prompt 迁移/优化**：输入一段现有 system prompt + 一组 query-response 数据集 + 一份 YAML 配置，输出一版更优的 prompt 和对比指标。
- **优化器**：
  - **MIPROv2**（DSPy 内置贝叶斯优化）：用 bootstrapped few-shot 示例 + 指令候选 + 超参搜索，把 prompt 当可调参数、把指标当目标函数。
  - **PDO**（Prompt Duel Optimizer，见 [arXiv:2510.13907](https://arxiv.org/abs/2510.13907)）：免标签的 prompt 优化，用 dueling bandits + Thompson sampling，在 BIG-bench Hard 与 MS MARCO 上达到 SOTA。
- **可插拔 metric**：支持 `LLM-as-Judge`（让裁判 LLM 按维度打分）、文件路径动态加载自定义 metric 类、以及「N/A 维度权重再分配」防作弊机制。
- **多推理后端**：经 LiteLLM 统一接入 OpenRouter / vLLM / NVIDIA NIMs 等。

**标准数据格式**（`StandardJSONAdapter` 自动处理）：

```json
[ { "question": "输入", "answer": "期望输出" } ]
```

自定义格式可继承 `DatasetAdapter`。详见 [docs](docs/)。

---

## 2. 优化运营工作区（本仓库主要实践）

日常的 prompt 迭代不靠记忆和对话，靠**可回溯的脚本 + 存档 + 版本族**。核心约定：

### 版本族管理

以 `use-cases/parent-child-coach/` 的「亲子沟通弹窗」为例，prompt 分四条路线、以版本族为单位管理：

| 路线 | 说明 | 最新生产版 |
|------|------|-----------|
| **A 轨 v1.x** | 原始对抗性修复路线 | v1.13（验证通过） |
| **A 轨 v2.x** | 专家手工标注基线 | v2.3（已确认劣于 v4.0.x，保留作回退） |
| **A 轨 v4.0.x** | DSPy MIPROv2 + 手工迭代，**当前生产** | **v4.0.23** |
| **B 轨 星灵多智能体** | 感知/生产/总控三层多智能体 | 独立仓库 |

**版本号铁律**（见 `CLAUDE.md`）：文件名 = 内部标题版本号；改版必须同步更新代码引用；一个版本号 = 一个文件 = 一条 commit；中间版本确认不回退即清理。

### 双版本对比方法论

任何「新版本是否晋升」都必须跑**同 judge、同 case、同 n** 的横向对比，而不是口头判断：

- 对比脚本统一放各 use-case 根目录：`compare_v151_v30.py`、`compare_v154_v155.py`、`compare_v155_v20.py`、`compare_v20_v21.py`、`compare_four_versions.py` 等。
- 裁判统一用**国产模型**（优先 DeepSeek，Claude 只做 Judge 不参与生成）。
- 结果以 JSON 存档进 `results/`，逐条可回溯「谁赢、赢在哪、输在哪」。
- 结论也回填进内存记忆（如「v2.1 未通过 Δ=-0.847」「v1.13 66.7% 决胜率」），避免重复踩坑。

### 运行方式示例

```bash
# 用 PDO / MIPROv2 优化一段通用弹窗 prompt
prompt-ops migrate --config use-cases/general-popup-native/config_pdo.yaml \
                   --output-dir use-cases/general-popup-native/results

# 横向对比两个版本（同 judge 同 case 同 n）
python use-cases/general-dialogue-popup/compare_v155_v20.py
```

---

## 3. use-cases 清单

| 目录 | 内容 | 状态 |
|------|------|------|
| `parent-child-coach/` | 亲子沟通现场弹窗——**主战场**，四条版本族 + 多智能体 + 真实管线 | 活跃，生产 v4.0.23 |
| `general-dialogue-popup/` | 通用对话弹窗（上下级/同事/情侣等），双版本对比脚本齐全 | 活跃 |
| `general-popup-native/` | 用原生优化器对比 PDO vs MIPROv2 优化通用弹窗 | 活跃 |
| `web-of-lies-pdo/` | PDO 在逻辑推理任务上的演示（Web of Lies） | 演示 |
| `ms-marco-pdo/` | PDO 在信息检索（MS MARCO）上的评测 | 演示 |
| `facility-support-analyzer/` | 设施客服消息分类 prompt 迁移示例 | 样例 |
| `hotpotqa/` | HotpotQA 多跳推理基准 | 样例 |

---

## 4. 快速开始（框架）

```bash
conda create -n prompt-ops python=3.10 && conda activate prompt-ops
pip install -e .          # 推荐源码安装
prompt-ops create my-project && cd my-project
# 在 .env 里配置 OPENROUTER_API_KEY 等
prompt-ops migrate        # 默认读 config.yaml，结果输出到 results/
```

详细教程见 [docs/basic/readme.md](docs/basic/readme.md)。

---

## 5. 文档与索引

- [CONCEPTS.md](CONCEPTS.md)：本项目专属概念词典（Master/Production/Validator 智能体、诊断式/鼓励式弹窗、FC_TONE_OFF、P2 引语检查等）。
- [docs/](docs/)：框架使用指南（Quick Start / 配置 / 数据集适配 / metric 选择 / 推理后端）。
- 各 use-case 根目录的 `README*.md`：对应场景的部署与运行说明。

## 开发

```bash
pip install -e ".[dev]"
pytest                          # 跑测试
black . && isort . && mypy src   # 格式化与类型检查
```

## 致谢

底层框架基于 [meta-llama/prompt-ops](https://github.com/meta-llama/prompt-ops)（MIT），并借鉴 [DSPy](https://github.com/stanfordnlp/dspy) 的优化思路。

## License

MIT — 见 [LICENSE](LICENSE)。
