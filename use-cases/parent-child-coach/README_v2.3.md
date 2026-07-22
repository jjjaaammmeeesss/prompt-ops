# v2.3 亲子沟通教练 · 现场弹窗 — 独立部署包

> **v2.3 = 2025 年三位专家手工标注的单 prompt 基线**
> 架构：窗口模式（~300字切窗逐窗分析，LLM 自主判断 diagnostic/empowering tone）

## 文件清单

| 文件 | 说明 |
|------|------|
| `system_prompt_v2.3.txt` | 核心提示词（55行，4641字节） |
| `runner_v23.py` | 自包含运行器（无需测试智能体依赖） |
| `README_v2.3.md` | 本说明文件 |

## 快速开始

### 1. 安装依赖

```bash
pip install litellm pydantic pyyaml
```

### 2. 配置 API Key

方式一：环境变量
```bash
export DEEPSEEK_API_KEY="sk-xxx"
# 或
export V23_API_KEY="sk-xxx"
export V23_API_BASE="https://api.deepseek.com/v1"
export V23_MODEL="deepseek/deepseek-chat"
```

方式二：config.yaml（与本文件同目录）
```yaml
models:
  gen: "deepseek/deepseek-chat"
  gen_api_base: "https://api.deepseek.com/v1"
  gen_api_key_env: "DEEPSEEK_API_KEY"
```

### 3. 运行

```bash
# 命令行直接输入对话
python runner_v23.py --dialogue "妈妈：作业写完了吗？\n孩子：还没……"

# 从文件读取
python runner_v23.py --file dialogue.txt

# JSON 输出（供程序消费）
python runner_v23.py --file dialogue.txt --json

# 强制指定 tone
python runner_v23.py --file dialogue.txt --tone 诊断式
```

### 4. Python 调用

```python
from runner_v23 import V23Runner

runner = V23Runner(
    model="deepseek/deepseek-chat",
    api_key="sk-xxx",
    api_base="https://api.deepseek.com/v1",
)

result = runner.run(dialogue_text="妈妈：你怎么又在玩手机？\n孩子：我刚拿起来！")

for popup in result["popups"]:
    print(f"[{popup['tone']}] 窗口 {popup['window_range']}")
    print(popup["text"])
    print()
```

## 对话格式

支持两种输入格式：

**格式一：带说话人前缀**（推荐）
```
妈妈：作业写完了吗？
孩子：还没……我在休息一下。
妈妈：你已经休息一个小时了！
```

**格式二：纯文本**（按行切句）
```
作业写完了吗？
还没……我在休息一下。
你已经休息一个小时了！
```

## 输出结构

```json
{
  "popups": [
    {
      "window_range": "0-2",
      "window_indices": [0, 1, 2],
      "tone": "诊断式",
      "text": "完整的弹窗文本...",
      "raw_response": "LLM 原始输出"
    }
  ],
  "windows": [
    {"index": 0, "speaker": "妈妈", "text": "作业写完了吗？"},
    {"index": 1, "speaker": "孩子", "text": "还没……"}
  ],
  "raw_dialogue": "原始输入文本"
}
```

## v2.3 核心特征

- **单 prompt**：一份 `system_prompt_v2.3.txt` 承担全部逻辑
- **~300 字切窗**：对话按字数分块，每块独立分析，保证上下文精度
- **LLM 自主判 tone**：不预设 diagnostic/empowering，由 LLM 根据对话内容决定
- **双视角输出**：每个弹窗至少包含两种视角（家长已看到的 + 还没注意到的）
- **`——` 功能墙**：墙前全部洞察，墙后只有一句建议

## 与 v4.0.12 的关系

v4.0.12 = v2.3 基底 → DSPy MIPROv2 → 12 轮手工迭代的当前生产版本。

v4.0.12 是两阶段流水线（Stage1 周易三爻分析 → Stage2 弹窗生成），v2.3 是单 prompt。
对比评估脚本见测试智能体仓库的 `scripts/compare_v23_vs_v4012.py`。
