# v1.3 沟通现场弹窗 · 一般场景 — 独立部署包

> **v1.4 = 当前最优版本（v1.0 → v1.4 迭代结果）**
> 架构：窗口模式（~300字切窗逐窗分析，LLM 自主判断洞察视角：自己/对方/模式）+ 关键词提前触发
> 测试集（15 例，去人名，Claude Opus 评审）：mean=4.15，弹窗率=0.733，硬规则违规=2
> 场景：任意两人日常对话（同事、朋友、伴侣、家人……），服务对象是账号持有者（对话中的一方）

## 文件清单

| 文件 | 说明 |
|------|------|
| `system_prompt_v1.4.txt` | 当前最优核心提示词（v1.0 → v1.4 迭代结果） |
| `runner_v10.py` | 自包含运行器（默认加载 `system_prompt_v1.4.txt`） |
| `keyword_config.json` | 关键词触发配置（critical 立即提前分析；warning 预触发向前250字/向后等50字） |
| `judge_glm.py` | GLM-5.2 评审器（软维度打分 + 硬规则代码检查） |
| `evolve.py` | 自我迭代主循环（生成→评分→变异→择优保留） |
| `config.yaml` | 模型配置 |
| `data/sample_dialogues.json` | 5 条冒烟测试对话 |
| `data/test_dialogues.json` | 15 条迭代评估对话（含 3 条应保持安静的健康对话） |
| `results/evolve/` | 迭代结果（每轮评分 + final_report.json） |
| `README_v1.0.md` | 本说明文件 |

## 产品背景与硬约束

- **输入是录音转文字**：没有画面、没有人称标签，AI 需自行推断哪方是账号持有者（推断不了时对双方都友善，绝不贬低任何一方）
- **质量 > 数量**：弹窗的准确率和建议可用性优先于弹出量，宁缺毋滥
- **字数**：硬合规 180 字以内，200 字是前端窗口的绝对底线
- **双重机制**：关键词命中触发提前分析（critical 立即触发、向前最多300字/最少80字；warning 预触发、向前250字/向后等50字），与 ~300 字切窗主通道共用同一 prompt

## 三条核心使命

1. **看清自己** —— 话说重了 / 太软弱丢失边界 / 情绪上来了自己没察觉
2. **看见对方** —— 言下之意 / 没有明确表达的需求
3. **看见模式** —— 两人互动的循环（如一方逼迫一方退让、一个追一个躲）

## 与 v2.3（亲子场景）的异同

| 维度 | v2.3 亲子 | v1.0 一般场景 |
|------|-----------|---------------|
| 单 prompt 架构 | ✅ | ✅ 保留 |
| ~300 字切窗 | ✅ | ✅ 保留 |
| 价值观层 | 孩子是独立生命；先看见家长 | 生命立场：两人都是完整的人；平等协商合作为基准，不帮一方欺压/欺骗另一方 |
| tone 维度 | 诊断式 / 鼓励式 | 废掉，改为按洞察对象：**自己 / 对方 / 模式**（只在心里判断，不标注） |
| 双视角要求 | 必须有"已看到的 + 没看到的" | **取消**，核心是"没看到的"；认知中立、态度是帮助，不批判任何一方 |
| `——` 功能墙 | ✅ | ✅ 原样保留 |
| 分段硬规则 | ✅ | ✅ 保留 |
| 字数 | 诊断 ≤200 / 鼓励 ≤60 | **60–180 硬合规，200 为前端绝对底线** |
| 触发机制 | ~300字切窗单通道 | 双重：切窗 + 关键词提前触发（critical/warning 两级） |
| 安静信号 | 无 | "安好"两字（宁缺毋滥） |

## 自我迭代

```bash
export DEEPSEEK_API_KEY="sk-xxx"   # 生成弹窗 + prompt 变异（DeepSeek v4-pro）
export GLM_API_KEY="bce-v3/..."    # judge（百度千帆 glm-5.2）
python evolve.py --rounds 5 --patience 2
```

每轮：跑全部 15 条测试对话 → GLM-5.2 按 insight/suggestion/non_judgment/language 四维打分（硬规则代码检查扣分）→ 最低分案例反馈交给 DeepSeek v4 改写 prompt → 总分更高则保留新版本（`system_prompt_v1.N.txt`），连续 2 轮无提升自动停止。结果见 `results/evolve/`。

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
export V10_API_KEY="sk-xxx"
export V10_API_BASE="https://api.deepseek.com/v1"
export V10_MODEL="deepseek/deepseek-v4-pro"
```

方式二：修改同目录 `config.yaml`。

### 3. 运行

```bash
# 命令行直接输入对话
python runner_v10.py --dialogue "阿琳：方案改完了吗？\n老周：还在弄……"

# 从文件读取
python runner_v10.py --file dialogue.txt

# JSON 输出（供程序消费）
python runner_v10.py --file dialogue.txt --json

# 强制指定洞察视角
python runner_v10.py --file dialogue.txt --lens 模式
```

### 4. Python 调用

```python
from runner_v10 import V10Runner

runner = V10Runner(
    model="deepseek/deepseek-v4-pro",
    api_key="sk-xxx",
    api_base="https://api.deepseek.com/v1",
)

result = runner.run(dialogue_text="阿琳：你怎么又忘了回我消息？\n老周：我刚看到！")

for popup in result["popups"]:
    print(f"[{popup['lens']}] 窗口 {popup['window_range']}")
    print(popup["text"])
    print()
```

## 对话格式

支持两种输入格式（与 v2.3 相同）：

**格式一：带说话人前缀**（推荐）
```
阿琳：方案改完了吗？
老周：还在弄……马上好。
```

**格式二：纯文本**（按行切句）

## 输出结构

```json
{
  "popups": [
    {
      "window_range": "0-2",
      "window_indices": [0, 1, 2],
      "lens": "模式",
      "text": "完整的弹窗文本...",
      "raw_response": "LLM 原始输出"
    }
  ],
  "windows": [
    {"index": 0, "speaker": "阿琳", "text": "方案改完了吗？"}
  ],
  "raw_dialogue": "原始输入文本"
}
```

> 注：`lens` 字段由 runner 关键词启发式粗判，仅作展示参考；prompt 本身不要求 LLM 输出视角标签。
