# 正则句式匹配快速路径实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 在现有 critical/warning 关键词快速路径中增加逐句正则匹配；正则命中后按 warning 窗口策略处理，并跳过该句的 warning 关键词检查。

**架构：** `keyword_config.json` 新增顶层 `patterns` 对象，按四种句式类别保存 11 条正则字符串。`V10Runner._match_pattern()` 逐类别、逐表达式调用 `re.search()`；`_chunk_windows()` 保持 critical 最高优先级，在未命中 critical 时先查 pattern，再决定是否查 warning keyword，并让 pattern 与 warning 共用既有“向前 250 字 + 向后 50 字”分支。

**技术栈：** Python 3.10+、标准库 `re` / `json`、现有 `V10Runner`。

---

## 约定与范围

- 只改 `use-cases/general-dialogue-popup/keyword_config.json` 与 `use-cases/general-dialogue-popup/runner_v10.py`；不引入依赖、不重构切窗算法。
- `patterns` 的四个键固定为 `stealth_but`、`masking`、`helpless_story`、`controlling`，数组数量分别为 4、2、3、2；所有 pattern 均属于 warning 级。
- pattern 命中返回 `pattern:warning:<category>:<regex>`。该格式保留级别、类别和实际命中的配置项，便于日志与结果追踪。
- 单句优先级固定为 `critical keyword > pattern > warning keyword > normal window`。critical 命中时不查 pattern；pattern 命中时不查 warning keyword。
- `_match_pattern()` 接收 `_chunk_windows()` 已构造的单句 `sentence`（含可选说话人前缀），不跨句匹配。
- pattern 使用配置中的原始字符串；JSON 中反斜杠必须双写，例如正则中的 `\s` 在 JSON 文件内写成 `\\s`。

## 计划修改后的文件结构

- `use-cases/general-dialogue-popup/keyword_config.json`：新增四类 warning 级正则配置。
- `use-cases/general-dialogue-popup/runner_v10.py`：导入 `re`、增加 `_match_pattern()`、调整 `_chunk_windows()` 的匹配顺序与 warning 分支触发值。

### Task 1：定义并静态校验 11 条正则配置

**文件与位置：**

- 修改：`use-cases/general-dialogue-popup/keyword_config.json:2-3`，在 `_description` 后、`critical` 前新增顶层 `patterns` 字段。
- 不改现有 `critical`（当前 3-28 行）和 `warning`（当前 29-87 行）数组内容。

- [ ] **Step 1：加入四类正则数组**

核心结构（JSON 中的反斜杠按 JSON 规则转义）：

```json
{
  "_description": "...",
  "patterns": {
    "stealth_but": [
      "不是.{0,12}(?:怪|说|针对)你[，,、\\s]*(?:但是|但|不过)",
      "我(?:没有|没想|不是想).{0,12}[，,、\\s]*(?:但是|但|不过)",
      "(?:我知道|我理解|我明白).{0,12}[，,、\\s]*(?:但是|但|不过)",
      "(?:别误会|你别多想).{0,8}[，,、\\s]*(?:但是|但|不过)"
    ],
    "masking": [
      "(?:没事|没关系|我没事).{0,8}(?:随便|算了|不用了|你忙吧)",
      "(?:挺好|很好|可以|行啊).{0,8}(?:反正|至少|总比|就这样)"
    ],
    "helpless_story": [
      "(?:我还能|我又能|能让我)怎(?:么|样)(?:办|做|说)?",
      "(?:反正|无论|不管).{0,12}(?:都没用|也没用|都一样|没人管)",
      "(?:我也|我已经).{0,8}(?:没办法|不知道怎么办|无能为力|尽力了)"
    ],
    "controlling": [
      "你(?:必须|一定要|最好|应该|得).{1,24}",
      "(?:不许|不准|别再|马上|立刻).{1,24}"
    ]
  },
  "critical": ["..."],
  "warning": ["..."]
}
```

实现注意：

- `{0,n}` 给变体留出有限间隔，同时避免 `.*` 跨过过多内容造成误报。
- controlling 的 `.{1,24}` 要求命令词后存在被控制的动作/内容，避免只有“马上”这类残句就触发。
- 不给表达式添加 `^`/`$`，以兼容 `_chunk_windows()` 传入的 `说话人：正文` 格式。

- [ ] **Step 2：验证 JSON 可解析、四类数量准确、每条 regex 可编译**

运行：

```powershell
@'
import json
import re
from pathlib import Path

path = Path("use-cases/general-dialogue-popup/keyword_config.json")
config = json.loads(path.read_text(encoding="utf-8"))
expected = {
    "stealth_but": 4,
    "masking": 2,
    "helpless_story": 3,
    "controlling": 2,
}
assert set(config["patterns"]) == set(expected)
assert {name: len(items) for name, items in config["patterns"].items()} == expected
for category, patterns in config["patterns"].items():
    for pattern in patterns:
        re.compile(pattern)
print("PASS: JSON valid; 11/11 regexes compiled")
'@ | python -
```

预期：退出码为 0，输出 `PASS: JSON valid; 11/11 regexes compiled`。

**验证方法：** 除上述脚本外，确认 `critical`、`warning` 的数组长度及原始内容在 diff 中没有变化。

**风险：低。** 主要风险是 JSON 反斜杠漏转义、尾逗号导致解析失败、类别拼写或数量不符；编译脚本能拦截语法问题和数量偏差，但不能证明语义不会误报。

### Task 2：增加逐句 `_match_pattern()` 匹配器

**文件与位置：**

- 修改：`use-cases/general-dialogue-popup/runner_v10.py:25-32`，标准库导入区加入 `import re`。
- 修改：`use-cases/general-dialogue-popup/runner_v10.py:258-270`，在 `_match_critical()` 与 `_match_warning()` 之间新增 `_match_pattern()`。

- [ ] **Step 1：先用最小对象测试描述期望行为**

不实例化 `V10Runner`（避免 prompt/config 文件依赖），通过 `__new__` 注入配置；此脚本在实现方法前应因缺少 `_match_pattern` 而失败：

```powershell
@'
import importlib.util
from pathlib import Path

module_path = Path("use-cases/general-dialogue-popup/runner_v10.py")
spec = importlib.util.spec_from_file_location("runner_v10", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

runner = module.V10Runner.__new__(module.V10Runner)
runner._keywords = {
    "patterns": {
        "stealth_but": [r"我知道.{0,12}(?:但是|但|不过)"],
        "masking": [],
        "helpless_story": [],
        "controlling": [],
    }
}
assert runner._match_pattern("小王：我知道你很忙，但是这件事不能再拖了") == (
    r"pattern:warning:stealth_but:我知道.{0,12}(?:但是|但|不过)"
)
assert runner._match_pattern("小王：我知道你很忙，所以明天再说") is None
print("PASS: pattern matcher")
'@ | python -
```

预期（实现前）：非零退出，错误包含 `AttributeError`。

- [ ] **Step 2：实现最小匹配逻辑**

核心代码：

```python
import re

# ...

def _match_pattern(self, text: str) -> str | None:
    """逐类检查 warning 级正则；仅匹配当前句。"""
    for category, patterns in self._keywords.get("patterns", {}).items():
        for pattern in patterns:
            if pattern and re.search(pattern, text):
                return f"pattern:warning:{category}:{pattern}"
    return None
```

实现注意：

- 使用 `.get("patterns", {})`，让没有新字段的旧配置保持“不命中 pattern”而不是抛 `KeyError`。
- 使用 `re.search(pattern, text)`，不使用 `re.match()`，因为正文前可能有 `说话人：`。
- 按 JSON 中类别顺序与数组顺序返回首个命中，保证结果确定性。
- 不在运行时吞掉 `re.error`；Task 1 的 `re.compile()` 是配置发布门禁，若配置绕过门禁仍应快速暴露错误，而非静默漏检。

- [ ] **Step 3：复跑匹配器脚本**

运行 Step 1 的同一命令。

预期：退出码为 0，输出 `PASS: pattern matcher`。

**验证方法：** 增加一次兼容性断言：`runner._keywords = {"critical": [], "warning": []}` 时，`runner._match_pattern("任意文本") is None`。

**风险：中。** 正则按句逐条搜索会增加少量 CPU 开销；表达式灾难性回溯会放大该风险。当前 11 条表达式只有有限量词且没有嵌套无限量词，风险可控。另一风险是配置值不是对象/数组时出现类型错误；本计划不增加配置 schema，以保持改动最小，并以静态校验作为发布门禁。

### Task 3：接入 `_chunk_windows()` 的触发优先级

**文件与位置：**

- 修改：`use-cases/general-dialogue-popup/runner_v10.py:272-282`，docstring 增加 pattern 规则与优先级说明。
- 修改：`use-cases/general-dialogue-popup/runner_v10.py:323-357`，在 while 循环中将 pattern 检查置于 `_match_warning()` 前，并让 pattern 复用 warning 窗口分支。

- [ ] **Step 1：写行为级回归脚本，覆盖“pattern 命中后不查 warning keyword”**

核心测试脚本：

```powershell
@'
from pathlib import Path
import importlib.util

path = Path("use-cases/general-dialogue-popup/runner_v10.py")
spec = importlib.util.spec_from_file_location("runner_v10", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

runner = module.V10Runner.__new__(module.V10Runner)
runner.window_size = 300
runner.critical_min_context = 80
runner.warning_backward_chars = 250
runner.warning_forward_chars = 50
runner._keywords = {
    "critical": ["你滚"],
    "warning": ["但是"],
    "patterns": {
        "stealth_but": [r"我知道.{0,12}(?:但是|但|不过)"],
        "masking": [],
        "helpless_story": [],
        "controlling": [],
    },
}

calls = 0
original = runner._match_warning
def counted_warning(text):
    global calls
    calls += 1
    return original(text)
runner._match_warning = counted_warning

windows = [
    module.TestWindow(window_index=0, speaker="甲", text="前文" * 80),
    module.TestWindow(window_index=1, speaker="乙", text="我知道你着急，但是先听我说"),
    module.TestWindow(window_index=2, speaker="甲", text="后文" * 20),
]
chunks = runner._chunk_windows(windows)
pattern_chunk = next(chunk for chunk in chunks if chunk[2].startswith("pattern:warning:"))
assert pattern_chunk[2].startswith("pattern:warning:stealth_but:")
assert calls == 0, "pattern 命中句不应再检查 warning keyword"
assert len(pattern_chunk[0].splitlines()[0]) <= 253  # 250 字前文 + “甲：”
assert "后文" in pattern_chunk[0]
print("PASS: pattern precedence and warning strategy")
'@ | python -
```

预期（接入前）：非零退出，原因是没有 `pattern:warning:` 触发窗或 warning 被调用。

- [ ] **Step 2：按显式短路顺序接入**

核心代码：

```python
sentence, _ = sentences[i]
crit = self._match_critical(sentence)
pattern = self._match_pattern(sentence) if not crit else None
warn = self._match_warning(sentence) if not crit and not pattern else None
warning_trigger = pattern or warn

if crit:
    # 保持现有 critical 分支不变
    ...

if warning_trigger:
    # warning / pattern：向前 250，向后等 50
    start = find_start(i, self.warning_backward_chars, last_end)
    end = i
    after = 0
    while end + 1 < n and after < self.warning_forward_chars:
        end += 1
        after += lengths[end]
    chunk_text = "\n".join(s for s, _ in sentences[start:end + 1])
    chunk_indices = [idx for _, idx in sentences[start:end + 1]]
    chunks.append((chunk_text, chunk_indices, warning_trigger))
    last_end = end
    i = end + 1
    continue
```

同时把 docstring 规则写清楚：

```python
"""
规则：
  - 正常窗口：无触发时按 ~window_size 切窗
  - critical keyword：最高优先级，沿用 critical 上下文策略
  - warning pattern：逐句 re.search；命中后跳过 warning keyword
  - warning keyword：仅在 critical/pattern 均未命中时检查
  - pattern 与 warning keyword 均向前取 250 字、再向后等 50 字
"""
```

- [ ] **Step 3：复跑行为级回归脚本**

运行 Step 1 的同一命令。

预期：退出码为 0，输出 `PASS: pattern precedence and warning strategy`。

**验证方法：** 再补三条独立断言：

1. 同一句同时含 critical 与 pattern 时，trigger 以 `keyword:critical:` 开头。
2. 只含 warning keyword 时，trigger 仍以 `keyword:warning:` 开头。
3. 无任何命中时，trigger 仍为 `window`。

**风险：中。** 最容易出错的是用 `pattern or warn` 但仍无条件提前计算 `warn`，违反“pattern 命中后不查关键词”；必须用条件表达式短路。其次是误把 pattern 接到 critical 分支，导致使用错误的上下文范围，或改变 `last_end/i` 推进造成重复/遗漏窗口。

### Task 4：对全部 11 条 pattern 做有效性与代表性语义验证

**文件与位置：**

- 检查：`use-cases/general-dialogue-popup/keyword_config.json` 的 `patterns` 整段。
- 检查：`use-cases/general-dialogue-popup/runner_v10.py` 的 `_match_pattern()` 与 `_chunk_windows()`。
- 本任务不新增测试文件，使用一次性标准库脚本完成配置级验收。

- [ ] **Step 1：逐条执行 `re.compile()` 与 `re.search()`**

核心脚本结构：

```powershell
@'
import json
import re
from pathlib import Path

config = json.loads(
    Path("use-cases/general-dialogue-popup/keyword_config.json").read_text(encoding="utf-8")
)
positive = {
    "stealth_but": [
        "我不是怪你，但是你应该早点告诉我",
        "我没有想责备你，不过这次确实耽误了",
        "我知道你已经尽力了，但结果还是不行",
        "你别多想，不过这件事我不会答应",
    ],
    "masking": [
        "我没事，你忙吧",
        "行啊，反正你已经决定了",
    ],
    "helpless_story": [
        "我又能怎么办",
        "反正说什么都没用",
        "我已经尽力了",
    ],
    "controlling": [
        "你必须今天把它做完",
        "不准再联系他",
    ],
}
negative = {
    "stealth_but": ["我知道你很忙，所以我们改天聊"],
    "masking": ["没事，我们一起解决"],
    "helpless_story": ["我有办法，我们先试一次"],
    "controlling": ["你可以考虑明天再做"],
}

total = 0
for category, patterns in config["patterns"].items():
    assert len(patterns) == len(positive[category])
    for pattern, sample in zip(patterns, positive[category]):
        compiled = re.compile(pattern)
        assert compiled.search(sample), (category, pattern, sample)
        total += 1
    for sample in negative[category]:
        assert not any(re.search(pattern, sample) for pattern in patterns), (category, sample)

assert total == 11
print("PASS: 11/11 regexes valid and representative samples matched")
'@ | python -
```

预期：退出码为 0，输出 `PASS: 11/11 regexes valid and representative samples matched`。

- [ ] **Step 2：执行 Python 语法校验**

运行：

```powershell
python -m py_compile use-cases/general-dialogue-popup/runner_v10.py
```

预期：退出码为 0，无输出。

- [ ] **Step 3：检查最终差异范围**

运行：

```powershell
git diff -- use-cases/general-dialogue-popup/keyword_config.json use-cases/general-dialogue-popup/runner_v10.py
```

预期：仅包含 `patterns` 配置、`import re`、`_match_pattern()`、`_chunk_windows()` 的优先级/docstring 调整；不应有 prompt、模型调用、CLI 或现有关键词内容变化。

**验证方法：** 上述三步全部通过才算完成；特别记录 11/11 编译与正例命中，不能只以 JSON 可读取代替正则验证。

**风险：中。** `re.compile()` 只能验证语法，正反例只能覆盖代表性语义，不能完全排除真实对话中的误报/漏报。上线前应以历史对话样本离线观察四类触发分布；这属于验收建议，不在本次最小代码改动范围内。

## 完成标准

- `keyword_config.json` 存在四类、共 11 条 pattern，数量严格为 4/2/3/2。
- 每条 pattern 都通过 `re.compile()`，并各自命中一个代表性正例。
- `_match_pattern()` 使用逐句 `re.search()`，返回 `pattern:warning:<category>:<regex>` 或 `None`。
- `_chunk_windows()` 的执行顺序是 critical → pattern → warning keyword，且 pattern 命中句不会调用 `_match_warning()`。
- pattern 触发复用 warning 的向前 250 字、向后 50 字策略。
- 旧配置没有 `patterns` 时不会因 `_match_pattern()` 抛错；原 critical、warning、normal window 行为通过回归断言。
- 最终 diff 不包含任务范围外的文件或行为改动。
