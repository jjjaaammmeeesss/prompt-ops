"""快慢通道窗口规格 —— 唯一权威参数（所有版本统一标准）。

这是「亲子沟通洞见弹窗」触发层窗口逻辑的单一真源（single source of truth）。
生产 SUT、测试管线（run_v418_pipeline / run_demo）、realtime 原型一律从这里取值，
避免各版本参数漂移。

规格原文（用户确认，无歧义）：

- 慢通道：缓冲凑满 300 字才分析
- 快通道 critical（最严重，severity≥4）：命中当下就弹，向前取最多 300 字，
  最少 80 字，少于 80 取消
- 快通道一般严重（warning + opportunity 一致逻辑）：向前取 250 字，
  命中后等缓冲再进 50 字再试图分析，总字数少于 80 取消
- "向后等 50 字"：真实录音实时语音转写流式场景，关键词命中后不着急，
  等转写再累积 50 个字，窗口 buffer[trigger_pos-250 : trigger_pos+50] 再弹窗分析
"""

# 触发层级：critical = severity ≥ 该阈值（决策1 选 B）
CRITICAL_SEVERITY_MIN = 4

# 慢通道：缓冲凑满多少字才分析
SLOW_THRESHOLD_CHARS = 300

# 快通道 critical：命中当下弹，向前取最多多少字
FAST_CRITICAL_FORWARD = 300

# 快通道一般严重：向前取多少字
FAST_GENERAL_FORWARD = 250

# 快通道一般严重：命中后等缓冲再进多少字再试图分析
FAST_GENERAL_WAIT = 50

# 快通道下限：总字数少于多少取消
FAST_MIN_CHARS = 80


def is_critical(severity: int) -> bool:
    """severity ≥ CRITICAL_SEVERITY_MIN 视为 critical（当下就弹）。"""
    return severity >= CRITICAL_SEVERITY_MIN
