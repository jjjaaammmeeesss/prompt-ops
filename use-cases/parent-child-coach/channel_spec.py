"""快慢通道窗口规格 —— 唯一权威参数（所有版本统一标准）。

这是「亲子沟通洞见弹窗」触发层窗口逻辑的单一真源（single source of truth）。
生产 SUT、测试管线（run_v418_pipeline / run_demo）、realtime 原型一律从这里取值，
避免各版本参数漂移。

规格原文（用户确认，无歧义）：

- 慢通道：缓冲凑满 300 字才分析
- 快通道：三类词（严重/警告/机会）统一，命中当下即刻分析，向前取最多 150 字，
  同时参考前 900 字背景（含 150 分析窗口），最少 80 字，少于 80 取消；
  命中后 80 字内新关键词不再扫描（保护窗口，新增有效 <80 则忽略）
"""

# 触发层级：critical = severity ≥ 该阈值（决策1 选 B）
CRITICAL_SEVERITY_MIN = 4

# 慢通道：缓冲凑满多少字才分析
SLOW_THRESHOLD_CHARS = 300

# 快通道统一：三类词（严重/警告/机会）命中当下即刻分析，向前取最多多少字
FAST_CRITICAL_FORWARD = 150

# 快通道参考背景：触发点前多少字（含 150 分析窗口，背景 = 900 - 150 = 750）
FAST_BACKGROUND = 900

# 快通道下限 / 保护窗口：总字数少于多少取消；命中后多少字内新关键词不再扫描
FAST_MIN_CHARS = 80


def is_critical(severity: int) -> bool:
    """severity ≥ CRITICAL_SEVERITY_MIN 视为 critical（当下就弹）。"""
    return severity >= CRITICAL_SEVERITY_MIN
