"""Meta-Optimizer: 三策略自进化对比框架。

Level 1 (内圈): harness 优化 — mutate → evaluate → keep/discard
Level 1.5 (中圈): 搜索控制 — 失败源分类 + 预算分配 + 收敛敏感度
Level 2 (外圈): 机制比较 — Explore → Critique → Specify → Generate
"""

__version__ = "0.1.0"
