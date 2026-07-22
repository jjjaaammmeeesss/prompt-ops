"""[历史实验] 语法检查 + import + 字段验证。"""
import ast
import sys
from pathlib import Path

files = [
    "D:/星灵-soul-手搓/亲子沟通洞见/路线A_自上而下法_鼓励版/src/case_memory.py",
    "D:/星灵-soul-手搓/亲子沟通洞见/路线A_自上而下法_鼓励版/src/perception_agent.py",
    "D:/星灵-soul-手搓/亲子沟通洞见/路线A_自上而下法_鼓励版/src/master_agent.py",
]
for f in files:
    ast.parse(open(f, encoding="utf-8").read())
    name = Path(f).name
    print(f"OK: {name}")

sys.path.insert(0, "D:/星灵-soul-手搓/亲子沟通洞见/路线A_自上而下法_鼓励版")
sys.path.insert(0, "D:/prompt-ops/use-cases/parent-child-coach")

from src.case_memory import PerceptionReport
from src.perception_agent import PerceptionAgent
from src.master_agent import MasterAgent
from src.multi_agent_orchestrator import MultiAgentOrchestrator

p = PerceptionReport()
print(f"PerceptionReport: positive_moment={p.positive_moment!r}, positive_moment_category={p.positive_moment_category!r}")

print("all imports OK")
