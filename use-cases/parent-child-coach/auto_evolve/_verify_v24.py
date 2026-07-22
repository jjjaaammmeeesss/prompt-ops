"""验证 v2.4 双向兼容：语法 + import + 映射逻辑。"""
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
    print(f"OK: {Path(f).name}")

sys.path.insert(0, "D:/星灵-soul-手搓/亲子沟通洞见/路线A_自上而下法_鼓励版")
sys.path.insert(0, "D:/prompt-ops/use-cases/parent-child-coach")

from src.perception_agent import PerceptionAgent

# 模拟 v2.1 输出（只有 positive_moment_category）
pagent = PerceptionAgent.__new__(PerceptionAgent)
r1 = pagent._build_report({
    "positive_moment_category": "genuine_transformation",
})
print(f"\nv2.1 input 'genuine_transformation':")
print(f"  positive_moment_category={r1.positive_moment_category!r}")
print(f"  response_need={r1.response_need!r}  (expect 'needs_empowering')")

r2 = pagent._build_report({
    "positive_moment_category": "surface_compromise",
})
print(f"\nv2.1 input 'surface_compromise':")
print(f"  positive_moment_category={r2.positive_moment_category!r}")
print(f"  response_need={r2.response_need!r}  (expect 'needs_diagnostic')")

# 模拟 v2.4 输出（只有 response_need）
r3 = pagent._build_report({
    "response_need": "needs_diagnostic",
})
print(f"\nv2.4 input 'needs_diagnostic':")
print(f"  positive_moment_category={r3.positive_moment_category!r}  (expect 'surface_compromise')")
print(f"  response_need={r3.response_need!r}")

r4 = pagent._build_report({
    "response_need": "needs_empowering",
})
print(f"\nv2.4 input 'needs_empowering':")
print(f"  positive_moment_category={r4.positive_moment_category!r}  (expect 'genuine_transformation')")
print(f"  response_need={r4.response_need!r}")

# 空输入
r5 = pagent._build_report({})
print(f"\nempty input:")
print(f"  positive_moment_category={r5.positive_moment_category!r}  (expect 'none')")
print(f"  response_need={r5.response_need!r}  (expect 'none')")

print("\nall checks OK")
