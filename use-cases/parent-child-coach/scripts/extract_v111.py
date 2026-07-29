"""提取 v1.11 的 System Prompt 和 User Prompt，保存为独立 txt 文件。"""
import re
from pathlib import Path

MD_PATH = Path("D:/星灵-soul-手搓/亲子沟通洞见/路线A_自上而下法_鼓励版/prompts/_candidates/prompt_A轨_v1.11_候选版.md")
OUT_DIR = Path(__file__).resolve().parent.parent  # use-cases/parent-child-coach/

content = MD_PATH.read_text(encoding="utf-8")

# System Prompt
sys_match = re.search(r'## 2\. System Prompt\n\n```\n(.*?)\n```', content, re.DOTALL)
if not sys_match:
    raise ValueError("Cannot extract System Prompt")
system_prompt = sys_match.group(1)
(OUT_DIR / "system_prompt_v1.11_sys.txt").write_text(system_prompt, encoding="utf-8")
print(f"System Prompt: {len(system_prompt)} chars -> system_prompt_v1.11_sys.txt")

# User Prompt template
user_match = re.search(r'## 3\. User Prompt\n\n````\n(.*?)\n````', content, re.DOTALL)
if not user_match:
    raise ValueError("Cannot extract User Prompt")
user_prompt = user_match.group(1)
(OUT_DIR / "system_prompt_v1.11_user.txt").write_text(user_prompt, encoding="utf-8")
print(f"User Prompt:   {len(user_prompt)} chars -> system_prompt_v1.11_user.txt")
