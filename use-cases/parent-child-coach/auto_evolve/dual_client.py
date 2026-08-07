"""双模型 Client 管理 —— 千帆（task）+ 星鸾（judge）分离。

用法:
    from auto_evolve.dual_client import init_clients, task_client, judge_client

    init_clients()
    # task_client 生成弹窗，judge_client 评分
"""

import os
from openai import OpenAI
from anthropic import Anthropic

task_client: OpenAI | None = None
task_model: str = ""
judge_client: Anthropic | None = None
judge_model: str = ""


def _load_env():
    r"""从 parent-child-coach\.env 加载环境变量。"""
    from pathlib import Path
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def init_clients() -> tuple[OpenAI, str, Anthropic, str]:
    """初始化双模型 client。返回 (task_client, task_model, judge_client, judge_model)。

    幂等：如果已初始化，直接返回已有实例。
    """
    global task_client, task_model, judge_client, judge_model

    if task_client is not None and judge_client is not None:
        return task_client, task_model, judge_client, judge_model

    _load_env()

    # Task: 百度千帆 DeepSeek-v4（OpenAI 兼容）
    task_api_key = os.environ.get("QIANFAN_API_KEY", "")
    task_base_url = os.environ.get("QIANFAN_BASE_URL", "https://qianfan.baidubce.com/v2")
    task_model = os.environ.get("QIANFAN_MODEL", "deepseek-v4-pro")

    task_client = OpenAI(
        api_key=task_api_key,
        base_url=task_base_url,
        max_retries=2,
        timeout=120,
    )

    # Judge: 星鸾 Claude Opus 4.7（Anthropic 原生 API）
    judge_api_key = os.environ.get("XINGLUAN_AUTH_TOKEN", "")
    judge_base_url = os.environ.get("XINGLUAN_BASE_URL", "https://luanapi.xingluan.cn")
    judge_model = os.environ.get("XINGLUAN_MODEL", "claude-opus-4-7")

    judge_client = Anthropic(
        api_key=judge_api_key,
        base_url=judge_base_url,
        max_retries=2,
        timeout=120,
    )

    return task_client, task_model, judge_client, judge_model


def call_judge_claude(prompt: str, max_tokens: int = 600) -> str:
    """用星鸾 Claude 做 judge。返回原始响应文本。"""
    global judge_client, judge_model
    if judge_client is None:
        init_clients()

    resp = judge_client.messages.create(
        model=judge_model,
        max_tokens=max_tokens,
        temperature=0.1,
        system="你是亲子教育领域的评估专家。只输出严格 JSON，不含其他文本。",
        messages=[{"role": "user", "content": prompt}],
    )
    # Anthropic 返回 content 列表，取第一个 text block
    for block in resp.content:
        if block.type == "text":
            return block.text
    return ""
