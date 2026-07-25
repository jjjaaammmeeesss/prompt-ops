"""策略适配器：三种多智能体拓扑的 MVP 实现。

每个适配器是一个独立的 process_window() 实现，共享 PerceptionAgent + ProductionAgent，
仅在决策层拓扑不同。适配器可被 evaluate_with_prompt 直接调用。
"""

from .senate import SenateAdapter
from .teacher_student import TeacherStudentAdapter
from .saga import SAGAAdapter

__all__ = ["SenateAdapter", "TeacherStudentAdapter", "SAGAAdapter"]


def get_adapter(strategy: str, llm_client, model: str = "deepseek-chat",
                harness_dir: str = "", prompt_base_dir: str = ""):
    """工厂函数：根据策略名返回对应的适配器实例。

    Args:
        strategy: "senate" | "teacher_student" | "saga"
        llm_client: OpenAI 客户端
        model: 模型名
        harness_dir: 策略 harness 目录（含 harness.md + harness.py）
        prompt_base_dir: 星灵 prompts 目录

    Returns:
        适配器实例（有 process_window 方法）
    """
    if strategy == "senate":
        return SenateAdapter(llm_client, model, harness_dir, prompt_base_dir)
    elif strategy == "teacher_student":
        return TeacherStudentAdapter(llm_client, model, harness_dir, prompt_base_dir)
    elif strategy == "saga":
        return SAGAAdapter(llm_client, model, harness_dir, prompt_base_dir)
    else:
        raise ValueError(f"未知策略: {strategy}")
