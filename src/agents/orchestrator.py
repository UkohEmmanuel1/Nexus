from collections.abc import Callable
from typing import Any

from src.inference.engine import InferenceEngine

from .planner import StepExecutor, TaskPlanner


class AgentOrchestrator:
    def __init__(
        self,
        engine: InferenceEngine,
        tools: dict[str, Callable],
        max_iterations: int = 10,
    ):
        self.engine = engine
        self.tools = tools
        self.planner = TaskPlanner(engine)
        self.executor = StepExecutor(engine, tools)
        self.max_iterations = max_iterations

    def run(self, task: str) -> dict[str, Any]:
        plan = self.planner.create_execution_plan(task)
        context = {"results": {}, "accumulated": ""}

        for step in plan["sub_tasks"][: self.max_iterations]:
            result = self.executor.execute_step(step, context)
            step_id = str(step.get("id", 0))
            context["results"][step_id] = {
                "description": step.get("description"),
                "result": result["result"],
                "status": result["status"],
            }
            context["accumulated"] += f"\nStep {step_id}: {result['result']}"

            if result.get("status") == "done":
                break

        final_prompt = (
            f"Original task: {task}\n\n"
            f"Results from all steps:\n{context['accumulated']}\n\n"
            "Provide a comprehensive final answer based on all the above results."
        )
        final_answer = self.engine.generate(final_prompt, max_new_tokens=1024, temperature=0.3)

        return {
            "task": task,
            "plan": plan,
            "steps": context["results"],
            "final_answer": final_answer,
        }

    def run_stream(self, task: str):
        plan = self.planner.create_execution_plan(task)
        yield {"type": "plan", "data": plan}

        context = {"results": {}, "accumulated": ""}

        for step in plan["sub_tasks"][: self.max_iterations]:
            yield {"type": "step_start", "data": step}
            result = self.executor.execute_step(step, context)
            step_id = str(step.get("id", 0))
            context["results"][step_id] = {
                "description": step.get("description"),
                "result": result["result"],
                "status": result["status"],
            }
            context["accumulated"] += f"\nStep {step_id}: {result['result']}"
            yield {"type": "step_result", "data": result}

            if result.get("status") == "done":
                break

        final_prompt = (
            f"Original task: {task}\n\n"
            f"Results from all steps:\n{context['accumulated']}\n\n"
            "Provide a comprehensive final answer."
        )
        final_answer = self.engine.generate(final_prompt, max_new_tokens=1024, temperature=0.3)
        yield {"type": "final", "data": final_answer}


def create_orchestrator(
    engine: InferenceEngine, extra_tools: dict[str, Callable] = None
) -> AgentOrchestrator:
    from .tools import get_function_map

    tools = get_function_map()
    if extra_tools:
        tools.update(extra_tools)
    return AgentOrchestrator(engine, tools)
