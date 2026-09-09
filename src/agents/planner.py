import json
import re
from collections.abc import Callable
from typing import Any

from src.inference.engine import InferenceEngine


class TaskPlanner:
    def __init__(self, engine: InferenceEngine):
        self.engine = engine

    def decompose(self, task: str, max_steps: int = 5) -> list[dict[str, str]]:
        prompt = (
            "You are a task planner. Break down the following complex task into "
            "a sequence of smaller, executable sub-tasks. Each sub-task should be "
            "self-contained and achievable with a single tool or reasoning step.\n\n"
            f"Task: {task}\n\n"
            "Respond with a JSON list of sub-tasks, where each sub-task has:\n"
            "- 'id': step number\n"
            "- 'description': what to do\n"
            "- 'tool': the tool needed ('reason', 'code', 'search', 'calculator', or 'complete')\n"
            "- 'depends_on': list of step IDs this depends on\n\n"
            "JSON:"
        )

        response = self.engine.generate(prompt, max_new_tokens=1024, temperature=0.3)
        json_match = re.search(r"\[.*\]", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        return [{"id": 1, "description": task, "tool": "reason", "depends_on": []}]

    def create_execution_plan(self, task: str) -> dict[str, Any]:
        sub_tasks = self.decompose(task)
        return {
            "original_task": task,
            "sub_tasks": sub_tasks,
            "total_steps": len(sub_tasks),
        }


class StepExecutor:
    def __init__(self, engine: InferenceEngine, tools: dict[str, Callable]):
        self.engine = engine
        self.tools = tools

    def execute_step(self, step: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        tool = step.get("tool", "reason")
        description = step.get("description", "")

        if tool == "complete":
            return {"result": context.get("accumulated", ""), "status": "done"}

        context_str = (
            json.dumps(context.get("results", {}), indent=2)
            if context.get("results")
            else "No prior results."
        )

        prompt = (
            f"Sub-task: {description}\n"
            f"Tool to use: {tool}\n"
            f"Context from previous steps:\n{context_str}\n\n"
            f"Execute this sub-task{' using the appropriate tool' if tool != 'reason' else ''} and provide the result.",  # noqa: E501
        )

        if tool in self.tools and tool != "reason":
            response = self.engine.generate(prompt, max_new_tokens=512, temperature=0.3)
            try:
                tool_result = self.tools[tool](response)
                return {"result": tool_result, "status": "success"}
            except Exception as e:
                return {"result": f"Tool error: {e}", "status": "error"}
        else:
            response = self.engine.generate(prompt, max_new_tokens=1024, temperature=0.5)
            return {"result": response, "status": "success"}
