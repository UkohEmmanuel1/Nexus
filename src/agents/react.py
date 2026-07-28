import json
import re
from collections.abc import Callable

from src.inference.engine import InferenceEngine
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ReActAgent:
    def __init__(
        self,
        engine: InferenceEngine,
        tools: dict[str, Callable],
        max_steps: int = 10,
    ):
        self.engine = engine
        self.tools = tools
        self.max_steps = max_steps

    def run(self, task: str) -> str:
        system_prompt = (
            "You are a helpful AI assistant with access to tools. "
            "You must use the following format:\n\n"
            "Thought: your reasoning about what to do\n"
            "Action: tool_name\n"
            "Action Input: json arguments\n"
            "Observation: result of the action\n"
            "... (repeat Thought/Action/Observation as needed) ...\n"
            "Thought: I now know the answer\n"
            "Final Answer: your final response\n\n"
            f"Available tools: {', '.join(self.tools.keys())}\n"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]

        for step in range(self.max_steps):
            prompt = self.engine._format_chat(messages)
            response = self.engine.generate(prompt, max_new_tokens=512, temperature=0.3)

            messages.append({"role": "assistant", "content": response})

            if "Final Answer:" in response:
                final_match = re.search(r"Final Answer:\s*(.*)", response, re.DOTALL)
                if final_match:
                    return final_match.group(1).strip()
                return response

            action_match = re.search(r"Action:\s*(\w+)", response)
            action_input_match = re.search(r"Action Input:\s*(.*?)(?:\n|$)", response, re.DOTALL)

            if action_match and action_input_match:
                action = action_match.group(1)
                action_input_str = action_input_match.group(1).strip()

                if action in self.tools:
                    try:
                        action_input = json.loads(action_input_str)
                        result = self.tools[action](**action_input) if isinstance(action_input, dict) else self.tools[action](action_input)
                    except Exception as e:
                        result = f"Error: {e}"
                else:
                    result = f"Unknown tool: {action}. Available: {list(self.tools.keys())}"

                messages.append({"role": "user", "content": f"Observation: {result}"})
            else:
                break

        return "Max steps reached without final answer."


agent = ReActAgent
