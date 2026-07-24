import json
from typing import Any, Callable, Dict, List, Optional

from src.inference.engine import InferenceEngine


class FunctionCallingAgent:
    def __init__(
        self,
        engine: InferenceEngine,
        functions: List[Dict],
        function_map: Dict[str, Callable],
    ):
        self.engine = engine
        self.functions = functions
        self.function_map = function_map

    def run(self, user_message: str, max_turns: int = 5) -> str:
        messages = [{"role": "user", "content": user_message}]
        system = self._build_system_prompt()
        messages.insert(0, {"role": "system", "content": system})

        for _ in range(max_turns):
            prompt = self.engine._format_chat(messages)
            response = self.engine.generate(prompt, max_new_tokens=512, temperature=0.3)

            if "<function_call>" not in response:
                return response

            messages.append({"role": "assistant", "content": response})

            fc_match = self._extract_function_call(response)
            if not fc_match:
                return response

            name, args = fc_match
            result = self._execute_function(name, args)
            messages.append({
                "role": "user",
                "content": f"<function_result>\n{json.dumps(result, indent=2)}\n</function_result>",
            })

        return "Max turns reached."

    def _build_system_prompt(self) -> str:
        func_descs = []
        for f in self.functions:
            func_descs.append(f"{f['name']}: {f['description']}\nParams: {json.dumps(f.get('parameters', {}))}")
        return (
            "You have access to the following functions. To call a function, "
            "respond with <function_call>name\njson_args</function_call>.\n\n"
            + "\n".join(func_descs)
        )

    def _extract_function_call(self, text: str) -> Optional[tuple[str, dict]]:
        import re
        match = re.search(r"<function_call>(.*?)\n(.*?)</function_call>", text, re.DOTALL)
        if match:
            name = match.group(1).strip()
            try:
                args = json.loads(match.group(2).strip())
                return name, args
            except json.JSONDecodeError:
                return name, {}
        return None

    def _execute_function(self, name: str, args: dict) -> Any:
        func = self.function_map.get(name)
        if func:
            try:
                return func(**args)
            except Exception as e:
                return {"error": str(e)}
        return {"error": f"Unknown function: {name}"}


agent = FunctionCallingAgent
