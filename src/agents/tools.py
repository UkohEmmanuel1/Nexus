import datetime
import math
from typing import Any


def calculator(expression: str) -> dict[str, Any]:
    try:
        result = eval(expression, {"__builtins__": {}}, math.__dict__)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


def current_datetime(format_str: str = "%Y-%m-%d %H:%M:%S") -> dict[str, str]:
    return {"datetime": datetime.datetime.now().strftime(format_str)}


def python_repl(code: str) -> dict[str, Any]:
    try:
        local_vars = {}
        exec(code, {"__builtins__": __builtins__}, local_vars)
        return {"output": str(local_vars)}
    except Exception as e:
        return {"error": str(e)}


def search_web(query: str) -> dict[str, Any]:
    try:
        import requests
        response = requests.get(
            f"https://api.duckduckgo.com/?q={query}&format=json",
            timeout=10,
        )
        return {"results": response.json().get("AbstractText", "No results found")}
    except ImportError:
        return {"error": "requests not installed"}
    except Exception as e:
        return {"error": str(e)}


TOOL_REGISTRY: dict[str, tuple[callable, str]] = {
    "calculator": (calculator, "Evaluate a mathematical expression"),
    "current_datetime": (current_datetime, "Get the current date and time"),
    "python_repl": (python_repl, "Execute Python code and return the result"),
    "search_web": (search_web, "Search the web for information"),
}

TOOL_DEFINITIONS = [
    {
        "name": name,
        "description": desc,
        "parameters": {"type": "object", "properties": {}},
    }
    for name, (_, desc) in TOOL_REGISTRY.items()
]


def get_function_map() -> dict[str, callable]:
    return {name: func for name, (func, _) in TOOL_REGISTRY.items()}
