"""Public components loaded on demand to keep serving dependencies small."""

from importlib import import_module

_EXPORTS = {
    "CodeExecutor": ("code_executor", "CodeExecutor"),
    "FileSearch": ("file_search", "FileSearch"),
    "file_search": ("file_search", "file_search"),
    "FunctionCallingAgent": ("function_calling", "FunctionCallingAgent"),
    "AgentOrchestrator": ("orchestrator", "AgentOrchestrator"),
    "create_orchestrator": ("orchestrator", "create_orchestrator"),
    "StepExecutor": ("planner", "StepExecutor"),
    "TaskPlanner": ("planner", "TaskPlanner"),
    "ReActAgent": ("react", "ReActAgent"),
    "SearchGrounding": ("search_grounding", "SearchGrounding"),
    "search_grounding": ("search_grounding", "search_grounding"),
    "TOOL_DEFINITIONS": ("tools", "TOOL_DEFINITIONS"),
    "TOOL_REGISTRY": ("tools", "TOOL_REGISTRY"),
    "get_function_map": ("tools", "get_function_map"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(f"{__name__}.{module}"), attribute)
    globals()[name] = value
    return value
