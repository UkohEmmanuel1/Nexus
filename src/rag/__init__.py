"""Public components loaded on demand to keep serving dependencies small."""

from importlib import import_module

_EXPORTS = {
    "RAGGenerator": ("generator", "RAGGenerator"),
    "Retriever": ("retriever", "Retriever"),
    "WebIndex": ("web_index", "WebIndex"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(f"{__name__}.{module}"), attribute)
    globals()[name] = value
    return value
