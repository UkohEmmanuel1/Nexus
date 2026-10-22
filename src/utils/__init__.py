"""Public components loaded on demand to keep serving dependencies small."""

from importlib import import_module

_EXPORTS = {
    "load_checkpoint": ("checkpoint", "load_checkpoint"),
    "save_checkpoint": ("checkpoint", "save_checkpoint"),
    "KeyManager": ("key_manager", "KeyManager"),
    "get_logger": ("logging", "get_logger"),
    "setup_logging": ("logging", "setup_logging"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(f"{__name__}.{module}"), attribute)
    globals()[name] = value
    return value
