import ast
import logging
import subprocess
import sys
import textwrap
from typing import Any

logger = logging.getLogger(__name__)

ALLOWED_IMPORTS = {
    'math', 'random', 'datetime', 'json', 'collections', 'itertools',
    'functools', 'statistics', 'typing', 'decimal', 'fractions', 're',
}

RESTRICTED_BUILTINS = {
    'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'bytearray', 'bytes',
    'chr', 'complex', 'dict', 'dir', 'divmod', 'enumerate', 'filter',
    'float', 'format', 'frozenset', 'getattr', 'hasattr', 'hash', 'hex',
    'id', 'int', 'isinstance', 'issubclass', 'iter', 'len', 'list', 'map',
    'max', 'min', 'next', 'object', 'oct', 'ord', 'pow', 'print', 'range',
    'repr', 'reversed', 'round', 'set', 'slice', 'sorted', 'str', 'sum',
    'super', 'tuple', 'type', 'vars', 'zip',
}


class CodeExecutor:
    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def execute(self, code: str) -> dict[str, Any]:
        result = {"stdout": "", "stderr": "", "error": None, "success": False}

        wrapped = textwrap.dedent(f"""\
import sys
import io
from typing import Any, Dict, List, Optional, Tuple, Union

_safe_builtins = {{}}
for _name in {repr(list(RESTRICTED_BUILTINS))}:
    if hasattr(__builtins__, _name):
        _safe_builtins[_name] = getattr(__builtins__, _name)

_safe_modules = {{}}
for _mod_name in {repr(list(ALLOWED_IMPORTS))}:
    try:
        _safe_modules[_mod_name] = __import__(_mod_name)
    except ImportError:
        pass

_stdout = io.StringIO()
_stderr = io.StringIO()
sys.stdout = _stdout
sys.stderr = _stderr

_globals = {{'__builtins__': _safe_builtins, **_safe_modules}}
_locals = {{}}

try:
    compiled = compile({repr(code)}, '<sandbox>', 'exec')
    exec(compiled, _globals, _locals)
    result_obj = _locals.get('result', None)
except Exception as e:
    import traceback
    traceback.print_exc(file=_stderr)

sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

print('__STDOUT_MARKER__')
print(_stdout.getvalue())
print('__STDERR_MARKER__')
print(_stderr.getvalue())
""")

        try:
            proc = subprocess.run(
                [sys.executable, "-c", wrapped],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={},
            )

            output = proc.stdout
            stdout_marker = "__STDOUT_MARKER__"
            stderr_marker = "__STDERR_MARKER__"

            if stdout_marker in output and stderr_marker in output:
                parts = output.split(stdout_marker)
                if len(parts) > 1:
                    inner = parts[1]
                    if stderr_marker in inner:
                        inner_parts = inner.split(stderr_marker)
                        result["stdout"] = inner_parts[0].strip()
                        result["stderr"] = inner_parts[1].strip()
                    else:
                        result["stdout"] = inner.strip()
                result["success"] = True

            if proc.stderr:
                result["stderr"] += "\n" + proc.stderr.strip()

        except subprocess.TimeoutExpired:
            result["error"] = f"Execution timed out after {self.timeout}s"
        except Exception as e:
            result["error"] = str(e)

        return result

    def validate_code(self, code: str) -> bool:
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        base = alias.name.split('.')[0]
                        if base not in ALLOWED_IMPORTS and not base.startswith('_'):
                            raise ValueError(f"Import not allowed: {alias.name}")
                if isinstance(node, ast.ImportFrom):
                    if node.module:
                        base = node.module.split('.')[0]
                        if base not in ALLOWED_IMPORTS and not base.startswith('_'):
                            raise ValueError(f"Import not allowed: {node.module}")
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute):
                        dangerous = {'system', 'popen', 'run', 'call', 'check_output', 'exec', 'eval', 'compile'}
                        if node.func.attr in dangerous:
                            if isinstance(node.func.value, ast.Name) and node.func.value.id in ('os', 'subprocess', 'sys'):
                                raise ValueError(f"Dangerous call: {node.func.value.id}.{node.func.attr}")
            return True
        except SyntaxError as e:
            raise ValueError(f"Syntax error: {e}")

    def __call__(self, code: str) -> dict[str, Any]:
        return self.execute(code)
