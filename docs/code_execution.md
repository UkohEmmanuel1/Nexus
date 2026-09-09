# Code Execution

Execute Python code safely in an isolated sandbox. The code executor validates, runs, and returns results from user-submitted code with strict security controls.

## Architecture

```
User Code
    │
    ▼
AST Validation (static analysis)
    │
    ▼
Subprocess Execution (sandbox)
    │
    ▼
Result Collection (stdout, stderr, error)
    │
    ▼
Return Dict
```

## Security Model

### Import Whitelist
Only the following modules can be imported:

| Module | Purpose |
|--------|---------|
| `math` | Mathematical functions |
| `random` | Random number generation |
| `datetime` | Date/time operations |
| `json` | JSON parsing |
| `collections` | Specialized data structures |
| `itertools` | Iterator tools |
| `functools` | Higher-order functions |
| `statistics` | Statistical functions |
| `typing` | Type hints |
| `decimal` | Decimal arithmetic |
| `fractions` | Rational numbers |
| `re` | Regular expressions |

### Blocked Operations
The AST validator blocks:
- `os.system`, `os.popen`, `os.exec*`
- `subprocess.run`, `subprocess.Popen`, `subprocess.call`
- `builtins.exec`, `builtins.eval`, `builtins.compile`
- `shutil.*`, `sys.exit`, `importlib.*`

### Runtime Restrictions
- **Timeout**: configurable (default 30s)
- **Subprocess isolation**: code runs in a separate Python process
- **No side effects**: filesystem and network access are blocked
- **Memory limit**: via `resource.RLIMIT_AS` on Unix

## Usage

### Python API
```python
from src.agents.code_executor import CodeExecutor

executor = CodeExecutor(timeout=30)

# Simple execution
result = executor.execute("x = 1 + 1\nprint(x)")
print(result)
# {'stdout': '2', 'stderr': '', 'error': None, 'success': True}

# Code with error
result = executor.execute("print(1/0)")
print(result['stderr'])  # Contains traceback

# Code validation
try:
    executor.validate_code("import os\nos.system('ls')")
except ValueError as e:
    print(f"Blocked: {e}")
```

### REST API
```json
POST /v1/code/execute
{
  "code": "print('Hello, World!')"
}
```

Response:
```json
{
  "stdout": "Hello, World!",
  "stderr": "",
  "error": null,
  "success": true
}
```

### Agent Integration
The code executor can be registered as a tool in the agent system:

```python
from src.agents.orchestrator import create_orchestrator
from src.agents.code_executor import CodeExecutor

executor = CodeExecutor()
orchestrator = create_orchestrator(engine, extra_tools={"execute_python": executor.execute})
result = orchestrator.run("Calculate 15! using Python")
```

## Configuration

```python
executor = CodeExecutor(
    timeout=30,      # Max execution time in seconds
)
```

## Testing

```python
# Valid code
result = executor.execute("x = [i**2 for i in range(10)]\nprint(sum(x))")
assert result['success']

# Blocked import
try:
    executor.execute("import os\nos.listdir('.')")
except ValueError:
    pass  # Blocked at validation stage

# Timeout
result = executor.execute("while True: pass")
assert result['error']  # "Execution timed out after 30s"
```

## Security Considerations

1. **The sandbox is not cryptographically secure** — it's designed to prevent accidental damage, not to withstand a determined attacker.

2. **For production deployments**, consider:
   - Running code in Docker containers
   - Using gVisor or Firecracker micro-VMs
   - Network-level isolation
   - Per-user rate limiting

3. **Logging**: All executed code is logged with timestamps for audit trails.

4. **Resource monitoring**: The executor tracks CPU and memory usage per execution.
