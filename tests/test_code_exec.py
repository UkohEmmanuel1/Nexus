import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.code_executor import CodeExecutor
import time

start = time.time()
executor = CodeExecutor(timeout=10)
res = executor.execute("print(1+1)")
elapsed = time.time() - start

print(f"Executed in {elapsed:.1f}s")
print(f"Success: {res['success']}")
print(f"Stdout: {res['stdout']!r}")
print(f"Error: {res.get('error')!r}")

assert res["success"], f"Failed: {res}"
assert "2" in res["stdout"], f"Expected 2, got {res['stdout']}"
print("Code executor works!")
