"""Local test-only MCP server: no network or third-party code."""

import json
import os
import sys

initialized = False
for line in sys.stdin:
    message = json.loads(line)
    method = message.get("method")
    if method == "notifications/initialized":
        initialized = True
        continue
    if "id" not in message:
        continue
    if method == "initialize":
        result = {
            "protocolVersion": "2025-11-25",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "echo-test", "version": "1"},
        }
    elif not initialized:
        result = {"isError": True, "content": [{"type": "text", "text": "not initialized"}]}
    elif method == "tools/list":
        result = {
            "tools": [
                {
                    "name": "echo",
                    "description": "Echo test input",
                    "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}},
                }
            ]
        }
    elif method == "tools/call":
        args = message.get("params", {}).get("arguments", {})
        result = {
            "content": [{"type": "text", "text": args.get("text", "hello")}],
            "credential_leaked": bool(os.environ.get("HF_TOKEN")),
        }
    else:
        result = {}
    print(json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}), flush=True)
