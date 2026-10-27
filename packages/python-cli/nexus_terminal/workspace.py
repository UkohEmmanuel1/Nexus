"""Explicitly selected skill instructions and consent-based stdio MCP tools."""

import json
import os
import pathlib
import queue
import re
import subprocess
import tempfile
import threading


def valid_name(name):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
        raise ValueError(
            "Use a name with letters, numbers, underscores or hyphens (1-64 characters)"
        )
    return name


class Store:
    def __init__(self, directory):
        self.directory = pathlib.Path(directory)
        self.path = self.directory / "extensions.json"

    def state(self):
        try:
            with self.path.open(encoding="utf-8") as handle:
                raw = handle.read(262145)
            if len(raw) > 262144:
                raise ValueError()
            data = json.loads(raw)
            if not isinstance(data, dict) or any(
                not isinstance(data.get(k, {}), dict) for k in ("skills", "mcp")
            ):
                raise ValueError()
            return {"skills": data.get("skills", {}), "mcp": data.get("mcp", {})}
        except FileNotFoundError:
            return {"skills": {}, "mcp": {}}
        except (OSError, ValueError):
            raise ValueError("Cannot read extensions.json; repair it or move it aside") from None

    def save(self, state):
        data = json.dumps(state, indent=2)
        if len(data) > 262144:
            raise ValueError("Extension configuration exceeds 256 KB")
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.directory, delete=False
            ) as handle:
                path = pathlib.Path(handle.name)
                handle.write(data)
            os.replace(path, self.path)
        finally:
            if path:
                path.unlink(missing_ok=True)

    def add_skill(self, source, name):
        valid_name(name)
        source = pathlib.Path(source)
        if not source.is_file():
            raise ValueError("Skill source must be a local UTF-8 SKILL.md file")
        with source.open("rb") as handle:
            raw = handle.read(32769)
        if len(raw) > 32768 or b"\x00" in raw:
            raise ValueError("Skill must be UTF-8 text under 32 KB")
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeError:
            raise ValueError("Skill must be UTF-8 text") from None
        if not text.strip():
            raise ValueError("Skill cannot be empty")
        state = self.state()
        if name in state["skills"]:
            raise ValueError("Skill already exists; remove it first to replace it")
        state["skills"][name] = {"enabled": True, "content": text}
        self.save(state)

    def toggle_skill(self, name, enabled):
        state = self.state()
        if valid_name(name) not in state["skills"]:
            raise ValueError("Unknown skill")
        state["skills"][name]["enabled"] = enabled
        self.save(state)

    def add_mcp(self, name, command, args):
        valid_name(name)
        if not command or not isinstance(command, str) or any(c in command for c in "\r\n\x00"):
            raise ValueError("Provide an executable name or absolute path")
        if (
            not isinstance(args, list)
            or len(args) > 64
            or any(not isinstance(a, str) or "\x00" in a or len(a) > 4096 for a in args)
        ):
            raise ValueError("MCP arguments must be at most 64 individual strings")
        if pathlib.Path(command).suffix.lower() in {".bat", ".cmd"}:
            raise ValueError(
                "Use a real executable (for example node.exe plus a script), not a batch wrapper"
            )
        state = self.state()
        if name in state["mcp"]:
            raise ValueError("MCP server already exists; remove it first to replace it")
        state["mcp"][name] = {"command": command, "args": args}
        self.save(state)

    def remove(self, kind, name):
        state = self.state()
        if valid_name(name) not in state[kind]:
            raise ValueError("Unknown extension")
        del state[kind][name]
        self.save(state)


def skill_context(directory):
    parts = []
    for name, skill in Store(directory).state()["skills"].items():
        valid_name(name)
        if not isinstance(skill, dict) or not isinstance(skill.get("content"), str):
            raise ValueError("Invalid saved skill")
        if skill.get("enabled"):
            parts.append(f"User-selected skill {name}:\n{skill['content']}")
    text = "\n\n".join(parts)
    if len(text) > 32768:
        raise ValueError("Enabled skills exceed 32 KB; disable some skills")
    return text


class MCPClient:
    """Bounded legacy MCP (2024/2025) client; never launches a shell."""

    def __init__(self, config, timeout=20):
        self.config, self.timeout = config, timeout
        self.process = None
        self.incoming = queue.Queue(maxsize=64)
        self.serial = 0

    def __enter__(self):
        command, args = self.config["command"], self.config.get("args", [])
        if pathlib.Path(command).suffix.lower() in {".cmd", ".bat"}:
            raise ValueError("MCP batch wrappers are not supported; use the underlying executable")
        env = {
            k: v
            for k, v in os.environ.items()
            if k.upper()
            in {
                "PATH",
                "PATHEXT",
                "SYSTEMROOT",
                "WINDIR",
                "TEMP",
                "TMP",
                "HOME",
                "USERPROFILE",
                "APPDATA",
                "LOCALAPPDATA",
                "LANG",
                "LC_ALL",
                "PYTHONUTF8",
            }
        }
        env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
        self.process = subprocess.Popen(
            [command, *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
            shell=False,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )

        def read():
            while True:
                raw = self.process.stdout.readline(1048577)
                if not raw or len(raw) > 1048576:
                    break
                try:
                    item = json.loads(raw)
                except (ValueError, UnicodeError):
                    item = {"invalid": True}
                try:
                    self.incoming.put(item, timeout=1)
                except queue.Full:
                    break
            try:
                self.incoming.put(None, timeout=1)
            except queue.Full:
                pass

        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()
        try:
            result = self.request(
                "initialize",
                {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": "vonk", "version": "0.3.0"},
                },
            )
            if result.get("protocolVersion") not in {
                "2024-11-05",
                "2025-03-26",
                "2025-06-18",
                "2025-11-25",
            }:
                raise ValueError(
                    "Unsupported MCP version; this release supports legacy 2024/2025 stdio servers"
                )
            self.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
            return self
        except BaseException:
            self.close()
            raise

    def send(self, message):
        raw = (json.dumps(message) + "\n").encode("utf-8")
        if len(raw) > 65536:
            raise ValueError("MCP request exceeds 64 KB")
        self.process.stdin.write(raw)
        self.process.stdin.flush()

    def request(self, method, params):
        import time

        self.serial += 1
        request_id = self.serial
        self.send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        deadline = time.monotonic() + self.timeout
        for _ in range(128):
            try:
                message = self.incoming.get(timeout=max(0.001, deadline - time.monotonic()))
            except queue.Empty:
                raise ValueError("MCP request timed out") from None
            if not isinstance(message, dict) or message.get("invalid"):
                raise ValueError("MCP server exited or emitted invalid JSON")
            if "method" in message and "id" in message:
                self.send(
                    {
                        "jsonrpc": "2.0",
                        "id": message["id"],
                        "error": {"code": -32601, "message": "Unsupported client request"},
                    }
                )
                continue
            if message.get("id") != request_id:
                continue
            if "error" in message:
                raise ValueError("MCP server rejected the request")
            result = message.get("result")
            if not isinstance(result, dict):
                raise ValueError("Invalid MCP result")
            return result
        raise ValueError("Too many MCP messages")

    def close(self):
        if self.process:
            if self.process.stdin:
                try:
                    self.process.stdin.close()
                except OSError:
                    pass
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=1)
            self.reader.join(timeout=1)
            self.process.stdout.close()

    def __exit__(self, *args):
        self.close()


def invoke_mcp(store, name, method, params, yes=False, interactive=False):
    config = store.state()["mcp"].get(valid_name(name))
    if not isinstance(config, dict):
        raise ValueError("Unknown MCP server")
    if not yes:
        if not interactive:
            raise ValueError(
                "MCP starts a local program; review it and pass --yes to authorize this invocation"
            )
        print("MCP program:", config["command"], json.dumps(config.get("args", [])))
        if method == "tools/call":
            print("Tool request:", json.dumps(params))
        if input("Run this trusted program/request? [y/N] ").strip().lower() != "y":
            raise ValueError("MCP invocation cancelled")
    with MCPClient(config) as client:
        result = client.request(method, params)
    from .ui import panel

    panel("MCP RESULT", [json.dumps(result, ensure_ascii=True)])
    return result


def extension_command(args, directory, interactive):
    store = Store(directory)
    try:
        if args.command == "skills":
            if args.action == "list":
                for name, item in store.state()["skills"].items():
                    print(f"{name}: {'enabled' if item.get('enabled') else 'disabled'}")
                if not store.state()["skills"]:
                    print("No skills. Add one with: vonk skills add PATH --name NAME")
            elif args.action == "add":
                store.add_skill(args.value or "", args.name or "")
                print("Skill added and enabled.")
            elif args.action == "remove":
                store.remove("skills", args.value)
            else:
                store.toggle_skill(args.value, args.action == "enable")
        else:
            if args.action == "list":
                for name, item in store.state()["mcp"].items():
                    print(f"{name}: {item['command']} (on demand)")
                if not store.state()["mcp"]:
                    print("No MCP servers. Add: vonk mcp add NAME EXECUTABLE ARG...")
            elif args.action == "add":
                if not args.values:
                    raise ValueError("Usage: vonk mcp add NAME EXECUTABLE ARG...")
                store.add_mcp(args.name, args.values[0], args.values[1:])
                print("MCP server saved; no program executed.")
            elif args.action == "remove":
                store.remove("mcp", args.name)
            elif args.action == "tools":
                invoke_mcp(store, args.name, "tools/list", {}, args.yes, interactive)
            else:
                if not 1 <= len(args.values) <= 2:
                    raise ValueError("Usage: vonk mcp call NAME TOOL [JSON_OBJECT] --yes")
                arguments = json.loads(args.values[1]) if len(args.values) == 2 else {}
                if not isinstance(arguments, dict):
                    raise ValueError("Tool arguments must be a JSON object")
                result = invoke_mcp(
                    store,
                    args.name,
                    "tools/call",
                    {"name": args.values[0], "arguments": arguments},
                    args.yes,
                    interactive,
                )
                if result.get("isError"):
                    return 1
        return 0
    except (ValueError, OSError) as exc:
        from .ui import clean

        print("Vonk: " + clean(str(exc)))
        return 2
