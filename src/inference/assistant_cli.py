"""Lightweight Nexus terminal assistant (no torch dependency)."""
import argparse
import getpass
import json
import os
import pathlib
import sys
import urllib.error
import urllib.request

from src.inference.providers import PROVIDERS, EXAMPLE_MODELS

CONFIG = pathlib.Path.home() / ".config" / "nexus" / "config.json"
KEYS = pathlib.Path.home() / ".config" / "nexus" / "keys.json"
DEFAULT = {"provider": "deepseek", "model": "deepseek-chat"}


def load(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def save(path, data, secret=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if secret and os.name != "nt":
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as handle:
            json.dump(data, handle, indent=2)
        os.chmod(path, 0o600)
    else:
        path.write_text(json.dumps(data, indent=2))


def request_chat(provider, model, messages, key):
    from src.inference.providers import complete_byok
    import asyncio
    result = asyncio.run(complete_byok(
        provider=provider, model=model, messages=messages, provider_key=key
    ))
    return result["response"]


def main():
    parser = argparse.ArgumentParser(prog="nexus-ai", description="Nexus AI coding terminal")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("models", help="List supported API providers and example models")
    login = sub.add_parser("login", help="Save a provider API key on this machine")
    login.add_argument("provider", choices=sorted(PROVIDERS))
    choose = sub.add_parser("use", help="Select provider and model")
    choose.add_argument("provider", choices=sorted(PROVIDERS))
    choose.add_argument("model")
    chat = sub.add_parser("chat", help="Chat about code and your project")
    chat.add_argument("prompt", nargs="*")
    chat.add_argument("--file", action="append", default=[], help="Attach a local text file (repeatable)")
    chat.add_argument("--provider", choices=sorted(PROVIDERS))
    chat.add_argument("--model")
    sub.add_parser("logout", help="Remove locally saved provider API keys")
    args = parser.parse_args()
    if args.command == "models":
        for p in PROVIDERS:
            print(f"{p}: {', '.join(EXAMPLE_MODELS.get(p, []))} (example IDs)")
        return
    if args.command == "login":
        key = getpass.getpass(f"{args.provider} API key: ").strip()
        if not key:
            parser.error("API key cannot be empty")
        keys = load(KEYS)
        keys[args.provider] = key
        save(KEYS, keys, secret=True)
        print(f"Saved locally for {args.provider}. Treat this device as trusted.")
        return
    if args.command == "logout":
        KEYS.unlink(missing_ok=True)
        print("Saved provider credentials deleted.")
        return
    if args.command == "use":
        save(CONFIG, {"provider": args.provider, "model": args.model})
        print(f"Using {args.provider} / {args.model}")
        return
    if args.command is None:
        args = parser.parse_args(["chat"])
    if args.command != "chat":
        parser.print_help()
        return
    settings = DEFAULT | load(CONFIG)
    provider = args.provider or settings["provider"]
    model = args.model or settings["model"]
    if provider not in PROVIDERS:
        parser.error("Unsupported provider")
    key = os.environ.get(f"{provider.upper()}_API_KEY") or load(KEYS).get(provider)
    if not key:
        print(f"No key for {provider}. Run: nexus-ai login {provider}", file=sys.stderr)
        sys.exit(2)
    context = []
    for file in args.file:
        path = pathlib.Path(file).resolve()
        if not path.is_file() or path.stat().st_size > 60000:
            parser.error(f"File missing or over 60 KB: {file}")
        context.append(f"File: {path.name}\n```\n{path.read_text(errors='replace')}\n```")
    messages = [{"role": "system", "content": "You are Nexus, a practical coding assistant. Explain changes precisely; never claim to have edited files.\n" + "\n".join(context)}]
    def ask(prompt):
        messages.append({"role": "user", "content": prompt})
        try:
            answer = request_chat(provider, model, messages, key)
        except Exception as exc:
            messages.pop()
            print(f"Request failed: {type(exc).__name__}", file=sys.stderr)
            return
        print(f"\nNexus: {answer}\n")
        messages.append({"role": "assistant", "content": answer})
    if args.prompt:
        ask(" ".join(args.prompt))
        return
    print(f"Nexus AI | {provider}/{model} | /exit to quit | /clear to reset")
    while True:
        try:
            prompt = input("nexus> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if prompt in {"/exit", "/quit"}:
            break
        if prompt == "/clear":
            messages = messages[:1]
            print("Conversation cleared.")
        elif prompt:
            ask(prompt)


if __name__ == "__main__":
    main()
