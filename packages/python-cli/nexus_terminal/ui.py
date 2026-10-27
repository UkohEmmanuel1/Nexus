"""Portable interactive workspace; no credentials are needed to open it."""

import pathlib
import shutil
import sys
import textwrap


def clean(text):
    return "".join(c for c in str(text) if ord(c) >= 32 and not 127 <= ord(c) <= 159)


def panel(title, lines):
    width = max(40, min(96, shutil.get_terminal_size((80, 24)).columns - 2))
    print("\n+" + "-" * (width - 2) + "+")
    for line in [title, "", *lines]:
        for part in textwrap.wrap(clean(line), width - 6, replace_whitespace=False) or [""]:
            print("| " + part.ljust(width - 4) + " |")
    print("+" + "-" * (width - 2) + "+")


def pick(title, options):
    panel(title, [f" {i + 1}. {label}" for i, label in enumerate(options)] + [" 0. Back"])
    value = input("  Select > ").strip()
    if value in {"", "0", "/back", "/exit"}:
        return None
    if value.isdigit() and 1 <= int(value) <= len(options):
        return int(value) - 1
    print("  Choose one of the numbers shown.")
    return None


def model_picker(api, config):
    entries = []
    for provider, models in api.EXAMPLE_MODELS.items():
        for model in models:
            note = "local, no API bill" if provider == "ollama" else "your API key"
            if provider == "openrouter" and model == "openrouter/free":
                note = "free tier; account/quota required"
            entries.append((provider, model, f"{provider} / {model} [{note}]"))
    selected = pick("SELECT MODEL", [e[2] for e in entries])
    if selected is not None:
        provider, model, _ = entries[selected]
        api.save(config, {"provider": provider, "model": model})
        print(f"  Selected {provider} / {model}")


def connect_provider(api, config, keys):
    providers = list(api.PROVIDERS)
    chosen = pick(
        "CONNECT A PROVIDER",
        [p + (" - local, no key" if p == "ollama" else " - enter your API key") for p in providers],
    )
    if chosen is None:
        return
    provider = providers[chosen]
    if provider != "ollama":
        if not sys.stdin.isatty():
            print(
                "  Run 'vonk login PROVIDER' in an interactive terminal, or set its key variable."
            )
            return
        if api.run(["login", provider]) != 0:
            return
    api.save(config, {"provider": provider, "model": api.EXAMPLE_MODELS[provider][0]})
    print("  Provider selected. Open Models to choose another model.")


def free_models(api, config):
    selected = pick(
        "FREE MODEL OPTIONS",
        [
            "Local Ollama - no provider API key or per-token bill; runs on your computer",
            "OpenRouter free router - own free-tier API key and rate limits apply",
        ],
    )
    if selected == 0:
        api.save(config, {"provider": "ollama", "model": "qwen2.5-coder:1.5b"})
        panel(
            "LOCAL OLLAMA SELECTED",
            [
                "Install Ollama from https://ollama.com if needed.",
                "Run: ollama pull qwen2.5-coder:1.5b",
                "Keep Ollama running, then choose Start chat. Download/hardware are required.",
            ],
        )
    elif selected == 1:
        api.save(config, {"provider": "openrouter", "model": "openrouter/free"})
        print("  Free router selected. Connect OpenRouter with your own API key.")


def skills_menu(directory):
    from .workspace import Store

    store = Store(directory)
    action = pick(
        "SKILLS",
        ["List installed skills", "Add a local SKILL.md", "Enable a skill", "Disable a skill"],
    )
    if action is None:
        return
    if action == 0:
        for name, skill in store.state()["skills"].items():
            print(f"  {name}: {'enabled' if skill['enabled'] else 'disabled'}")
        if not store.state()["skills"]:
            print("  No skills yet. Add a SKILL.md file to customize the assistant.")
    elif action == 1:
        path = input("  Path to SKILL.md > ").strip().strip('"')
        name = input("  Skill name > ").strip()
        store.add_skill(path, name)
        print("  Skill added and enabled. It will be included in your next chat.")
    else:
        store.toggle_skill(input("  Skill name > ").strip(), action == 2)


def mcp_menu(directory):
    from .workspace import Store, invoke_mcp

    store = Store(directory)
    action = pick(
        "MCP CONNECTIONS",
        ["List configured servers", "Add a stdio server", "Connect and list tools"],
    )
    if action is None:
        return
    if action == 0:
        servers = store.state()["mcp"]
        for name, item in servers.items():
            print(f"  {name}: {clean(item['command'])} (starts only when requested)")
        if not servers:
            print("  No MCP servers. Add a trusted program with its individual arguments.")
    elif action == 1:
        name = input("  Server name > ").strip()
        command = input("  Executable path/name > ").strip().strip('"')
        args = []
        print("  Enter one argument per line. Empty line finishes.")
        while value := input("  Argument > "):
            args.append(value)
        store.add_mcp(name, command, args)
        print("  Saved. No program has been executed.")
    else:
        name = input("  Server name > ").strip()
        invoke_mcp(store, name, "tools/list", {}, yes=False, interactive=True)


def dashboard(api, connect=False):
    if sys.stdin.isatty() and sys.stdout.isatty():
        from .tui import run

        return run(api, connect)
    config, keys = api.config_paths()
    if connect:
        connect_provider(api, config, keys)
    while True:
        try:
            settings = api.load(config)
            provider = settings.get("provider", "deepseek")
            model = settings.get("model", api.EXAMPLE_MODELS.get(provider, ["not selected"])[0])
            key, source = (
                api.credential(provider, keys)
                if provider in api.PROVIDERS
                else (None, "not configured")
            )
            panel(
                f"V O N K    {api.VERSION}  /  DEVELOPER WORKSPACE",
                [
                    f"Project    {pathlib.Path.cwd().name}",
                    f"Model      {provider} / {model}",
                    "Connection "
                    + (
                        "Ready - " + source
                        if key
                        else "Not connected; choose Connect or Free models"
                    ),
                    "",
                    "1  Start chat             2  Select model",
                    "3  Connect provider       4  Free models",
                    "5  Skills                 6  MCP connections",
                    "7  Help                   0  Exit",
                    "",
                    "Type a number or /models, /connect, /skills, /mcp, /help, /exit.",
                ],
            )
            command = input("  vonk > ").strip()
            if command in {"0", "/exit", "/quit", "exit", "quit"}:
                return 0
            if command in {"1", "/chat", "chat"}:
                if not key:
                    panel(
                        "CONNECT BEFORE CHATTING",
                        [
                            "Choose 3 to connect your provider, or 4 for local/free-tier options.",
                            "You can browse Vonk without any API key.",
                        ],
                    )
                else:
                    api.run(["chat", "--plain"])
            elif command in {"2", "/model", "/models"}:
                model_picker(api, config)
            elif command in {"3", "/connect"}:
                connect_provider(api, config, keys)
            elif command in {"4", "/free"}:
                free_models(api, config)
            elif command in {"5", "/skills"}:
                skills_menu(config.parent)
            elif command in {"6", "/mcp"}:
                mcp_menu(config.parent)
            elif command in {"7", "/help"}:
                panel(
                    "WORKSPACE HELP",
                    [
                        "Connect uses your own provider key. Free local models use Ollama.",
                        "Skills add instructions to chat; they do not execute code.",
                        "MCP runs trusted local server programs after confirmation.",
                        "Chat is read-only: attach files for review; no automatic source edits.",
                        "Command help: vonk --help. Plain chat: vonk chat --plain.",
                    ],
                )
            elif command:
                print("  Select Start chat to send a message, or choose a workspace command.")
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print("\n  Workspace closed.")
            return 130
        except Exception as exc:
            message = (
                str(exc)
                if isinstance(exc, (ValueError, api.CLIError))
                else "Operation failed; check your configuration and try again"
            )
            print("  " + clean(message))
            return 2
