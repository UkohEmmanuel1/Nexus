# Vonk terminal assistant

## Interactive workspace (0.4.0)

Run `vonk`, `vonk chat`, or `vonk ui` in an interactive terminal to open the
full-screen workspace. Vonk uses a dark green background, lime green accents,
and amber highlights, with a wordmark, conversation area, model status, and a
persistent message composer. The terminal tab title shows `Vonk · PROJECT`.
It opens without an API key. Type a message directly, or `/` for the command
palette. Use arrow keys and Enter to choose models and providers. Credentials
are entered in a hidden field. `vonk connect` opens provider setup directly.

| Key or command | Action |
|---|---|
| Enter / Alt+Enter | Send / insert a new line |
| Ctrl+P or `/` | Command palette |
| F2 or `/models` | Select model |
| F3 or `/connect` | Connect provider |
| `/free` | Choose local or hosted free-tier options |
| `/skills` | List skills and toggle them with Enter |
| `/mcp` | View connections and MCP command instructions |
| `/new` | Clear conversation |
| PageUp / PageDown | Scroll conversation |
| Esc | Close palette or cancel request |
| Ctrl+D or `/exit` | Exit |

Add skills and MCP servers with the CLI commands below from another terminal.
The UI does not automatically start MCP programs. Windows CMD, PowerShell, and
modern ANSI terminals are supported. Piped input retains the basic menu without
full-screen control sequences; `vonk chat --plain` explicitly selects plain chat.

Free local chat uses Ollama at `127.0.0.1:11434`, without a provider key or API
bill. Install Ollama and run `ollama pull qwen2.5-coder:1.5b`, then choose the
local model in Vonk. Hardware and model downloads are required. The OpenRouter
free router is another option, with the user's own free-tier key and quotas.
Vonk does not promise anonymous free hosted inference.

```bash
vonk skills add ./SKILL.md --name review
vonk skills list
vonk skills disable review
vonk mcp add local -- python ./server.py
vonk mcp tools local --yes
vonk mcp call local tool_name '{"argument":"value"}' --yes
```

Skills are local UTF-8 instruction files (32 KB each), imported into your user
configuration and included in chat only when enabled. Enabled skills are capped
at 32 KB in total. They do not execute code.

MCP currently supports legacy 2024/2025 stdio servers, tool listing, and explicit
tool calls. Remote HTTP, OAuth, and autonomous model-selected tool calls are not
implemented in this release. Adding/listing a server does not start it. Starting
its configured executable requires a confirmation, or `--yes` for a reviewed
invocation. Provider-key environment variables are not forwarded to it. MCP
programs still run with your OS user's filesystem/network permissions; they are
not sandboxed. Use real executables (for example `node.exe server.js`) rather
than Windows `.cmd`/`.bat` wrappers. Pass `--` before server arguments that
contain option flags when adding a server with the Python client.

Plain/script chat is available with `vonk chat --plain`, or a one-shot prompt
such as `vonk chat "Explain this code"`. Both packages use the same user
configuration directory. Python uses prompt-toolkit; npm uses Node's terminal
APIs without runtime dependencies. Publishing remains paused for local testing.

Public registry installation is pending authorization/publication. The agreed
Python package is `vonk-ai-terminal`, keeping the `vonk` executable. See
[release instructions](releasing-cli.md) for the exact publishing status/checks.

Vonk offers Python and npm clients for provider-backed coding chat and read-only
file review. They do not edit files, execute commands, or download model weights.

## Install

From this repository (Python 3.10+):

```bash
python -m pip install ./packages/python-cli
vonk --version
```

Or install directly from GitHub:

```bash
python -m pip install "git+https://github.com/UkohEmmanuel1/Vonk.git#subdirectory=packages/python-cli"
```

For Node.js 20.3+ users, from the repository root:

```bash
npm install -g ./packages/npm-cli
vonk --version
```

Choose one installation for the `vonk` command. Both expose the same executable
name and share configuration. Version 0.4.0 is prepared in source; publishing to
npm or PyPI is a separate step and has not been performed here.

## Qwen, DeepSeek, and MiMo on Hugging Face

| Alias | Hugging Face model |
| --- | --- |
| `qwen` | [Qwen2.5 Coder 32B Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct) |
| `deepseek` | [DeepSeek V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) |
| `mimo` | [MiMo V2.5](https://huggingface.co/XiaomiMiMo/MiMo-V2.5) |

These are open-weight models, called through hosted Hugging Face Inference
Providers. Your HF token needs Inference Providers permission and sufficient
quota. Access, provider availability, costs, and model licenses still apply.
An open-weight model does not imply that hosted inference is free or offline.
The presets were checked against the official model cards/catalog on 2026-10-27.

```bash
vonk models huggingface
vonk login huggingface
vonk use huggingface qwen
vonk chat "Explain dependency injection in Python"
vonk use huggingface deepseek
vonk use huggingface mimo
```

Full model IDs are also accepted. One request can override the saved selection:

```bash
vonk chat --provider huggingface --model mimo --max-tokens 2048 "Review this approach"
```

## Big Pickle through OpenCode

Big Pickle is an OpenCode model ID, not a verified Hugging Face open-weight
repository. It uses a separate provider and credential:

```bash
vonk login opencode
vonk use opencode big-pickle
vonk chat "Explain this error"
```

The client uses OpenCode's documented OpenAI-compatible endpoint. OpenCode's
[current access rules](https://opencode.ai/v2/docs/console/models/) restrict
free-tier chat models to the OpenCode client; Big Pickle access from third-party
clients may therefore be rejected. Vonk does not spoof client identities or bypass
access controls. Routing is covered by automated tests; live access requires a
permitted account and has not been verified without credentials.

## Commands and credentials

```bash
vonk models                   # all providers
vonk status                   # selected model and key source, never the key
vonk use PROVIDER MODEL
vonk chat --file src/app.py "Review this file"
vonk chat --temperature 0.5 --max-tokens 1024 "Write a test example"
vonk logout huggingface       # remove one saved key
vonk logout                   # remove all saved keys
```

Interactive chat supports `/exit`, `/quit`, `/clear`, and `/help`. Both clients
handle EOF; Ctrl+C cancels with exit code 130. Successful one-shot chat exits 0,
request failures exit 1, and configuration/usage failures exit 2. Conversations
are bounded; the oldest turns are removed as the history limit is approached.

Environment variables override saved keys: `HUGGINGFACE_API_KEY` or `HF_TOKEN`
for Hugging Face, `OPENCODE_API_KEY` for OpenCode, and `PROVIDER_API_KEY` for other
providers (for example `DEEPSEEK_API_KEY`). Login requires an interactive terminal
and hides key entry in both implementations. Scripts should use environment
variables instead. Logout does not remove environment variables.

Configuration lives at `~/.config/vonk/config.json`; saved plaintext credentials
live at `~/.config/vonk/keys.json`. POSIX credential files are owner-readable only;
Windows uses the account's directory permissions. Prefer environment variables
or a credential manager on shared devices. `VONK_CONFIG_DIR` overrides the
directory, which is also useful for isolated tests.

Attach at most eight UTF-8 files totaling 60 KB. Credential files, binary files,
and unreadable files are rejected. Attached content is sent to the selected
provider; attach only files you intend to share. Original files are never changed.
Upstream errors are sanitized and terminal escape sequences are removed from
displayed responses. There are no automatic generation retries that could cause
duplicate billed requests.

## Test

```bash
python -m pytest tests/test_assistant_cli.py tests/test_providers.py -q
npm --prefix packages/npm-cli test
```

CI exercises both distributions and adds Windows/Linux CLI jobs. Automated tests
use synthetic credentials and intercepted HTTP requests; they do not prove live
model availability or generate provider charges.
