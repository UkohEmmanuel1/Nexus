# Vonk terminal assistant

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
name and share configuration. Version 0.2.0 is prepared in source; publishing to
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
