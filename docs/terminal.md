# Nexus AI — terminal coding assistant

Nexus now has two lightweight CLI implementations with the same main commands.
Both offer multi-provider BYOK text chat, model selection, local provider-key
configuration, and optional read-only code file context. Neither requires
GPU hardware or downloads large local weights.

## Install with pip (Python 3.10+)

```bash
pip install "git+https://github.com/UkohEmmanuel1/Nexus.git#subdirectory=packages/python-cli"
nexus-ai models
nexus-ai login deepseek
nexus-ai use deepseek deepseek-chat
nexus-ai chat
nexus-ai chat "Explain dependency injection in Python"
nexus-ai chat --file src/app.py "Review this file"
```

Note: published PyPI installation is **not yet available**. The command above
installs directly from the GitHub source.

## Install with npm (Node.js 20+)

```bash
git clone https://github.com/UkohEmmanuel1/Nexus.git
cd Nexus/packages/npm-cli
npm install -g .
nexus-ai models
nexus-ai login deepseek
nexus-ai use deepseek deepseek-chat
nexus-ai chat
nexus-ai chat --file src/app.ts "Review this file"
```

Note: published npm installation is **not yet available**. A package name in
source is not proof that the package has been published on npm.

## Configure an API key without saving

Environment variables take precedence over saved keys:
`DEEPSEEK_API_KEY`, `OPENAI_API_KEY`, `GROQ_API_KEY`,
`ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, etc.
On Linux/macOS, e.g. `export DEEPSEEK_API_KEY="..."`.
On Windows PowerShell: `$env:DEEPSEEK_API_KEY="..."`.

`nexus-ai login PROVIDER` saves a plaintext API key **on the user's
device**, at `~/.config/nexus/keys.json`. On POSIX systems the CLI attempts
to restrict file permissions to owner only. Environment variables or an OS
credential manager are preferable for shared machines. The npm CLI's login
prompt does not hide typing, so prefer environment variables there.

## Scope compared with OpenCode

This is a **coding chat MVP**, not yet an autonomous coding agent. It can
read specifically attached small text files (`--file`), maintain interactive
conversation history, and offer code suggestions. It **does not** edit source
files, run shell commands, index an entire repository, use LSP, track Git
changes, or propose/apply reviewed diffs. Those need a separate permission
and sandbox architecture.

Both clients call model providers directly with HTTPS. The Nexus hosted API
also supports `/v1/chat/byok` as a separate deployment-based entrypoint.
Models are examples, not guarantees; check your provider's actual catalog,
pricing, region and account permissions.

## Validate locally

From repository root:
```bash
python -m compileall packages/python-cli/nexus_terminal src/inference/assistant_cli.py
node --check packages/npm-cli/bin/nexus-ai.js
python -m pytest tests/test_providers.py
```
Live generation requires a valid upstream API key and available model quota.
