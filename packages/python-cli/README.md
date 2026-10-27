# Vonk

An interactive developer terminal with model selection, provider connections,
skills, and on-demand stdio MCP tools. Run `vonk` or `vonk chat` to open the
full-screen green-and-amber workspace. Type directly into the composer; `/` or
Ctrl+P opens commands, F2 selects models, F3 connects providers, Esc cancels a
request, and Ctrl+D exits. The terminal tab shows `Vonk · PROJECT`. No API key
is needed to browse the interface. Plain chat: `vonk chat --plain`.

Python installation: `pip install vonk-ai-terminal` after registry publication.
For the current source release, clone the repository and run
`python -m pip install ./packages/python-cli`.

Use Free models for local Ollama (no provider key or API bill) or OpenRouter's
account-based free tier. Local models need Ollama, downloaded weights and your
own hardware. Hosted premium models use your own provider key; fees and quotas
are controlled by that provider. Hugging Face includes Qwen, DeepSeek and MiMo.
Big Pickle uses OpenCode and is subject to its client/account restrictions.

Commands:

```text
vonk
vonk connect
vonk models
vonk skills add ./SKILL.md --name review
vonk skills disable review
vonk mcp add demo -- python ./server.py
vonk mcp tools demo --yes
vonk mcp call demo tool_name '{"argument":"value"}' --yes
```

Skills add user-selected instructions to model context. MCP support currently
covers legacy 2024/2025 stdio servers. Starting an MCP program or invoking a tool
requires confirmation, or explicit `--yes` authorization for scripts. Trusted
MCP programs run with your OS user permissions; this is not a sandbox.

Core chat reviews explicitly attached files and does not edit source files or
run autonomous tool loops. MCP tools are explicit user actions. API credentials
are stored only when you use login; saved local keys are plaintext. Never put
keys in source control. See the [full guide](https://github.com/UkohEmmanuel1/Vonk/blob/main/docs/terminal.md).
