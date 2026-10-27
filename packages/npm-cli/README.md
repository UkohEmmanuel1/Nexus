# Vonk

Interactive developer terminal: model selection, provider connection, free local
models, paid models with your own key, skills, and stdio MCP tools.

The interactive UI fills the terminal with a green-and-amber theme, Vonk
wordmark, conversation history, and persistent composer. Type a prompt directly.
`/` or Ctrl+P opens commands; F2 selects models, F3 connects providers, Esc
cancels a request, and Ctrl+D exits. The terminal tab shows `Vonk · PROJECT`.
Plain chat remains available with `vonk chat --plain`.

After publication: `npm install -g vonk`, then run `vonk` or `vonk chat`.
For the current source candidate, clone the repository and run
`npm install -g ./packages/npm-cli`. Node.js 20.3+ is required.

The workspace opens without a key. Free local models use Ollama, installed
weights and your own hardware. Hosted free tiers require their own account/quota.
Hosted premium models use your provider key. Hugging Face includes Qwen,
DeepSeek, and MiMo; Big Pickle uses OpenCode with its access restrictions.

```text
vonk connect
vonk models
vonk skills add ./SKILL.md --name review
vonk skills disable review
vonk mcp add demo node ./server.js
vonk mcp tools demo --yes
vonk mcp call demo tool_name '{"argument":"value"}' --yes
```

Skills add instructions, not executable code. MCP currently supports legacy
2024/2025 stdio servers and requires confirmation or explicit `--yes` permission
to launch programs/call tools. MCP programs run with your OS user permissions;
this is not a sandbox. Core chat does not autonomously edit files or run tools.

Credentials are saved locally only when you use login; saved keys are plaintext.
See the [full guide](https://github.com/UkohEmmanuel1/Vonk/blob/main/docs/terminal.md).
