import { createInterface } from "node:readline";
import { basename } from "node:path";
import { Store, invokeMcp } from "./workspace.js";

function clean(value) {
    return String(value).replace(/[\x00-\x1f\x7f-\x9f]/g, "");
}
export function panel(title, lines) {
    const width = Math.max(
        40,
        Math.min(96, (process.stdout.columns || 80) - 2),
    );
    console.log("\n+" + "-".repeat(width - 2) + "+");
    for (const line of [title, "", ...lines]) {
        const text = clean(line);
        for (let i = 0; i < Math.max(1, text.length); i += width - 6)
            console.log(
                "| " + text.slice(i, i + width - 6).padEnd(width - 4) + " |",
            );
    }
    console.log("+" + "-".repeat(width - 2) + "+");
}
function reader() {
    let rl, iterator;
    return {
        close() {
            rl?.close();
            rl = undefined;
            iterator = undefined;
        },
        async read(prompt) {
            if (!rl) {
                rl = createInterface({
                    input: process.stdin,
                    output: process.stdout,
                    terminal: Boolean(process.stdin.isTTY),
                });
                iterator = rl[Symbol.asyncIterator]();
            }
            process.stdout.write(prompt);
            const result = await iterator.next();
            if (result.done) {
                const error = Error("EOF");
                error.eof = true;
                throw error;
            }
            return result.value.trim();
        },
    };
}
async function pick(input, title, options) {
    panel(title, [...options.map((x, i) => ` ${i + 1}. ${x}`), " 0. Back"]);
    const value = await input.read("  Select > ");
    if (!value || ["0", "/back", "/exit"].includes(value)) return null;
    const n = Number(value);
    return Number.isInteger(n) && n >= 1 && n <= options.length ? n - 1 : null;
}
export async function extensionCommand(kind, args, directory, api) {
    const [action = "list", value, ...rest] = args,
        store = new Store(directory);
    try {
        if (kind === "skills") {
            if (action === "list") {
                const state = await store.state();
                for (const [name, item] of Object.entries(state.skills))
                    console.log(
                        clean(name) +
                            ": " +
                            (item.enabled ? "enabled" : "disabled"),
                    );
                if (!Object.keys(state.skills).length)
                    console.log(
                        "No skills. Add: vonk skills add PATH --name NAME",
                    );
            } else if (action === "add") {
                if (rest[0] !== "--name" || rest.length !== 2)
                    throw Error("Usage: vonk skills add PATH --name NAME");
                await store.addSkill(value, rest[1]);
                console.log("Skill added and enabled.");
            } else if (["enable", "disable"].includes(action))
                await store.toggleSkill(value, action === "enable");
            else if (action === "remove") await store.remove("skills", value);
            else
                throw Error(
                    "Skills actions: list, add, enable, disable, remove",
                );
        } else {
            if (action === "list") {
                const state = await store.state();
                for (const [name, item] of Object.entries(state.mcp))
                    console.log(
                        clean(name + ": " + item.command) + " (on demand)",
                    );
                if (!Object.keys(state.mcp).length)
                    console.log(
                        "No MCP servers. Add: vonk mcp add NAME EXECUTABLE ARG...",
                    );
            } else if (action === "add") {
                const values = rest[0] === "--" ? rest.slice(1) : rest;
                await store.addMcp(value, values[0], values.slice(1));
                console.log("MCP server saved; no program executed.");
            } else if (action === "remove") await store.remove("mcp", value);
            else if (["tools", "call"].includes(action)) {
                const yes = rest.includes("--yes"),
                    values = rest.filter((x) => x !== "--yes");
                if (
                    action === "call" &&
                    (values.length < 1 || values.length > 2)
                )
                    throw Error(
                        "Usage: vonk mcp call NAME TOOL [JSON_OBJECT] --yes",
                    );
                const arguments_ =
                    action === "call" && values[1] ? JSON.parse(values[1]) : {};
                if (
                    !arguments_ ||
                    typeof arguments_ !== "object" ||
                    Array.isArray(arguments_)
                )
                    throw Error("Tool arguments must be a JSON object");
                const input = reader();
                try {
                    const result = await invokeMcp(
                        store,
                        value,
                        action === "call" ? "tools/call" : "tools/list",
                        action === "call"
                            ? { name: values[0], arguments: arguments_ }
                            : {},
                        async (config, method, params) => {
                            if (yes) return true;
                            if (!process.stdin.isTTY) return false;
                            console.log(
                                clean(
                                    "MCP program: " +
                                        config.command +
                                        " " +
                                        JSON.stringify(config.args),
                                ),
                            );
                            if (method === "tools/call")
                                console.log(
                                    clean(
                                        "Tool request: " +
                                            JSON.stringify(params),
                                    ),
                                );
                            return (
                                (
                                    await input.read(
                                        "Run this trusted program/request? [y/N] ",
                                    )
                                ).toLowerCase() === "y"
                            );
                        },
                    );
                    console.log(
                        api.terminalText(JSON.stringify(result, null, 2)),
                    );
                    if (result.isError) return 1;
                } finally {
                    input.close();
                }
            } else throw Error("MCP actions: list, add, remove, tools, call");
        }
        return 0;
    } catch (error) {
        console.error("Vonk: " + clean(error.message));
        return 2;
    }
}
export async function dashboard(api, directory, connect = false) {
    if (process.stdin.isTTY && process.stdout.isTTY) {
        const { terminalWorkspace } = await import("./tui.js");
        return await terminalWorkspace(api, directory, connect);
    }
    const input = reader(),
        config = api.config,
        keys = api.keysPath;
    async function providerPicker() {
        const providers = Object.keys(api.endpoints);
        const n = await pick(
            input,
            "CONNECT A PROVIDER",
            providers.map(
                (x) =>
                    x +
                    (x === "ollama"
                        ? " - local, no key"
                        : " - enter your API key"),
            ),
        );
        if (n === null) return;
        const provider = providers[n];
        if (provider !== "ollama") {
            if (!process.stdin.isTTY) {
                console.log(
                    "Run vonk login PROVIDER in an interactive terminal or set its key variable.",
                );
                return;
            }
            input.close();
            if ((await api.main(["login", provider])) !== 0) return;
        }
        await api.save(config, { provider, model: api.examples[provider] });
        console.log("Provider selected. Open Models to choose another model.");
    }
    try {
        if (connect) await providerPicker();
        while (true) {
            const settings = await api.load(config),
                provider = settings.provider || "deepseek",
                model =
                    settings.model || api.examples[provider] || "not selected";
            const [key, source] = Object.hasOwn(api.endpoints, provider)
                ? await api.credential(provider, keys)
                : [null, "not configured"];
            panel(`V O N K    ${api.VERSION}  /  DEVELOPER WORKSPACE`, [
                "Project    " + basename(process.cwd()),
                "Model      " + provider + " / " + model,
                "Connection " +
                    (key
                        ? "Ready - " + source
                        : "Not connected - choose Connect or Free models"),
                "",
                "1  Start chat             2  Select model",
                "3  Connect provider       4  Free models",
                "5  Skills                 6  MCP connections",
                "7  Help                   0  Exit",
                "",
                "Type a number or /models, /connect, /skills, /mcp, /help, /exit.",
            ]);
            const command = await input.read("  vonk > ");
            if (["0", "/exit", "/quit", "exit", "quit"].includes(command))
                return 0;
            try {
                if (["1", "/chat", "chat"].includes(command)) {
                    if (!key)
                        panel("CONNECT BEFORE CHATTING", [
                            "Choose 3 to connect your provider, or 4 for local/free-tier options.",
                            "You can browse Vonk without any API key.",
                        ]);
                    else {
                        input.close();
                        await api.main(["chat", "--plain"]);
                    }
                } else if (["2", "/model", "/models"].includes(command)) {
                    const entries = [];
                    for (const [p, example] of Object.entries(api.examples)) {
                        const models =
                            p === "huggingface"
                                ? Object.values(api.hfModels)
                                : [example];
                        for (const m of models) entries.push([p, m]);
                    }
                    const n = await pick(
                        input,
                        "SELECT MODEL",
                        entries.map(([p, m]) => `${p} / ${m}`),
                    );
                    if (n !== null)
                        await api.save(config, {
                            provider: entries[n][0],
                            model: entries[n][1],
                        });
                } else if (["3", "/connect"].includes(command))
                    await providerPicker();
                else if (["4", "/free"].includes(command)) {
                    const n = await pick(input, "FREE MODEL OPTIONS", [
                        "Local Ollama - no API key or per-token bill; uses your hardware",
                        "OpenRouter free router - own free-tier API key and rate limits",
                    ]);
                    if (n === 0) {
                        await api.save(config, {
                            provider: "ollama",
                            model: "qwen2.5-coder:1.5b",
                        });
                        panel("LOCAL OLLAMA SELECTED", [
                            "Install Ollama from https://ollama.com if needed.",
                            "Run: ollama pull qwen2.5-coder:1.5b",
                            "Keep Ollama running, then choose Start chat. Download/hardware required.",
                        ]);
                    } else if (n === 1) {
                        await api.save(config, {
                            provider: "openrouter",
                            model: "openrouter/free",
                        });
                        console.log(
                            "Connect OpenRouter with your own free-tier key.",
                        );
                    }
                } else if (["5", "/skills"].includes(command)) {
                    const store = new Store(directory);
                    const n = await pick(input, "SKILLS", [
                        "List installed skills",
                        "Add a local SKILL.md",
                        "Enable a skill",
                        "Disable a skill",
                    ]);
                    if (n === 0)
                        await extensionCommand(
                            "skills",
                            ["list"],
                            directory,
                            api,
                        );
                    else if (n === 1) {
                        const path = (
                            await input.read("  Path to SKILL.md > ")
                        ).replace(/^"|"$/g, "");
                        const name = await input.read("  Skill name > ");
                        await store.addSkill(path, name);
                        console.log("Skill added and enabled.");
                    } else if (n !== null)
                        await store.toggleSkill(
                            await input.read("  Skill name > "),
                            n === 2,
                        );
                } else if (["6", "/mcp"].includes(command)) {
                    const n = await pick(input, "MCP CONNECTIONS", [
                        "List configured servers",
                        "Add a stdio server",
                        "Connect and list tools",
                    ]);
                    if (n === 0)
                        await extensionCommand("mcp", ["list"], directory, api);
                    else if (n === 1) {
                        const name = await input.read("  Server name > "),
                            executable = (
                                await input.read("  Executable path/name > ")
                            ).replace(/^"|"$/g, "");
                        const args = [];
                        console.log(
                            "Enter one argument per line. Empty line finishes.",
                        );
                        while (true) {
                            const arg = await input.read("  Argument > ");
                            if (!arg) break;
                            args.push(arg);
                        }
                        await new Store(directory).addMcp(
                            name,
                            executable,
                            args,
                        );
                        console.log("Saved. No program has been executed.");
                    } else if (n === 2) {
                        const name = await input.read("  Server name > ");
                        input.close();
                        await extensionCommand(
                            "mcp",
                            ["tools", name],
                            directory,
                            api,
                        );
                    }
                } else if (["7", "/help"].includes(command))
                    panel("WORKSPACE HELP", [
                        "Connect uses your provider key. Local/free-tier options are under Free models.",
                        "Skills add instructions; they do not execute code. MCP runs trusted local programs after confirmation.",
                        "Chat is read-only. Attach files for review. Command help: vonk --help. Plain chat: vonk chat --plain.",
                    ]);
                else if (command)
                    console.log(
                        "Select Start chat to send a message, or choose a workspace command.",
                    );
            } catch (error) {
                if (error.eof) return 0;
                console.error("Vonk: " + clean(error.message));
            }
        }
    } catch (error) {
        if (error.eof) return 0;
        console.error("Vonk: " + clean(error.message));
        return 2;
    } finally {
        input.close();
    }
}
