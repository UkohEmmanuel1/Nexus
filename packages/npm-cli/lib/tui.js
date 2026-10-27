import { emitKeypressEvents } from "node:readline";
import { basename } from "node:path";
import { Store, skillContext } from "./workspace.js";

export const colors = {
    background: "#101713", surface: "#19231c", text: "#e2eadf",
    muted: "#8b9c8e", green: "#9de275", amber: "#efbd62",
};
export const commands = [
    ["/models", "Select a model"], ["/connect", "Connect a provider"],
    ["/free", "Set up a free model"], ["/skills", "Manage skill instructions"],
    ["/mcp", "View MCP connections"], ["/new", "Start a new conversation"],
    ["/help", "Keyboard shortcuts and commands"], ["/exit", "Close Vonk"],
];
const wordmark = [
    "██╗   ██╗ ██████╗ ███╗   ██╗██╗  ██╗",
    "██║   ██║██╔═══██╗████╗  ██║██║ ██╔╝",
    "██║   ██║██║   ██║██╔██╗ ██║█████╔╝ ",
    "╚██╗ ██╔╝██║   ██║██║╚██╗██║██╔═██╗",
    " ╚████╔╝ ╚██████╔╝██║ ╚████║██║  ██╗",
    "  ╚═══╝   ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝",
];
function safe(value) {
    return String(value).replace(/\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))/g, "")
        .replace(/[\x00-\x08\x0b-\x1f\x7f-\x9f]/g, "");
}
function ansi(hex, bg = false) {
    const n = Number.parseInt(hex.slice(1), 16);
    return `\x1b[${bg ? 48 : 38};2;${n >> 16};${(n >> 8) & 255};${n & 255}m`;
}
function wrap(text, width) {
    return safe(text).split("\n").flatMap(line => {
        const chars = Array.from(line.replaceAll("\t", "    "));
        if (!chars.length) return [""];
        const result = [];
        for (let i = 0; i < chars.length; i += width) result.push(chars.slice(i, i + width).join(""));
        return result;
    });
}
export function renderFrame(state, columns = 80, rows = 24) {
    const width = Math.max(1, columns - 1), height = Math.max(1, rows);
    const lines = [];
    const line = (text = "", color = "text", surface = false) => {
        const clipped = Array.from(safe(text).replaceAll("\n", " ")).slice(0, width).join("");
        lines.push(ansi(surface ? colors.surface : colors.background, true) + ansi(colors[color])
            + clipped.padEnd(width));
    };
    line(`  V O N K   ${state.version}   /   ${state.project}`, "green");
    line();
    const options = state.options || [];
    const overlay = state.mode === "key" ? 3 : state.mode ? Math.min(9, options.length + 2) : 2;
    const room = Math.max(0, height - overlay - 7);
    const hero = state.home && !state.mode && width >= 42 && room >= 13;
    if (hero) {
        for (let i = 0; i < wordmark.length; i++)
            line(wordmark[i].padStart(Math.floor((width + wordmark[i].length) / 2)), i < 3 ? "green" : "amber");
        line();
    }
    const contentRoom = Math.max(0, room - (hero ? 7 : 0));
    const content = wrap(state.transcript || "", Math.max(1, width - 4));
    const end = Math.max(0, content.length - (state.scroll || 0));
    const visible = content.slice(Math.max(0, end - contentRoom), end);
    for (const text of visible) line("  " + text);
    for (let i = visible.length; i < contentRoom; i++) line();
    if (state.mode === "key") {
        line("  CONNECT PROVIDER · API key is hidden · Esc cancels", "amber");
        line("  " + state.notice, "muted");
        line();
    } else if (state.mode) {
        line(`  ${state.mode.toUpperCase()}   Up/Down select · Enter confirm · Esc close`, "amber");
        const start = Math.max(0, state.index - 5);
        for (let i = start; i < Math.min(start + 7, options.length); i++)
            line((i === state.index ? "  > " : "    ") + options[i][1],
                i === state.index ? "green" : "muted", i === state.index);
        if (!options.length) line("  No matching command. Esc returns to chat.", "muted");
    } else {
        line("  " + state.notice, "muted");
        line();
    }
    while (lines.length < height - 5) line();
    line(`  ${state.provider} / ${state.model}   ${state.busy ? "thinking..." : state.connected ? "connected" : "/connect to add your key"}`, "amber");
    const text = state.mode === "key" ? "*".repeat(Array.from(state.draft).length) : state.draft;
    const draftLines = wrap(text, Math.max(1, width - 6)).slice(-3);
    for (let i = 0; i < 3; i++) line((i === 0 ? "  > " : "    ") + (draftLines[i] || ""), "text", true);
    line("  Enter send  Alt+Enter newline  / commands  F2 models  Esc cancel  Ctrl+D exit", "muted");
    const inputWidth = Math.max(1, width - 6);
    const prefix = Array.from(text).slice(0, state.cursor ?? Array.from(text).length).join("");
    const prefixLines = wrap(prefix, inputWidth);
    const allLines = wrap(text, inputWidth);
    const row = Math.min(height - 1, height - 3 + prefixLines.length - 1 - Math.max(0, allLines.length - 3));
    const column = Math.min(width, 5 + Array.from(prefixLines.at(-1)).length);
    return "\x1b[H" + lines.slice(0, height).join("\r\n") + "\x1b[0m"
        + `\x1b[${Math.max(1, row)};${column}H\x1b[?25h`;
}

export class TerminalSession {
    constructor(api, directory) {
        this.api = api;
        this.store = new Store(directory);
        this.directory = directory;
        this.messages = [];
        this.state = {
            version: api.VERSION, project: basename(process.cwd()),
            provider: "deepseek", model: "deepseek-chat", connected: false,
            draft: "", cursor: 0, mode: null, index: 0, options: [], scroll: 0,
            notice: "Type a message to begin. / opens commands.", busy: false, home: true,
            transcript: "\nVonk\n\nYour project. Your models. Your flow.\n\n"
                + "Write a prompt below, or open /models and /connect.\n"
                + "Free local chat: /free. Skills: /skills. Connections: /mcp.\n",
        };
        this.draw = () => {};
        this.finish = () => {};
    }
    async refresh() {
        const settings = await this.api.load(this.api.config);
        const p = settings.provider || "deepseek";
        if (!Object.hasOwn(this.api.endpoints, p)) throw Error("Unsupported provider. Run vonk use PROVIDER MODEL.");
        this.state.provider = p;
        this.state.model = settings.model || this.api.examples[p];
        [this.key] = await this.api.credential(p, this.api.keysPath);
        this.state.connected = Boolean(this.key);
    }
    append(role, text) {
        this.state.home = false;
        this.state.transcript = (this.state.transcript + `\n${role}\n\n${safe(text)}\n`).slice(-120000);
        this.state.scroll = 0;
    }
    palette(mode, options) {
        this.state.mode = mode;
        this.state.options = options;
        this.state.index = 0;
    }
    async select(provider, model) {
        await this.api.save(this.api.config, { provider, model });
        await this.refresh();
        this.messages = [];
        this.state.notice = "Model changed. A new conversation starts with your next prompt.";
    }
    async command(cmd) {
        if (this.state.busy) return;
        this.state.mode = null;
        this.state.draft = "";
        if (["/exit", "/quit"].includes(cmd)) this.finish();
        else if (["/models", "/model"].includes(cmd)) {
            const entries = [];
            for (const [p, example] of Object.entries(this.api.examples))
                for (const m of p === "huggingface" ? Object.values(this.api.hfModels) : [example])
                    entries.push([[p, m], `${p.padEnd(13)} ${m}`]);
            this.palette("models", entries);
        } else if (cmd === "/connect")
            this.palette("connect", Object.keys(this.api.endpoints).map(p => [p, p + (p === "ollama" ? " · local, no key" : " · your API key")]));
        else if (cmd === "/free")
            this.palette("free", [
                [["ollama", "qwen2.5-coder:1.5b"], "Ollama · local / no API bill"],
                [["openrouter", "openrouter/free"], "OpenRouter · free tier / own key"],
            ]);
        else if (["/new", "/clear"].includes(cmd)) {
            this.messages = [];
            this.state.transcript = "";
            this.append("Vonk", "New conversation. What are we working on?");
            this.state.home = true;
        } else if (cmd === "/skills") {
            const skills = Object.entries((await this.store.state()).skills);
            if (skills.length)
                this.palette("skills", skills.map(([n, s]) => [[n, s.enabled], `${n} · ${s.enabled ? "enabled" : "disabled"}`]));
            else this.append("Skills", "No skills installed. Add one in another terminal:\nvonk skills add PATH/TO/SKILL.md --name review\nThen /skills lets you enable or disable it here.");
        } else if (cmd === "/mcp") {
            const entries = Object.entries((await this.store.state()).mcp);
            this.append("MCP connections", entries.map(([n, s]) => `${n} · ${s.command} · on demand`).join("\n") || "No MCP connections configured.");
            this.append("MCP commands", "vonk mcp add NAME EXECUTABLE ARG...\nvonk mcp tools NAME\nvonk mcp call NAME TOOL [JSON]\nRun these in another terminal; each invocation needs your permission.");
        } else this.append("Commands", commands.map(([cmd, label]) => `${cmd.padEnd(12)} ${label}`).join("\n")
            + "\n\nEnter sends · Alt+Enter adds a line · PageUp/PageDown scroll\nCtrl+P commands · F2 models · F3 connect · Esc cancel · Ctrl+D exit\nChat reviews and explains code; it does not automatically edit files.");
        this.draw();
    }
    async choose() {
        const { mode, options, index } = this.state;
        if (!options.length) return await this.command(this.state.draft);
        const value = options[index][0];
        this.state.mode = null;
        this.state.draft = "";
        if (mode === "commands") await this.command(value);
        else if (mode === "models" || mode === "free") {
            await this.select(...value);
            if (mode === "free") this.append("Free models", "Local: install Ollama and run `ollama pull qwen2.5-coder:1.5b`. Keep Ollama running. OpenRouter requires your own free-tier key and quotas.");
        } else if (mode === "connect") {
            if (value === "ollama") await this.select(value, this.api.examples[value]);
            else {
                this.loginProvider = value;
                this.palette("key", []);
                this.state.notice = `Connect ${value}: enter your API key. Saved locally; environment keys take precedence.`;
            }
        } else if (mode === "skills") {
            await this.store.toggleSkill(value[0], !value[1]);
            await this.command("/skills");
        }
        this.draw();
    }
    async login() {
        const key = this.state.draft.trim();
        if (!/^[\x21-\x7e]{1,512}$/.test(key)) {
            this.state.draft = "";
            this.state.notice = "API key must be 1-512 printable characters without spaces.";
            return;
        }
        const keys = await this.api.load(this.api.keysPath);
        keys[this.loginProvider] = key;
        await this.api.save(this.api.keysPath, keys, true);
        await this.select(this.loginProvider, this.api.examples[this.loginProvider]);
        this.state.mode = null;
        this.state.draft = "";
        this.state.notice = "Provider key saved locally. Ready to chat.";
    }
    async send() {
        const prompt = this.state.draft.trim();
        if (!prompt || this.state.busy) return;
        if (prompt.startsWith("/")) return await this.command(prompt);
        if (prompt.length > 16000) { this.state.notice = "Prompt exceeds 16,000 characters."; return; }
        if (!this.key) { this.state.notice = "Connect with /connect, or choose a local model with /free."; return; }
        this.state.draft = "";
        this.state.busy = true;
        this.append("You", prompt);
        this.cancellation = new AbortController();
        this.draw();
        try {
            if (!this.messages.length) this.messages = [{ role: "system", content: this.api.systemPrompt + "\n" + await skillContext(this.directory) }];
            const turns = [...this.messages, { role: "user", content: prompt }];
            while (turns.length > 99 || turns.reduce((n, m) => n + m.content.length, 0) > 120000) turns.splice(1, 2);
            const answer = await this.api.complete(this.state.provider, this.state.model, turns, this.key, 2048, 0.7, this.cancellation.signal);
            if (this.cancellation.signal.aborted) return;
            this.messages = [...turns, { role: "assistant", content: answer }];
            this.append("Vonk", answer);
            this.state.notice = "Ready. /new starts fresh.";
        } catch (error) {
            this.append("Connection", this.cancellation.signal.aborted ? "Request cancelled." : "Request failed. Check your connection and provider settings.");
        } finally {
            this.state.busy = false;
            this.draw();
        }
    }
    async keypress(text, key = {}) {
        const s = this.state;
        const chars = Array.from(s.draft);
        const insert = value => {
            const content = Array.from(value);
            chars.splice(s.cursor, 0, ...content);
            s.cursor += content.length;
            s.draft = chars.join("").slice(0, s.mode === "key" ? 512 : 16001);
        };
        if (key.ctrl && key.name === "d") { this.cancellation?.abort(); this.finish(); return; }
        if (key.name === "escape" || (key.ctrl && key.name === "c")) {
            if (s.busy) this.cancellation?.abort();
            else if (!s.draft && !s.mode && key.ctrl) this.finish();
            s.mode = null; s.draft = "";
        } else if (key.ctrl && key.name === "p" && !s.busy) { s.draft = "/"; s.cursor = 1; s.mode = "commands"; }
        else if (key.name === "f2") await this.command("/models");
        else if (key.name === "f3") await this.command("/connect");
        else if (key.name === "pageup") s.scroll += 8;
        else if (key.name === "pagedown") s.scroll = Math.max(0, s.scroll - 8);
        else if (["up", "down"].includes(key.name) && s.mode) {
            s.index = Math.max(0, Math.min(s.options.length - 1, s.index + (key.name === "up" ? -1 : 1)));
        } else if (key.name === "return" || key.name === "enter") {
            if (key.meta && !s.mode) insert("\n");
            else if (s.mode === "key") await this.login();
            else if (s.mode) await this.choose();
            else void this.send().catch(() => { s.busy = false; this.draw(); });
        } else if (key.name === "left") s.cursor = Math.max(0, s.cursor - 1);
        else if (key.name === "right") s.cursor = Math.min(chars.length, s.cursor + 1);
        else if (key.name === "home" || (key.ctrl && key.name === "a")) s.cursor = 0;
        else if (key.name === "end" || (key.ctrl && key.name === "e")) s.cursor = chars.length;
        else if (key.name === "backspace") {
            if (s.cursor > 0) chars.splice(--s.cursor, 1);
            s.draft = chars.join("");
        } else if (key.name === "delete") {
            chars.splice(s.cursor, 1);
            s.draft = chars.join("");
        }
        else if (text && !key.ctrl && !key.meta && !["up", "down", "left", "right", "home", "end", "delete", "tab"].includes(key.name))
            insert(safe(text));
        s.cursor = Math.min(s.cursor, Array.from(s.draft).length);
        if (s.mode === "commands" && !s.draft.startsWith("/")) s.mode = null;
        if (!s.mode && s.draft.startsWith("/") && !s.busy) s.mode = "commands";
        if (s.mode === "commands") {
            s.options = commands.filter(([cmd]) => cmd.startsWith(s.draft.trim())).map(([cmd, label]) => [cmd, `${cmd.padEnd(12)} ${label}`]);
            s.index = Math.max(0, Math.min(s.index, s.options.length - 1));
        }
        this.draw();
    }
}

export async function terminalWorkspace(api, directory, connect = false) {
    const session = new TerminalSession(api, directory);
    await session.refresh();
    if (connect) await session.command("/connect");
    const input = process.stdin, output = process.stdout;
    const wasRaw = input.isRaw;
    emitKeypressEvents(input);
    let closed = false;
    session.draw = () => {
        if (!closed) output.write(renderFrame(session.state, output.columns, output.rows));
    };
    const resize = () => session.draw();
    const onKey = (text, key) => {
        void session.keypress(text, key).catch(() => {
            session.state.notice = "Operation failed. Check configuration and try again.";
            session.draw();
        });
    };
    try {
        output.write("\x1b[22;0t\x1b]0;Vonk · " + safe(session.state.project) + "\x07\x1b[?1049h\x1b[?25l");
        input.setRawMode(true);
        input.resume();
        input.on("keypress", onKey);
        output.on("resize", resize);
        return await new Promise(resolve => {
            session.finish = () => { closed = true; session.cancellation?.abort(); resolve(0); };
            session.draw();
        });
    } finally {
        closed = true;
        input.removeListener("keypress", onKey);
        output.removeListener("resize", resize);
        input.setRawMode(Boolean(wasRaw));
        input.pause();
        output.write("\x1b[0m\x1b[?25h\x1b[?1049l\x1b[23;0t");
    }
}
