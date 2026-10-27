import {
    readFile,
    writeFile,
    mkdir,
    rename,
    unlink,
    stat,
} from "node:fs/promises";
import { join, extname } from "node:path";
import { randomUUID } from "node:crypto";
import { spawn } from "node:child_process";

export function validName(name) {
    if (
        typeof name !== "string" ||
        !/^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(name)
    )
        throw Error(
            "Use a name with letters, numbers, underscores or hyphens (1-64 characters)",
        );
    return name;
}
export class Store {
    constructor(directory) {
        this.directory = directory;
        this.path = join(directory, "extensions.json");
    }
    async state() {
        try {
            const raw = await readFile(this.path, "utf8");
            if (raw.length > 262144) throw Error();
            const data = JSON.parse(raw);
            if (!data || typeof data !== "object" || Array.isArray(data))
                throw Error();
            for (const name of ["skills", "mcp"])
                if (
                    data[name] &&
                    (typeof data[name] !== "object" ||
                        Array.isArray(data[name]))
                )
                    throw Error();
            return { skills: data.skills || {}, mcp: data.mcp || {} };
        } catch (error) {
            if (error.code === "ENOENT") return { skills: {}, mcp: {} };
            throw Error(
                "Cannot read extensions.json; repair it or move it aside",
            );
        }
    }
    async save(state) {
        const raw = JSON.stringify(state, null, 2);
        if (raw.length > 262144)
            throw Error("Extension configuration exceeds 256 KB");
        await mkdir(this.directory, { recursive: true, mode: 0o700 });
        const temporary = join(this.directory, ".extensions-" + randomUUID());
        try {
            await writeFile(temporary, raw, { mode: 0o600, flag: "wx" });
            await rename(temporary, this.path);
        } finally {
            await unlink(temporary).catch((error) => {
                if (error.code !== "ENOENT") throw error;
            });
        }
    }
    async addSkill(source, name) {
        validName(name);
        const metadata = await stat(source);
        if (!metadata.isFile() || metadata.size > 32768)
            throw Error("Skill must be a local text file under 32 KB");
        const raw = await readFile(source);
        if (raw.length > 32768 || raw.includes(0))
            throw Error("Skill must be UTF-8 text under 32 KB");
        let content;
        try {
            content = new TextDecoder("utf-8", { fatal: true }).decode(raw);
        } catch {
            throw Error("Skill must be UTF-8 text");
        }
        if (!content.trim()) throw Error("Skill cannot be empty");
        const state = await this.state();
        if (Object.hasOwn(state.skills, name))
            throw Error("Skill already exists; remove it first to replace it");
        Object.defineProperty(state.skills, name, {
            value: { enabled: true, content },
            enumerable: true,
            configurable: true,
            writable: true,
        });
        await this.save(state);
    }
    async toggleSkill(name, enabled) {
        const state = await this.state();
        if (!Object.hasOwn(state.skills, validName(name)))
            throw Error("Unknown skill");
        state.skills[name].enabled = enabled;
        await this.save(state);
    }
    async addMcp(name, command, args) {
        validName(name);
        if (typeof command !== "string" || !command || /[\r\n\0]/.test(command))
            throw Error("Provide an executable name or absolute path");
        if (
            !Array.isArray(args) ||
            args.length > 64 ||
            args.some(
                (a) =>
                    typeof a !== "string" ||
                    a.length > 4096 ||
                    a.includes("\0"),
            )
        )
            throw Error("Invalid MCP arguments");
        if ([".cmd", ".bat"].includes(extname(command).toLowerCase()))
            throw Error(
                "Use an executable (node.exe plus a script), not a batch wrapper",
            );
        const state = await this.state();
        if (Object.hasOwn(state.mcp, name))
            throw Error(
                "MCP server already exists; remove it first to replace it",
            );
        Object.defineProperty(state.mcp, name, {
            value: { command, args },
            enumerable: true,
            configurable: true,
            writable: true,
        });
        await this.save(state);
    }
    async remove(kind, name) {
        const state = await this.state();
        if (!Object.hasOwn(state[kind], validName(name)))
            throw Error("Unknown extension");
        delete state[kind][name];
        await this.save(state);
    }
}
export async function skillContext(directory) {
    const state = await new Store(directory).state(),
        parts = [];
    for (const [name, skill] of Object.entries(state.skills)) {
        validName(name);
        if (!skill || typeof skill.content !== "string")
            throw Error("Invalid saved skill");
        if (skill.enabled)
            parts.push(`User-selected skill ${name}:\n${skill.content}`);
    }
    const result = parts.join("\n\n");
    if (result.length > 32768)
        throw Error("Enabled skills exceed 32 KB; disable some skills");
    return result;
}
export async function invokeMcp(store, name, method, params, authorize) {
    const config = (await store.state()).mcp[validName(name)];
    if (
        !config ||
        typeof config.command !== "string" ||
        !Array.isArray(config.args)
    )
        throw Error("Unknown or invalid MCP server");
    if (!(await authorize(config, method, params)))
        throw Error(
            "MCP invocation cancelled; authorize with --yes after reviewing the command",
        );
    if ([".cmd", ".bat"].includes(extname(config.command).toLowerCase()))
        throw Error("MCP batch wrappers are not supported");
    const env = Object.fromEntries(
        Object.entries(process.env).filter(([k]) =>
            [
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
            ].includes(k.toUpperCase()),
        ),
    );
    env.PYTHONIOENCODING = "utf-8";
    env.PYTHONUTF8 = "1";
    const child = spawn(config.command, config.args, {
        stdio: ["pipe", "pipe", "ignore"],
        shell: false,
        windowsHide: true,
        env,
    });
    let serial = 0,
        buffer = "",
        failure;
    const waiting = new Map();
    function fail(message) {
        failure = Error(message);
        for (const item of waiting.values()) {
            clearTimeout(item.timer);
            item.reject(failure);
        }
        waiting.clear();
    }
    function send(message) {
        const raw = JSON.stringify(message) + "\n";
        if (Buffer.byteLength(raw) > 65536)
            throw Error("MCP request exceeds 64 KB");
        child.stdin.write(raw);
    }
    child.on("error", () => fail("Cannot start MCP executable"));
    child.on("exit", () => fail("MCP server exited"));
    child.stdin.on("error", () => fail("MCP input stream closed"));
    child.stdout.setEncoding("utf8");
    child.stdout.on("data", (chunk) => {
        buffer += chunk;
        if (Buffer.byteLength(buffer) > 1048576) {
            fail("MCP response too large");
            child.kill();
            return;
        }
        let index;
        while ((index = buffer.indexOf("\n")) !== -1) {
            const line = buffer.slice(0, index);
            buffer = buffer.slice(index + 1);
            let message;
            try {
                message = JSON.parse(line);
            } catch {
                fail("MCP server emitted invalid JSON");
                return;
            }
            if (!message || typeof message !== "object") {
                fail("Invalid MCP response");
                return;
            }
            if (message.method && message.id !== undefined) {
                send({
                    jsonrpc: "2.0",
                    id: message.id,
                    error: {
                        code: -32601,
                        message: "Unsupported client request",
                    },
                });
                continue;
            }
            const item = waiting.get(message.id);
            if (!item) continue;
            clearTimeout(item.timer);
            waiting.delete(message.id);
            if (
                message.error ||
                !message.result ||
                typeof message.result !== "object"
            )
                item.reject(Error("MCP server rejected the request"));
            else item.resolve(message.result);
        }
    });
    function request(method, params) {
        if (failure) return Promise.reject(failure);
        return new Promise((resolve, reject) => {
            const id = ++serial;
            const timer = setTimeout(() => {
                waiting.delete(id);
                reject(Error("MCP request timed out"));
            }, 20000);
            waiting.set(id, { resolve, reject, timer });
            try {
                send({ jsonrpc: "2.0", id, method, params });
            } catch (error) {
                clearTimeout(timer);
                waiting.delete(id);
                reject(error);
            }
        });
    }
    try {
        const initialized = await request("initialize", {
            protocolVersion: "2025-11-25",
            capabilities: {},
            clientInfo: { name: "vonk", version: "0.3.0" },
        });
        if (
            !["2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"].includes(
                initialized.protocolVersion,
            )
        )
            throw Error(
                "Unsupported MCP version; this release supports legacy 2024/2025 stdio servers",
            );
        send({ jsonrpc: "2.0", method: "notifications/initialized" });
        return await request(method, params);
    } finally {
        for (const item of waiting.values()) clearTimeout(item.timer);
        child.stdin.end();
        if (child.exitCode === null) {
            const timer = setTimeout(() => child.kill(), 500);
            timer.unref();
        }
    }
}
