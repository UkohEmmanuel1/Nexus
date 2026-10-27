import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, writeFile, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import { Store, skillContext, invokeMcp } from "../lib/workspace.js";
import { main, complete } from "../bin/nexus-ai.js";

async function isolated(t) {
    const directory = await mkdtemp(join(tmpdir(), "vonk-workspace-"));
    const env = { ...process.env };
    process.env.VONK_CONFIG_DIR = directory;
    t.after(async () => {
        for (const name of Object.keys(process.env))
            if (!(name in env)) delete process.env[name];
        Object.assign(process.env, env);
        await rm(directory, { recursive: true, force: true });
    });
    return directory;
}
test("first launch and vonk chat show a workspace instead of rejecting missing keys", async (t) => {
    const directory = await isolated(t);
    const executable = fileURLToPath(
        new URL("../bin/nexus-ai.js", import.meta.url),
    );
    for (const args of [[], ["chat"], ["ui"]]) {
        const env = { ...process.env, VONK_CONFIG_DIR: directory };
        delete env.DEEPSEEK_API_KEY;
        const result = spawnSync(process.execPath, [executable, ...args], {
            encoding: "utf8",
            env,
            input: "1\n/help\n0\n",
        });
        assert.equal(result.status, 0, result.stderr);
        for (const label of [
            "V O N K",
            "Select model",
            "Connect provider",
            "Free models",
            "Skills",
            "MCP",
            "CONNECT BEFORE CHATTING",
        ])
            assert.ok(result.stdout.includes(label), label);
        assert.ok(!result.stderr.includes("No key for deepseek"));
    }
});
test("free local selection persists and does not require a key", async (t) => {
    const directory = await isolated(t);
    const result = spawnSync(
        process.execPath,
        [fileURLToPath(new URL("../bin/nexus-ai.js", import.meta.url)), "ui"],
        {
            encoding: "utf8",
            env: { ...process.env, VONK_CONFIG_DIR: directory },
            input: "4\n1\n0\n",
        },
    );
    assert.equal(result.status, 0, result.stderr);
    assert.equal(
        JSON.parse(await readFile(join(directory, "config.json"))).provider,
        "ollama",
    );
    let captured;
    assert.equal(
        await complete(
            "ollama",
            "qwen2.5-coder:1.5b",
            [{ role: "user", content: "hi" }],
            null,
            64,
            0.7,
            async (url, options) => {
                captured = { url, options };
                return new Response(
                    JSON.stringify({
                        choices: [{ message: { content: "local answer" } }],
                    }),
                );
            },
        ),
        "local answer",
    );
    assert.equal(captured.url, "http://127.0.0.1:11434/v1/chat/completions");
    assert.ok(!captured.options.headers.Authorization.includes("undefined"));
});
test("skills add, enable, disable, remove and real chat context", async (t) => {
    const directory = await isolated(t),
        store = new Store(directory),
        source = join(directory, "SKILL.md");
    await writeFile(source, "Always include a test explanation.");
    await store.addSkill(source, "review");
    assert.ok((await skillContext(directory)).includes("Always include"));
    process.env.HF_TOKEN = "synthetic-key";
    let messages;
    const log = t.mock.method(console, "log", () => {});
    assert.equal(
        await main(["chat", "--provider", "huggingface", "hello"], {
            complete: async (_p, _m, m) => {
                messages = m;
                return "done";
            },
        }),
        0,
    );
    assert.ok(messages[0].content.includes("Always include"));
    await store.toggleSkill("review", false);
    assert.equal(await skillContext(directory), "");
    await store.remove("skills", "review");
    assert.deepEqual((await store.state()).skills, {});
    log.mock.restore();
});
test("extension names and sizes cannot escape configuration or accept batch wrappers", async (t) => {
    const directory = await isolated(t),
        store = new Store(directory),
        source = join(directory, "SKILL.md");
    await writeFile(source, "x".repeat(32769));
    await assert.rejects(store.addSkill(source, "large"));
    await assert.rejects(store.addMcp("../escape", "node", []));
    await assert.rejects(store.addMcp("batch", "bad.cmd", []));
});
test("real MCP stdio handshake, listing and call with provider credentials excluded", async (t) => {
    const directory = await isolated(t),
        store = new Store(directory);
    const server = fileURLToPath(
        new URL("../../../tests/fixtures/mcp_echo_server.py", import.meta.url),
    );
    await store.addMcp(
        "echo",
        process.platform === "win32" ? "python" : "python3",
        [server],
    );
    await assert.rejects(
        invokeMcp(store, "echo", "tools/list", {}, async () => false),
        /cancelled/,
    );
    const tools = await invokeMcp(
        store,
        "echo",
        "tools/list",
        {},
        async () => true,
    );
    assert.equal(tools.tools[0].name, "echo");
    process.env.HF_TOKEN = "not-for-mcp";
    const result = await invokeMcp(
        store,
        "echo",
        "tools/call",
        { name: "echo", arguments: { text: "MCP works" } },
        async () => true,
    );
    assert.equal(result.content[0].text, "MCP works");
    assert.equal(result.credential_leaked, false);
});
