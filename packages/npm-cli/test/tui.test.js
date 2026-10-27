import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { TerminalSession, renderFrame } from "../lib/tui.js";
import { endpoints, examples, hfModels, load, save } from "../bin/nexus-ai.js";

async function session(t, complete = async () => "Hello") {
    const directory = await mkdtemp(join(tmpdir(), "vonk-tui-"));
    t.after(() => rm(directory, { recursive: true, force: true }));
    const api = {
        VERSION: "test", config: join(directory, "config.json"),
        keysPath: join(directory, "keys.json"), load, save, endpoints, examples, hfModels,
        credential: async p => [p === "ollama" ? "local" : null],
        complete, systemPrompt: "Coding assistant",
    };
    const ui = new TerminalSession(api, directory);
    await ui.refresh();
    return ui;
}

test("fullscreen command palette selects a model and sends with conversation context", async t => {
    const calls = [];
    const ui = await session(t, async (...args) => { calls.push(args); return "Works"; });
    for (const char of "/free") await ui.keypress(char, { name: char });
    assert.equal(ui.state.mode, "commands");
    await ui.keypress("\r", { name: "return" });
    assert.equal(ui.state.mode, "free");
    await ui.keypress("\r", { name: "return" });
    assert.equal(ui.state.provider, "ollama");
    assert.equal((await load(ui.api.config)).model, "qwen2.5-coder:1.5b");
    ui.state.draft = "Hello";
    await ui.send();
    ui.state.draft = "Again";
    await ui.send();
    assert.equal(calls[1][2].length, 4);
    assert.ok(ui.state.transcript.includes("Works"));
    await ui.command("/new");
    assert.deepEqual(ui.messages, []);
});

test("composer editing, missing credentials, masking and small terminal rendering", async t => {
    const ui = await session(t);
    for (const char of "helo") await ui.keypress(char, { name: char });
    await ui.keypress("", { name: "left" });
    await ui.keypress("l", { name: "l" });
    assert.equal(ui.state.draft, "hello");
    await ui.send();
    assert.equal(ui.state.draft, "hello");
    assert.ok(ui.state.notice.includes("/connect"));
    ui.state.mode = "key";
    ui.state.draft = "synthetic-secret";
    const frame = renderFrame(ui.state, 40, 16);
    assert.ok(!frame.includes("synthetic-secret"));
    assert.ok(frame.includes("********"));
    assert.equal(frame.split("\r\n").length, 16);
    assert.ok(frame.includes("38;2;157;226;117"));
    assert.ok(!renderFrame({ ...ui.state, project: "\x1b]52;c;secret\x07" }).includes("]52;c;secret"));
});

test("Esc cancels an in-flight request without adding a response to history", async t => {
    let ready;
    const started = new Promise(resolve => { ready = resolve; });
    const ui = await session(t, (...args) => new Promise((_, reject) => {
        args[6].addEventListener("abort", () => reject(Error("synthetic-private-detail")));
        ready();
    }));
    await ui.select("ollama", examples.ollama);
    ui.state.draft = "hello";
    const request = ui.send();
    await started;
    await ui.keypress("", { name: "escape" });
    await request;
    assert.equal(ui.state.busy, false);
    assert.ok(ui.state.transcript.includes("Request cancelled"));
    assert.equal(ui.messages.length, 1);
    assert.ok(!ui.state.transcript.includes("synthetic-private-detail"));
});
