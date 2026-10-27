import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, writeFile, stat, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PassThrough, Writable } from "node:stream";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import {
    main,
    complete,
    hfModels,
    endpoints,
    CLIError,
    fileContext,
    readSecret,
    terminalText,
} from "../bin/nexus-ai.js";

async function isolated(t) {
    const directory = await mkdtemp(join(tmpdir(), "vonk-cli-test-"));
    const previous = { ...process.env };
    process.env.VONK_CONFIG_DIR = directory;
    for (const provider of Object.keys(endpoints))
        delete process.env[provider.toUpperCase() + "_API_KEY"];
    delete process.env.HF_TOKEN;
    const output = [],
        errors = [];
    t.mock.method(console, "log", (...args) => output.push(args.join(" ")));
    t.mock.method(console, "error", (...args) => errors.push(args.join(" ")));
    t.after(async () => {
        for (const key of Object.keys(process.env))
            if (!(key in previous)) delete process.env[key];
        Object.assign(process.env, previous);
        await rm(directory, { recursive: true, force: true });
    });
    return { directory, output, errors };
}

test("discover, select, inspect, login and selectively logout", async (t) => {
    const { directory, output } = await isolated(t);
    assert.equal(await main(["models", "huggingface"]), 0);
    for (const alias of ["qwen", "deepseek", "mimo"])
        assert.ok(output.join("\n").includes(alias));
    assert.equal(await main(["use", "huggingface", "mimo"]), 0);
    assert.equal(
        JSON.parse(await readFile(join(directory, "config.json"))).model,
        "XiaomiMiMo/MiMo-V2.5",
    );
    assert.equal(
        await main(["login", "huggingface"], {
            readSecret: async () => "synthetic_saved_secret",
        }),
        0,
    );
    assert.equal(
        await main(["login", "opencode"], {
            readSecret: async () => "synthetic_opencode_secret",
        }),
        0,
    );
    process.env.HF_TOKEN = "synthetic_env_secret";
    assert.equal(await main(["status"]), 0);
    assert.ok(output.join("\n").includes("HF_TOKEN"));
    assert.ok(!output.join("\n").includes("synthetic_"));
    if (process.platform !== "win32")
        assert.equal(
            (await stat(join(directory, "keys.json"))).mode & 0o777,
            0o600,
        );
    assert.equal(await main(["logout", "huggingface"]), 0);
    assert.deepEqual(JSON.parse(await readFile(join(directory, "keys.json"))), {
        opencode: "synthetic_opencode_secret",
    });
    assert.equal(await main(["use", "opencode"]), 0);
    assert.equal(
        JSON.parse(await readFile(join(directory, "config.json"))).model,
        "big-pickle",
    );
});

for (const [provider, alias, expected] of [
    ["huggingface", "qwen", "Qwen/Qwen2.5-Coder-32B-Instruct"],
    ["huggingface", "deepseek", "deepseek-ai/DeepSeek-V3.2"],
    ["huggingface", "mimo", "XiaomiMiMo/MiMo-V2.5"],
    ["opencode", "big-pickle", "big-pickle"],
]) {
    test(`routes ${alias} to its correct provider and model`, async () => {
        let captured;
        const answer = await complete(
            provider,
            alias,
            [{ role: "user", content: "hi" }],
            "test_key",
            64,
            0.7,
            async (url, options) => {
                captured = { url, ...options };
                return new Response(
                    JSON.stringify({
                        choices: [{ message: { content: "Answer: 4" } }],
                    }),
                );
            },
        );
        assert.equal(answer, "Answer: 4");
        assert.equal(captured.url, endpoints[provider]);
        assert.equal(JSON.parse(captured.body).model, expected);
        assert.equal(captured.headers.Authorization, "Bearer test_key");
        assert.equal(captured.redirect, "error");
    });
}

test("changing provider resets the default model and forwards generation settings", async (t) => {
    await isolated(t);
    assert.equal(await main(["use", "deepseek", "deepseek-chat"]), 0);
    process.env.HF_TOKEN = "test_key";
    let captured;
    assert.equal(
        await main(
            [
                "chat",
                "--provider",
                "huggingface",
                "--max-tokens",
                "64",
                "--temperature",
                "0.2",
                "hi",
            ],
            {
                complete: async (...args) => {
                    captured = args;
                    return "hello";
                },
            },
        ),
        0,
    );
    assert.equal(captured[1], hfModels.qwen);
    assert.equal(captured[4], 64);
    assert.equal(captured[5], 0.2);
});

test("one-shot upstream errors return failure and never expose raw provider responses", async (t) => {
    const { errors } = await isolated(t);
    process.env.HF_TOKEN = "test_key";
    assert.equal(
        await main(["chat", "--provider", "huggingface", "hi"], {
            complete: async () => {
                throw new Error("PRIVATE_KEY_OR_PROMPT");
            },
        }),
        1,
    );
    assert.ok(!errors.join("\n").includes("PRIVATE_KEY_OR_PROMPT"));
});

for (const status of [401, 403, 404, 429, 500]) {
    test(`upstream HTTP ${status} produces a sanitized error`, async () => {
        await assert.rejects(
            complete(
                "huggingface",
                "qwen",
                [{ role: "user", content: "hi" }],
                "test_key",
                64,
                0.7,
                async () => new Response("PRIVATE_KEY_OR_PROMPT", { status }),
            ),
            (error) => {
                assert.ok(error instanceof CLIError);
                assert.ok(!error.message.includes("PRIVATE_KEY_OR_PROMPT"));
                return true;
            },
        );
    });
}

test("timeouts, malformed JSON, and empty completions are handled", async () => {
    for (const fetcher of [
        async () => {
            throw new Error("PRIVATE");
        },
        async () => new Response("invalid JSON"),
        async () => new Response(JSON.stringify({ choices: [] })),
        async () =>
            new Response(
                JSON.stringify({ choices: [{ message: { content: "" } }] }),
            ),
    ])
        await assert.rejects(
            complete(
                "huggingface",
                "qwen",
                [{ role: "user", content: "hi" }],
                "test_key",
                64,
                0.7,
                fetcher,
            ),
            CLIError,
        );
});

test("Anthropic gets system separately and text blocks are decoded", async () => {
    const answer = await complete(
        "anthropic",
        "test-model",
        [
            { role: "system", content: "rules" },
            { role: "user", content: "hi" },
        ],
        "test_key",
        64,
        0.7,
        async (_url, options) => {
            const body = JSON.parse(options.body);
            assert.equal(body.system, "rules");
            assert.deepEqual(body.messages, [{ role: "user", content: "hi" }]);
            assert.equal(options.headers["x-api-key"], "test_key");
            return new Response(
                JSON.stringify({ content: [{ type: "text", text: "hello" }] }),
            );
        },
    );
    assert.equal(answer, "hello");
});

test("missing keys, malformed config, missing flags and invalid limits fail clearly", async (t) => {
    const { directory, errors } = await isolated(t);
    for (const argv of [
        ["chat", "hi"],
        ["chat", "--file"],
        ["chat", "--unknown"],
        ["chat", "--max-tokens", "0", "hi"],
        ["chat", "--temperature", "NaN", "hi"],
        ["use"],
        ["login", "unknown"],
        ["use", "huggingface", "big-pickle"],
    ])
        assert.equal(await main(argv), 2);
    for (const content of ["[]", "broken JSON"]) {
        await writeFile(join(directory, "config.json"), content);
        assert.equal(await main(["status"]), 2);
    }
    assert.ok(!errors.join("\n").includes("at main"));
});

test("file attachments are read-only, UTF-8, and bounded in total", async (t) => {
    const { directory } = await isolated(t);
    const path = join(directory, "example.py");
    await writeFile(path, "hello");
    assert.ok((await fileContext([path])).includes("hello"));
    assert.equal(await readFile(path, "utf8"), "hello");
    for (const data of [
        Buffer.alloc(60001, 65),
        Buffer.from([0]),
        Buffer.from([255]),
    ]) {
        await writeFile(path, data);
        await assert.rejects(fileContext([path]), CLIError);
    }
    await assert.rejects(fileContext([join(directory, ".env")]), CLIError);
    await assert.rejects(fileContext([directory]), CLIError);
});

test("key input is hidden and model output cannot execute terminal commands", async () => {
    const input = new PassThrough();
    input.isTTY = true;
    input.setRawMode = () => {};
    let output = "";
    const sink = new Writable({
        write(chunk, _encoding, done) {
            output += chunk.toString();
            done();
        },
    });
    const reading = readSecret(input, sink);
    input.write("synthetic_hidden_key\r");
    assert.equal(await reading, "synthetic_hidden_key");
    assert.ok(!output.includes("synthetic_hidden_key"));
    input.end();
    assert.equal(
        terminalText("\x1b[31mhello\x1b[0m\x1b]52;c;payload\x07\x9b"),
        "hello",
    );
});

test("actual executable supports help/version and rejects piped credential login", () => {
    const executable = new URL("../bin/nexus-ai.js", import.meta.url);
    for (const option of ["--help", "--version"]) {
        const result = spawnSync(
            process.execPath,
            [fileURLToPath(executable), option],
            { encoding: "utf8" },
        );
        assert.equal(result.status, 0, result.stderr);
        assert.ok(result.stdout.includes("Vonk"));
    }
    const result = spawnSync(
        process.execPath,
        [fileURLToPath(executable), "login", "huggingface"],
        { encoding: "utf8", input: "not-a-real-key\n" },
    );
    assert.equal(result.status, 2);
    assert.ok(result.stderr.includes("interactive terminal"));
    assert.ok(!result.stdout.includes("not-a-real-key"));
});

test("pending requests honor cancellation without leaking error details", async () => {
    const cancellation = new AbortController();
    const request = complete(
        "huggingface",
        "qwen",
        [{ role: "user", content: "hi" }],
        "test_key",
        64,
        0.7,
        async (_url, options) =>
            new Promise((_resolve, reject) => {
                options.signal.addEventListener(
                    "abort",
                    () => reject(new Error("PRIVATE")),
                    { once: true },
                );
                queueMicrotask(() => cancellation.abort());
            }),
        cancellation.signal,
    );
    await assert.rejects(
        request,
        (error) => error instanceof CLIError && error.message === "Cancelled",
    );
});

test("piped interactive chat preserves turns and clears history", async (t) => {
    const { directory } = await isolated(t);
    const cliURL = new URL("../bin/nexus-ai.js", import.meta.url).href;
    const script = `import {main} from ${JSON.stringify(cliURL)};
        process.exitCode = await main(['chat', '--provider', 'huggingface'], {
            complete: async (_p, _m, messages) => 'history:' + messages.length,
        });`;
    const result = spawnSync(
        process.execPath,
        ["--input-type=module", "-e", script],
        {
            encoding: "utf8",
            input: "first\nsecond\n/clear\nfresh\n/exit\n",
            env: {
                ...process.env,
                HF_TOKEN: "synthetic_key",
                VONK_CONFIG_DIR: directory,
            },
        },
    );
    assert.equal(result.status, 0, result.stderr);
    assert.equal((result.stdout.match(/history:2/g) || []).length, 2);
    assert.equal((result.stdout.match(/history:4/g) || []).length, 1);
});
