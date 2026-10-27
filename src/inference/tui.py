"""Full-screen chat workspace. Imported only in interactive terminals."""

import asyncio
import pathlib
from types import SimpleNamespace

from prompt_toolkit.application import Application, in_terminal
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import ConditionalContainer, HSplit, Layout, Window, WindowAlign
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.output import ColorDepth
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import TextArea

from .workspace import Store, skill_context

PALETTE = {
    "background": "#101713",
    "surface": "#19231c",
    "text": "#e2eadf",
    "muted": "#8b9c8e",
    "green": "#9de275",
    "amber": "#efbd62",
}
COMMANDS = [
    ("/models", "Select a model"),
    ("/connect", "Connect a provider"),
    ("/free", "Set up a free model"),
    ("/skills", "Manage skill instructions"),
    ("/mcp", "View MCP connections"),
    ("/new", "Start a new conversation"),
    ("/help", "Keyboard shortcuts and commands"),
    ("/exit", "Close Vonk"),
]
LOGO = "V O N K"
WORDMARK = [
    "██╗   ██╗ ██████╗ ███╗   ██╗██╗  ██╗",
    "██║   ██║██╔═══██╗████╗  ██║██║ ██╔╝",
    "██║   ██║██║   ██║██╔██╗ ██║█████╔╝ ",
    "╚██╗ ██╔╝██║   ██║██║╚██╗██║██╔═██╗",
    " ╚████╔╝ ╚██████╔╝██║ ╚████║██║  ██╗",
    "  ╚═══╝   ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝",
]


class Workspace:
    def __init__(self, api, *, input=None, output=None):
        self.api = api
        self.config, self.keys = api.config_paths()
        self.store = Store(self.config.parent)
        self.messages = []
        self.transcript = ""
        self.home = True
        self.busy = False
        self.task = None
        self.modal = None
        self.index = 0
        self.notice = "Type a message to begin. / opens commands."
        self.refresh_selection()
        self.history = TextArea(
            read_only=True, scrollbar=True, wrap_lines=True, style="class:conversation"
        )
        self.composer = TextArea(
            height=3, multiline=True, wrap_lines=True, prompt="  > ", style="class:composer"
        )
        self.composer.buffer.on_text_changed += self.changed
        bindings = KeyBindings()

        @bindings.add("enter")
        def submit(event):
            if self.modal:
                self.choose()
            else:
                self.send()

        @bindings.add("escape", "enter")
        def newline(event):
            if not self.modal:
                self.composer.buffer.insert_text("\n")

        @bindings.add("c-p")
        def commands(event):
            if not self.busy:
                self.composer.text = "/"

        @bindings.add("f2")
        def models(event):
            self.command("/models")

        @bindings.add("f3")
        def connect(event):
            self.command("/connect")

        @bindings.add("escape")
        def cancel(event):
            if self.task and not self.task.done():
                self.task.cancel()
            self.modal = None
            self.composer.text = ""

        @bindings.add("up", filter=Condition(lambda: bool(self.modal)))
        def up(event):
            self.index = max(0, self.index - 1)

        @bindings.add("down", filter=Condition(lambda: bool(self.modal)))
        def down(event):
            self.index = min(len(self.options()) - 1, self.index + 1)

        @bindings.add("pageup")
        def page_up(event):
            self.history.buffer.cursor_up(count=8)

        @bindings.add("pagedown")
        def page_down(event):
            self.history.buffer.cursor_down(count=8)

        @bindings.add("c-c")
        def interrupt(event):
            if self.busy:
                cancel(event)
            elif self.composer.text or self.modal:
                cancel(event)
            else:
                event.app.exit(result=0)

        @bindings.add("c-d")
        def exit_app(event):
            if self.task:
                self.task.cancel()
            event.app.exit(result=0)

        self.app = Application(
            layout=Layout(
                HSplit(
                    [
                        Window(FormattedTextControl(self.header), height=2),
                        ConditionalContainer(
                            Window(
                                FormattedTextControl(self.hero), height=7, align=WindowAlign.CENTER
                            ),
                            filter=Condition(lambda: self.home and not self.modal),
                        ),
                        self.history,
                        Window(FormattedTextControl(self.overlay), height=condition_height(self)),
                        Window(FormattedTextControl(self.status), height=1),
                        self.composer,
                        Window(FormattedTextControl(self.footer), height=1),
                    ]
                ),
                focused_element=self.composer,
            ),
            key_bindings=bindings,
            full_screen=True,
            color_depth=ColorDepth.TRUE_COLOR,
            mouse_support=True,
            style=Style.from_dict(
                {
                    "": f"bg:{PALETTE['background']} {PALETTE['text']}",
                    "conversation": f"bg:{PALETTE['background']}",
                    "composer": f"bg:{PALETTE['surface']} {PALETTE['text']}",
                    "green": PALETTE["green"],
                    "amber": PALETTE["amber"],
                    "muted": PALETTE["muted"],
                    "selected": f"bg:{PALETTE['green']} #101713 bold",
                    "scrollbar.background": f"bg:{PALETTE['surface']}",
                    "scrollbar.button": f"bg:{PALETTE['muted']}",
                }
            ),
            input=input,
            output=output,
        )
        self.append(
            "Vonk",
            "Your project. Your models. Your flow.\n\n"
            "Write a prompt below, or open /models and /connect.\n"
            "Free local chat: /free. Skills: /skills. Connections: /mcp.",
        )
        self.home = True

    def hero(self):
        return [
            ("class:green" if i < 3 else "class:amber", line + "\n")
            for i, line in enumerate(WORDMARK)
        ]

    def refresh_selection(self):
        self.provider, self.model = self.api.selection(
            SimpleNamespace(provider=None, model=None), self.api.load(self.config)
        )
        self.key, _ = self.api.credential(self.provider, self.keys)

    def header(self):
        return [
            ("class:green bold", f"  {LOGO}"),
            ("class:amber", f"   {self.api.VERSION}"),
            ("class:muted", f"   /   {self.api.terminal_text(pathlib.Path.cwd().name)}\n"),
        ]

    def status(self):
        return [
            ("class:green", "  " + self.provider),
            ("class:muted", " / " + self.api.terminal_text(self.model)),
            (
                "class:amber",
                "   thinking..."
                if self.busy
                else "   connected"
                if self.key
                else "   /connect to add your key",
            ),
        ]

    def footer(self):
        return [
            (
                "class:muted",
                "  Enter send  Alt+Enter newline  / commands  F2 models  Esc cancel  Ctrl+D exit",
            )
        ]

    def changed(self, _):
        if self.modal and self.modal[0] == "commands":
            self.index = 0
            if not self.composer.text.startswith("/"):
                self.modal = None
        elif self.composer.text.startswith("/") and not self.modal and not self.busy:
            self.modal = ("commands", [])
            self.index = 0

    def options(self):
        if not self.modal:
            return []
        if self.modal[0] == "commands":
            return [
                (cmd, f"{cmd:12} {label}")
                for cmd, label in COMMANDS
                if cmd.startswith(self.composer.text.strip())
            ]
        return self.modal[1]

    def overlay(self):
        if not self.modal:
            return [("class:muted", "  " + self.api.terminal_text(self.notice))]
        options = self.options()
        start = max(0, self.index - 5)
        result = [
            (
                "class:amber",
                f"  {self.modal[0].upper()}   Up/Down select · Enter confirm · Esc close\n",
            )
        ]
        for i, (_, label) in enumerate(options[start : start + 7], start):
            result.append(
                (
                    "class:selected" if i == self.index else "class:muted",
                    "  " + self.api.terminal_text(label) + "\n",
                )
            )
        if not options:
            result.append(("class:muted", "  No matching command. Esc returns to chat."))
        return result

    def append(self, role, text):
        self.home = False
        self.transcript += f"\n  {role}\n\n{self.api.terminal_text(str(text))}\n"
        self.transcript = self.transcript[-120000:]
        self.history.buffer.set_document(
            Document(self.transcript, cursor_position=len(self.transcript)), bypass_readonly=True
        )

    def show(self, title, options):
        if self.busy:
            return
        self.modal, self.index = (title, options), 0

    def choose(self):
        options = self.options()
        if not options:
            self.send()
            return
        mode = self.modal[0]
        value = options[min(self.index, len(options) - 1)][0]
        self.modal = None
        self.composer.text = ""
        if mode == "commands":
            self.command(value)
        elif mode == "models":
            self.select(*value)
        elif mode == "connect":
            if value == "ollama":
                self.select(value, self.api.EXAMPLE_MODELS[value][0])
            else:
                self.task = self.app.create_background_task(self.login(value))
        elif mode == "free":
            self.select(*value)
            self.append(
                "Free models",
                "Local: install Ollama and run `ollama pull qwen2.5-coder:1.5b`. "
                "Keep Ollama running. OpenRouter requires your own free-tier key and quotas.",
            )
        elif mode == "skills":
            self.store.toggle_skill(value[0], not value[1])
            self.command("/skills")

    def select(self, provider, model):
        self.api.save(self.config, {"provider": provider, "model": model})
        self.refresh_selection()
        self.messages = []
        self.notice = "Model changed. A new conversation will start with your next prompt."

    async def login(self, provider):
        from prompt_toolkit.shortcuts import input_dialog

        async with in_terminal():
            key = await input_dialog(
                title=f"Connect {provider}",
                text="Your API key (saved locally; environment keys take precedence)",
                password=True,
                style=self.app.style,
            ).run_async()
        if key is None:
            return
        key = key.strip()
        if not key or len(key) > 512 or any(ord(c) < 33 or ord(c) > 126 for c in key):
            self.append("Connection", "API key must be 1-512 printable characters without spaces.")
            return
        keys = self.api.load(self.keys)
        keys[provider] = key
        self.api.save(self.keys, keys, secret=True)
        self.select(provider, self.api.EXAMPLE_MODELS[provider][0])
        self.notice = "Provider key saved locally. Ready to chat."
        self.app.invalidate()

    def command(self, command):
        if self.busy:
            self.notice = "Press Esc to cancel the current request first."
            return
        self.modal = None
        self.composer.text = ""
        if command in {"/exit", "/quit"}:
            self.app.exit(result=0)
        elif command in {"/models", "/model"}:
            self.show(
                "models",
                [
                    ((p, m), f"{p:13} {m}")
                    for p, models in self.api.EXAMPLE_MODELS.items()
                    for m in models
                ],
            )
        elif command == "/connect":
            self.show(
                "connect",
                [
                    (p, p + (" · local, no key" if p == "ollama" else " · your API key"))
                    for p in self.api.PROVIDERS
                ],
            )
        elif command == "/free":
            self.show(
                "free",
                [
                    (("ollama", "qwen2.5-coder:1.5b"), "Ollama · local / no API bill"),
                    (("openrouter", "openrouter/free"), "OpenRouter · free tier / own key"),
                ],
            )
        elif command in {"/new", "/clear"}:
            self.messages, self.transcript = [], ""
            self.append("Vonk", "New conversation. What are we working on?")
            self.home = True
        elif command == "/skills":
            skills = self.store.state()["skills"]
            if skills:
                self.show(
                    "skills",
                    [
                        (
                            (name, item.get("enabled", False)),
                            name + (" · enabled" if item.get("enabled") else " · disabled"),
                        )
                        for name, item in skills.items()
                    ],
                )
            else:
                self.append(
                    "Skills",
                    "No skills installed. Add one in another terminal:\n"
                    "vonk skills add PATH/TO/SKILL.md --name review\n"
                    "Then /skills lets you enable or disable it here.",
                )
        elif command == "/mcp":
            servers = self.store.state()["mcp"]
            self.append(
                "MCP connections",
                "\n".join(f"{n} · {v['command']} · on demand" for n, v in servers.items())
                or "No MCP connections configured.",
            )
            self.append(
                "MCP commands",
                "vonk mcp add NAME EXECUTABLE ARG...\n"
                "vonk mcp tools NAME\nvonk mcp call NAME TOOL [JSON]\n"
                "Run these in another terminal; each invocation needs your permission.",
            )
        else:
            self.append(
                "Commands",
                "\n".join(f"{c:12} {label}" for c, label in COMMANDS)
                + "\n\nEnter sends · Alt+Enter adds a line · PageUp/PageDown scroll\n"
                "Ctrl+P commands · F2 models · F3 connect · Esc cancel · Ctrl+D exit\n"
                "Chat reviews and explains code; it does not automatically edit files.",
            )

    def send(self):
        prompt = self.composer.text.strip()
        if not prompt or self.busy:
            return
        if prompt.startswith("/"):
            self.command(prompt)
            return
        if len(prompt) > 16000:
            self.notice = "Prompt exceeds 16,000 characters. Shorten it before sending."
            return
        if not self.key:
            self.notice = "Connect a provider with /connect, or choose a local model with /free."
            self.modal = None
            return
        self.composer.text = ""
        self.append("You", prompt)
        self.busy = True
        self.task = self.app.create_background_task(self.generate(prompt))

    async def generate(self, prompt):
        try:
            context = skill_context(self.config.parent)
            if not self.messages:
                self.messages = [{"role": "system", "content": self.api.SYSTEM + "\n" + context}]
            turns = self.messages + [{"role": "user", "content": prompt}]
            while len(turns) > 99 or sum(len(m["content"]) for m in turns) > 120000:
                del turns[1:3]
            result = await self.api.complete_byok(
                provider=self.provider,
                model=self.model,
                messages=turns,
                provider_key=self.key,
                max_tokens=2048,
                temperature=0.7,
            )
            answer = result["response"]
            self.messages = turns + [{"role": "assistant", "content": answer}]
            self.append("Vonk", answer)
            self.notice = "Ready. /new starts fresh."
        except asyncio.CancelledError:
            self.append("Vonk", "Request cancelled.")
        except Exception as exc:
            self.append(
                "Connection",
                getattr(
                    exc, "detail", "Request failed. Check your connection and provider settings."
                ),
            )
        finally:
            self.busy = False
            self.app.invalidate()


def condition_height(workspace):
    """A callable keeps the palette compact and respects the terminal height."""

    def height():
        if not workspace.modal:
            return 2
        return min(9, len(workspace.options()) + 2)

    return height


def run(api, connect=False):
    workspace = Workspace(api)
    if connect:
        workspace.command("/connect")
    workspace.app.output.set_title("Vonk · " + api.terminal_text(pathlib.Path.cwd().name))
    try:
        return workspace.app.run()
    finally:
        workspace.app.output.clear_title()
