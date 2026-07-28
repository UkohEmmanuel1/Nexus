

class ConversationMemory:
    def __init__(self, max_tokens: int = 4096, summarizer=None):
        self.messages: list[dict[str, str]] = []
        self.max_tokens = max_tokens
        self.summarizer = summarizer
        self.summary: str | None = None

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def get_context(self) -> list[dict[str, str]]:
        if self.summary:
            summary_msg = {"role": "system", "content": f"Conversation summary: {self.summary}"}
            recent = self.messages[-4:] if len(self.messages) > 4 else self.messages
            return [summary_msg] + recent
        return self.messages[-8:] if len(self.messages) > 8 else self.messages

    def clear(self):
        self.messages = []
        self.summary = None

    def __len__(self) -> int:
        return len(self.messages)
