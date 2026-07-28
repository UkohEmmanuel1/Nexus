
from src.inference.engine import InferenceEngine

from .retriever import Retriever


class RAGGenerator:
    def __init__(
        self,
        engine: InferenceEngine,
        retriever: Retriever,
        system_prompt: str | None = None,
    ):
        self.engine = engine
        self.retriever = retriever
        self.system_prompt = system_prompt or (
            "You are a helpful assistant. Answer the user's question based on the "
            "provided context from retrieved documents. If the context doesn't contain "
            "enough information, say so."
        )

    def generate(self, query: str, top_k: int = 5, **kwargs) -> str:
        docs = self.retriever.retrieve(query, top_k=top_k)
        context = "\n\n".join(docs)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ]

        prompt = self.engine._format_chat(messages)
        return self.engine.generate(prompt, **kwargs)

    def generate_stream(self, query: str, top_k: int = 5, **kwargs):
        docs = self.retriever.retrieve(query, top_k=top_k)
        context = "\n\n".join(docs)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ]

        prompt = self.engine._format_chat(messages)
        yield from self.engine.generate(prompt, stream=True, **kwargs)
