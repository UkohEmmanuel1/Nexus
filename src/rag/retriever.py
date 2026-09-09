import numpy as np

from .index import DocumentIndex


class Retriever:
    def __init__(
        self,
        index: DocumentIndex,
        embedding_model=None,
        top_k: int = 5,
    ):
        self.index = index
        self.embedding_model = embedding_model
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[str]:
        k = top_k or self.top_k
        if self.embedding_model is not None:
            query_emb = self.embedding_model.encode([query])
        else:
            query_emb = np.random.randn(1, self.index.embedding_dim)

        results = self.index.search(query_emb, top_k=k)
        return [doc for doc, _ in results]

    def retrieve_with_scores(self, query: str, top_k: int | None = None) -> list[tuple[str, float]]:
        k = top_k or self.top_k
        if self.embedding_model is not None:
            query_emb = self.embedding_model.encode([query])
        else:
            query_emb = np.random.randn(1, self.index.embedding_dim)

        return self.index.search(query_emb, top_k=k)
