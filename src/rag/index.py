import pickle
from pathlib import Path

import numpy as np


class DocumentIndex:
    def __init__(self, embedding_dim: int = 4096):
        self.embedding_dim = embedding_dim
        self.documents: list[str] = []
        self.embeddings: np.ndarray | None = None

    def add_documents(self, documents: list[str], embeddings: np.ndarray):
        self.documents.extend(documents)
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        if self.embeddings is None or len(self.embeddings) == 0:
            return []

        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        doc_norms = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-10)
        scores = np.dot(doc_norms, query_norm.T).flatten()

        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(self.documents[i], float(scores[i])) for i in top_indices]

    def save(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"documents": self.documents, "embeddings": self.embeddings}, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.documents = data["documents"]
        self.embeddings = data["embeddings"]
