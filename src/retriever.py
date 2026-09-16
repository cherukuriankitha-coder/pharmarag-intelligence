"""Semantic retrieval component for PharmaRAG Intelligence."""
from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

from document_pipeline import Chunk


class SemanticRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.chunks: List[Chunk] = []
        self.embeddings = None

    def fit(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        self.embeddings = self.model.encode(
            [c.text for c in chunks], normalize_embeddings=True
        )

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        if self.embeddings is None:
            raise RuntimeError("Call fit() before search().")
        query_vector = self.model.encode([query], normalize_embeddings=True)[0]
        scores = np.dot(self.embeddings, query_vector)
        indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in indices]
