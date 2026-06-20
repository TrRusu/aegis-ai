"""Caches RAG responses by semantic similarity of past queries.
"""

import math
from langchain_core.embeddings import Embeddings

DEFAULT_THRESHOLD = 0.92


class SemanticCache:

    def __init__(self, embeddings: Embeddings, threshold: float = DEFAULT_THRESHOLD):
        self._embeddings = embeddings
        self._threshold = threshold
        self._entries: list[tuple[list[float], str]] = []

    def check(self, query: str) -> str | None:
        if not self._entries:
            return None
        query_vector = self._embeddings.embed_query(query)
        best_score = -1.0
        best_response = None
        for vector, response in self._entries:
            score = self._cosine_similarity(query_vector, vector)
            if score > best_score:
                best_score = score
                best_response = response
        if best_score >= self._threshold:
            return best_response
        return None

    def store(self, query: str, response: str) -> None:
        vector = self._embeddings.embed_query(query)
        self._entries.append((vector, response))

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
