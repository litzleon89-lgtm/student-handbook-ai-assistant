import json
from pathlib import Path

import numpy as np

from app.models import Chunk, SearchResult


class VectorStore:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray):
        if len(chunks) != len(vectors):
            raise ValueError("Each chunk must have one embedding")
        self.chunks = chunks
        self.vectors = np.asarray(vectors, dtype=np.float32)

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> list[SearchResult]:
        if not self.chunks:
            return []
        query = np.asarray(query_vector, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        vector_norms = np.linalg.norm(self.vectors, axis=1)
        scores = (self.vectors @ query) / np.maximum(vector_norms * query_norm, 1e-12)
        indexes = np.argsort(scores)[::-1][:top_k]
        return [SearchResult(self.chunks[i], float(scores[i])) for i in indexes]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        (directory / "chunks.json").write_text(
            json.dumps([chunk.__dict__ for chunk in self.chunks], indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> "VectorStore":
        chunks_data = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        chunks = [Chunk(**item) for item in chunks_data]
        vectors = np.load(directory / "vectors.npy")
        return cls(chunks, vectors)
