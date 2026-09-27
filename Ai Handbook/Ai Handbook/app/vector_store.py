from pathlib import Path
from uuid import uuid4

import chromadb
import numpy as np

from app.models import Chunk, SearchResult


class VectorStore:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray):
        if len(chunks) != len(vectors):
            raise ValueError("Each chunk must have one embedding")
        self.chunks = chunks
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.client = chromadb.EphemeralClient()
        self.collection = self.client.create_collection(
            name="handbook_" + uuid4().hex, metadata={"hnsw:space": "cosine"}
        )
        if chunks:
            self._add_chunks(chunks, self.vectors)

    def _add_chunks(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self.collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=np.asarray(vectors, dtype=np.float32).tolist(),
            metadatas=[{"page": chunk.page} for chunk in chunks],
        )

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> list[SearchResult]:
        if not self.chunks:
            return []
        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1).tolist()
        response = self.collection.query(query_embeddings=query, n_results=min(top_k, len(self.chunks)))
        results = []
        for text, chunk_id, metadata, distance in zip(
            response["documents"][0], response["ids"][0], response["metadatas"][0], response["distances"][0]
        ):
            results.append(SearchResult(Chunk(text, int(metadata["page"]), chunk_id), 1.0 - float(distance)))
        return results

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        persistent_client = chromadb.PersistentClient(path=str(directory))
        try:
            persistent_client.delete_collection("handbook")
        except chromadb.errors.NotFoundError:
            pass
        collection = persistent_client.create_collection("handbook", metadata={"hnsw:space": "cosine"})
        if self.chunks:
            collection.add(
                ids=[chunk.chunk_id for chunk in self.chunks],
                documents=[chunk.text for chunk in self.chunks],
                embeddings=self.vectors.tolist(),
                metadatas=[{"page": chunk.page} for chunk in self.chunks],
            )

    @classmethod
    def load(cls, directory: Path) -> "VectorStore":
        client = chromadb.PersistentClient(path=str(directory))
        collection = client.get_collection("handbook")
        data = collection.get(include=["documents", "embeddings", "metadatas"])
        instance = cls.__new__(cls)
        instance.client = client
        instance.collection = collection
        instance.chunks = [
            Chunk(text, int(metadata["page"]), chunk_id)
            for text, metadata, chunk_id in zip(data["documents"], data["metadatas"], data["ids"])
        ]
        instance.vectors = np.asarray(data["embeddings"], dtype=np.float32)
        return instance
