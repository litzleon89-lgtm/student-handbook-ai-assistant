from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    text: str
    page: int
    chunk_id: str


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float
