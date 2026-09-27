import re
from pathlib import Path

from app.config import INDEX_DIR, RELEVANCE_THRESHOLD, TOP_K
from app.embeddings import Embedder, SentenceTransformerEmbedder
from app.generator import AnswerGenerator, create_generator
from app.ingest import load_and_chunk
from app.models import SearchResult
from app.vector_store import VectorStore

STOP_WORDS = {"a", "an", "are", "do", "does", "for", "from", "how", "i", "is", "it", "of", "should", "the", "to", "what", "when", "who", "with"}


class RAGAssistant:
    def __init__(self, store: VectorStore, embedder: Embedder, generator: AnswerGenerator, threshold: float = RELEVANCE_THRESHOLD):
        self.store = store
        self.embedder = embedder
        self.generator = generator
        self.threshold = threshold

    @classmethod
    def from_index(cls, index_dir: Path = INDEX_DIR, model_name: str = "all-MiniLM-L6-v2") -> "RAGAssistant":
        return cls(VectorStore.load(index_dir), SentenceTransformerEmbedder(model_name), create_generator())

    @classmethod
    def build_index(cls, pdf_path: Path, index_dir: Path = INDEX_DIR, model_name: str = "all-MiniLM-L6-v2") -> int:
        embedder = SentenceTransformerEmbedder(model_name)
        chunks = load_and_chunk(pdf_path)
        if not chunks:
            raise ValueError("No text could be extracted from the handbook")
        VectorStore(chunks, embedder.encode([chunk.text for chunk in chunks])).save(index_dir)
        return len(chunks)

    def ask(self, question: str) -> tuple[str, int | None, list[SearchResult]]:
        query_vector = self.embedder.encode([question])[0]
        candidates = self.store.search(query_vector, max(TOP_K, 12))
        query_terms = {term for term in re.findall(r"[a-z0-9]+", question.lower()) if term not in STOP_WORDS}
        results = sorted(
            candidates,
            key=lambda result: 0.35 * result.score + 0.65 * (
                len(query_terms & set(re.findall(r"[a-z0-9]+", result.chunk.text.lower()))) / max(len(query_terms), 1)
            ),
            reverse=True,
        )[:TOP_K]
        relevant = [result for result in results if result.score >= self.threshold]
        if not relevant:
            from app.config import NOT_AVAILABLE
            return NOT_AVAILABLE, None, []
        answer = self.generator.generate(question, relevant)
        return answer, relevant[0].chunk.page, relevant
