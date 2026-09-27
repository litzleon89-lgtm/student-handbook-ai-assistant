import numpy as np

from app.config import NOT_AVAILABLE
from app.generator import ContextAnswerGenerator
from app.models import Chunk, SearchResult
from app.rag import RAGAssistant
from app.vector_store import VectorStore


class FakeEmbedder:
    def encode(self, texts):
        return np.array([[1.0, 0.0] if "attendance" in text.lower() else [0.0, 1.0] for text in texts])


class FakeGenerator:
    def generate(self, question, results):
        return f"Handbook answer: {results[0].chunk.text}"


def make_assistant(threshold=0.3):
    chunks = [Chunk("Attendance requires 80 percent participation.", 12, "p12-c0"), Chunk("The library closes at 8 PM.", 4, "p4-c0")]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
    return RAGAssistant(VectorStore(chunks, vectors), FakeEmbedder(), FakeGenerator(), threshold)


def test_ask_returns_answer_and_page():
    answer, page, results = make_assistant().ask("What is the attendance requirement?")
    assert "80 percent" in answer
    assert page == 12
    assert results[0].chunk.page == 12


def test_ask_returns_not_available_when_below_threshold():
    answer, page, results = make_assistant(threshold=1.01).ask("What is the attendance requirement?")
    assert answer == NOT_AVAILABLE
    assert page is None
    assert results == []


def test_offline_generator_keeps_adjacent_chunks_from_same_page():
    results = [
        SearchResult(Chunk("CPU and memory requirements", 5, "p5-c0"), 0.9),
        SearchResult(Chunk("Storage and internet requirements", 5, "p5-c1"), 0.8),
    ]
    answer = ContextAnswerGenerator().generate("What hardware do I need?", results)
    assert "CPU and memory" in answer
    assert "Storage and internet" in answer
