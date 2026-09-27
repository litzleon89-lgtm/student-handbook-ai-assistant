import numpy as np

from app.config import NOT_AVAILABLE
from openai import OpenAIError

from app.generator import ContextAnswerGenerator, OpenAIAnswerGenerator
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


def test_openai_errors_fall_back_to_retrieved_handbook_passage():
    class FailingCompletions:
        def create(self, **kwargs):
            raise OpenAIError("temporarily unavailable")

    class FakeClient:
        class chat:
            completions = FailingCompletions()

    generator = OpenAIAnswerGenerator.__new__(OpenAIAnswerGenerator)
    generator.client = FakeClient()
    generator.model = "test-model"
    result = SearchResult(Chunk("Bootcamp outcomes include programming fundamentals.", 4, "p4-c0"), 0.9)

    assert generator.generate("What are the outcomes?", [result]) == "[Page 4] Bootcamp outcomes include programming fundamentals."


def test_openai_generator_uses_retrieved_context():
    class FakeCompletions:
        def create(self, **kwargs):
            assert kwargs["messages"][1]["content"].find("Attendance requires 80 percent") >= 0
            return type("Response", (), {
                "choices": [type("Choice", (), {"message": type("Message", (), {"content": "The requirement is 80 percent."})()})()]
            })()

    class FakeClient:
        class chat:
            completions = FakeCompletions()

    generator = OpenAIAnswerGenerator.__new__(OpenAIAnswerGenerator)
    generator.client = FakeClient()
    generator.model = "test-model"
    result = SearchResult(Chunk("Attendance requires 80 percent participation.", 12, "p12-c0"), 0.9)

    assert generator.generate("What is attendance?", [result]) == "The requirement is 80 percent."
    assert generator.last_used_fallback is False


def test_chroma_vector_store_persists_and_reloads(tmp_path):
    chunks = [Chunk("Attendance requires 80 percent participation.", 12, "p12-c0")]
    store = VectorStore(chunks, np.array([[1.0, 0.0]]))
    store.save(tmp_path)

    loaded = VectorStore.load(tmp_path)
    results = loaded.search(np.array([1.0, 0.0]), top_k=1)

    assert len(results) == 1
    assert results[0].chunk.text == chunks[0].text
    assert results[0].chunk.page == 12
    assert results[0].score > 0.99
