import os
from typing import Protocol

from app.config import NOT_AVAILABLE, OPENAI_MODEL
from app.models import SearchResult


class AnswerGenerator(Protocol):
    def generate(self, question: str, results: list[SearchResult]) -> str: ...


class OpenAIAnswerGenerator:
    def __init__(self, model: str = OPENAI_MODEL):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model

    def generate(self, question: str, results: list[SearchResult]) -> str:
        context = "\n\n".join(f"[Page {r.chunk.page}] {r.chunk.text}" for r in results)
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": "Answer only from the supplied handbook context. If the context does not answer the question, reply exactly: " + NOT_AVAILABLE},
                {"role": "user", "content": f"Handbook context:\n{context}\n\nQuestion: {question}"},
            ],
        )
        return response.choices[0].message.content.strip()


class ContextAnswerGenerator:
    """Offline fallback that returns the best handbook passage without inventing facts."""

    def generate(self, question: str, results: list[SearchResult]) -> str:
        if not results:
            return NOT_AVAILABLE
        passages = []
        for result in results[:2]:
            passages.append(f"[Page {result.chunk.page}] {result.chunk.text}")
        return "\n\n".join(passages)


def create_generator() -> AnswerGenerator:
    return OpenAIAnswerGenerator() if os.getenv("OPENAI_API_KEY") else ContextAnswerGenerator()
