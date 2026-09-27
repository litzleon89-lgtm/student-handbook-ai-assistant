from typing import Protocol

from app.config import (
    LLM_ENABLED,
    LLM_PROVIDER,
    NOT_AVAILABLE,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OPENAI_MODEL,
)
from app.models import SearchResult


class AnswerGenerator(Protocol):
    def generate(self, question: str, results: list[SearchResult]) -> str: ...


class OpenAIAnswerGenerator:
    def __init__(self, model: str = OPENAI_MODEL, base_url: str | None = None, api_key: str | None = None):
        from openai import OpenAI
        client_options = {}
        if base_url:
            client_options["base_url"] = base_url
        if api_key:
            client_options["api_key"] = api_key
        self.client = OpenAI(**client_options)
        self.model = model
        self.last_used_fallback = False

    def generate(self, question: str, results: list[SearchResult]) -> str:
        context = "\n\n".join(f"[Page {r.chunk.page}] {r.chunk.text}" for r in results)
        self.last_used_fallback = False
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": "Answer only from the supplied handbook context. If the context does not answer the question, reply exactly: " + NOT_AVAILABLE},
                    {"role": "user", "content": f"Handbook context:\n{context}\n\nQuestion: {question}"},
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as error:
            from openai import OpenAIError
            if not isinstance(error, OpenAIError):
                raise
            self.last_used_fallback = True
            return ContextAnswerGenerator().generate(question, results)


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
    if LLM_PROVIDER == "ollama":
        return OpenAIAnswerGenerator(
            model=OLLAMA_MODEL,
            base_url=f"{OLLAMA_BASE_URL.rstrip('/')}/v1",
            api_key="ollama",
        )
    if LLM_ENABLED and LLM_PROVIDER == "openai":
        return OpenAIAnswerGenerator()
    return ContextAnswerGenerator()
