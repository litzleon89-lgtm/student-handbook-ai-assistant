import argparse
import csv
from pathlib import Path

from app.config import INDEX_DIR, LLM_ENABLED
from app.generator import ContextAnswerGenerator
from app.rag import RAGAssistant

QUESTIONS = [
    "What outcomes should I expect from the bootcamp?",
    "What laptop hardware and internet connection do I need?",
    "What technologies are taught in the curriculum sprint plan, from HTML and CSS to React, NodeJS, Python, and AI agents?",
    "When does the bootcamp start and how long does it run?",
    "How do I get access to my learning calendar?",
    "When are the live classes and are they recorded?",
    "What tutor support is available if I cannot attend a live class?",
    "What are the bootcamp fees and payment deadlines?",
    "What career and placement support does Zaio provide?",
    "What communication channels should students use?",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate handbook retrieval and answers.")
    parser.add_argument(
        "--passages-only",
        action="store_true",
        help="Use retrieved handbook passages without calling the OpenAI API.",
    )
    args = parser.parse_args()
    assistant = RAGAssistant.from_index(
        INDEX_DIR,
        generator=ContextAnswerGenerator() if args.passages_only else None,
    )
    output = Path("evaluation_results.csv")
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["question", "source", "answer", "answer_mode"])
        writer.writeheader()
        for question in QUESTIONS:
            answer, page, _ = assistant.ask(question)
            generator = assistant.generator
            used_fallback = isinstance(generator, ContextAnswerGenerator) or getattr(generator, "last_used_fallback", False)
            answer_mode = "retrieved_passage_fallback" if used_fallback else "openai_llm"
            writer.writerow({
                "question": question,
                "source": f"Page {page}" if page else "Not found",
                "answer": answer,
                "answer_mode": answer_mode,
            })
    print(f"Wrote {len(QUESTIONS)} results to {output}")


if __name__ == "__main__":
    main()
