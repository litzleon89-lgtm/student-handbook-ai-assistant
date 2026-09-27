import csv
from pathlib import Path

from app.config import INDEX_DIR
from app.rag import RAGAssistant

QUESTIONS = [
    "What outcomes should I expect from the bootcamp?",
    "What laptop hardware and internet connection do I need?",
    "What topics are covered in weeks 1-10, 11-15, 16-20, 21-24, and 25-30?",
    "When does the bootcamp start and how long does it run?",
    "How do I get access to my learning calendar?",
    "When are the live classes and are they recorded?",
    "What tutor support is available if I cannot attend a live class?",
    "What are the bootcamp fees and payment deadlines?",
    "What career and placement support does Zaio provide?",
    "What communication channels should students use?",
]


def main() -> None:
    assistant = RAGAssistant.from_index(INDEX_DIR)
    output = Path("evaluation_results.csv")
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["question", "source", "answer"])
        writer.writeheader()
        for question in QUESTIONS:
            answer, page, _ = assistant.ask(question)
            writer.writerow({"question": question, "source": f"Page {page}" if page else "Not found", "answer": answer})
    print(f"Wrote {len(QUESTIONS)} results to {output}")


if __name__ == "__main__":
    main()
