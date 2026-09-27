from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import HANDBOOK_PATH, INDEX_DIR
from app.rag import RAGAssistant

app = FastAPI(title="Student Handbook AI Assistant", version="1.0.0")
assistant: RAGAssistant | None = None


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)


class AskResponse(BaseModel):
    answer: str
    source: str | None


def get_assistant() -> RAGAssistant:
    global assistant
    if assistant is None:
        try:
            assistant = RAGAssistant.from_index(INDEX_DIR)
        except FileNotFoundError as error:
            raise HTTPException(status_code=503, detail="The handbook index is not available. Run the ingestion command first.") from error
    return assistant


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "index": "ready" if INDEX_DIR.exists() else "missing"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be blank")
    answer, page, _ = get_assistant().ask(question)
    return AskResponse(answer=answer, source=f"Page {page}" if page else None)


def require_handbook() -> Path:
    if not HANDBOOK_PATH.exists():
        raise SystemExit(f"Handbook not found at {HANDBOOK_PATH}. Add the PDF and try again.")
    return HANDBOOK_PATH
