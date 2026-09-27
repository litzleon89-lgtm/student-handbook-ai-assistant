from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from pydantic import BaseModel, Field
from starlette.responses import FileResponse, HTMLResponse

from app.config import HANDBOOK_PATH, INDEX_DIR, LLM_ENABLED, LLM_PROVIDER
from app.rag import RAGAssistant

app = FastAPI(title="Student Handbook AI Assistant", version="1.0.0", docs_url=None)
assistant: RAGAssistant | None = None
FAVICON_PATH = Path(__file__).parent / "static" / "favicon.svg"


@app.get("/favicon.svg", include_in_schema=False)
def favicon() -> FileResponse:
    return FileResponse(FAVICON_PATH, media_type="image/svg+xml")


@app.get("/docs", include_in_schema=False)
def docs() -> HTMLResponse:
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - API docs",
        swagger_favicon_url="/favicon.svg",
    )


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
    return {
        "status": "ok",
        "index": "ready" if INDEX_DIR.exists() else "missing",
        "answer_mode": (
            "ollama_local_with_fallback" if LLM_PROVIDER == "ollama"
            else "openai_configured_with_fallback" if LLM_ENABLED
            else "passage_only_no_llm"
        ),
    }


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
