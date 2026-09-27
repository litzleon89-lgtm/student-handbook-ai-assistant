import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from app import main


class StubAssistant:
    def ask(self, question):
        return "Attendance requires 80 percent participation.", 12, []


def test_ask_endpoint_returns_json(monkeypatch):
    monkeypatch.setattr(main, "assistant", StubAssistant())
    response = TestClient(main.app).post("/ask", json={"question": " attendance? "})
    assert response.status_code == 200
    assert response.json() == {"answer": "Attendance requires 80 percent participation.", "source": "Page 12"}


def test_ask_endpoint_rejects_blank_question(monkeypatch):
    monkeypatch.setattr(main, "assistant", StubAssistant())
    response = TestClient(main.app).post("/ask", json={"question": "   "})
    assert response.status_code == 400
    assert "blank" in response.json()["detail"]


def test_health_reports_answer_mode():
    response = TestClient(main.app).get("/health")
    assert response.status_code == 200
    assert response.json()["answer_mode"] in {
        "openai_configured_with_fallback",
        "passage_only_no_api_key",
    }


def test_docs_and_custom_favicon_are_served():
    client = TestClient(main.app)
    docs_response = client.get("/docs")
    favicon_response = client.get("/favicon.svg")

    assert docs_response.status_code == 200
    assert "/favicon.svg" in docs_response.text
    assert favicon_response.status_code == 200
    assert "<svg" in favicon_response.text
