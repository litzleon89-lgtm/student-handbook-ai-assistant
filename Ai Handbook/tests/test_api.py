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
