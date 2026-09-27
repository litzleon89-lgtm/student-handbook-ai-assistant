from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
HANDBOOK_PATH = Path(os.getenv("HANDBOOK_PATH", ROOT_DIR / "data" / "handbook.pdf"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", ROOT_DIR / "artifacts" / "index"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai" if os.getenv("OPENAI_API_KEY") else "none").lower()
LLM_ENABLED = LLM_PROVIDER == "ollama" or (LLM_PROVIDER == "openai" and bool(os.getenv("OPENAI_API_KEY")))
TOP_K = int(os.getenv("TOP_K", "4"))
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.30"))
NOT_AVAILABLE = "I am sorry, that information is not available in the handbook."
