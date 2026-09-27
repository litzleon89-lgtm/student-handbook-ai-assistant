from app.config import HANDBOOK_PATH, INDEX_DIR, EMBEDDING_MODEL
from app.main import require_handbook
from app.rag import RAGAssistant


if __name__ == "__main__":
    handbook = require_handbook()
    count = RAGAssistant.build_index(handbook, INDEX_DIR, EMBEDDING_MODEL)
    print(f"Indexed {count} chunks from {handbook} into {INDEX_DIR}")
