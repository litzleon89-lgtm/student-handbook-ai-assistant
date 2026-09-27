# Student Handbook AI Assistant

A retrieval-augmented generation (RAG) API that answers student questions from a handbook PDF and returns the supporting page.

Source handbook: `data/handbook.pdf` (October 2025 Fullstack AI Engineer Bootcamp handbook, 26 pages).

## Features

- Extracts text from every PDF page with `pypdf`.
- Splits text into overlapping, page-aware chunks.
- Generates `all-MiniLM-L6-v2` sentence embeddings.
- Stores vectors in a local NumPy vector store with JSON chunk metadata.
- Retrieves the most relevant chunks using cosine similarity.
- Uses an OpenAI chat model when `OPENAI_API_KEY` is configured.
- Uses a safe offline passage-only fallback when no API key is configured.
- Returns a fixed not-available response when retrieval is below the relevance threshold.
- Provides JSON requests and responses for future n8n integration.

## Setup

1. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. The October 2025 handbook PDF is included at `data/handbook.pdf`. Its text layer is normalized during extraction to handle character-spaced PDF text.

3. Build the index:

   ```powershell
   python -m scripts.ingest
   ```

   The embedding model is downloaded on first use. To use another model, set `EMBEDDING_MODEL`.

4. Optional: create `.env` with an OpenAI key for LLM-generated answers:

   ```text
   OPENAI_API_KEY=your-key
   OPENAI_MODEL=gpt-4o-mini
   ```

5. Start the API:

   ```powershell
   uvicorn app.main:app --reload
   ```

## API

`POST /ask`

Request:

```json
{"question":"What are the live class times?"}
```

Response:

```json
{"answer":"...","source":"Page 11"}
```

Invalid JSON or missing/empty questions receive a 4xx response. If the index has not been built, the API returns `503` with an actionable error. `GET /health` reports whether the index directory exists.
## Testing

Run the unit tests:

```powershell
pytest -q
```

The tests cover chunking, page metadata, vector retrieval, not-available behavior, JSON responses, and invalid questions. They do not require a PDF, embedding download, or OpenAI key.

After indexing the real handbook, generate the required 10-question evidence table:

```powershell
python -m scripts.evaluate
```

This writes `evaluation_results.csv` with `question`, `source`, and `answer` columns. Questions cover outcomes, laptop requirements, curriculum, dates, the learning calendar, classes, tutor support, fees, placement, and communications.

## Project structure

```text
app/                 Application and RAG components
data/                Put handbook.pdf here
scripts/ingest.py    Extract, chunk, embed, and persist the index
scripts/evaluate.py  Run ten documented questions
tests/                Unit and API tests
artifacts/index/     Generated vector database files (ignored by git)
```
