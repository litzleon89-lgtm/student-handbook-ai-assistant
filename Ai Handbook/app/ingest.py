import json
import re
from pathlib import Path

from pypdf import PdfReader

from app.models import Chunk


def normalize_pdf_text(text: str) -> str:
    text = text.replace("\u00c2\u00a0", " ").replace("\u00a0", " ")
    normalized_lines = []
    for line in text.splitlines():
        tokens = line.split()
        single_character_tokens = sum(len(token) == 1 for token in tokens)
        if len(tokens) >= 3 and single_character_tokens / len(tokens) >= 0.5:
            line = re.sub(r"(?<=\S) (?=\S)", "", line)
            line = re.sub(r" {2,}", " ", line)
        normalized_lines.append(line)
    return "\n".join(normalized_lines)


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"Handbook not found: {pdf_path}")
    reader = PdfReader(str(pdf_path))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = normalize_pdf_text(page.extract_text() or "")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if text:
            pages.append((page_number, text))
    return pages


def extract_accessible_pages(source_path: Path) -> list[tuple[int, str]]:
    records = json.loads(source_path.read_text(encoding="utf-8-sig"))
    return [(int(record["page"]), record["text"]) for record in records if record.get("text")]


def chunk_pages(pages: list[tuple[int, str]], chunk_size: int = 900, overlap: int = 150) -> list[Chunk]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller than chunk_size")

    chunks = []
    for page_number, text in pages:
        start = 0
        chunk_number = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                boundary = max(text.rfind("\n", start, end), text.rfind(" ", start, end))
                if boundary > start + chunk_size // 2:
                    end = boundary
            content = text[start:end].strip()
            if content:
                chunks.append(Chunk(content, page_number, f"p{page_number}-c{chunk_number}"))
                chunk_number += 1
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
    return chunks


def load_and_chunk(pdf_path: Path, chunk_size: int = 900, overlap: int = 150) -> list[Chunk]:
    pages = extract_pages(pdf_path)
    if not pages:
        accessible_source = pdf_path.with_name("handbook_accessible.json")
        if accessible_source.exists():
            pages = extract_accessible_pages(accessible_source)
    return chunk_pages(pages, chunk_size, overlap)
