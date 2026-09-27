from app.ingest import chunk_pages, normalize_pdf_text


def test_normalizes_character_spaced_pdf_text():
    assert normalize_pdf_text("W h a t  t i m e  a r e  c l a s s e s?") == "What time are classes?"
    assert normalize_pdf_text("C S S   G i t  Projects \u00c2\u00a0 Outcomes") == "CSS Git Projects Outcomes"


def test_chunks_preserve_page_and_overlap():
    chunks = chunk_pages([(7, "one two three four five six seven eight nine ten")], chunk_size=25, overlap=5)
    assert chunks
    assert all(chunk.page == 7 for chunk in chunks)
    assert chunks[0].chunk_id == "p7-c0"
    assert any(first.text[-5:] in second.text for first, second in zip(chunks, chunks[1:]))


def test_invalid_chunk_settings_raise():
    try:
        chunk_pages([(1, "text")], chunk_size=10, overlap=10)
    except ValueError:
        return
    raise AssertionError("Expected invalid overlap to raise")
