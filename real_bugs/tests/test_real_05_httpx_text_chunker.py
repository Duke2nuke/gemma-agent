import real_05_httpx_text_chunker as mod


def test_empty_content_yields_nothing():
    assert mod.TextChunker().decode("") == []


def test_plain_content():
    assert mod.TextChunker().decode("hello") == ["hello"]


def test_chunked_content():
    chunker = mod.TextChunker(chunk_size=5)
    assert chunker.decode("hello world") == ["hello", " worl"]
    assert chunker.flush() == ["d"]
