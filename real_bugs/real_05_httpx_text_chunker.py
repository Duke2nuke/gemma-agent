"""Split decoded text content into fixed-size chunks.

Extracted from httpx, httpx/_decoders.py, as it was before commit
1e11096 ("Fixed `iter_text` adding an empty string", James Braza,
2023-12-11, fixing issue #2998).

The bug: ``decode("")`` returns ``[""]`` instead of ``[]``, so
iterating the text of an empty response yields one empty string.
"""

import io
import typing


class TextChunker:
    """
    Handles returning text content in fixed-size chunks.
    """

    def __init__(self, chunk_size: typing.Optional[int] = None) -> None:
        self._buffer = io.StringIO()
        self._chunk_size = chunk_size

    def decode(self, content: str) -> typing.List[str]:
        if self._chunk_size is None:
            return [content]

        self._buffer.write(content)
        if self._buffer.tell() >= self._chunk_size:
            value = self._buffer.getvalue()
            chunks = [
                value[i : i + self._chunk_size]
                for i in range(0, len(value), self._chunk_size)
            ]
            if len(chunks[-1]) == self._chunk_size:
                self._buffer.seek(0)
                self._buffer.truncate()
                return chunks
            else:
                self._buffer.seek(0)
                self._buffer.write(chunks[-1])
                self._buffer.truncate()
                return chunks[:-1]
        else:
            return []

    def flush(self) -> typing.List[str]:
        value = self._buffer.getvalue()
        self._buffer.seek(0)
        self._buffer.truncate()
        return [value] if value else []
