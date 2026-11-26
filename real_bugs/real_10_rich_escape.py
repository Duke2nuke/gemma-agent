"""Escape text so that it won't be interpreted as markup.

Extracted from rich, rich/markup.py, as it was before commit 4b1293e
("fix at escape", Will McGugan, 2022-01-02, released in 10.16.2).

The bug: the escape regex does not include ``@`` in the set of tag
starting characters, so ``[@click]...[/@click]`` markup is not
escaped and is still interpreted as a tag.
"""

import re
from typing import Callable, Match

_ReStringMatch = Match[str]  # regex match object
_ReSubCallable = Callable[[_ReStringMatch], str]  # Callable invoked by re.sub
_EscapeSubMethod = Callable[[_ReSubCallable, str], str]  # Sub method of a compiled re


def escape(
    markup: str, _escape: _EscapeSubMethod = re.compile(r"(\\*)(\[[a-z#\/].*?\])").sub
) -> str:
    """Escapes text so that it won't be interpreted as markup.

    Args:
        markup (str): Content to be inserted in to markup.

    Returns:
        str: Markup with square brackets escaped.
    """

    def escape_backslashes(match: Match[str]) -> str:
        """Called by re.sub replace matches."""
        backslashes, text = match.groups()
        return f"{backslashes}{backslashes}\\{text}"

    markup = _escape(escape_backslashes, markup)
    return markup
