"""Parse console markup into (position, text, tag) tuples.

``Tag`` and ``_parse`` are extracted from rich, rich/markup.py, as
they were before commit 71325bb ("Fix issue with markup escaping",
Darren Burns, 2022-02-10).

The bug: the tag regex ``[a-z#\\/@].*?`` lets ``.*?`` span across a
``[``, so in ``"[foo bar [baz] qux]"`` the whole ``"[foo bar [baz]"``
is read as one tag instead of plain text ``"[foo bar "`` followed by
the tag ``[baz]``.
"""

import re
from typing import Iterable, NamedTuple, Optional, Tuple

RE_TAGS = re.compile(
    r"""((\\*)\[([a-z#\/@].*?)\])""",
    re.VERBOSE,
)


class Tag(NamedTuple):
    """A tag in console markup."""

    name: str
    """The tag name. e.g. 'bold'."""
    parameters: Optional[str]
    """Any additional parameters after the name."""

    def __str__(self) -> str:
        return (
            f"[{self.name}]"
            if self.parameters is None
            else f"[{self.name}={self.parameters}]"
        )


def _parse(markup: str) -> Iterable[Tuple[int, Optional[str], Optional[Tag]]]:
    """Parse markup in to an iterable of tuples of (position, text, tag).

    Args:
        markup (str): A string containing console markup

    """
    position = 0
    _divmod = divmod
    _Tag = Tag
    for match in RE_TAGS.finditer(markup):
        full_text, escapes, tag_text = match.groups()
        start, end = match.span()
        if start > position:
            yield start, markup[position:start], None
        if escapes:
            backslashes, escaped = _divmod(len(escapes), 2)
            if backslashes:
                # Literal backslashes
                yield start, "\\" * backslashes, None
                start += backslashes * 2
            if escaped:
                # Escape of tag
                yield start, full_text[len(escapes) :], None
                position = end
                continue
        text, equals, parameters = tag_text.partition("=")
        yield start, None, _Tag(text, parameters if equals else None)
        position = end
    if position < len(markup):
        yield position, markup[position:], None
