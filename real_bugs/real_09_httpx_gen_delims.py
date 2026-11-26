"""Percent-encode the path, query and fragment of a URL.

The ``quote`` / ``percent_encode`` helpers are extracted from httpx,
httpx/_urlparse.py. ``encode_url_parts`` is the quoting step of
``urlparse`` in the same file, as it was before PR #2701 ("Fix for
gen-delims escaping behaviour in path/query/fragment", Tom Christie,
2023-05-09).

The bug: the ``safe`` character sets passed to ``quote`` are wrong
for each component. ``:`` ``[`` ``]`` ``@`` are dropped from the path
set, ``:`` ``[`` ``]`` ``@`` from the query set, and ``:`` ``#``
``[`` ``]`` ``@`` from the fragment set, so characters that are legal
delimiters in that component get percent-encoded.
"""

import re
import typing

# https://datatracker.ietf.org/doc/html/rfc3986.html#section-2.3
UNRESERVED_CHARACTERS = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
)
SUB_DELIMS = "!$&'()*+,;="

PERCENT_ENCODED_REGEX = re.compile("%[A-Fa-f0-9]{2}")


def percent_encode(char: str) -> str:
    """
    Replace every character in a string with the percent-encoded representation.

    Characters outside the ASCII range are represented with their a percent-encoded
    representation of their UTF-8 byte sequence.

    For example:

        percent_encode(" ") == "%20"
    """
    return "".join([f"%{byte:02x}" for byte in char.encode("utf-8")]).upper()


def quote(string: str, safe: str = "/") -> str:
    NON_ESCAPED_CHARS = UNRESERVED_CHARACTERS + safe
    if string.count("%") == len(PERCENT_ENCODED_REGEX.findall(string)):
        # If all occurances of '%' are valid '%xx' escapes, then treat
        # percent as a non-escaping character.
        NON_ESCAPED_CHARS += "%"

    return "".join(
        [char if char in NON_ESCAPED_CHARS else percent_encode(char) for char in string]
    )


def encode_url_parts(
    path: str,
    query: typing.Optional[str],
    fragment: typing.Optional[str],
) -> typing.Tuple[str, typing.Optional[str], typing.Optional[str]]:
    """Percent-encode a URL's path, query and fragment components."""
    parsed_path: str = quote(path, safe=SUB_DELIMS + ":@/")
    parsed_query: typing.Optional[str] = (
        None if query is None else quote(query, safe=SUB_DELIMS + "/?")
    )
    parsed_fragment: typing.Optional[str] = (
        None if fragment is None else quote(fragment, safe=SUB_DELIMS + "/?")
    )
    return parsed_path, parsed_query, parsed_fragment
