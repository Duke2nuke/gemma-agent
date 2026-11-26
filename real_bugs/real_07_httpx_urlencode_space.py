"""Build a query string from (key, value) pairs.

The ``quote`` / ``percent_encode`` helpers are extracted from httpx,
httpx/_urlparse.py. ``build_query_string`` is the behaviour of
``QueryParams.__str__`` in httpx/_urls.py as it was before PR #2543
("Use '%20' for encoding spaces in query parameters", Tom Christie,
2023-01-10, fixing issue #2536).

The bug: the standard library ``urlencode`` is used, which encodes
spaces as ``+`` and ``/`` as ``%2F``. httpx query parameters should
use the same escaping as the rest of the URL: spaces as ``%20`` and
``/`` left alone.
"""

import re
import typing
from urllib.parse import urlencode as _stdlib_urlencode

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


def build_query_string(items: typing.List[typing.Tuple[str, str]]) -> str:
    return _stdlib_urlencode(items)
