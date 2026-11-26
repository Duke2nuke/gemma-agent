"""Coerce a primitive data type into a string value.

Extracted from httpx, httpx/_utils.py, as it was before PR #2523
("Raise `TypeError` on invalid query params", Tom Christie,
2022-12-30).

The bug: any value at all is silently stringified, so
``QueryParams({"a": b"bytes"})`` produces ``a=b'bytes'`` instead of
raising a ``TypeError``.
"""


def primitive_value_to_str(value):
    """
    Coerce a primitive data type into a string value.

    Note that we prefer JSON-style 'true'/'false' for boolean values here.
    """
    if value is True:
        return "true"
    elif value is False:
        return "false"
    elif value is None:
        return ""
    return str(value)
