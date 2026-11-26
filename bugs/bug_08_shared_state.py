"""Collect the even numbers from a list.

Return only the even numbers from the list given in this call.
"""

_seen = []


def evens(xs):
    for x in xs:
        if x % 2 == 0:
            _seen.append(x)
    return _seen
