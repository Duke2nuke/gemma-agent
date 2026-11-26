"""Encode a hostname for use in a URL.

Extracted from httpx, httpx/_urlparse.py, as it was before PR #2886
("Fix encode host", robinaly, 2023-10-10).

The bug: the IPv4-detection regex uses unescaped ``.`` which matches
any character, so a hostname like ``023b76x43144`` is treated as an
IPv4 address and rejected as invalid.

Note: the IDNA branch below is the real code; ``idna`` is imported
lazily here so this module stays importable without the package
installed. The tested paths (IPv4-style, valid IPv4, ASCII hostname)
are identical to the real code.
"""

import ipaddress
import re
import typing

try:
    import idna
except ImportError:  # pragma: no cover - the real code imports idna at module level
    idna = None

# https://datatracker.ietf.org/doc/html/rfc3986.html#section-2.3
UNRESERVED_CHARACTERS = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
)
SUB_DELIMS = "!$&'()*+,;="

PERCENT_ENCODED_REGEX = re.compile("%[A-Fa-f0-9]{2}")

# We use these simple regexs as a first pass before handing off to
# the stdlib 'ipaddress' module for IP address validation.
IPv4_STYLE_HOSTNAME = re.compile(r"^[0-9]+.[0-9]+.[0-9]+.[0-9]+$")
IPv6_STYLE_HOSTNAME = re.compile(r"^\[.*\]$")


class InvalidURL(Exception):
    pass


def percent_encode(char: str) -> str:
    return "".join([f"%{byte:02x}" for byte in char.encode("utf-8")]).upper()


def quote(string: str, safe: str = "/") -> str:
    NON_ESCAPED_CHARS = UNRESERVED_CHARACTERS + safe
    if string.count("%") == len(PERCENT_ENCODED_REGEX.findall(string)):
        NON_ESCAPED_CHARS += "%"

    return "".join(
        [char if char in NON_ESCAPED_CHARS else percent_encode(char) for char in string]
    )


def encode_host(host: str) -> str:
    if not host:
        return ""

    elif IPv4_STYLE_HOSTNAME.match(host):
        # Validate IPv4 hostnames like #.#.#.#
        #
        # From https://datatracker.ietf.org/doc/html/rfc3986/#section-3.2.2
        #
        # IPv4address = dec-octet "." dec-octet "." dec-octet "." dec-octet
        try:
            ipaddress.IPv4Address(host)
        except ipaddress.AddressValueError:
            raise InvalidURL(f"Invalid IPv4 address: {host!r}")
        return host

    elif IPv6_STYLE_HOSTNAME.match(host):
        # Validate IPv6 hostnames like [...]
        #
        # From https://datatracker.ietf.org/doc/html/rfc3986/#section-3.2.2
        #
        # "A host identified by an Internet Protocol literal address, version 6
        # [RFC3513] or later, is distinguished by enclosing the IP literal
        # within square brackets ("[" and "]").  This is the only place where
        # square bracket characters are allowed in the URI syntax."
        try:
            ipaddress.IPv6Address(host[1:-1])
        except ipaddress.AddressValueError:
            raise InvalidURL(f"Invalid IPv6 address: {host!r}")
        return host[1:-1]

    elif host.isascii():
        # Regular ASCII hostnames
        #
        # From https://datatracker.ietf.org/doc/html/rfc3986/#section-3.2.2
        #
        # reg-name    = *( unreserved / pct-encoded / sub-delims )
        return quote(host.lower(), safe=SUB_DELIMS)

    # IDNA hostnames
    if idna is None:
        raise InvalidURL(f"Invalid IDNA hostname: {host!r}")
    try:
        return idna.encode(host.lower()).decode("ascii")
    except idna.IDNAError:
        raise InvalidURL(f"Invalid IDNA hostname: {host!r}")
