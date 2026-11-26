"""Return the Requests tuple auth for a given url from netrc.

Extracted from requests, src/requests/utils.py, as it was before
PR #7205 ("Fix empty netrc entry usage", Nate Prewitt, 2026-02-12).

The bug: an empty netrc entry (e.g. a bare ``default`` line, which
parses to ``(None, None, None)``) is truthy as a tuple, so
``get_netrc_auth`` returns ``(None, None)`` instead of ``None``.
"""

import os
from urllib.parse import urlparse

NETRC_FILES = (".netrc", "_netrc")


def get_netrc_auth(url, raise_errors=False):
    """Returns the Requests tuple auth for a given url from netrc."""

    netrc_file = os.environ.get("NETRC")
    if netrc_file is not None:
        netrc_locations = (netrc_file,)
    else:
        netrc_locations = (f"~/{f}" for f in NETRC_FILES)

    try:
        from netrc import NetrcParseError, netrc

        netrc_path = None

        for f in netrc_locations:
            loc = os.path.expanduser(f)
            if os.path.exists(loc):
                netrc_path = loc
                break

        # Abort early if there isn't one.
        if netrc_path is None:
            return

        ri = urlparse(url)
        host = ri.hostname

        try:
            _netrc = netrc(netrc_path).authenticators(host)
            if _netrc:
                # Return with login / password
                login_i = 0 if _netrc[0] else 1
                return (_netrc[login_i], _netrc[2])
        except (NetrcParseError, OSError):
            # If there was a parsing error or a permissions issue reading the file,
            # we'll just skip netrc auth unless explicitly asked to raise errors.
            if raise_errors:
                raise

    # App Engine hackiness.
    except (ImportError, AttributeError):
        pass
