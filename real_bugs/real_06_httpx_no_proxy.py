"""Get proxy information from the environment.

Extracted from httpx, httpx/_utils.py, as it was before PR #2659
("fix: NO_PROXY should support IPv4, IPv6 and localhost",
Jiayun Shen, 2023-04-19).

The bug: every NO_PROXY entry is treated as a domain suffix, so
``NO_PROXY=127.0.0.1`` becomes the mount ``all://*127.0.0.1`` instead
of ``all://127.0.0.1``; IPv6 addresses and ``localhost`` are wrong
the same way.
"""

import os
import typing
from urllib.request import getproxies


def get_environment_proxies() -> typing.Dict[str, typing.Optional[str]]:
    """Gets proxy information from the environment"""

    # urllib.request.getproxies() falls back on System
    # Registry and Config for proxies on Windows and macOS.
    # We don't want to propagate non-HTTP proxies into
    # our configuration such as 'TRAVIS_APT_PROXY'.
    proxy_info = getproxies()
    mounts: typing.Dict[str, typing.Optional[str]] = {}

    for scheme in ("http", "https", "all"):
        if proxy_info.get(scheme):
            hostname = proxy_info[scheme]
            mounts[f"{scheme}://"] = (
                hostname if "://" in hostname else f"http://{hostname}"
            )

    no_proxy_hosts = [host.strip() for host in proxy_info.get("no", "").split(",")]
    for hostname in no_proxy_hosts:
        # See https://curl.haxx.se/libcurl/c/CURLOPT_NOPROXY.html for details
        # on how names in `NO_PROXY` are handled.
        if hostname == "*":
            # If NO_PROXY=* is used or if "*" occurs as any one of the comma
            # separated hostnames, then we should just bypass any information
            # from HTTP_PROXY, HTTPS_PROXY, ALL_PROXY, and always ignore
            # proxies.
            return {}
        elif hostname:
            # NO_PROXY=.google.com is marked as "all://*.google.com,
            #   which disables "www.google.com" but not "google.com"
            # NO_PROXY=google.com is marked as "all://*google.com,
            #   which disables "www.google.com" and "google.com".
            #   (But not "wwwgoogle.com")
            mounts[f"all://*{hostname}"] = None

    return mounts
