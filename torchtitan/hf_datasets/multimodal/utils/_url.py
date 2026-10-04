# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""URL validation and fetching for multimodal image loading.

The image decoder accepts HTTP(S) URLs from dataset samples. Those URLs are
untrusted input, so fetching them must not reach private, loopback, or
otherwise non-public hosts (SSRF defense). This module validates the URL
*before* any connection is opened and re-validates every redirect hop, since
a redirect can silently move the request to a host that was safe on the
first hop.
"""

import ipaddress
import logging
import socket
from collections.abc import Iterable
from ipaddress import IPv4Address, IPv6Address
from urllib.parse import urljoin, urlparse

import requests

logger = logging.getLogger(__name__)

# Only http and https are allowed. file:// and other schemes could read local
# files or reach internal services through non-network transports.
_ALLOWED_SCHEMES = frozenset({"http", "https"})

# Hostnames that should never be fetched regardless of what they resolve to.
_BLOCKED_HOSTNAMES = frozenset({"localhost", "ip6-localhost", "ip6-loopback"})

# Cap on redirect hops to avoid unbounded fetches on redirect loops.
_MAX_REDIRECTS = 10


def _is_blocked_ip(ip: IPv4Address | IPv6Address) -> bool:
    """Return True if the resolved address is private/loopback/etc.

    Covers IPv4 and IPv6, including IPv4-mapped IPv6 addresses (e.g.
    ``::ffff:127.0.0.1``), which ``ipaddress`` normalizes to an IPv4Address
    and so are caught by the IPv4 checks below.
    """
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _resolve_hostnames(hostnames: Iterable[str]) -> set[IPv4Address | IPv6Address]:
    """Resolve each hostname to its addresses, skipping unresolvable names.

    DNS failures are swallowed so a bad hostname is reported as a blocked URL
    rather than as a confusing socket error.
    """
    addresses: set[IPv4Address | IPv6Address] = set()
    for hostname in hostnames:
        try:
            infos = socket.getaddrinfo(hostname, None)
        except socket.gaierror:
            continue
        for _, _, _, _, sockaddr in infos:
            addresses.add(ipaddress.ip_address(sockaddr[0]))
    return addresses


def is_safe_url(url: str) -> bool:
    """Return True if ``url`` is an http(s) URL to a public address.

    The hostname is resolved and every resolved address is checked against
    private/loopback/link-local/multicast/reserved/unspecified ranges. A URL
    is rejected if the scheme is not http(s), the hostname is a known
    loopback alias, or any resolved address is blocked.
    """
    parsed = urlparse(url)
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        return False
    if not parsed.hostname:
        return False
    if parsed.hostname.lower() in _BLOCKED_HOSTNAMES:
        return False
    addresses = _resolve_hostnames({parsed.hostname})
    if not addresses:
        # Nothing resolved: do not attempt the fetch.
        return False
    return not any(_is_blocked_ip(ip) for ip in addresses)


def fetch_url(url: str, *, timeout: float = 10) -> requests.Response:
    """Fetch ``url`` after validating it is a safe, public http(s) URL.

    Redirects are followed, but every hop is re-validated before the
    connection is opened, so a redirect cannot move the request onto a host
    that was rejected on the first hop.

    Raises:
        ValueError: if the URL (or any redirect target) is not a public
            http(s) URL, or the redirect chain loops or exceeds the hop cap.
        requests.RequestException: on network/HTTP failures.
    """
    seen: set[str] = set()
    for _ in range(_MAX_REDIRECTS + 1):
        if not is_safe_url(url):
            raise ValueError(
                f"Refusing to fetch {url!r}: only public http(s) URLs are "
                f"allowed, and the resolved address must not be private, "
                f"loopback, link-local, multicast, reserved, or unspecified."
            )
        response = requests.get(url, timeout=timeout, allow_redirects=False)
        if not (response.is_redirect or response.is_permanent_redirect):
            return response
        location = response.headers.get("Location")
        if not location:
            raise ValueError(
                f"Redirect from {url!r} has no Location header; refusing."
            )
        url = urljoin(url, location)
        if url in seen:
            raise ValueError(f"Redirect loop detected at {url!r}; refusing.")
        seen.add(url)
    raise ValueError(
        f"Redirect chain exceeded {_MAX_REDIRECTS} hops; refusing to fetch."
    )