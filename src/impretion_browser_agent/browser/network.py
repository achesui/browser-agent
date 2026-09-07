"""Public-destination network policy for the worker (T021).

Mirrors ``network_policy.rs`` exactly: only HTTPS to globally public
destinations is allowed. Loopback, private, link-local, multicast, reserved,
documentation, shared, benchmark and unspecified addresses are blocked, as is
the ``localhost`` name. A destination is allowed only when its hostname
passes the name gate and *every* resolved IP is public, so a redirect or DNS
rebinding that mixes in one non-public address blocks the whole destination.

Resolution happens here, at request time, with a fresh lookup per request:
re-resolving on the desktop would only add a TOCTOU gap. Reasons are static
strings that never echo the URL or host.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable
from typing import Any

NETWORK_POLICY_VERSION = 2

_REASON_NOT_HTTPS = "browser URL is not an HTTPS destination"
_REASON_NON_PUBLIC = "browser URL host is not a public destination"


class NetworkBlocked(Exception):
    """A destination violates the policy. Carries only a static reason."""

    def __init__(self, reason: str = _REASON_NON_PUBLIC) -> None:
        super().__init__(reason)
        self.reason = reason


def is_public_ipv4(addr: ipaddress.IPv4Address) -> bool:
    """Explicit ranges mirroring the desktop policy (no version-dependent
    stdlib classifiers)."""
    raw = int(addr)
    a = raw >> 24
    b = (raw >> 16) & 0xFF
    c = (raw >> 8) & 0xFF
    return not (
        raw == 0  # 0.0.0.0 (unspecified)
        or a == 0  # 0.0.0.0/8 ("this network")
        or a == 127  # 127.0.0.0/8 (loopback)
        or a == 10  # 10.0.0.0/8 (private)
        or (a == 172 and 16 <= b <= 31)  # 172.16.0.0/12 (private)
        or (a == 192 and b == 168)  # 192.168.0.0/16 (private)
        or (a == 169 and b == 254)  # 169.254.0.0/16 (link-local)
        or raw == 0xFFFFFFFF  # 255.255.255.255 (broadcast)
        or a >= 224 and a <= 239  # 224.0.0.0/4 (multicast)
        or (a == 192 and b == 0 and c == 2)  # 192.0.2.0/24 (documentation)
        or (a == 198 and b == 51 and c == 100)  # 198.51.100.0/24 (documentation)
        or (a == 203 and b == 0 and c == 113)  # 203.0.113.0/24 (documentation)
        or a >= 240  # 240.0.0.0/4 (reserved)
        or (a == 100 and (b & 0b1100_0000) == 0b0100_0000)  # 100.64.0.0/10 (shared)
        or (a == 198 and b in (18, 19))  # 198.18.0.0/15 (benchmarking)
    )


def is_public_ipv6(addr: ipaddress.IPv6Address) -> bool:
    """Explicit ranges mirroring the desktop policy."""
    # IPv4-mapped addresses inherit the inner IPv4 verdict: a mapped private
    # address never names a public host.
    if addr.ipv4_mapped is not None:
        return is_public_ipv4(addr.ipv4_mapped)
    raw = int(addr)
    if raw == 0:  # :: (unspecified)
        return False
    if raw == 1:  # ::1 (loopback)
        return False
    if (raw >> 120) == 0xFF:  # ff00::/8 (multicast)
        return False
    top = raw >> 112
    if (top & 0xFFC0) == 0xFE80:  # fe80::/10 (link-local)
        return False
    if (top & 0xFE00) == 0xFC00:  # fc00::/7 (unique-local)
        return False
    if (raw >> 96) == 0x20010DB8:  # 2001:db8::/32 (documentation)
        return False
    return True


def is_public_ip(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(addr, ipaddress.IPv4Address):
        return is_public_ipv4(addr)
    return is_public_ipv6(addr)


def all_ips_public(ips: list[Any]) -> bool:
    """True only when every address is public. Empty carries no evidence."""
    return len(ips) > 0 and all(is_public_ip(ip) for ip in ips)


def is_allowed_host(host: object) -> bool:
    """Name gate: literal IPs must be public; names must be well-formed DNS
    and not ``localhost``. Resolution itself happens per request."""
    if not isinstance(host, str):
        return False
    host = host.strip().rstrip(".")
    if not host:
        return False
    try:
        return is_public_ip(ipaddress.ip_address(host))
    except ValueError:
        pass
    if (
        host.startswith("[")
        or host.endswith("]")
        or "%" in host
        or any(char.isspace() for char in host)
    ):
        return False
    return host.lower() != "localhost"


def split_host(url: object) -> str:
    """Extract the URL host without any policy verdict. Returns ``""`` when
    no host can be extracted. Used for in-memory block records: the record
    keeps the host even when the host itself is the blocked part."""
    if not isinstance(url, str):
        return ""
    _, separator, rest = url.strip().partition("://")
    if not separator:
        return ""
    authority = rest.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
    hostport = authority.rsplit("@", 1)[-1]
    if hostport.startswith("["):
        inner, bracket, _ = hostport[1:].partition("]")
        return inner if bracket else ""
    if hostport.count(":") > 1:
        return ""
    host = hostport.rsplit(":", 1)[0] if ":" in hostport else hostport
    return host.strip().rstrip(".")


def split_url(url: object) -> tuple[str, str]:
    """Split one URL into ``(scheme, host)`` or raise :class:`NetworkBlocked`
    with a static reason. Mirrors the desktop authority parsing."""
    if not isinstance(url, str):
        raise NetworkBlocked(_REASON_NOT_HTTPS)
    url = url.strip()
    scheme, separator, rest = url.partition("://")
    if not separator or scheme.lower() != "https":
        raise NetworkBlocked(_REASON_NOT_HTTPS)
    authority = rest.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
    hostport = authority.rsplit("@", 1)[-1]
    if hostport.startswith("["):
        inner, bracket, _ = hostport[1:].partition("]")
        if not bracket:
            raise NetworkBlocked(_REASON_NON_PUBLIC)
        host = inner
    else:
        if hostport.count(":") > 1:
            raise NetworkBlocked(_REASON_NON_PUBLIC)
        host = hostport.rsplit(":", 1)[0] if ":" in hostport else hostport
    if not is_allowed_host(host):
        raise NetworkBlocked(_REASON_NON_PUBLIC)
    return scheme.lower(), host


def default_resolver(host: str) -> list[Any]:
    """Resolve a host to unique IPs. Failures resolve to no evidence."""
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except OSError:
        return []
    addresses: list[Any] = []
    for info in infos:
        try:
            addr = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if addr not in addresses:
            addresses.append(addr)
    return addresses


def is_destination_allowed(
    url: object,
    resolve: Callable[[str], list[Any]] = default_resolver,
) -> bool:
    """Full request-time verdict: shape, name gate, then every resolved IP."""
    try:
        _, host = split_url(url)
    except NetworkBlocked:
        return False
    try:
        ips = resolve(host)
    except Exception:
        return False
    return all_ips_public(ips)
