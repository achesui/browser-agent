"""T021 policy tests mirroring network_policy.rs case for case."""

from __future__ import annotations

import ipaddress

import pytest

from impretion_browser_agent.browser.network import (
    NETWORK_POLICY_VERSION,
    NetworkBlocked,
    all_ips_public,
    default_resolver,
    is_allowed_host,
    is_destination_allowed,
    is_public_ip,
    split_host,
    split_url,
)


def v4(address: str):
    return ipaddress.ip_address(address)


def test_policy_version_is_pinned() -> None:
    assert NETWORK_POLICY_VERSION == 2


def test_public_ipv4_addresses_pass() -> None:
    for ip in ["8.8.8.8", "1.1.1.1", "93.184.216.34"]:
        assert is_public_ip(v4(ip)), ip


def test_blocked_ipv4_ranges_fail() -> None:
    for ip in [
        "127.0.0.1", "127.1.2.3",
        "10.0.0.1", "172.16.0.1", "172.31.255.254", "192.168.1.1",
        "169.254.169.254",
        "224.0.0.1", "239.255.255.255",
        "0.0.0.0", "0.1.2.3", "255.255.255.255", "240.0.0.1",
        "192.0.2.1", "198.51.100.2", "203.0.113.3",
        "100.64.0.1", "100.127.255.254",
        "198.18.0.1", "198.19.255.255",
    ]:
        assert not is_public_ip(v4(ip)), ip
    assert is_public_ip(v4("172.15.0.1"))
    assert is_public_ip(v4("172.32.0.1"))
    assert is_public_ip(v4("100.63.255.255"))
    assert is_public_ip(v4("100.128.0.1"))
    assert is_public_ip(v4("198.17.255.255"))
    assert is_public_ip(v4("198.20.0.1"))


def test_blocked_ipv6_addresses_fail() -> None:
    for ip in [
        "::1", "::", "fe80::1", "fc00::1", "fd00::1", "ff02::1", "ff05::1",
        "2001:db8::1", "::ffff:127.0.0.1", "::ffff:10.0.0.1",
    ]:
        assert not is_public_ip(ipaddress.ip_address(ip)), ip
    for ip in ["2001:4860:4860::8888", "2606:4700:4700::1111"]:
        assert is_public_ip(ipaddress.ip_address(ip)), ip


def test_multi_ip_destinations_require_every_address_public() -> None:
    assert all_ips_public([v4("8.8.8.8"), v4("1.1.1.1")])
    assert all_ips_public([v4("8.8.8.8")])
    assert not all_ips_public([v4("8.8.8.8"), v4("10.0.0.1")])
    assert not all_ips_public([])
    # Redirect/rebinding mixing one private address blocks the destination.
    assert all_ips_public([v4("93.184.216.34")])
    assert not all_ips_public([v4("93.184.216.34"), v4("192.168.1.1")])
    assert not all_ips_public([v4("93.184.216.34"), v4("127.0.0.1")])


def test_localhost_names_are_blocked_case_insensitively() -> None:
    for host in ["localhost", "LOCALHOST", "LocalHost", "localhost.", " localhost "]:
        assert not is_allowed_host(host), host
    assert is_allowed_host("example.com")
    assert is_allowed_host("sub.example.com.")
    assert not is_allowed_host("")
    assert not is_allowed_host("   ")


def test_literal_ip_hosts_follow_the_address_policy() -> None:
    assert is_allowed_host("8.8.8.8")
    assert is_allowed_host("2001:4860:4860::8888")
    assert not is_allowed_host("127.0.0.1")
    assert not is_allowed_host("10.1.2.3")
    assert not is_allowed_host("169.254.169.254")
    assert not is_allowed_host("::1")
    assert not is_allowed_host("[::1]")


def test_policy_urls_require_public_http_destinations() -> None:
    for url, host in [
        ("https://example.com/docs/page", "example.com"),
        ("https://example.com:8080/a", "example.com"),
        ("https://8.8.8.8/x", "8.8.8.8"),
        ("https://[2001:4860:4860::8888]/y", "2001:4860:4860::8888"),
        ("https://user:pass@example.com/x", "example.com"),
    ]:
        assert split_url(url) == (url.split("://")[0], host)
    # HTTP is rejected: only HTTPS carries browser traffic.
    for url in [
        "http://example.com/",
        "http://example.com:8080/a",
        "http://8.8.8.8/x",
        "http://[2001:4860:4860::8888]/y",
    ]:
        with pytest.raises(NetworkBlocked):
            split_url(url)
    for url in [
        "ftp://example.com/file",
        "file:///etc/passwd",
        "data:text/plain,hello",
        "wss://example.com/session",
        "ws://127.0.0.1:9222/devtools/browser/abc",
        "javascript:alert(1)",
        "not a url",
        "",
    ]:
        with pytest.raises(NetworkBlocked):
            split_url(url)
    for url in [
        "https://localhost/admin",
        "https://localhost./admin",
        "https://127.0.0.1/",
        "https://10.0.0.1/",
        "https://192.168.0.1/",
        "https://169.254.169.254/latest/meta-data",
        "https://[::1]/",
        "https://[fe80::1]/",
        "https://[fe80::1%25eth0]/",
        "https://[2001:db8::1]/x",
        "https://2001:4860:4860::8888/y",
    ]:
        with pytest.raises(NetworkBlocked):
            split_url(url)
    # Reasons never echo the URL.
    with pytest.raises(NetworkBlocked) as excinfo:
        split_url("https://169.254.169.254/latest/meta-data?token=secret")
    assert "169.254" not in str(excinfo.value)


def test_split_host_extracts_without_verdict() -> None:
    assert split_host("http://169.254.169.254/latest/meta-data") == "169.254.169.254"
    assert split_host("https://user:pass@example.com:8443/x") == "example.com"
    assert split_host("http://[fe80::1]/") == "fe80::1"
    assert split_host("data:text/plain,hello") == ""
    assert split_host("not a url") == ""
    assert split_host(None) == ""


def test_full_verdict_uses_fresh_resolution_per_request() -> None:
    calls: list[str] = []

    def resolve(host: str):
        calls.append(host)
        return [v4("93.184.216.34")]

    assert is_destination_allowed("https://example.com/a", resolve) is True
    assert calls == ["example.com"]

    def rebind(host: str):
        return [v4("93.184.216.34"), v4("192.168.1.1")]

    assert is_destination_allowed("https://example.com/a", rebind) is False

    def failing(host: str):
        raise OSError("dns down")

    assert is_destination_allowed("https://example.com/a", failing) is False
    assert is_destination_allowed("http://127.0.0.1/", resolve) is False
    assert is_destination_allowed("ftp://example.com/x", resolve) is False


def test_default_resolver_returns_parsed_addresses() -> None:
    ips = default_resolver("localhost")
    assert any(str(ip).startswith("127.") or ip == ipaddress.ip_address("::1") for ip in ips)
    assert default_resolver("nonexistent.invalid") == []
