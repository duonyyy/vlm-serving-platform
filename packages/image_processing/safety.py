"""Safety checks and SSRF prevention for image loading."""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

# Restricted private/loopback/metadata IP networks
BLOCKED_NETWORKS = [
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),  # Link-local & Cloud metadata
    ipaddress.ip_network("100.64.0.0/10"),  # Carrier-grade NAT
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),  # IPv6 Unique local
    ipaddress.ip_network("fe80::/10"),  # IPv6 Link-local
]


def is_safe_url(url: str, allowlist: list[str] | None = None) -> tuple[bool, str]:
    """Validate that a URL is safe to fetch and not attempting SSRF.

    Returns:
        (is_safe, reason)
    """
    if not url or not isinstance(url, str):
        return False, "URL must be a non-empty string"

    # Data URIs are self-contained and handled separately
    if url.startswith("data:"):
        return True, "Data URI"

    try:
        parsed = urlparse(url)
    except Exception as exc:
        return False, f"Invalid URL syntax: {exc}"

    if parsed.scheme not in {"http", "https"}:
        return False, f"Unsupported URL scheme: {parsed.scheme}; only http/https allowed"

    hostname = parsed.hostname
    if not hostname:
        return False, "URL hostname is missing"

    # Reject localhost directly
    if hostname.lower() in {"localhost", "127.0.0.1", "::1"}:
        return False, f"Access to localhost/loopback is blocked: {hostname}"

    # Check hostname against explicit allowlist if provided
    if allowlist:
        if not any(hostname.lower() == domain.lower() or hostname.lower().endswith("." + domain.lower()) for domain in allowlist):
            return False, f"Hostname {hostname} is not in allowed domains list"

    # Resolve IP addresses to prevent DNS rebinding / internal routing
    try:
        addr_infos = socket.getaddrinfo(hostname, parsed.port or (443 if parsed.scheme == "https" else 80), proto=socket.IPPROTO_TCP)
        for _, _, _, _, sockaddr in addr_infos:
            ip_str = sockaddr[0]
            ip_obj = ipaddress.ip_address(ip_str)
            for net in BLOCKED_NETWORKS:
                if ip_obj in net:
                    return False, f"Resolved IP {ip_str} belongs to blocked/private range {net}"
    except socket.gaierror as exc:
        return False, f"Failed to resolve hostname '{hostname}': {exc}"
    except Exception as exc:
        return False, f"Network validation error for '{hostname}': {exc}"

    return True, "URL is safe"
