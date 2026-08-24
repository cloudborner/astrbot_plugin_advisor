from __future__ import annotations

import ipaddress
import socket
import urllib.parse
import urllib.request


class RejectRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("redirects are not allowed")


PUBLIC_HTTPS_OPENER = urllib.request.build_opener(RejectRedirectHandler())


def validate_public_https_url(url: str, *, label: str = "URL") -> str:
    parsed = urllib.parse.urlparse(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.fragment
    ):
        raise ValueError(f"{label} must be a public HTTPS URL")
    hostname = parsed.hostname.lower().rstrip(".")
    if (
        hostname in {"localhost", "localhost.localdomain"}
        or hostname.endswith((".local", ".internal", ".localhost"))
        or "." not in hostname
    ):
        raise ValueError(f"local {label} hosts are not allowed")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        try:
            resolved = socket.getaddrinfo(
                hostname, parsed.port or 443, type=socket.SOCK_STREAM
            )
        except socket.gaierror as exc:
            raise ValueError(f"{label} host could not be resolved") from exc
        if not resolved:
            raise ValueError(f"{label} host returned no addresses")
        for item in resolved:
            address = ipaddress.ip_address(item[4][0])
            if not address.is_global:
                raise ValueError(f"{label} host resolved to a non-global IP")
    else:
        if not address.is_global:
            raise ValueError(f"private or non-global {label} IP is not allowed")
    return url
