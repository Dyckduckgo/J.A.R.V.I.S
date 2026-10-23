"""SSRF hardening for fetch_url: an approved public URL must not reach local or
private addresses through redirects or DNS (security review finding #1)."""

from __future__ import annotations

import socket
import threading
from pathlib import Path

import httpx
import pytest

from jarvis.errors import ToolError
from jarvis.tools.base import ToolContext
from jarvis.tools.builtin.web import FetchUrlTool, is_visibly_local, resolves_to_non_public

PUBLIC_IP = "93.184.215.14"


@pytest.fixture
def fake_dns(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Deterministic DNS: names map to the given IPs; unknown names fail."""
    table = {"nice.example": PUBLIC_IP, "rebind.example": "127.0.0.1"}

    def getaddrinfo(host, *args, **kwargs):
        if host in table:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (table[host], 0))]
        raise socket.gaierror("not found")

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    return table


def make_ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(workspace=tmp_path, cancel_event=threading.Event(), http_timeout=5)


def redirecting_tool(target: str, hops: list[str]) -> FetchUrlTool:
    def handler(request: httpx.Request) -> httpx.Response:
        hops.append(str(request.url))
        if request.url.host == "nice.example":
            return httpx.Response(302, headers={"Location": target})
        return httpx.Response(200, text="INTERNAL-SECRET", headers={"content-type": "text/plain"})

    return FetchUrlTool(transport=httpx.MockTransport(handler))


@pytest.mark.parametrize(
    "target",
    [
        "http://169.254.169.254/latest/meta-data/iam/",
        "http://127.0.0.1:8080/admin",
        "http://localhost/",
        "http://[::1]/",
        "http://10.0.0.5/",
        "http://192.168.1.1/",
        "http://rebind.example/",  # public-looking name that resolves to loopback
    ],
)
def test_redirect_into_local_network_is_blocked(tmp_path: Path, fake_dns: dict, target: str) -> None:
    hops: list[str] = []
    tool = redirecting_tool(target, hops)
    with pytest.raises(ToolError, match="local-network or private address"):
        tool.run(tool.validate({"url": "http://nice.example/page"}), make_ctx(tmp_path))
    assert len(hops) == 1, f"the internal address must never be contacted: {hops}"


def test_redirect_between_public_hosts_still_works(tmp_path: Path, fake_dns: dict) -> None:
    fake_dns["other.example"] = "8.8.8.8"
    hops: list[str] = []
    result = redirecting_tool("http://other.example/final", hops).run({"url": "http://nice.example/page", "max_chars": 15000}, make_ctx(tmp_path))
    assert result.ok and "INTERNAL-SECRET" in result.content and len(hops) == 2


def test_public_name_resolving_to_loopback_is_blocked_on_first_hop(tmp_path: Path, fake_dns: dict) -> None:
    calls: list[str] = []
    tool = FetchUrlTool(transport=httpx.MockTransport(lambda r: calls.append(str(r.url)) or httpx.Response(200, text="x")))
    with pytest.raises(ToolError):
        tool.run({"url": "http://rebind.example/", "max_chars": 15000}, make_ctx(tmp_path))
    assert calls == []


def test_explicitly_local_url_is_allowed_and_flagged(tmp_path: Path) -> None:
    tool = FetchUrlTool(transport=httpx.MockTransport(lambda r: httpx.Response(200, text="dev server", headers={"content-type": "text/plain"})))
    args = tool.validate({"url": "http://127.0.0.1:5000/health"})
    assert "local network" in tool.describe_call(args)
    assert tool.run(args, make_ctx(tmp_path)).ok
    assert "local network" not in tool.describe_call(tool.validate({"url": "https://example.com/"}))


@pytest.mark.parametrize(
    ("host", "local"),
    [
        ("127.0.0.1", True),
        ("::1", True),
        ("[::1]", True),
        ("::ffff:127.0.0.1", True),
        ("169.254.169.254", True),
        ("10.1.2.3", True),
        ("172.16.0.1", True),
        ("192.168.0.10", True),
        ("100.64.0.1", True),  # carrier-grade NAT
        ("0.0.0.0", True),
        ("fd00::1", True),
        ("localhost", True),
        ("api.localhost", True),
        ("8.8.8.8", False),
        ("example.com", False),
    ],
)
def test_is_visibly_local(host: str, local: bool) -> None:
    assert is_visibly_local(host) is local


def test_unresolvable_host_is_not_treated_as_local(fake_dns: dict) -> None:
    assert resolves_to_non_public("does-not-exist.example") is False
    assert resolves_to_non_public("rebind.example") is True
    assert resolves_to_non_public("nice.example") is False
