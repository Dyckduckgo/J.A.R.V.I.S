"""fetch_url against a real local HTTP server (plus a few httpx.MockTransport cases)."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
from builtin_helpers import call

from jarvis.errors import ToolArgumentError, ToolError
from jarvis.tools.base import ToolContext
from jarvis.tools.builtin.web import MAX_DOWNLOAD_BYTES, USER_AGENT, FetchUrlTool, html_to_text

PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title> Test   Page </title>
<style>body { color: red; }</style>
<script>var SECRET_SCRIPT = "do not show";</script></head>
<body>
<noscript>NOSCRIPT fallback</noscript>
<h1>Welcome</h1>
<p>Hello &amp; welcome to the   <b>test</b> page.</p>
<ul><li>item one</li><li>item two</li></ul>
<svg><title>icon title</title><text>SVG TEXT</text></svg>
<table><tr><td>cell A</td><td>cell B</td></tr></table>
</body></html>"""

BIG_BODY = b"x" * (3 * 1024 * 1024)


class _Handler(BaseHTTPRequestHandler):
    seen_user_agents: list[str] = []

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - silence test output
        pass

    def _send(self, status: int, body: bytes, content_type: str, extra: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (ConnectionError, OSError):
            pass  # the client may stop reading early (size cap)

    def _redirect(self, location: str) -> None:
        self.send_response(302)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        _Handler.seen_user_agents.append(self.headers.get("User-Agent", ""))
        path = self.path
        if path == "/page.html":
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/missing":
            self._send(404, b"Not here, sorry.", "text/plain")
        elif path == "/broken":
            self._send(500, b"<html><body><p>Internal oops</p></body></html>", "text/html")
        elif path == "/plain":
            self._send(200, b"just some text\nsecond line", "text/plain; charset=utf-8")
        elif path == "/json":
            self._send(200, b'{"answer": 42, "items": [1, 2]}', "application/json")
        elif path == "/latin1":
            self._send(200, "<p>caf\u00e9</p>".encode("latin-1"), "text/html; charset=iso-8859-1")
        elif path == "/image.png":
            self._send(200, b"\x89PNG\r\n\x1a\n" + b"\x00" * 100, "image/png")
        elif path == "/big":
            self._send(200, BIG_BODY, "text/plain")
        elif path == "/redirect":
            self._redirect("/page.html")
        elif path == "/loop":
            self._redirect("/loop")
        elif path == "/to-file":
            self._redirect("file:///etc/passwd")
        else:
            self._send(404, b"unknown", "text/plain")


@pytest.fixture(scope="module")
def server() -> Iterator[str]:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.fixture(autouse=True)
def _no_proxies(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)


def test_html_page_is_converted_to_readable_text(server: str, ctx: ToolContext) -> None:
    result = call(FetchUrlTool(), {"url": f"{server}/page.html"}, ctx)
    assert result.ok, result.content
    text = result.content
    assert f"URL: {server}/page.html" in text
    assert "Status: 200 OK" in text
    assert "Title: Test Page" in text
    assert "Welcome" in text
    assert "Hello & welcome to the test page." in text
    assert "- item one\n" in text and "- item two" in text
    assert "cell A cell B" in text
    for hidden in ("SECRET_SCRIPT", "color: red", "NOSCRIPT", "SVG TEXT", "icon title"):
        assert hidden not in text
    assert _Handler.seen_user_agents[-1] == USER_AGENT


def test_redirect_reports_final_url(server: str, ctx: ToolContext) -> None:
    result = call(FetchUrlTool(), {"url": f"{server}/redirect"}, ctx)
    assert result.ok
    assert f"URL: {server}/page.html" in result.content


def test_redirect_loop_fails(server: str, ctx: ToolContext) -> None:
    with pytest.raises(ToolError, match="Too many redirects"):
        call(FetchUrlTool(), {"url": f"{server}/loop"}, ctx)


def test_redirect_to_non_http_scheme_is_blocked(server: str, ctx: ToolContext) -> None:
    with pytest.raises(ToolError):
        call(FetchUrlTool(), {"url": f"{server}/to-file"}, ctx)


def test_404_is_a_failure_with_status_and_excerpt(server: str, ctx: ToolContext) -> None:
    result = call(FetchUrlTool(), {"url": f"{server}/missing"}, ctx)
    assert not result.ok
    assert "Status: 404 Not Found" in result.content
    assert "HTTP error 404" in result.content
    assert "Not here, sorry." in result.content


def test_500_is_a_failure(server: str, ctx: ToolContext) -> None:
    result = call(FetchUrlTool(), {"url": f"{server}/broken"}, ctx)
    assert not result.ok
    assert "HTTP error 500" in result.content and "Internal oops" in result.content


def test_plain_text_and_json_are_returned_as_is(server: str, ctx: ToolContext) -> None:
    plain = call(FetchUrlTool(), {"url": f"{server}/plain"}, ctx)
    assert plain.content.endswith("\n\njust some text\nsecond line")
    data = call(FetchUrlTool(), {"url": f"{server}/json"}, ctx)
    assert data.ok and data.content.endswith('{"answer": 42, "items": [1, 2]}')


def test_declared_charset_is_honoured(server: str, ctx: ToolContext) -> None:
    result = call(FetchUrlTool(), {"url": f"{server}/latin1"}, ctx)
    assert "caf\u00e9" in result.content


def test_binary_content_is_only_described(server: str, ctx: ToolContext) -> None:
    result = call(FetchUrlTool(), {"url": f"{server}/image.png"}, ctx)
    assert result.ok
    assert "Binary content (image/png, 108 bytes) was not downloaded or shown." in result.content
    assert "PNG" not in result.content.split("Content-Type")[0]


def test_download_is_capped_at_2_mb_and_text_at_max_chars(server: str, ctx: ToolContext) -> None:
    result = call(FetchUrlTool(), {"url": f"{server}/big", "max_chars": 100_000}, ctx)
    assert result.ok
    assert "stopped at the 2 MB size limit" in result.content
    assert f"Showing the first 100,000 of {MAX_DOWNLOAD_BYTES:,} characters." in result.content
    header, body = result.content.split("\n\n", 1)
    assert body == "x" * 100_000
    assert len(header) < 1000


def test_default_max_chars(server: str, ctx: ToolContext) -> None:
    result = call(FetchUrlTool(), {"url": f"{server}/big"}, ctx)
    assert result.content.split("\n\n", 1)[1] == "x" * 15_000


def test_cancellation_stops_download(server: str, ctx: ToolContext) -> None:
    ctx.cancel_event.set()
    result = call(FetchUrlTool(), {"url": f"{server}/big"}, ctx)
    assert not result.ok and "cancelled" in result.content


def test_connection_failure_is_a_tool_error(ctx: ToolContext) -> None:
    # (A real refused connection takes ~2 s on Windows loopback, so it is simulated.)
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("[WinError 10061] connection refused", request=request)

    with pytest.raises(ToolError, match="Could not connect to example.com"):
        call(FetchUrlTool(transport=httpx.MockTransport(handler)), {"url": "https://example.com/"}, ctx)


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/file.txt",
        "file:///C:/Windows/win.ini",
        "file:///etc/passwd",
        "javascript:alert(1)",
        "data:text/html,<script>alert(1)</script>",
        "example.com/page",
        "//example.com/page",
        "http://user:secret@example.com/",
        "https://token@example.com/",
        "http:///no-host",
        "http://exa mple.com/",
        "http://example.com/\nHost: evil",
        "http://example.com:99999/",
        "",
    ],
)
def test_unsupported_or_dangerous_urls_are_rejected(url: str, ctx: ToolContext) -> None:
    tool = FetchUrlTool()
    with pytest.raises(ToolArgumentError):
        tool.validate({"url": url})
    with pytest.raises(ToolArgumentError):
        tool.run({"url": url, "max_chars": 1000}, ctx)  # run() re-checks (defence in depth)


@pytest.mark.parametrize("args", [{}, {"url": 5}, {"url": "https://a.b", "max_chars": 99}, {"url": "https://a.b", "max_chars": 100_001}])
def test_argument_validation(args: dict) -> None:
    with pytest.raises(ToolArgumentError):
        FetchUrlTool().validate(args)


def test_describe_call_shows_the_full_url() -> None:
    url = "https://example.com/search?q=" + "a" * 300
    tool = FetchUrlTool()
    assert tool.describe_call(tool.validate({"url": url})) == f"Download (HTTP GET) {url}"


# --- MockTransport: failure modes that are hard to produce with a real server ----


def test_timeout_is_reported(ctx: ToolContext) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated", request=request)

    with pytest.raises(ToolError, match="timed out after 5 s"):
        call(FetchUrlTool(transport=httpx.MockTransport(handler)), {"url": "https://example.com/"}, ctx)


def test_missing_content_type_is_sniffed(ctx: ToolContext) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html><title>Sniffed</title><p>body text</p></html>")

    result = call(FetchUrlTool(transport=httpx.MockTransport(handler)), {"url": "https://example.com/"}, ctx)
    assert "Title: Sniffed" in result.content and "body text" in result.content


def test_meta_charset_is_used_when_header_has_none(ctx: ToolContext) -> None:
    body = '<html><head><meta charset="windows-1252"></head><p>na\u00efve \u20ac</p></html>'.encode("cp1252")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, headers={"Content-Type": "text/html"})

    result = call(FetchUrlTool(transport=httpx.MockTransport(handler)), {"url": "https://example.com/"}, ctx)
    assert "na\u00efve \u20ac" in result.content


def test_html_to_text_keeps_preformatted_line_breaks() -> None:
    title, text = html_to_text("<title>T</title><pre>line 1\nline 2</pre><p>after</p>")
    assert title == "T"
    assert "line 1\nline 2" in text and text.endswith("after")
