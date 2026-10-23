"""Fetch a web page over HTTP(S) (always requires the user's confirmation:
network egress could leak data to the destination)."""

from __future__ import annotations

import codecs
import ipaddress
import re
import socket
import ssl
import threading
import time
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlsplit

import httpx

from jarvis.errors import ToolArgumentError, ToolError
from jarvis.tools.base import RiskLevel, Tool, ToolContext, ToolResult, truncate
from jarvis.tools.builtin._display import reveal_hidden

MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024
MAX_URL_LENGTH = 2048
MAX_FETCH_CHARS = 100_000
MAX_REDIRECTS = 5
USER_AGENT = "JARVIS-Desktop/1.0 (personal desktop assistant; fetch_url tool; python-httpx)"
ACCEPT = "text/html,application/xhtml+xml,text/plain;q=0.9,application/json;q=0.9,*/*;q=0.5"

_TEXT_TYPES = {
    "application/json",
    "application/ld+json",
    "application/xml",
    "application/javascript",
    "application/x-javascript",
    "application/ecmascript",
    "application/rss+xml",
    "application/atom+xml",
    "application/x-yaml",
    "application/yaml",
    "application/toml",
    "application/x-ndjson",
    "application/csv",
}
_HTML_TYPES = {"text/html", "application/xhtml+xml"}
_META_CHARSET = re.compile(rb"""<meta[^>]+charset\s*=\s*["']?\s*([A-Za-z0-9_.:-]+)""", re.IGNORECASE)


def check_web_url(url: Any, *, argument: str = "url") -> str:
    """Accept only absolute http(s) URLs without embedded credentials."""
    if not isinstance(url, str):
        raise ToolArgumentError(f"Argument '{argument}' must be a string.")
    text = url.strip()
    if not text:
        raise ToolArgumentError(f"Argument '{argument}' must not be empty.")
    if len(text) > MAX_URL_LENGTH:
        raise ToolArgumentError(f"The URL is too long (maximum {MAX_URL_LENGTH} characters).")
    if any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in text):
        raise ToolArgumentError("The URL must not contain spaces or control characters.")
    parts = urlsplit(text)
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        shown = f"'{scheme}:'" if scheme else "no scheme"
        raise ToolArgumentError(f"Only http:// and https:// URLs are allowed (got {shown}).")
    if "@" in parts.netloc:
        raise ToolArgumentError("URLs with embedded credentials (user:password@host) are not allowed.")
    if not parts.hostname:
        raise ToolArgumentError("The URL has no host name.")
    try:
        parts.port  # noqa: B018 - raises ValueError for an invalid port
        httpx.URL(text)
    except (ValueError, httpx.InvalidURL) as exc:
        raise ToolArgumentError(f"Invalid URL: {exc}") from exc
    return text


_ssl_lock = threading.Lock()
_ssl_context: ssl.SSLContext | None = None


def _shared_ssl_context() -> ssl.SSLContext:
    """httpx's default verifying context, built once: creating it costs ~0.5 s on Windows."""
    global _ssl_context
    with _ssl_lock:
        if _ssl_context is None:
            _ssl_context = httpx.create_ssl_context()
        return _ssl_context


def _is_non_public_ip(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])  # drop IPv6 zone id
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return not ip.is_global or ip.is_multicast


def is_visibly_local(host: str) -> bool:
    """True if the host *as written* is loopback/private/link-local (IP literal or localhost)."""
    host = host.strip("[]").rstrip(".").lower()
    return host == "localhost" or host.endswith(".localhost") or _is_non_public_ip(host)


def resolves_to_non_public(host: str) -> bool:
    """True if ``host`` is, or resolves to, a non-public address. DNS failure -> False
    (the connection itself will then fail)."""
    if is_visibly_local(host):
        return True
    try:
        infos = socket.getaddrinfo(host.strip("[]"), None, proto=socket.IPPROTO_TCP)
    except (OSError, UnicodeError):
        return False
    return any(_is_non_public_ip(str(info[4][0])) for info in infos)


def _make_request_guard(allow_local: bool):
    """Re-check every hop, including redirects.

    Unless the user approved a URL that is visibly local (``localhost``,
    ``127.0.0.1``, ``192.168.x.x``...), no hop may reach a loopback, private,
    link-local or otherwise non-public address. This stops an approved public
    URL from redirecting (or DNS-resolving) into the local network or a cloud
    metadata endpoint without the user having seen that address.
    """

    def guard(request: httpx.Request) -> None:
        if request.url.scheme not in ("http", "https"):
            raise ToolError(f"Blocked a request to a non-HTTP URL: {request.url}")
        if request.url.userinfo:
            raise ToolError("Blocked a redirect to a URL with embedded credentials.")
        if not allow_local and resolves_to_non_public(request.url.host):
            raise ToolError(
                f"Blocked a request to {request.url.host}: it is a local-network or private address that you "
                "did not approve. Ask for that address explicitly if you need it."
            )

    return guard


def _parse_content_type(header: str) -> tuple[str, str | None]:
    mime, _, params = header.partition(";")
    charset = None
    for param in params.split(";"):
        key, _, value = param.partition("=")
        if key.strip().lower() == "charset":
            charset = value.strip().strip("\"'") or None
    return mime.strip().lower(), charset


def _classify(mime: str) -> str:
    if mime in _HTML_TYPES:
        return "html"
    if mime.startswith("text/") or mime in _TEXT_TYPES or mime.endswith(("+json", "+xml")):
        return "text"
    if not mime:
        return "unknown"
    return "binary"


def _valid_codec(name: str | None) -> str | None:
    if not name:
        return None
    try:
        return codecs.lookup(name).name
    except LookupError:
        return None


def decode_body(body: bytes, charset: str | None, is_html: bool) -> str:
    if body.startswith(codecs.BOM_UTF8):
        return body[len(codecs.BOM_UTF8) :].decode("utf-8", errors="replace")
    encoding = _valid_codec(charset)
    if encoding is None and is_html:
        match = _META_CHARSET.search(body[:4096])
        if match:
            encoding = _valid_codec(match.group(1).decode("ascii", errors="ignore"))
    return body.decode(encoding or "utf-8", errors="replace")


class _HTMLToText(HTMLParser):
    SKIP = {"script", "style", "noscript", "template", "svg", "canvas", "iframe", "object"}
    BLOCK = {
        "address", "article", "aside", "blockquote", "br", "dd", "details", "div", "dl", "dt", "fieldset",
        "figcaption", "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "li",
        "main", "nav", "ol", "p", "pre", "section", "summary", "table", "tbody", "thead", "tfoot", "tr", "ul",
    }  # fmt: skip

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._title: list[str] = []
        self._skip_depth = 0
        self._in_title = False
        self._pre_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.SKIP:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = True
            return
        if tag == "pre":
            self._pre_depth += 1
        if tag in self.BLOCK:
            self._parts.append("\n")
        if tag == "li":
            self._parts.append("- ")
        elif tag in ("td", "th"):
            self._parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = False
            return
        if tag == "pre":
            self._pre_depth = max(0, self._pre_depth - 1)
        if tag in self.BLOCK and tag not in ("br", "hr"):
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self._title.append(data)
            return
        if self._pre_depth:
            self._parts.append(data.replace("\n", "\x00"))  # protect line breaks inside <pre>
        else:
            self._parts.append(re.sub(r"\s+", " ", data))

    @property
    def title(self) -> str:
        return " ".join("".join(self._title).split())

    def text(self) -> str:
        raw = "".join(self._parts)
        lines = [re.sub(r"[ \t\f\v\r]+", " ", line).strip() for line in raw.split("\n")]
        text = "\n".join(line.replace("\x00", "\n") for line in lines)
        return re.sub(r"\n{3,}", "\n\n", text).strip()


def html_to_text(html: str) -> tuple[str, str]:
    """Return (title, readable text) for an HTML document."""
    parser = _HTMLToText()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # noqa: BLE001 - html.parser is lenient; never fail the fetch on markup
        pass
    return parser.title, parser.text()


class FetchUrlTool(Tool):
    name = "fetch_url"
    description = (
        "Download a web page (http/https GET) and return its readable text: HTML is converted to plain text, "
        "plain text and JSON are returned as-is, other content types are only described. Reports the final "
        "URL after redirects and the HTTP status. Requires the user's approval."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Absolute http:// or https:// URL.",
                "minLength": 1,
                "maxLength": MAX_URL_LENGTH,
            },
            "max_chars": {
                "type": "integer",
                "description": f"Maximum characters of text to return (default 15000, max {MAX_FETCH_CHARS}).",
                "default": 15_000,
                "minimum": 100,
                "maximum": MAX_FETCH_CHARS,
            },
        },
        "required": ["url"],
        "additionalProperties": False,
    }
    risk = RiskLevel.CONSEQUENTIAL

    def __init__(self, transport: httpx.BaseTransport | None = None) -> None:
        self._transport = transport

    def validate(self, args: dict[str, Any]) -> dict[str, Any]:
        result = super().validate(args)
        result["url"] = check_web_url(result["url"])
        return result

    def describe_call(self, args: dict[str, Any], ctx: ToolContext | None = None) -> str:
        url = str(args.get("url", ""))
        note = " (an address on this computer or local network)" if is_visibly_local(urlsplit(url).hostname or "") else ""
        return f"Download (HTTP GET) {reveal_hidden(url)}{note}"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        url = check_web_url(args["url"])
        max_chars = int(args.get("max_chars", 15_000))
        client_kwargs: dict[str, Any] = {
            "timeout": ctx.http_timeout,
            "follow_redirects": True,
            "max_redirects": MAX_REDIRECTS,
            "headers": {"User-Agent": USER_AGENT, "Accept": ACCEPT},
            "event_hooks": {"request": [_make_request_guard(is_visibly_local(urlsplit(url).hostname or ""))]},
        }
        if self._transport is not None:
            client_kwargs["transport"] = self._transport
        else:
            client_kwargs["verify"] = _shared_ssl_context()
        overall_deadline = time.monotonic() + max(30.0, 3 * float(ctx.http_timeout))
        try:
            with httpx.Client(**client_kwargs) as client, client.stream("GET", url) as response:
                status = response.status_code
                reason = response.reason_phrase or ""
                final_url = str(response.url)
                content_type = response.headers.get("content-type", "")
                mime, charset = _parse_content_type(content_type)
                kind = _classify(mime)
                body = b""
                stopped: str | None = None
                if kind != "binary":
                    body, stopped = self._read_capped(response, ctx, overall_deadline)
                declared_length = response.headers.get("content-length")
        except httpx.TimeoutException as exc:
            raise ToolError(f"The request to {url} timed out after {ctx.http_timeout:g} s.") from exc
        except httpx.TooManyRedirects as exc:
            raise ToolError(f"Too many redirects (more than {MAX_REDIRECTS}) starting from {url}.") from exc
        except httpx.UnsupportedProtocol as exc:
            raise ToolError(f"Redirected to an unsupported URL scheme: {exc}") from exc
        except httpx.ConnectError as exc:
            raise ToolError(f"Could not connect to {urlsplit(url).hostname}: {exc}") from exc
        except httpx.InvalidURL as exc:
            raise ToolError(f"Invalid URL: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ToolError(f"HTTP request to {url} failed: {exc}") from exc

        if stopped == "cancelled":
            return ToolResult.failure("Download cancelled by the user.")

        if kind == "unknown":
            kind = "html" if body.lstrip()[:15].lower().startswith((b"<!doctype html", b"<html")) else "text"
            if b"\x00" in body[:8192]:
                kind = "binary"

        header = [f"URL: {final_url}", f"Status: {status} {reason}".rstrip()]
        if content_type:
            header.append(f"Content-Type: {content_type}")

        if kind == "binary":
            size = f"{int(declared_length):,} bytes" if declared_length and declared_length.isdigit() else "unknown size"
            header.append(f"Binary content ({mime or 'unknown type'}, {size}) was not downloaded or shown.")
            text = ""
        else:
            decoded = decode_body(body, charset, kind == "html")
            if kind == "html":
                title, text = html_to_text(decoded)
                if title:
                    header.append(f"Title: {title}")
            else:
                text = decoded.strip()
            if stopped:
                header.append(f"Note: download {stopped}; the content below is incomplete.")

        if status >= 400:
            excerpt = text[:1000]
            message = "\n".join(header) + f"\n\nHTTP error {status}." + (f"\n\n{excerpt}" if excerpt else "")
            return ToolResult.failure(truncate(message))

        total = len(text)
        if total > max_chars:
            text = text[:max_chars]
            header.append(f"Showing the first {max_chars:,} of {total:,} characters.")
        output = "\n".join(header) + ("\n\n" + text if text else ("" if kind == "binary" else "\n\n(empty body)"))
        return ToolResult.success(truncate(output, max_chars + 2000))

    @staticmethod
    def _read_capped(response: httpx.Response, ctx: ToolContext, deadline: float) -> tuple[bytes, str | None]:
        chunks: list[bytes] = []
        size = 0
        for chunk in response.iter_bytes():
            if ctx.cancel_event.is_set():
                return b"".join(chunks), "cancelled"
            room = MAX_DOWNLOAD_BYTES - size
            if len(chunk) > room:
                chunks.append(chunk[:room])
                return b"".join(chunks), f"stopped at the {MAX_DOWNLOAD_BYTES // (1024 * 1024)} MB size limit"
            chunks.append(chunk)
            size += len(chunk)
            if time.monotonic() > deadline:
                return b"".join(chunks), "stopped because it took too long"
        return b"".join(chunks), None
