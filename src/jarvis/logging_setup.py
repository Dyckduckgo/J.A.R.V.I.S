"""Logging configuration with secret redaction.

Every handler uses :class:`RedactingFormatter`, which scrubs API keys from the
fully formatted record (message, arguments *and* traceback text) before it is
written anywhere.
"""

from __future__ import annotations

import logging
import re
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

_KEY_PATTERNS = (
    re.compile(r"gsk_[A-Za-z0-9]{8,}"),  # Groq API keys
    re.compile(r"(?i)(authorization[\"']?\s*[:=]\s*[\"']?bearer\s+)[^\s\"',}]+"),
    re.compile(r"(?i)((?:api[_-]?key|secret|token|password)[\"']?\s*[:=]\s*[\"']?)[^\s\"',}]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),  # common provider key shape (defence in depth)
)
_REDACTED = "[REDACTED]"

_known_secrets: set[str] = set()
_known_secrets_lock = threading.Lock()

LOG_FORMAT = "%(asctime)s %(levelname)-8s [%(threadName)s] %(name)s: %(message)s"


def register_secret(value: str | None) -> None:
    """Make sure an exact secret value is never written to the logs."""
    if value and len(value) >= 8:
        with _known_secrets_lock:
            _known_secrets.add(value)


def redact(text: str) -> str:
    with _known_secrets_lock:
        secrets = tuple(_known_secrets)
    for secret in secrets:
        text = text.replace(secret, _REDACTED)
    for pattern in _KEY_PATTERNS:
        if pattern.groups:
            text = pattern.sub(lambda m: m.group(1) + _REDACTED, text)
        else:
            text = pattern.sub(_REDACTED, text)
    return text


class RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


def configure_logging(log_file: Path | None, *, level: int = logging.INFO, console: bool = True) -> None:
    """Configure the root logger. Safe to call more than once."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, "_jarvis_handler", False):
            root.removeHandler(handler)
            handler.close()

    formatter = RedactingFormatter(LOG_FORMAT)
    handlers: list[logging.Handler] = []
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=5, encoding="utf-8")
        handlers.append(file_handler)
    if console and sys.stderr is not None:  # stderr is None under pythonw / frozen GUI builds
        handlers.append(logging.StreamHandler(sys.stderr))

    for handler in handlers:
        handler.setFormatter(formatter)
        handler._jarvis_handler = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    root.setLevel(level)

    # Third-party HTTP stacks are chatty at INFO/DEBUG and may echo headers.
    for noisy in ("httpx", "httpcore", "groq", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
