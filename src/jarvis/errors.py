"""Exception hierarchy.

Every exception carries a ``user_message`` that is safe to show in the GUI:
it never contains secrets or raw tracebacks. The plain ``str(exc)`` may hold
more technical detail and is meant for the log file.
"""

from __future__ import annotations


class JarvisError(Exception):
    """Base class for all expected application errors."""

    def __init__(self, message: str, *, user_message: str | None = None) -> None:
        super().__init__(message)
        self.user_message = user_message or message


class ConfigError(JarvisError):
    """Settings could not be loaded, validated or saved."""


class SecretStoreError(JarvisError):
    """The OS credential store is unavailable or rejected an operation."""


class StorageError(JarvisError):
    """The conversation database failed."""


class OperationCancelled(JarvisError):
    """The user stopped the running operation."""

    def __init__(self, message: str = "Operation cancelled by user.") -> None:
        super().__init__(message)


# --- LLM ---------------------------------------------------------------------


class LLMError(JarvisError):
    """Base class for failures talking to the language model provider."""

    retryable: bool = False


class MissingAPIKeyError(LLMError):
    def __init__(self) -> None:
        super().__init__(
            "No Groq API key configured.",
            user_message=(
                "No Groq API key is configured. Open Settings and enter your key, "
                "or set the GROQ_API_KEY environment variable."
            ),
        )


class LLMAuthenticationError(LLMError):
    """The API key was rejected (HTTP 401/403)."""


class LLMRateLimitError(LLMError):
    retryable = True


class LLMConnectionError(LLMError):
    """Network failure or timeout."""

    retryable = True


class LLMServiceError(LLMError):
    """The provider returned a 5xx error."""

    retryable = True


class LLMRequestError(LLMError):
    """The provider rejected the request (other 4xx, e.g. unknown model)."""


class LLMToolCallError(LLMRequestError):
    """The model produced a malformed tool call (Groq ``tool_use_failed``)."""

    retryable = True


# --- Tools -------------------------------------------------------------------


class ToolError(JarvisError):
    """A tool failed in an expected way; the message is returned to the model."""


class ToolArgumentError(ToolError):
    """Tool arguments were missing, malformed or out of range."""
