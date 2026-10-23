"""Tool abstraction used by the agent.

A tool declares a JSON-schema for its arguments and an intrinsic
:class:`RiskLevel`. Tools may *escalate* risk per call (for example reading a
file outside the workspace) via :meth:`Tool.assess_risk`; the approval policy
never lets a consequential call run without the user's explicit consent.
"""

from __future__ import annotations

import math
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Protocol

from jarvis.errors import ToolArgumentError


class RiskLevel(str, Enum):
    SAFE = "safe"
    """Read-only or confined to the app's own data: executed automatically."""

    CONSEQUENTIAL = "consequential"
    """Changes files, runs programs, contacts the network, or reads outside the
    workspace: always requires explicit user confirmation."""


@dataclass(frozen=True)
class Note:
    id: int
    content: str
    tags: str
    created_at: float


class NotesRepository(Protocol):
    def add_note(self, content: str, tags: str = "") -> Note: ...

    def search_notes(self, query: str = "", limit: int = 20) -> list[Note]: ...


@dataclass
class ToolContext:
    """Per-turn environment handed to every tool invocation."""

    workspace: Path
    cancel_event: threading.Event = field(default_factory=threading.Event)
    shell_timeout: int = 60
    http_timeout: float = 20.0
    notes: NotesRepository | None = None

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    content: str
    """Text returned to the model (and shown, possibly truncated, in the GUI)."""

    @classmethod
    def success(cls, content: str) -> ToolResult:
        return cls(True, content)

    @classmethod
    def failure(cls, content: str) -> ToolResult:
        return cls(False, content)


MAX_TOOL_OUTPUT_CHARS = 20_000


def truncate(text: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} characters]"


class Tool(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    parameters: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}, "required": []}
    risk: ClassVar[RiskLevel] = RiskLevel.SAFE

    def schema(self) -> dict[str, Any]:
        """OpenAI/Groq function-calling schema."""
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }

    def validate(self, args: dict[str, Any]) -> dict[str, Any]:
        """Validate ``args`` against :attr:`parameters`; returns args with defaults applied."""
        return validate_arguments(self.parameters, args)

    def assess_risk(self, args: dict[str, Any], ctx: ToolContext) -> RiskLevel:
        """Risk of this particular call. Override to escalate (never to downgrade
        an intrinsically consequential tool)."""
        return self.risk

    def describe_call(self, args: dict[str, Any], ctx: ToolContext | None = None) -> str:
        """Human-readable description shown in the GUI and approval dialog.

        ``ctx`` (when given) lets tools show absolute paths for relative arguments.
        """
        if not args:
            return f"{self.name}()"
        rendered = ", ".join(f"{k}={_short(v)}" for k, v in args.items())
        return f"{self.name}({rendered})"

    @abstractmethod
    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        """Execute with already-validated ``args``.

        Raise :class:`jarvis.errors.ToolError` for expected failures; the agent
        converts any exception into a failed :class:`ToolResult`.
        """


def _short(value: Any, limit: int = 60) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


def validate_arguments(schema: dict[str, Any], args: Any) -> dict[str, Any]:
    """Minimal JSON-schema validation for flat tool argument objects.

    Supports ``type``, ``required``, ``properties``, ``enum``, ``default``,
    ``minimum``/``maximum``, ``minLength``/``maxLength`` and
    ``additionalProperties: false``. Raises :class:`ToolArgumentError`.
    """
    if not isinstance(args, dict):
        raise ToolArgumentError("Arguments must be a JSON object.")
    properties: dict[str, Any] = schema.get("properties", {})
    result = dict(args)

    unknown = set(result) - set(properties)
    if unknown and schema.get("additionalProperties", True) is False:
        raise ToolArgumentError(f"Unknown argument(s): {', '.join(sorted(unknown))}")

    for name in schema.get("required", []):
        if name not in result or result[name] is None:
            raise ToolArgumentError(f"Missing required argument '{name}'.")

    for name, spec in properties.items():
        if name not in result or result[name] is None:
            if "default" in spec:
                result[name] = spec["default"]
            else:
                result.pop(name, None)
            continue
        value = result[name]
        expected = spec.get("type")
        if expected:
            allowed = _JSON_TYPES.get(expected, (object,))
            # bool is a subclass of int in Python; reject it for numeric types.
            if not isinstance(value, allowed) or (expected in ("integer", "number") and isinstance(value, bool)):
                if expected == "integer" and isinstance(value, float) and value.is_integer():
                    value = int(value)
                    result[name] = value
                else:
                    raise ToolArgumentError(f"Argument '{name}' must be of type {expected}.")
        if "enum" in spec and value not in spec["enum"]:
            raise ToolArgumentError(f"Argument '{name}' must be one of {spec['enum']}.")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if isinstance(value, float) and not math.isfinite(value):
                raise ToolArgumentError(f"Argument '{name}' must be a finite number.")
            if "minimum" in spec and value < spec["minimum"]:
                raise ToolArgumentError(f"Argument '{name}' must be >= {spec['minimum']}.")
            if "maximum" in spec and value > spec["maximum"]:
                raise ToolArgumentError(f"Argument '{name}' must be <= {spec['maximum']}.")
        if isinstance(value, str):
            if "minLength" in spec and len(value) < spec["minLength"]:
                raise ToolArgumentError(f"Argument '{name}' must be at least {spec['minLength']} characters.")
            if "maxLength" in spec and len(value) > spec["maxLength"]:
                raise ToolArgumentError(f"Argument '{name}' must be at most {spec['maxLength']} characters.")
    return result
