"""Safe arithmetic evaluator (``ast`` based; never ``eval``)."""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable
from typing import Any

from jarvis.errors import ToolError
from jarvis.tools.base import RiskLevel, Tool, ToolContext, ToolResult, truncate

MAX_EXPRESSION_LENGTH = 1000
MAX_INT_BITS = 10_000  # roughly 3000 decimal digits
MAX_FACTORIAL_ARG = 1000
MAX_ROUND_DIGITS = 100
MAX_DEPTH = 100
MAX_NODES = 500

Number = int | float


def _check(value: Any) -> Number:
    """Reject results that are not finite real numbers or are absurdly large."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        if value.bit_length() > MAX_INT_BITS:
            raise ToolError(f"Result is too large (more than about {int(MAX_INT_BITS * math.log10(2))} digits).")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ToolError("Result is too large or not a finite number (numeric overflow).")
        return value
    if isinstance(value, complex):
        raise ToolError("Result is not a real number (for example, a fractional power of a negative number).")
    raise ToolError(f"Unsupported value of type {type(value).__name__}.")


def _pow(base: Number, exponent: Number) -> Number:
    if isinstance(base, int) and isinstance(exponent, int):
        if exponent < 0:
            return base**exponent  # float result; ZeroDivisionError for 0 ** -n
        if abs(base) > 1 and exponent * math.log2(abs(base)) > MAX_INT_BITS:
            raise ToolError(
                f"Result of {_short(base)} ** {_short(exponent)} would be too large "
                f"(more than about {int(MAX_INT_BITS * math.log10(2))} digits)."
            )
        return base**exponent
    return base**exponent  # float pow: OverflowError or complex are handled by the caller


def _factorial(n: Number) -> int:
    if isinstance(n, float):
        if not n.is_integer():
            raise ToolError("factorial() requires a whole number.")
        n = int(n)
    if n < 0:
        raise ToolError("factorial() is not defined for negative numbers.")
    if n > MAX_FACTORIAL_ARG:
        raise ToolError(f"factorial() argument is too large (maximum {MAX_FACTORIAL_ARG}).")
    return math.factorial(n)


def _round(value: Number, ndigits: Number | None = None) -> Number:
    if ndigits is None:
        return round(value)
    if isinstance(ndigits, float):
        if not ndigits.is_integer():
            raise ToolError("round() needs a whole number of digits.")
        ndigits = int(ndigits)
    if abs(ndigits) > MAX_ROUND_DIGITS:
        raise ToolError(f"round() digits must be between -{MAX_ROUND_DIGITS} and {MAX_ROUND_DIGITS}.")
    return round(value, ndigits)


def _minmax(fn: Callable[..., Number], name: str) -> Callable[..., Number]:
    def wrapper(*args: Number) -> Number:
        if not args:
            raise ToolError(f"{name}() needs at least one argument.")
        return fn(args)

    return wrapper


_BINARY: dict[type[ast.operator], Callable[[Number, Number], Number]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: _pow,
}

_UNARY: dict[type[ast.unaryop], Callable[[Number], Number]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_FUNCTIONS: dict[str, Callable[..., Number]] = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
    "fabs": math.fabs,
    "factorial": _factorial,
    "radians": math.radians,
    "degrees": math.degrees,
    "abs": abs,
    "round": _round,
    "min": _minmax(min, "min"),
    "max": _minmax(max, "max"),
}

_CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e, "tau": math.tau}

_OPERATOR_SYMBOLS = {
    ast.BitAnd: "&",
    ast.BitOr: "|",
    ast.LShift: "<<",
    ast.RShift: ">>",
    ast.MatMult: "@",
    ast.Invert: "~",
    ast.Not: "not",
}


def _short(value: Number) -> str:
    text = str(value)
    return text if len(text) <= 30 else text[:27] + "..."


class _Evaluator:
    def __init__(self) -> None:
        self.nodes = 0

    def eval(self, node: ast.AST, depth: int = 0) -> Number:
        self.nodes += 1
        if self.nodes > MAX_NODES:
            raise ToolError("Expression is too complex.")
        if depth > MAX_DEPTH:
            raise ToolError("Expression is nested too deeply.")

        if isinstance(node, ast.Expression):
            return self.eval(node.body, depth + 1)
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ToolError(f"Unsupported literal {value!r}; only numbers are allowed.")
            return _check(value)
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type is ast.BitXor:
                raise ToolError("'^' is not supported; use ** for exponentiation.")
            func = _BINARY.get(op_type)
            if func is None:
                raise ToolError(f"Unsupported operator '{_OPERATOR_SYMBOLS.get(op_type, op_type.__name__)}'.")
            left = self.eval(node.left, depth + 1)
            right = self.eval(node.right, depth + 1)
            return _check(func(left, right))
        if isinstance(node, ast.UnaryOp):
            unary = _UNARY.get(type(node.op))
            if unary is None:
                raise ToolError(
                    f"Unsupported operator '{_OPERATOR_SYMBOLS.get(type(node.op), type(node.op).__name__)}'."
                )
            return _check(unary(self.eval(node.operand, depth + 1)))
        if isinstance(node, ast.Name):
            if node.id in _CONSTANTS:
                return _CONSTANTS[node.id]
            if node.id in _FUNCTIONS:
                raise ToolError(f"'{node.id}' is a function; call it like {node.id}(x).")
            raise ToolError(f"Unknown name '{node.id}'. Available constants: {', '.join(_CONSTANTS)}.")
        if isinstance(node, ast.Attribute):
            name = self._math_attribute(node)
            if name in _CONSTANTS:
                return _CONSTANTS[name]
            raise ToolError(f"Unknown name 'math.{name}'.")
        if isinstance(node, ast.Call):
            return self._call(node, depth)
        raise ToolError(f"Unsupported syntax ({type(node).__name__}); only arithmetic expressions are allowed.")

    @staticmethod
    def _math_attribute(node: ast.Attribute) -> str:
        if isinstance(node.value, ast.Name) and node.value.id == "math":
            return node.attr
        raise ToolError("Attribute access is not supported.")

    def _call(self, node: ast.Call, depth: int) -> Number:
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = self._math_attribute(node.func)
        else:
            raise ToolError("Unsupported function call.")
        func = _FUNCTIONS.get(name)
        if func is None:
            raise ToolError(f"Unknown function '{name}'. Available functions: {', '.join(sorted(_FUNCTIONS))}.")
        if node.keywords:
            raise ToolError("Keyword arguments are not supported.")
        if any(isinstance(arg, ast.Starred) for arg in node.args):
            raise ToolError("Argument unpacking is not supported.")
        values = [self.eval(arg, depth + 1) for arg in node.args]
        try:
            return _check(func(*values))
        except TypeError as exc:
            raise ToolError(f"Invalid arguments for {name}(): {exc}") from exc


def evaluate(expression: str) -> Number:
    """Evaluate an arithmetic expression; raises :class:`ToolError` on any problem."""
    text = expression.strip()
    if not text:
        raise ToolError("The expression is empty.")
    if len(text) > MAX_EXPRESSION_LENGTH:
        raise ToolError(f"The expression is too long (maximum {MAX_EXPRESSION_LENGTH} characters).")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"Invalid expression: {exc.msg}.") from exc
    except (ValueError, RecursionError, MemoryError) as exc:
        raise ToolError("Invalid or overly complex expression.") from exc
    try:
        return _Evaluator().eval(tree)
    except ZeroDivisionError as exc:
        raise ToolError("Division by zero.") from exc
    except OverflowError as exc:
        raise ToolError("Result is too large (numeric overflow).") from exc
    except ValueError as exc:
        raise ToolError(f"Math error: {exc}.") from exc
    except (RecursionError, MemoryError) as exc:
        raise ToolError("Expression is too complex.") from exc


def format_number(value: Number) -> str:
    if isinstance(value, int):
        return str(value)
    if value.is_integer() and abs(value) < 1e16:
        return str(int(value))
    return format(value, ".15g")


class CalculateTool(Tool):
    name = "calculate"
    description = (
        "Evaluate an arithmetic expression exactly and safely. Supports numbers, + - * / // % ** "
        "(use ** for powers, not ^), parentheses, the constants pi, e, tau and the functions "
        "sqrt, sin, cos, tan, asin, acos, atan (radians), log(x[, base]), log10, log2, exp, floor, ceil, "
        "fabs, factorial, radians, degrees, abs, round(x[, digits]), min, max."
    )
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Arithmetic expression, e.g. '(3.5 + 2) * sqrt(16) / 7' or 'log(1000, 10)'.",
                "minLength": 1,
                "maxLength": MAX_EXPRESSION_LENGTH,
            }
        },
        "required": ["expression"],
        "additionalProperties": False,
    }
    risk = RiskLevel.SAFE

    def describe_call(self, args: dict[str, Any], ctx: ToolContext | None = None) -> str:
        return f"Calculate: {args.get('expression', '')}"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        expression = args["expression"].strip()
        value = evaluate(expression)
        return ToolResult.success(truncate(f"{expression} = {format_number(value)}"))
