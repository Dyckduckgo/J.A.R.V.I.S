"""calculate: correctness, rejection of non-arithmetic syntax and DoS guards."""

from __future__ import annotations

import time

import pytest
from builtin_helpers import call

from jarvis.errors import ToolArgumentError, ToolError
from jarvis.tools.base import ToolContext
from jarvis.tools.builtin.calculator import MAX_EXPRESSION_LENGTH, CalculateTool, evaluate, format_number


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 3 * 4", "14"),
        ("(2 + 3) * 4", "20"),
        ("7 / 2", "3.5"),
        ("7 // 2", "3"),
        ("-7 // 2", "-4"),
        ("7 % 3", "1"),
        ("2 ** 10", "1024"),
        ("2 ** -1", "0.5"),
        ("-3 + +2", "-1"),
        ("--4", "4"),
        ("0.1 + 0.2", "0.3"),
        ("1e3", "1000"),
        ("1_000 * 3", "3000"),
        ("10 ** 20", "100000000000000000000"),
        ("sqrt(16)", "4"),
        ("sin(pi / 2)", "1"),
        ("cos(0)", "1"),
        ("tan(0)", "0"),
        ("asin(1) * 2", "3.14159265358979"),
        ("acos(1)", "0"),
        ("atan(1) * 4", "3.14159265358979"),
        ("log(e)", "1"),
        ("log(1000, 10)", "3"),
        ("log10(1000)", "3"),
        ("log2(1024)", "10"),
        ("exp(0)", "1"),
        ("floor(2.7)", "2"),
        ("ceil(2.1)", "3"),
        ("fabs(-2.5)", "2.5"),
        ("factorial(5)", "120"),
        ("factorial(5.0)", "120"),
        ("radians(180)", "3.14159265358979"),
        ("degrees(pi)", "180"),
        ("abs(-7)", "7"),
        ("round(3.14159, 2)", "3.14"),
        ("round(2.5)", "2"),
        ("min(3, 1, 2)", "1"),
        ("max(1, 2.5)", "2.5"),
        ("tau / 2", "3.14159265358979"),
        ("e", "2.71828182845905"),
        ("math.sqrt(9) + math.pi - pi", "3"),
        ("1 / 3", "0.333333333333333"),
        ("2 ** 0.5", "1.4142135623731"),
    ],
)
def test_evaluates_arithmetic(expression: str, expected: str, ctx: ToolContext) -> None:
    result = call(CalculateTool(), {"expression": expression}, ctx)
    assert result.ok
    assert result.content == f"{expression} = {expected}"


def test_big_but_reasonable_integers_are_exact() -> None:
    assert evaluate("2 ** 100") == 2**100
    assert evaluate("factorial(100)") > 10**157
    assert len(format_number(evaluate("9" * 999))) == 999


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("1 / 0", "Division by zero"),
        ("5 % 0", "Division by zero"),
        ("0 ** -1", "Division by zero"),
        ("log(8, 1)", "Division by zero"),
        ("sqrt(-1)", "Math error"),
        ("log(0)", "Math error"),
        ("(-8) ** (1/3)", "not a real number"),
        ("x + 1", "Unknown name 'x'"),
        ("sqrt", "is a function"),
        ("__import__('os').system('calc')", "Attribute access is not supported"),
        ("__import__('os')", "Unknown function '__import__'"),
        ("open('secrets.txt')", "Unknown function"),
        ("().__class__", "Attribute access is not supported"),
        ("(1).real", "Attribute access is not supported"),
        ("math.inf", "Unknown name 'math.inf'"),
        ("os.path", "Attribute access is not supported"),
        ("'a' * 3", "only numbers are allowed"),
        ("True + 1", "only numbers are allowed"),
        ("1j * 2", "only numbers are allowed"),
        ("2 ^ 3", "use \\*\\* for exponentiation"),
        ("6 & 3", "Unsupported operator '&'"),
        ("1 << 4", "Unsupported operator '<<'"),
        ("~5", "Unsupported operator '~'"),
        ("not 1", "Unsupported operator 'not'"),
        ("1 < 2", "Unsupported syntax"),
        ("1 if 1 else 2", "Unsupported syntax"),
        ("[1, 2]", "Unsupported syntax"),
        ("lambda: 1", "Unsupported syntax"),
        ("sqrt(x=4)", "Keyword arguments"),
        ("max(*[1, 2])", "unpacking"),
        ("sqrt(1, 2)", "Invalid arguments for sqrt"),
        ("min()", "at least one argument"),
        ("factorial(-1)", "negative"),
        ("factorial(2.5)", "whole number"),
        ("round(1.5, 0.5)", "whole number"),
        ("2 +", "Invalid expression"),
        ("x = 1", "Invalid expression"),
        ("1; 2", "Invalid expression"),
        ("(" * 250 + "1" + ")" * 250, "Invalid"),
    ],
)
def test_rejects_invalid_or_unsafe_input(expression: str, message: str) -> None:
    with pytest.raises(ToolError, match=message):
        evaluate(expression)


@pytest.mark.parametrize(
    "expression",
    [
        "9 ** 9 ** 9",
        "2 ** 100000",
        "10 ** 10 ** 10",
        "(2 ** 5000) ** 3",
        "(-3) ** 99999",
        "factorial(10 ** 6)",
        "factorial(1001)",
        "10.0 ** 400",
        "2.5 ** 10 ** 9",
        "exp(1000)",
        "1e308 * 10",
        "10 ** 3000 / 3",
        "sqrt(10 ** 3000)",
        "round(10 ** 3000, -10 ** 9)",
        "factorial(1000) * factorial(1000)",
        "round(1.5, 10 ** 9)",
    ],
)
def test_resource_exhaustion_is_refused_quickly(expression: str) -> None:
    started = time.perf_counter()
    with pytest.raises(ToolError, match="too large|overflow|round\\(\\) digits"):
        evaluate(expression)
    assert time.perf_counter() - started < 1.0


def test_nesting_depth_is_limited() -> None:
    with pytest.raises(ToolError, match="nested too deeply|too complex"):
        evaluate("-" * 300 + "1")


def test_expression_length_is_limited(ctx: ToolContext) -> None:
    too_long = "1+" * (MAX_EXPRESSION_LENGTH // 2) + "1"
    with pytest.raises(ToolArgumentError, match="at most"):
        CalculateTool().validate({"expression": too_long})
    with pytest.raises(ToolError, match="too long"):
        evaluate(too_long)


def test_node_count_is_limited() -> None:
    with pytest.raises(ToolError, match="too complex"):
        # Wide rather than deep: 9 chains of 50 terms stay under the depth limit.
        evaluate("min(" + ",".join(["+".join(["1"] * 50)] * 9) + ")")


@pytest.mark.parametrize("args", [{}, {"expression": ""}, {"expression": 5}, {"expression": "1", "x": 1}])
def test_argument_validation(args: dict) -> None:
    with pytest.raises(ToolArgumentError):
        CalculateTool().validate(args)


def test_blank_expression_fails(ctx: ToolContext) -> None:
    with pytest.raises(ToolError, match="empty"):
        call(CalculateTool(), {"expression": "   "}, ctx)


def test_describe_call_shows_the_expression() -> None:
    assert CalculateTool().describe_call({"expression": "2 * pi"}) == "Calculate: 2 * pi"
