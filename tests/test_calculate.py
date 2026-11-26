"""Tests for the agent's calculate tool: valid arithmetic, invalid input, code injection."""

import pytest

from stockdesk.agent.calculate import CalculationError, calculate, calculate_tool


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("1 + 2 * 3", 7),
        ("(1 + 2) * 3", 9),
        ("2 ** 10", 1024),
        ("2 ** -1", 0.5),
        ("-5 + +3", -2),
        ("10 / 4", 2.5),
        ("(14694 - 13520) / 13520 * 100", pytest.approx(8.68343195)),
        ("round(2.675, 2)", 2.67),  # float rounding, as Python does it
        ("round(1234.5678)", 1235),
        ("abs(-3.5)", 3.5),
        ("round(abs(-1.23456), 3)", 1.235),
        ("  1.5e3 + 0.5 ", 1500.5),
    ],
)
def test_valid_arithmetic(expression: str, expected):
    assert calculate(expression) == expected


def test_tool_reply_is_the_number_without_float_noise():
    assert calculate_tool("0.1 + 0.2") == "0.3"
    assert calculate_tool("14694 - 13520") == "1174"
    assert calculate_tool("round(1174 / 13520 * 100, 2)") == "8.68"


@pytest.mark.parametrize(
    "expression, message",
    [
        ("1 / 0", "division by zero"),
        ("0 ** -1", "division by zero"),
        ("14,694 + 1", "thousands separators"),
        ("$100 + 5", "not a valid expression"),
        ("10 % 3", "not allowed"),
        ("10 // 3", "not allowed"),
        ("", "not a valid expression"),
        ("revenue - cost", "not allowed"),
        ("round(1.5, 2.5)", "round()"),
        ("abs(1, 2)", "abs()"),
        ("(-8) ** 0.5", "not a real number"),
        ("9 ** 9 ** 9", "exponent too large"),
        ("((2 ** 100) ** 100) ** 100", "result too large"),
        ("10.0 ** 100 * 10.0 ** 100 * 10.0 ** 100 * 10.0 ** 100", "not a finite number"),
        ("1" * 501, "too long"),
        ("1 < 2", "not allowed"),
    ],
)
def test_invalid_expressions_raise_a_clear_error(expression: str, message: str):
    with pytest.raises(CalculationError, match=message):
        calculate(expression)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo pwned')",
        "open('/etc/passwd').read()",
        "eval('1 + 1')",
        "exec('x = 1')",
        "(1).__class__.__bases__[0].__subclasses__()",
        "abs.__self__",
        "round.__call__(1.5)",
        "[x for x in (1, 2)]",
        "lambda: 1",
        "(x := 5)",
        "1 if 1 else 2",
        "'a' * 3",
        "b'a'",
        "True + 1",
        "None",
        "1j * 1j",
        "{1: 2}",
        "round(1.5, ndigits=1)",
        "round(*[1.5])",
        "getattr(1, 'real')",
        "import os",
        "1; 2",
    ],
)
def test_code_injection_is_rejected(expression: str):
    with pytest.raises(CalculationError):
        calculate(expression)


def test_tool_reply_on_error_starts_with_error():
    reply = calculate_tool("__import__('os').system('echo pwned')")
    assert reply.startswith("Error: ")
    assert calculate_tool("1 / 0") == "Error: division by zero"
