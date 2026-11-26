"""Safe arithmetic for the agent's calculate tool: parsed with ast, never eval()'d.

Allowed: int and float literals, + - * / **, unary + and -, parentheses, round() and abs().
Anything else (names, attributes, strings, comprehensions, other calls) is rejected.
"""

import ast
import math
import operator

MAX_EXPRESSION_CHARS = 500
MAX_EXPONENT = 100  # caps ** so "9 ** 9 ** 9" can't hang the process
MAX_INT_BITS = 1000  # caps intermediate ints so chained ** can't use unbounded memory

BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
FUNCTIONS = {"round": round, "abs": abs}


class CalculationError(ValueError):
    """The expression is not allowed or can't be evaluated; the message says why."""


def _check(value: int | float | complex) -> int | float:
    """Reject results that aren't real numbers of a sane size."""
    if isinstance(value, complex):
        raise CalculationError("result is not a real number")
    if isinstance(value, int) and value.bit_length() > MAX_INT_BITS:
        raise CalculationError("result too large")
    return value


def _eval(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant):
        # bool is an int subclass; reject it along with strings, None, complex, etc.
        if type(node.value) in (int, float):
            return node.value
        raise CalculationError(f"only numbers are allowed, got {node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in BINARY_OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise CalculationError(f"exponent too large (max {MAX_EXPONENT})")
        if isinstance(node.op, ast.Div) and right == 0:
            raise CalculationError("division by zero")
        return _check(BINARY_OPS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPS:
        return UNARY_OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
            raise CalculationError("only round() and abs() can be called")
        if node.keywords:
            raise CalculationError(f"{node.func.id}() takes positional arguments only")
        args = [_eval(arg) for arg in node.args]
        try:
            return FUNCTIONS[node.func.id](*args)
        except TypeError as e:
            raise CalculationError(f"{node.func.id}(): {e}") from None
    if isinstance(node, ast.Tuple):
        raise CalculationError("write numbers without thousands separators, e.g. 14694 not 14,694")
    raise CalculationError(f"{type(node).__name__} is not allowed")


def calculate(expression: str) -> int | float:
    """Evaluate an arithmetic expression; raise CalculationError if it isn't allowed."""
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise CalculationError(f"expression too long (max {MAX_EXPRESSION_CHARS} characters)")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError:
        raise CalculationError(
            "not a valid expression; use numbers, + - * / ** ( ), round() and abs() only"
        ) from None
    try:
        value = _eval(tree.body)
    except ZeroDivisionError:  # e.g. 0 ** -1
        raise CalculationError("division by zero") from None
    except OverflowError:
        raise CalculationError("result too large") from None
    except RecursionError:
        raise CalculationError("expression nested too deeply") from None
    if isinstance(value, float) and not math.isfinite(value):
        raise CalculationError("result is not a finite number")
    return value


def calculate_tool(expression: str) -> str:
    """The calculate tool's reply: the result, or "Error: ..." with the reason."""
    try:
        value = calculate(expression)
    except CalculationError as e:
        return f"Error: {e}"
    if isinstance(value, float):
        value = round(value, 10)  # hide float noise like 0.30000000000000004
    return str(value)
