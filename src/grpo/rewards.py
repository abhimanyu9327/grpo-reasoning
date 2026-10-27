import ast
import re
from collections import Counter
from fractions import Fraction

# Only the answer tag is required. The base model never closes </think> (measured 0/48),
# so requiring it would make every reward 0 and leave RL nothing to learn from.
# DOTALL lets "." match newlines.
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
ALLOWED_CHARS_RE = re.compile(r"^[0-9+\-*/()\s]+$")
MAX_EXPR_LEN = 100


def extract_answer(response):
    # Exactly one answer tag. Several answers would let the model "guess many times".
    if response.count("<answer>") != 1 or response.count("</answer>") != 1:
        return None
    match = ANSWER_RE.search(response)
    if match is None:
        return None
    expr = match.group(1).strip()
    # The model often writes "(8 - 3) + 5 = 10". Drop the claimed result:
    # we compute the value ourselves, so the claim is never trusted.
    if expr.count("=") == 1:
        left, right = expr.split("=")
        if not right.strip().isdigit():
            return None
        expr = left.strip()
    return expr


def parse_expression(expr):
    # Returns an ast tree, or None if the text is not a plain arithmetic expression.
    # We only *parse* the text with ast, we never run it.
    if not expr or len(expr) > MAX_EXPR_LEN or not ALLOWED_CHARS_RE.match(expr):
        return None
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        ok = isinstance(node, (ast.Expression, ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div))
        ok = ok or (isinstance(node, ast.Constant) and type(node.value) is int)
        if not ok:  # e.g. ** (Pow), // (FloorDiv), unary minus
            return None
    return tree


def numbers_in(tree):
    return [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)]


def evaluate(node):
    # Our own tiny calculator. Fraction keeps division exact (7/2 stays 7/2, not 3.4999...).
    if isinstance(node, ast.Expression):
        return evaluate(node.body)
    if isinstance(node, ast.Constant):
        return Fraction(node.value)
    left, right = evaluate(node.left), evaluate(node.right)
    if left is None or right is None:
        return None
    if isinstance(node.op, ast.Add):
        return left + right
    if isinstance(node.op, ast.Sub):
        return left - right
    if isinstance(node.op, ast.Mult):
        return left * right
    if right == 0:
        return None
    return left / right


def uses_numbers_correctly(used, given):
    # Each used number must come from the given list, and not more times than given.
    used_count, given_count = Counter(used), Counter(given)
    return all(used_count[n] <= given_count[n] for n in used_count)


def check(response, numbers, target):
    # Returns (format_ok, correct).
    expr = extract_answer(response)
    if expr is None:
        return False, False
    tree = parse_expression(expr)
    if tree is None:
        return False, False
    if not uses_numbers_correctly(numbers_in(tree), numbers):
        return True, False
    value = evaluate(tree)
    return True, value is not None and value == target


def compute_reward(response, numbers, target):
    format_ok, correct = check(response, numbers, target)
    if correct:
        return 1.0
    if format_ok:
        return 0.1
    return 0.0
