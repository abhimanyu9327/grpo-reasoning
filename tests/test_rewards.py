from grpo.rewards import compute_reward

NUMBERS = [3, 7, 12]
TARGET = 16  # e.g. 12 + 7 - 3


def wrap(expr):
    return f"<think>let me try some combinations</think>\n<answer>{expr}</answer>"


# (response, expected reward, why)
CASES = [
    # correct
    (wrap("12 + 7 - 3"), 1.0, "correct"),
    (wrap("(12 - 3) + 7"), 1.0, "correct with parentheses"),
    (wrap("7 + 12 - 3"), 1.0, "different order is fine"),
    (wrap("12 + 7 - 3") + " Hope this helps!", 1.0, "text after the answer is allowed"),
    ("<think>\nhmm <answer>12 + 7 - 3</answer>", 1.0, "unclosed think is fine, only the answer tag is required"),
    (wrap("12 + 7 - 3 = 16"), 1.0, "a trailing '= result' is dropped"),
    # wrong but well formatted
    (wrap("12 + 7"), 0.1, "wrong value"),
    (wrap("12 / 7"), 0.1, "fraction, wrong value"),
    (wrap("12 / (7 - 7)"), 0.1, "division by zero must not crash"),
    # cheating attempts
    (wrap("12 / 3 * 4"), 0.1, "4 is not one of the given numbers"),
    (wrap("7 + 7 + 3 - 1"), 0.1, "reuses 7 and invents 1"),
    (wrap("12 + 12 - 7 - 1"), 0.1, "reuses 12"),
    (wrap("16"), 0.1, "just writes the target"),
    (wrap("12 + 7 = 16"), 0.1, "claimed result is ignored, we compute 19 ourselves"),
    (wrap("__import__('os').system('ls')"), 0.0, "code injection is not parsed"),
    (wrap("2 ** 4"), 0.0, "power is not allowed"),
    (wrap("33 // 2"), 0.0, "floor division is not allowed"),
    (wrap("-3 + 7 + 12"), 0.0, "unary minus is not allowed"),
    (wrap("16.0"), 0.0, "decimals are not allowed"),
    (wrap("12 + 7") + "\n<answer>12 + 7 - 3</answer>", 0.0, "two answers: no guessing twice"),
    # malformed
    ("12 + 7 - 3", 0.0, "no tags at all"),
    ("<answer>12 + 7 - 3", 0.0, "answer tag not closed"),
    (wrap("= 16"), 0.0, "no expression before '='"),
    (wrap("12 + 7 - 3 = 16 = 16"), 0.0, "two '=' signs"),
    (wrap("12 + 7 - 3 = sixteen"), 0.0, "result after '=' is not a number"),
    (wrap("twelve plus seven minus three"), 0.0, "words instead of an expression"),
    (wrap(""), 0.0, "empty answer"),
    (wrap("(12 + 7 - 3"), 0.0, "unbalanced parentheses"),
]


def test_reward_examples():
    for response, expected, why in CASES:
        got = compute_reward(response, NUMBERS, TARGET)
        assert got == expected, f"{why}: got {got}, expected {expected} for {response!r}"
