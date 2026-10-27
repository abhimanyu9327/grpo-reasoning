import json
import os
import random

from grpo import config

# Difficulty decides how the hidden solution is built:
#   1: two of the three numbers with + or -   (third number is a distractor)
#   2: all three numbers with + or -
#   3: all three numbers with + - *
#   4: all three numbers with + - * /   (only exact integer division)
OPS_FOR_LEVEL = {1: "+-", 2: "+-", 3: "+-*", 4: "+-*/"}

# The assistant's reply is started with this text (a "prefill"). Without it the base
# model almost never uses the tags (measured: 0/8 format). It is part of the response.
RESPONSE_PREFIX = "<think>\n"

SYSTEM_PROMPT = (
    "You are a helpful assistant. You first think about the reasoning process "
    "and then give the user the answer."
)


def apply_op(a, b, op):
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if b == 0 or a % b != 0:  # keep everything an integer
        return None
    return a // b


def make_problem(rng, difficulty):
    # Build a random expression from the numbers, its value becomes the target.
    # This way the target is always reachable.
    ops = OPS_FOR_LEVEL[difficulty]
    while True:
        nums = [rng.randint(1, 20) for _ in range(3)]
        a, b, c = rng.sample(nums, 3)
        if difficulty == 1:
            target = apply_op(a, b, rng.choice(ops))
        else:
            op1, op2 = rng.choice(ops), rng.choice(ops)
            if rng.random() < 0.5:  # (a op1 b) op2 c
                left = apply_op(a, b, op1)
                target = None if left is None else apply_op(left, c, op2)
            else:                   # a op1 (b op2 c)
                right = apply_op(b, c, op2)
                target = None if right is None else apply_op(a, right, op1)
        # A positive target that is not just one of the numbers.
        if target is not None and target > 0 and target not in nums:
            return {"numbers": nums, "target": target}


def problem_key(p):
    return (tuple(sorted(p["numbers"])), p["target"])


def make_splits(difficulty=config.DIFFICULTY, seed=config.DATA_SEED,
                num_train=config.NUM_TRAIN, num_test=config.NUM_TEST):
    # One pool of unique problems, split into test first and train after,
    # so the two sets can never share a problem.
    rng = random.Random(seed)
    seen = set()
    pool = []
    while len(pool) < num_train + num_test:
        p = make_problem(rng, difficulty)
        if problem_key(p) not in seen:
            seen.add(problem_key(p))
            pool.append(p)
    test, train = pool[:num_test], pool[num_test:]
    assert not {problem_key(p) for p in test} & {problem_key(p) for p in train}
    return train, test


def make_prompt(problem, tokenizer):
    nums, target = problem["numbers"], problem["target"]
    user = (
        f"Using the numbers {nums}, create an expression that equals {target}. "
        "You can use +, -, *, / and parentheses. Each number can be used at most once. "
        "Show your reasoning inside <think> </think> tags, then write only the final "
        "expression inside <answer> </answer> tags, for example <answer>(1 + 2) * 3</answer>. "
        "Do not write '=' or the result inside the answer tags."
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text + RESPONSE_PREFIX


if __name__ == "__main__":
    # Write the splits to disk so they can be inspected by hand.
    train, test = make_splits()
    os.makedirs("data", exist_ok=True)
    for name, rows in [("train", train), ("test", test)]:
        with open(f"data/{name}.jsonl", "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    print(f"difficulty={config.DIFFICULTY}  train={len(train)}  test={len(test)}")
    print("example:", test[0])
