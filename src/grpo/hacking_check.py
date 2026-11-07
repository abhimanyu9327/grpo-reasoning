import argparse
import collections
import glob
import json
import os
import re

from transformers import AutoTokenizer

from grpo import config
from grpo.rewards import evaluate, extract_answer, numbers_in, parse_expression, uses_numbers_correctly
from grpo.task import RESPONSE_PREFIX

# Reads results/<run>/samples/*.json and counts the patterns we look for in reward hacking.
# One table row per saved step (64 answers each).


def outcome(r):
    expr = extract_answer(r["response"])
    tree = parse_expression(expr) if expr else None
    if tree is None:
        return "malformed", None
    if not uses_numbers_correctly(numbers_in(tree), r["numbers"]):
        return "misused", tree   # invented or reused a number (0.1 before the fix, 0.0 now)
    return ("correct" if evaluate(tree) == r["target"] else "wrong"), tree


def think_text(response):
    body = response[len(RESPONSE_PREFIX):]
    end = body.find("</think>")
    if end < 0:
        end = body.find("<answer>")
    return (body[:end] if end >= 0 else body).strip()


def template(r):
    # Replace the problem's own numbers with N, so the same answer shape counts once.
    text = r["response"]
    for n in sorted(set(r["numbers"] + [r["target"]]), key=lambda x: -len(str(x))):
        text = re.sub(rf"\b{n}\b", "N", text)
    return re.sub(r"\s+", " ", text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="grpo_d2")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)

    print("step | correct wrong misused malformed | sum_all_wrong | think_only_expr "
          "| tokens min/median/max | >=300 tok | distinct shapes")
    for path in sorted(glob.glob(os.path.join("results", args.run_name, "samples", "step_*.json"))):
        rows = json.load(open(path))
        counts = collections.Counter()
        sum_all_wrong = 0
        for r in rows:
            kind, tree = outcome(r)
            counts[kind] += 1
            # lazy default: just add all three numbers, whatever the target is
            if kind == "wrong" and sorted(numbers_in(tree)) == sorted(r["numbers"]) \
                    and evaluate(tree) == sum(r["numbers"]):
                sum_all_wrong += 1
        only_expr = sum(bool(re.fullmatch(r"[0-9+\-*/()\s=]+", think_text(r["response"]) or "x"))
                        for r in rows)
        lengths = sorted(len(tokenizer(r["response"])["input_ids"]) for r in rows)
        shapes = len(set(template(r) for r in rows))
        step = int(os.path.basename(path)[5:9])
        print(f"{step:4d} | {counts['correct']:7d} {counts['wrong']:5d} {counts['misused']:7d} "
              f"{counts['malformed']:9d} | {sum_all_wrong:13d} | {only_expr:15d} "
              f"| {lengths[0]:3d}/{lengths[len(lengths) // 2]:3d}/{lengths[-1]:3d}        "
              f"| {sum(n >= 300 for n in lengths):9d} | {shapes:3d}")


if __name__ == "__main__":
    main()
