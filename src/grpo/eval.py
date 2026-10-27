import argparse
import json
import math
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from grpo import config
from grpo.rewards import check
from grpo.task import RESPONSE_PREFIX, make_prompt, make_splits


def pick_device():
    if torch.cuda.is_available():
        return "cuda", torch.float16   # T4: fp16 only
    if torch.backends.mps.is_available():
        return "mps", torch.float32    # fp16 on MPS can be flaky; 0.5B fits in fp32 anyway
    return "cpu", torch.float32


def wilson_ci(successes, n, z=1.96):
    # 95% Wilson interval. Better than "p +- 1.96*sqrt(p(1-p)/n)" when p is near 0 or 1.
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, center - half), min(1.0, center + half)


def load_model(path, device, dtype):
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.padding_side = "left"  # decoder-only models must be padded on the left for generation
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate(model, tokenizer, prompts, device, max_new_tokens):
    batch = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    out = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False,
                         temperature=None, top_p=None, top_k=None,
                         repetition_penalty=config.EVAL_REPETITION_PENALTY,
                         pad_token_id=tokenizer.pad_token_id)
    new_tokens = out[:, batch["input_ids"].shape[1]:]
    texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    lengths = (new_tokens != tokenizer.pad_token_id).sum(dim=1).tolist()
    return texts, lengths


def evaluate_model(model, tokenizer, problems, device, batch_size, max_new_tokens):
    rows = []
    for i in range(0, len(problems), batch_size):
        chunk = problems[i:i + batch_size]
        prompts = [make_prompt(p, tokenizer) for p in chunk]
        texts, lengths = generate(model, tokenizer, prompts, device, max_new_tokens)
        for p, text, length in zip(chunk, texts, lengths):
            text = RESPONSE_PREFIX + text  # the prefill is part of the model's answer
            format_ok, correct = check(text, p["numbers"], p["target"])
            rows.append({**p, "response": text, "length": length,
                         "format_ok": format_ok, "correct": correct})
        print(f"  {len(rows)}/{len(problems)} done, correct so far: {sum(r['correct'] for r in rows)}")
    return rows


def summarize(rows):
    n = len(rows)
    num_correct = sum(r["correct"] for r in rows)
    low, high = wilson_ci(num_correct, n)
    return {
        "n": n,
        "accuracy": num_correct / n,
        "ci95": [low, high],
        "format_rate": sum(r["format_ok"] for r in rows) / n,
        "mean_response_length": sum(r["length"] for r in rows) / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=config.NUM_TEST)
    parser.add_argument("--difficulty", type=int, default=config.DIFFICULTY)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=config.MAX_NEW_TOKENS)
    parser.add_argument("--out", default="results/baseline.json")
    args = parser.parse_args()

    torch.manual_seed(0)
    device, dtype = pick_device()
    _, test = make_splits(difficulty=args.difficulty)  # held-out set only
    problems = test[:args.n]
    print(f"device={device} dtype={dtype} difficulty={args.difficulty} n={len(problems)}")

    model, tokenizer = load_model(config.MODEL_PATH, device, dtype)
    start = time.time()
    rows = evaluate_model(model, tokenizer, problems, device, args.batch_size, args.max_new_tokens)

    summary = summarize(rows)
    summary.update({"model": config.MODEL_PATH, "difficulty": args.difficulty,
                    "max_new_tokens": args.max_new_tokens, "decoding": "greedy",
                    "repetition_penalty": config.EVAL_REPETITION_PENALTY,
                    "seconds": round(time.time() - start, 1)})
    print(json.dumps(summary, indent=2))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "samples": rows}, f, indent=2)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
