.PHONY: test smoke train data baseline

test:
	uv run pytest -q tests

# Tiny 10-step training run on the Mac (G=4, 4 prompts, 128 tokens). Starts fresh every time.
smoke:
	rm -rf results/smoke
	uv run python -m grpo.train --smoke

# Real run (Kaggle T4). Re-running the same command resumes from the last checkpoint.
train:
	uv run python -m grpo.train

data:
	uv run python -m grpo.task

# Full baseline on the 200 held-out problems (use a GPU for this).
baseline:
	uv run python -m grpo.eval --out results/baseline.json
