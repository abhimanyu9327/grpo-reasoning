# grpo-reasoning

GRPO (Group Relative Policy Optimization) written from scratch to train a small LLM on a verifiable reasoning task (Countdown). Portfolio project for AI/ML engineer internship applications. It must be correct, measured, simple, and easy for me to explain.

## About me
I am a beginner in RL. I have never studied it. Always teach me after coding.

## How to explain things
- Explain like an AI professor teaching from scratch, in depth, in Hinglish (simple English mixed with Hindi words written in Roman script).
- Use small numeric examples and analogies. Define every term the first time it appears.
- Before each milestone, explain the concepts it needs. After each milestone, quiz me one question at a time and wait for my answer.

## Code style
- Simple, readable code like a strong student would write. No clever tricks, no heavy abstractions, no design patterns.
- Plain functions and small classes. Files under ~150 lines where possible.
- Short comments that explain WHY. No long docstrings.
- Do not add features I did not ask for.
- Structure: src/grpo/{task.py, rewards.py, rollout.py, grpo.py, train.py, eval.py, config.py}

## Rules
- Do NOT use TRL or any RL library. Use plain PyTorch, transformers (4.x, pinned), and peft for LoRA only.
- Model: Qwen2.5-0.5B-Instruct (weights may already exist at ~/mini-infer/weights/, otherwise download).
- Dev machine: MacBook Air M3 (MPS) for tiny smoke runs only. Real runs on a Kaggle T4 GPU: fp16 only, no bf16, no FlashAttention-2. Keep LoRA weights in fp32 and use autocast. Use a GradScaler if needed.
- Task: Countdown (given 3 numbers and a target, write an expression using each number at most once). Reward is computed by a Python verifier, never by a model. Never use eval() on model text. Parse the expression safely.
- Tests: only 3 in total (listed below). Do not add others.
- Log every training step to results/<run_name>/log.csv: step, mean_reward, accuracy, mean_response_length, kl, loss. Save sample completions every 20 steps.
- Every evaluation reports: number of problems, accuracy, and a 95% confidence interval. Fixed seeds. Evaluate on a held-out set that is never used for training.
- Never claim an improvement that was not measured. Be honest about small sample sizes and noise.
- Work one milestone at a time. After each milestone: run the tests, then stop, explain in Hinglish what you built, and tell me what to check by hand.
- After each milestone add a section to docs/LEARNING.md in plain language, so I can explain it in an interview.
- Use uv, pin dependencies, add a Makefile with: make test, make smoke.
- Do not run any git commands. I handle git myself.

## The only 3 tests
1. The reward function gives the right score on hand-written examples (correct, wrong, malformed, cheating attempts).
2. Group advantages: for each group they have mean ~0 and std ~1, and a group where all rewards are equal gives all-zero advantages (no crash).
3. One GRPO update on a tiny batch increases the log-probability of a positive-advantage sample.
