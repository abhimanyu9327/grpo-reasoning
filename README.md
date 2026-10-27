# grpo-reasoning

GRPO (Group Relative Policy Optimization) written from scratch in plain PyTorch, used to train
Qwen2.5-0.5B-Instruct on the Countdown task: given 3 numbers and a target, write an arithmetic
expression that hits the target, using each number at most once. The reward comes from a Python
verifier, not from a model.

## Status
- [x] G1: task generator, verifier, baseline evaluation
- [x] G2: rollouts, log-probs, group advantages
- [x] G3: GRPO loss (policy gradient + KL), LoRA training loop, resume
- [ ] Real training run on a T4
- [ ] Results

## Setup
```bash
uv sync
make test     # reward-function test
make data     # writes data/train.jsonl and data/test.jsonl to look at
make smoke    # 10-step training run on the Mac (G=4, 4 prompts, 128 tokens), pipeline check only
make train    # real run; re-run the same command to resume from the last checkpoint
make baseline # base model on all 200 held-out problems
```

## Baseline (base model, greedy, held-out set)
Qwen2.5-0.5B-Instruct, greedy decoding with repetition penalty 1.1 (Qwen's default), max 512 new tokens,
200 held-out problems (seed 1234). Plain greedy (penalty 1.0) gives 0.5% (`results/baseline_pure_greedy.json`).

| difficulty | n | accuracy | 95% CI (Wilson) | format rate |
|---|---|---|---|---|
| 1 (two numbers, + -) | 200 | 2.0% | 0.8% - 5.0% | 88.5% |
| 2 (all three numbers, + -) **chosen** | 200 | 5.0% | 2.7% - 9.0% | 86.5% |

Full outputs: `results/baseline.json` (difficulty 2) and `results/tune_d1.json`.

`uv run python -m grpo.rollout` prints one prompt with 8 sampled answers, rewards, and advantages.

## Training
Each run writes to `results/<run_name>/`: `config.json`, `log.csv` (step, mean_reward, accuracy,
mean_response_length, kl, loss), `samples/step_XXXX.json` every 20 steps, and LoRA checkpoints in
`checkpoints/step_XXXX/` every 50 steps. Any config field can be overridden: `--beta 0.02`.

On Kaggle, keep `results/<run_name>/` between sessions (for example save it as a dataset and copy it
back) so the run can resume.

## Results
TODO

## Layout
```
src/grpo/config.py   paths and settings
src/grpo/task.py     Countdown problem generator + prompt
src/grpo/rewards.py  safe verifier and reward
src/grpo/eval.py     greedy evaluation with 95% confidence interval
src/grpo/rollout.py  sample G answers per prompt, per-token log-probs
src/grpo/grpo.py     group advantages, KL estimator, GRPO loss
src/grpo/train.py    LoRA training loop, logging, checkpoints, resume
tests/               the 3 tests
docs/LEARNING.md     plain-language notes per milestone
```
