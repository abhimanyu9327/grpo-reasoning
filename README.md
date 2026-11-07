# grpo-reasoning

GRPO (Group Relative Policy Optimization) written from scratch in plain PyTorch, used to train
Qwen2.5-0.5B-Instruct on the Countdown task: given 3 numbers and a target, write an arithmetic
expression that hits the target, using each number at most once. The reward comes from a Python
verifier, not from a model.

## Status
- [x] G1: task generator, verifier, baseline evaluation
- [x] G2: rollouts, log-probs, group advantages
- [x] G3: GRPO loss (policy gradient + KL), LoRA training loop, resume
- [x] G4: real training run on a Kaggle T4 (200 steps, 78 min), via the Kaggle CLI
- [x] G5: reward hacking check of the real run (docs/REWARD_HACKING.md): no hacking found, one loophole, fixed in rewards.py
- [ ] Held-out evaluation of the trained adapter

## Setup
```bash
uv sync
make test     # reward-function test
make data     # writes data/train.jsonl and data/test.jsonl to look at
make smoke    # 10-step training run on the Mac (G=4, 4 prompts, 128 tokens), pipeline check only
make train    # real run; re-run the same command to resume from the last checkpoint
make baseline # base model on all 200 held-out problems
uv run python -m grpo.plot --run_name grpo_d2   # results/grpo_d2/plot.png from log.csv
```

## Baseline (base model, greedy, held-out set)
Qwen2.5-0.5B-Instruct, greedy decoding with repetition penalty 1.1 (Qwen's default), max 512 new tokens,
200 held-out problems (seed 1234). Plain greedy (penalty 1.0) gives 0.5% (`results/baseline_pure_greedy.json`).

| difficulty | n | accuracy | 95% CI (Wilson) | format rate (old rule) |
|---|---|---|---|---|
| 1 (two numbers, + -) | 200 | 2.0% | 0.8% - 5.0% | 80.5% (88.5%) |
| 2 (all three numbers, + -) **chosen** | 200 | 5.0% | 2.7% - 9.0% | 74.5% (86.5%) |

Format rate now also requires legal numbers (G5 fix); the new values were re-scored from the saved
answers, and accuracy did not change. Full outputs (scored with the old rule): `results/baseline.json` (difficulty 2) and `results/tune_d1.json`.

`uv run python -m grpo.rollout` prints one prompt with 8 sampled answers, rewards, and advantages.

## Training
Each run writes to `results/<run_name>/`: `config.json`, `log.csv` (step, mean_reward, accuracy,
mean_response_length, kl, loss), `samples/step_XXXX.json` every 20 steps, and LoRA checkpoints in
`checkpoints/step_XXXX/` every 50 steps. Any config field can be overridden: `--beta 0.02`.

On Kaggle the run is launched from the laptop with the Kaggle CLI (`kaggle/`, no token in the repo):
```bash
kaggle kernels push -p kaggle                                    # upload kaggle/run.py and start it on a T4
kaggle kernels status abhimanyu9327/grpo-reasoning-train         # QUEUED / RUNNING / COMPLETE / ERROR
kaggle kernels output abhimanyu9327/grpo-reasoning-train -p out  # download /kaggle/working + the log
```
To resume after a dead session, upload `results/<run_name>/` as a private Kaggle dataset and add it to
`dataset_sources` in `kaggle/kernel-metadata.json`; `run.py` copies it in and training continues.

Run `grpo_d2` (200 steps, T4, 21.7 s/step): training-rollout accuracy went from 2.7% (steps 1-20) to
48.2% (steps 181-200). These are sampled training answers, not the held-out set, so they are not
comparable to the baseline below. See `results/grpo_d2/plot.png` and docs/LEARNING.md (G4).

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
src/grpo/plot.py     plots log.csv to results/<run>/plot.png
src/grpo/hacking_check.py  counts reward-hacking patterns in saved samples
kaggle/              Kaggle kernel: metadata + run.py (clone, install, train)
tests/               the 3 tests
docs/LEARNING.md     plain-language notes per milestone
docs/REWARD_HACKING.md  reward hacking check with real examples
```
