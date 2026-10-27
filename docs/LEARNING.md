# Learning notes

## G1: Task, verifier, and baseline

**What is the task?** Countdown: you get 3 numbers (1-20) and a target, and you must write an
arithmetic expression (+ - * / and parentheses) that equals the target, using each number at most
once. Example: numbers [3, 7, 12], target 16, answer `12 + 7 - 3`.

**Why this task for RL?** Checking an answer is easy and exact, even though finding the answer is
hard. So the reward can come from a few lines of Python instead of a human or a judge model.
This is called a *verifiable reward*. A verifier cannot be fooled the way a judge model can.

**How are problems made?** I build a random expression from the 3 numbers and use its value as the
target. So every target is reachable by construction. Difficulty controls which operators the
hidden solution uses (1: two numbers with + -, 2: all three with + -, 3: adds *, 4: adds /).
All problems come from one pool with a fixed seed. The first 200 unique problems are the test
set and the next 2000 are the train set, so the two sets never overlap (checked with an assert).

**The verifier (rewards.py).**
1. Find exactly one `<answer>...</answer>`. Two answers are rejected so the model cannot guess twice.
2. If the answer ends with `= <number>`, drop that part. We compute the value ourselves and never
   trust the model's claimed result.
3. Allow only digits, `+ - * /`, parentheses, and spaces. Then parse with Python's `ast` module
   and allow only whole numbers and the four binary operators. `**`, `//`, unary minus, and
   decimals are rejected.
4. The tree is evaluated by our own small function using `Fraction`, so `7/2` stays exact.
   Division by zero gives "wrong", not a crash. `eval()` is never used, because `eval()` would
   run any code the model writes.
5. Each number may be used at most as many times as it is given.

Reward: 1.0 if correct, 0.1 if the format is fine but the answer is wrong, 0.0 if the format is broken.
The 0.1 tells the model "at least the format was right". It is small so that guessing
the format alone can never be worth more than a correct answer.

**Format decisions, all based on measurements.**
- Without help, 0/8 replies passed the format check (mostly no tags at all). So the assistant reply is started with
  `<think>\n` (a "prefill").
- An example expression in a separate "Format:" block made the model copy the example
  instead of solving the problem. The example was moved into the sentence instead.
- The model never closed `</think>` (0/48). Requiring it would make every reward 0, and then
  RL has nothing to learn from. So only the answer tag is required.

**Baseline (base model, greedy with Qwen's default repetition penalty 1.1, 200 held-out problems).**

| difficulty | accuracy | 95% CI | format rate |
|---|---|---|---|
| 1 | 2.0% (4/200) | 0.8% - 5.0% | 88.5% |
| 2 | 5.0% (10/200) | 2.7% - 9.0% | 86.5% |

I picked difficulty 2. The target range was 5-30%: at 0%, RL never sees a reward of 1, and at 80%
there is little left to improve. Difficulty 2 is at the low edge of that range.
Surprise: difficulty 1 is *harder* for this model than difficulty 2, because the model almost
always uses all three numbers. All 10 correct answers have the shape `a - b + c`, so the base
model is mostly pattern-guessing, not searching. Keep this in mind later: if accuracy goes up,
check whether the model learned to reason or just learned a better guessing pattern.

Note on decoding: plain greedy (repetition penalty 1.0) gives only 0.5% (1/200, CI 0.1%-2.8%)
with 36% format rate (`results/baseline_pure_greedy.json`). The penalty matters a lot for this
small model, so it is set explicitly in `config.py` and every before/after comparison uses 1.1.

**95% confidence interval (Wilson).** With only 200 problems, 5% could really be anywhere from
about 2.7% to 9%. A later result is only a real improvement if it is clearly outside this noise.
I used Wilson instead of the simple `p ± 1.96·sqrt(p(1-p)/n)` because the simple formula breaks
near 0% (for 0/8 it gives [0, 0], which is overconfident).

## G2: Rollouts, log-probs, and group advantages

**Rollout.** A rollout means: give the model a prompt and let it write answers. For every prompt
we sample G = 8 answers (a *group*) with temperature 1.0 and score each one with the verifier.
`num_return_sequences=8` keeps the 8 answers of one prompt next to each other in the batch.

**Plain sampling only.** Qwen's generation config turns on top-p 0.8, top-k 20 and repetition
penalty 1.1 by default. For training I switch them all off. Later the loss uses the model's
log-probabilities, and those describe the model's *real* distribution. If the answers were
sampled from a changed distribution, the math would be about a different distribution than the
one that produced the data.

**Completion mask.** All answers in a batch are padded to the same length. The mask is 1 for real
tokens up to and including the first end-of-sequence token, and 0 for padding after it, so
padding never counts in the loss.

**Per-token log-probs.** For each answer token we want log p(token | prompt + earlier tokens).
One forward pass over prompt + answer gives logits. The logit at position t predicts token t+1,
so we shift by one, take log_softmax, and pick the log-prob of the token that was actually
written. Two details:
- Left padding: position ids must start at 0 on the first real token (same as generate()).
  Checked by hand: batched-with-padding vs. alone differ by at most 2e-5 (fp32 noise).
- `logits_to_keep` asks only for the logits of the answer part. The vocabulary has ~150k tokens,
  so full logits for the prompt would waste a lot of memory.

**Group advantage.** advantage = (reward - group mean) / (group std + 1e-6), computed per prompt.
Example from a real rollout (numbers [3, 4, 10], target 17): rewards were
`[0, 0, 0, 0.1, 0, 0, 0, 0]`. Mean = 0.0125, std = 0.035, so the one formatted answer gets
+2.475 and the other seven get -0.354. "Better than your siblings" is pushed up, "worse" is pushed
down. If all 8 rewards are equal, every advantage is exactly 0 (not NaN): that group teaches nothing.
Dividing by std makes the scale the same for every prompt, so an easy prompt and a hard prompt
push the model with similar strength.

**Why no value network?** Older methods like PPO train a second network (the critic) to guess
"how good is this prompt on average". GRPO uses the mean reward of the group as that guess.
That saves a whole model's memory, which matters a lot on a 16 GB T4.

**Measured training signal (base model, T = 1.0, 16 train prompts x 8 samples):**
- correct: 1/128 samples (0.8%), right format: 27%
- groups with at least one correct answer: 1/16
- groups with different rewards (non-zero advantages): 15/16

So at the start almost all of the signal is "format vs. no format" (0.1 vs. 0), not "correct vs.
wrong". Sampling at T = 1.0 is much worse than greedy (5% correct, 86.5% format). This is a small
sample (16 prompts), so it is a rough picture, but it is the main risk for G3.

## G3: The GRPO loss and the training loop

**The loss, per token.**
```
loss_token = -(A * log_prob)  +  beta * KL_token
```
Each token of an answer gets that answer's advantage A. Average over the real tokens of each
answer (mask), then over all answers in the batch.
- A > 0 (better than the group): minimizing -(A * log_prob) pushes log_prob **up**.
- A < 0 (worse than the group): it pushes log_prob **down**.

**KL term.** KL measures how far the policy (base + LoRA) has moved from the reference
(the same base model with LoRA switched off, via `model.disable_adapter()`, so no second copy
in memory). Per-token estimator ("k3"): with x = ref_logp - logp, `KL = exp(x) - x - 1`.
It is always >= 0, exactly 0 when both models agree, and grows when the policy drifts.
The KL term is a leash: improve the reward, but don't forget how to write normal language.

**Why no PPO clipping here.** PPO uses the ratio r = p_new(token) / p_old(token), where
p_old is the model that sampled the answers. We sample with the current weights and then take
exactly ONE optimizer step on that batch, so at loss time p_new = p_old and r = 1 everywhere.
The gradient of r * A at r = 1 equals the gradient of A * log p, so clipping would never kick in.
Clipping only matters if we reuse a batch for several steps (then r drifts from 1, and clipping
stops one batch from pushing the model too far).

**Training loop (train.py), one step:**
1. Take the next 8 train problems (fixed shuffled order, so resume sees the same prompts).
2. Sample G = 8 answers each (64 answers) and score them with the verifier.
3. Advantages per group.
4. Loss on micro-batches of 4 answers, gradients added up, then one AdamW step.
5. Log to `results/<run>/log.csv`; samples every 20 steps; LoRA checkpoint every 50 steps.
Re-running the same command resumes from the newest checkpoint (Kaggle sessions die).

### Hyperparameters (Hinglish mein)

| Naam | Value | Kyun |
|---|---|---|
| `num_steps` | 200 | 200 × 8 = 1600 prompts, 2000 wale train set se kam. Koi prompt do baar nahi aata. T4 par time ke hisaab se bhi theek hai. |
| `prompts_per_step` | 8 | Ek step mein 8 alag sawaal. Zyada sawaal = gradient kam noisy, lekin har step slow. |
| `group_size` (G) | 8 | Har sawaal ke 8 jawab. G chhota hoga toh "sab barabar" wale groups zyada honge (signal zero). G bada hoga toh generation slow. 8 ek common beech ka raasta hai. |
| `max_new_tokens` | 512 | Base model ke jawab average ~100-200 tokens ke hain. 512 kaafi room deta hai. Isse lambe jawab kaat diye jaate hain. |
| `temperature` | 1.0 | Model ki asli distribution se sampling, taaki loss ke log-probs aur data match karein. G2 mein dekha ki is temperature par format sirf ~27% hai. Agar G4 mein signal kam mila toh ise badalne par sochenge (tab log-probs bhi usi temperature se nikalne honge). |
| `learning_rate` | 5e-5 | Ek step mein weights kitna hilein. 1e-5 se 5e-5 ki range mein upar wala end liya, kyunki LoRA ka B matrix zero se start hota hai aur LoRA ko full fine-tuning se thoda zyada push chahiye. RL ka gradient bahut noisy hota hai, isliye isse bahut zyada nahi. Ye ek starting guess hai, measure nahi kiya gaya. T4 run mein KL aur reward dekh kar check karenge. |
| `beta` | 0.04 | KL ka weight, yani "leash" kitni tight hai. DeepSeekMath GRPO paper ki value hai. Zyada beta = model base se kam hatega (safe, lekin slow learning). Kam beta = tez learning, lekin model ajeeb text likhne lag sakta hai (reward hacking ka risk). |
| `max_grad_norm` | 1.0 | Gradient clipping: agar saare gradients ka combined size (norm) 1.0 se zyada ho, toh use scale karke 1.0 kar do. RL mein kabhi kabhi ek batch bahut bada gradient deta hai, aur ye ek kharab step ko model barbaad karne se rokta hai. |
| `lora_rank` | 16 | LoRA ke chhote matrices ki "chaudai". Rank 16 = 8.8M trainable parameters (poore model ka 1.75%, measured). Is chhote task ke liye kaafi hai. |
| `lora_alpha` | 32 | LoRA ka output alpha/rank = 2 se multiply hota hai. Common rule: alpha = 2 × rank. |
| LoRA dropout | 0.0 | Dropout se loss ke log-probs sampling waale log-probs se alag ho jaate. RL mein ye galat hai. |
| LoRA targets | q,k,v,o + gate,up,down | Attention aur MLP, dono ki saari projections. Model ko "kya dekhna hai" (attention) aur "kya calculate karna hai" (MLP) dono badalne ki azaadi milti hai. |
| `micro_batch_size` | 4 | 64 jawab ek saath backward karne ki memory nahi hai. 4-4 karke gradients jodte hain, aur result wahi hota hai jo ek bade batch se aata. Mac par 16 ek saath karne se out of memory hua tha. |
| AdamW weight_decay | 0.0 | Run chhota hai aur LoRA weights zero ke paas se shuru hote hain. Decay ki zaroorat nahi. |
| `seed` | 0 | Prompts ka order aur har step ki sampling fixed hai, toh run reproducible hai. |
| `samples_every` / `save_every` | 20 / 50 | Har 20 steps par jawab save hote hain (haath se padhne ke liye). Har 50 steps par checkpoint (Kaggle session mar jaaye toh resume). |

**fp16 on the T4.** The base model is loaded in fp16 (half the memory). LoRA weights are kept in
fp32 and the forward pass runs under `autocast`, so big matmuls are fast in fp16 while the
weights we update stay precise. fp16 gradients can become so small they round to 0
("underflow"), so a `GradScaler` multiplies the loss by a big number before backward and divides
the gradients back before the optimizer step. On the Mac everything is fp32 and the scaler is off.

**Checked by hand.**
- Test 3 also fails when the advantage sign is flipped (the good answer's log-prob went down),
  so the test really checks the direction of the update.
- Resume: copied the smoke run, deleted its step-10 checkpoint, re-ran with `--num_steps 6`.
  It resumed from step 5, dropped log rows 6-10, and the redone step 6 matched the original
  step 6 exactly (reward 0.15, kl 0.114915, loss 0.029292).
- Smoke run (10 steps, Mac M3, fp32): works end to end, but took 57 minutes (~5 min/step,
  heavy memory swapping). 16 answers in one backward ran out of MPS memory, so the smoke run
  uses micro-batches of 2. The smoke run is not meant to learn: 16 answers per step is far
  too few to read anything from the reward numbers.
- The loss value is not a progress metric in GRPO. Advantages in a group sum to zero, so the
  loss stays near 0 and jumps around (smoke: between -0.18 and +0.03). Watch mean_reward,
  accuracy, and kl instead.
