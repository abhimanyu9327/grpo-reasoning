import argparse

import torch
import torch.nn.functional as F

from grpo import config
from grpo.rewards import compute_reward
from grpo.task import RESPONSE_PREFIX, make_prompt, make_splits


def completion_mask_from_ids(completion_ids, eos_ids):
    # 1 for real tokens up to and including the first EOS, 0 for the padding after it.
    is_eos = torch.isin(completion_ids, torch.tensor(eos_ids, device=completion_ids.device))
    # Count how many EOS tokens came strictly before each position.
    eos_before = is_eos.long().cumsum(dim=1) - is_eos.long()
    return (eos_before == 0).long()


@torch.no_grad()
def sample_completions(model, tokenizer, problems, group_size, max_new_tokens, temperature=1.0):
    prompts = [make_prompt(p, tokenizer) for p in problems]
    batch = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    # Plain sampling from the model's distribution: no top-p/top-k/repetition penalty
    # (Qwen's generation_config turns those on by default). RL needs the samples to
    # come from the same distribution whose log-probs we compute later.
    out = model.generate(
        **batch, do_sample=True, temperature=temperature, top_p=1.0, top_k=0,
        repetition_penalty=1.0, max_new_tokens=max_new_tokens,
        num_return_sequences=group_size, pad_token_id=tokenizer.pad_token_id,
    )
    # num_return_sequences keeps the G samples of one prompt next to each other:
    # rows 0..G-1 are prompt 0, rows G..2G-1 are prompt 1, ...
    prompt_len = batch["input_ids"].shape[1]
    completion_ids = out[:, prompt_len:]
    eos_ids = [tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.pad_token_id]
    completion_mask = completion_mask_from_ids(completion_ids, eos_ids)

    texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    texts = [RESPONSE_PREFIX + t for t in texts]  # the prefill is part of the answer
    rewards = []
    for i, text in enumerate(texts):
        p = problems[i // group_size]
        rewards.append(compute_reward(text, p["numbers"], p["target"]))

    return {
        "prompt_ids": batch["input_ids"].repeat_interleave(group_size, dim=0),
        "prompt_mask": batch["attention_mask"].repeat_interleave(group_size, dim=0),
        "completion_ids": completion_ids,
        "completion_mask": completion_mask,
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "texts": texts,
    }


def completion_log_probs(model, prompt_ids, prompt_mask, completion_ids, completion_mask):
    # log p(token_t | everything before it), for every completion token.
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    # With left padding, positions must start at 0 on the first real token,
    # exactly like generate() does. Otherwise the log-probs come out slightly wrong.
    position_ids = (attention_mask.cumsum(dim=1) - 1).clamp(min=0)
    num_completion = completion_ids.shape[1]
    # Only ask for the logits we need (saves a lot of memory with a 150k vocab).
    logits = model(input_ids=input_ids, attention_mask=attention_mask,
                   position_ids=position_ids, logits_to_keep=num_completion + 1).logits
    # The logit at position t predicts token t+1, so drop the last one.
    logits = logits[:, :-1, :].float()
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=completion_ids.unsqueeze(-1)).squeeze(-1)


if __name__ == "__main__":
    # Show one real rollout: 1 prompt, G completions, rewards and advantages.
    from grpo.eval import load_model, pick_device
    from grpo.grpo import group_advantages

    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0)  # which train problem
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device, dtype = pick_device()
    model, tokenizer = load_model(config.MODEL_PATH, device, dtype)
    train, _ = make_splits()
    problem = train[args.index]

    roll = sample_completions(model, tokenizer, [problem], config.GROUP_SIZE,
                              config.MAX_NEW_TOKENS, config.TEMPERATURE)
    advantages = group_advantages(roll["rewards"], config.GROUP_SIZE)
    lengths = roll["completion_mask"].sum(dim=1)

    print(f"numbers={problem['numbers']}  target={problem['target']}\n")
    for i in range(config.GROUP_SIZE):
        print(f"=== completion {i}  reward={roll['rewards'][i]:.1f}  "
              f"advantage={advantages[i]:+.3f}  tokens={lengths[i]}")
        print(roll["texts"][i].strip()[-400:], "\n")
    print("rewards:   ", [round(r, 2) for r in roll["rewards"].tolist()])
    print("advantages:", [round(a, 3) for a in advantages.tolist()])
