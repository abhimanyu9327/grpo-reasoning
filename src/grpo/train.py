import argparse
import contextlib
import csv
import json
import os
import random
from dataclasses import asdict, fields, replace

import torch
from peft import LoraConfig, PeftModel, get_peft_model

from grpo import config
from grpo.eval import load_model, pick_device
from grpo.grpo import group_advantages, grpo_loss
from grpo.rollout import completion_log_probs, sample_completions
from grpo.task import make_splits

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LOG_COLUMNS = ["step", "mean_reward", "accuracy", "mean_response_length", "kl", "loss"]


def amp_context(device):
    # T4: run matmuls in fp16, while the LoRA weights themselves stay fp32.
    if device == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def latest_checkpoint(run_dir):
    ckpt_root = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(ckpt_root):
        return 0, None
    steps = [int(name.split("_")[1]) for name in os.listdir(ckpt_root) if name.startswith("step_")]
    if not steps:
        return 0, None
    return max(steps), os.path.join(ckpt_root, f"step_{max(steps):04d}")


def add_lora(base, cfg, ckpt_path):
    if ckpt_path:
        model = PeftModel.from_pretrained(base, ckpt_path, is_trainable=True)
    else:
        # dropout 0: with dropout the log-probs in the loss would not match the ones that sampled
        lora = LoraConfig(r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=0.0,
                          target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
        model = get_peft_model(base, lora)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()  # LoRA weights in fp32, even when the base is fp16
    return model


def grpo_step(model, optimizer, scaler, roll, advantages, beta, max_grad_norm,
              micro_batch_size, device):
    # One optimizer step on one batch of rollouts. Gradients are summed over micro-batches,
    # each scaled by its share of the batch, so the result equals one big batch.
    n = roll["completion_ids"].shape[0]
    total_loss, total_kl = 0.0, 0.0
    optimizer.zero_grad()
    for start in range(0, n, micro_batch_size):
        end = min(start + micro_batch_size, n)
        args = (roll["prompt_ids"][start:end], roll["prompt_mask"][start:end],
                roll["completion_ids"][start:end], roll["completion_mask"][start:end])
        with amp_context(device):
            # Reference model = same weights with LoRA switched off. No second copy needed.
            with torch.no_grad(), model.disable_adapter():
                ref_logp = completion_log_probs(model, *args)
            logp = completion_log_probs(model, *args)
        loss, kl = grpo_loss(logp, ref_logp, advantages[start:end], args[3], beta)
        share = (end - start) / n
        scaler.scale(loss * share).backward()
        total_loss += loss.item() * share
        total_kl += kl.item() * share
    scaler.unscale_(optimizer)  # clip the real gradients, not the scaled ones
    trainable = [p for p in model.parameters() if p.requires_grad]
    torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
    scaler.step(optimizer)
    scaler.update()
    return total_loss, total_kl


def save_checkpoint(model, optimizer, scaler, step, run_dir):
    path = os.path.join(run_dir, "checkpoints", f"step_{step:04d}")
    model.save_pretrained(path)  # saves only the LoRA adapter (~35 MB)
    torch.save({"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict()},
               os.path.join(path, "trainer_state.pt"))


def prepare_log(path, resume_step):
    # On resume, drop rows logged after the checkpoint: those steps will be redone.
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            rows = [r for r in csv.DictReader(f) if int(r["step"]) <= resume_step]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def save_samples(run_dir, step, problems, roll, advantages, group_size):
    rows = []
    for i, text in enumerate(roll["texts"]):
        p = problems[i // group_size]
        rows.append({**p, "reward": roll["rewards"][i].item(),
                     "advantage": round(advantages[i].item(), 3), "response": text})
    os.makedirs(os.path.join(run_dir, "samples"), exist_ok=True)
    with open(os.path.join(run_dir, "samples", f"step_{step:04d}.json"), "w") as f:
        json.dump(rows, f, indent=2)


def train(cfg):
    run_dir = os.path.join("results", cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    device, dtype = pick_device()
    start_step, ckpt_path = latest_checkpoint(run_dir)
    base, tokenizer = load_model(config.MODEL_PATH, device, dtype)
    model = add_lora(base, cfg, ckpt_path)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=cfg.learning_rate, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    if ckpt_path:
        state = torch.load(os.path.join(ckpt_path, "trainer_state.pt"), map_location=device)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        print(f"resumed from {ckpt_path}")
    log_path = os.path.join(run_dir, "log.csv")
    prepare_log(log_path, start_step)

    # Fixed shuffled order of train problems; step s always uses the same prompts,
    # so a resumed run sees exactly what the original run would have seen.
    train_set, _ = make_splits()
    order = list(range(len(train_set)))
    random.Random(cfg.seed).shuffle(order)

    for step in range(start_step + 1, cfg.num_steps + 1):
        torch.manual_seed(cfg.seed * 100_000 + step)
        first = (step - 1) * cfg.prompts_per_step
        problems = [train_set[order[(first + j) % len(order)]] for j in range(cfg.prompts_per_step)]

        with amp_context(device):
            roll = sample_completions(model, tokenizer, problems, cfg.group_size,
                                      cfg.max_new_tokens, cfg.temperature)
        rewards = roll["rewards"]
        advantages = group_advantages(rewards, cfg.group_size).to(device)
        loss, kl = grpo_step(model, optimizer, scaler, roll, advantages, cfg.beta,
                             cfg.max_grad_norm, cfg.micro_batch_size, device)

        row = {"step": step, "mean_reward": round(rewards.mean().item(), 4),
               "accuracy": round((rewards == 1.0).float().mean().item(), 4),
               "mean_response_length": round(roll["completion_mask"].sum(1).float().mean().item(), 1),
               "kl": round(kl, 6), "loss": round(loss, 6)}
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=LOG_COLUMNS).writerow(row)
        print(row)

        if step % cfg.samples_every == 0:
            save_samples(run_dir, step, problems, roll, advantages, cfg.group_size)
        if step % cfg.save_every == 0:
            save_checkpoint(model, optimizer, scaler, step, run_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    for f in fields(config.TrainConfig):  # every config field can be overridden: --beta 0.02
        parser.add_argument(f"--{f.name}", type=type(f.default), default=None)
    args = vars(parser.parse_args())
    cfg = config.SMOKE if args.pop("smoke") else config.TrainConfig()
    cfg = replace(cfg, **{k: v for k, v in args.items() if v is not None})
    train(cfg)


if __name__ == "__main__":
    main()
