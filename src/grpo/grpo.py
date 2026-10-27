import torch


def group_advantages(rewards, group_size, eps=1e-6):
    # rewards: flat tensor of shape (num_prompts * G,), the G samples of one prompt are next
    # to each other. Each sample is compared only with the other answers to the SAME prompt.
    groups = rewards.view(-1, group_size)
    mean = groups.mean(dim=1, keepdim=True)
    std = groups.std(dim=1, keepdim=True)
    # If all rewards in a group are equal, std = 0 and (r - mean) = 0, so the result is 0
    # (not NaN): the group has nothing to teach.
    advantages = (groups - mean) / (std + eps)
    return advantages.view(-1)


def kl_per_token(logp, ref_logp):
    # "k3" estimator of KL(policy || reference), one number per token:
    #   x = ref_logp - logp,  kl = exp(x) - x - 1
    # It is always >= 0, is exactly 0 when the two models agree, and is unbiased
    # when the tokens were sampled from the policy (which they are).
    x = ref_logp - logp
    return torch.exp(x) - x - 1


def grpo_loss(logp, ref_logp, advantages, mask, beta):
    # logp, ref_logp, mask: (num_samples, num_tokens). advantages: (num_samples,)
    # Policy-gradient term: push up log-probs of tokens in good answers (A > 0),
    # push down tokens in bad answers (A < 0). Every token of an answer gets that answer's A.
    pg = -(advantages.unsqueeze(1) * logp)
    kl = kl_per_token(logp, ref_logp)
    per_token = pg + beta * kl
    # Average over the real tokens of each answer, then over answers.
    lengths = mask.sum(dim=1).clamp(min=1)
    loss = ((per_token * mask).sum(dim=1) / lengths).mean()
    mean_kl = ((kl * mask).sum(dim=1) / lengths).mean()
    return loss, mean_kl.detach()
