import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM

from grpo.grpo import group_advantages
from grpo.rollout import completion_log_probs
from grpo.train import LORA_TARGETS, grpo_step


def test_one_update_raises_logprob_of_good_sample():
    torch.manual_seed(0)
    # A tiny random Qwen2 so the test runs in a second without real weights.
    tiny = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    model = get_peft_model(Qwen2ForCausalLM(tiny), LoraConfig(r=4, lora_alpha=8, target_modules=LORA_TARGETS))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    # One prompt, two answers (G = 2). Answer 0 got reward 1, answer 1 got reward 0.
    prompt = torch.randint(0, 64, (1, 5)).repeat(2, 1)
    roll = {"prompt_ids": prompt, "prompt_mask": torch.ones_like(prompt),
            "completion_ids": torch.randint(0, 64, (2, 6)),
            "completion_mask": torch.ones(2, 6, dtype=torch.long)}
    advantages = group_advantages(torch.tensor([1.0, 0.0]), group_size=2)
    assert advantages[0] > 0

    def answer_logprob():
        with torch.no_grad():
            lp = completion_log_probs(model, roll["prompt_ids"], roll["prompt_mask"],
                                      roll["completion_ids"], roll["completion_mask"])
        return lp.sum(dim=1)

    before = answer_logprob()
    grpo_step(model, optimizer, scaler, roll, advantages, beta=0.04, max_grad_norm=1.0,
              micro_batch_size=2, device="cpu")
    after = answer_logprob()

    assert after[0] > before[0]  # the good answer became more likely
