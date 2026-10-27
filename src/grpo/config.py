import os
from dataclasses import dataclass

# Use local weights if they exist (Mac), otherwise download from the Hub (Kaggle).
LOCAL_WEIGHTS = os.path.expanduser("~/mini-infer/weights/Qwen2.5-0.5B-Instruct")
MODEL_PATH = os.environ.get(
    "GRPO_MODEL",
    LOCAL_WEIGHTS if os.path.isdir(LOCAL_WEIGHTS) else "Qwen/Qwen2.5-0.5B-Instruct",
)

# Data
DIFFICULTY = 2          # chosen by measuring the base model, see docs/LEARNING.md
DATA_SEED = 1234
NUM_TRAIN = 2000
NUM_TEST = 200

# Generation
MAX_NEW_TOKENS = 512
# Evaluation decoding: greedy with Qwen's default repetition penalty. Measured on the
# 200 held-out problems: 5.0% with 1.1, but 0.5% with plain greedy (1.0). Train-time
# sampling never uses it (see rollout.py). Before/after training use the same setting.
EVAL_REPETITION_PENALTY = 1.1

# Rollouts
GROUP_SIZE = 8          # G: completions sampled per prompt
TEMPERATURE = 1.0       # sample from the model's real distribution


@dataclass
class TrainConfig:
    run_name: str = "grpo_d2"
    seed: int = 0
    num_steps: int = 200            # 200 steps x 8 prompts = 1600 prompts, less than one pass over train
    prompts_per_step: int = 8
    group_size: int = GROUP_SIZE
    max_new_tokens: int = MAX_NEW_TOKENS
    temperature: float = TEMPERATURE
    learning_rate: float = 5e-5     # top of 1e-5..5e-5: LoRA starts at zero and needs a bit more push
    beta: float = 0.04              # KL weight (value from the DeepSeekMath GRPO paper)
    max_grad_norm: float = 1.0
    lora_rank: int = 16
    lora_alpha: int = 32            # LoRA output is scaled by alpha / rank = 2
    micro_batch_size: int = 4       # answers per forward/backward; 16 at once ran out of memory on the Mac
    samples_every: int = 20
    save_every: int = 50


# Tiny run for the Mac: proves the pipeline works end to end, not meant to learn.
SMOKE = TrainConfig(run_name="smoke", num_steps=10, prompts_per_step=4, group_size=4,
                    max_new_tokens=128, micro_batch_size=2, samples_every=5, save_every=5)
