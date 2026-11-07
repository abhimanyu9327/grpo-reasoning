import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")  # write a file, no window needed
import matplotlib.pyplot as plt

PANELS = ["mean_reward", "accuracy", "kl", "mean_response_length"]
WINDOW = 10  # moving average: one step has only 64 answers, so single steps are noisy


def moving_average(values, window):
    out = []
    for i in range(len(values)):
        chunk = values[max(0, i - window + 1):i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="grpo_d2")
    args = parser.parse_args()

    run_dir = os.path.join("results", args.run_name)
    with open(os.path.join(run_dir, "log.csv")) as f:
        rows = list(csv.DictReader(f))
    steps = [int(r["step"]) for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, name in zip(axes.flat, PANELS):
        values = [float(r[name]) for r in rows]
        ax.plot(steps, values, alpha=0.3, label="per step")
        ax.plot(steps, moving_average(values, WINDOW), label=f"{WINDOW}-step average")
        ax.set_title(name)
        ax.set_xlabel("step")
    axes[0, 0].legend()
    fig.suptitle(f"{args.run_name}: training rollouts (not the held-out set)")
    fig.tight_layout()

    out = os.path.join(run_dir, "plot.png")
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
