import argparse
import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grpo.plot import PANELS, WINDOW, moving_average

# One table + one plot comparing training runs, e.g. an ablation against the main run.
# Usage: python -m grpo.compare --name group_size --runs g6_main_s0 g6_g4
LAST = 50  # the table averages the last 50 steps, when the curves have mostly flattened


def load(run_name):
    with open(os.path.join("results", run_name, "log.csv")) as f:
        return list(csv.DictReader(f))


def mean_and_se(values):
    # SE = std / sqrt(n). Steps are not independent, so treat this as a rough lower bound on noise.
    n = len(values)
    mean = sum(values) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    return mean, std / math.sqrt(n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    args = parser.parse_args()
    logs = {run: load(run) for run in args.runs}

    lines = [f"| run | steps | " + " | ".join(f"{p} (last {LAST})" for p in PANELS) + " |",
             "|---|---|" + "---|" * len(PANELS)]
    for run, rows in logs.items():
        cells = []
        for p in PANELS:
            mean, se = mean_and_se([float(r[p]) for r in rows[-LAST:]])
            cells.append(f"{mean:.3f} ± {se:.3f}")
        lines.append(f"| {run} | {len(rows)} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    print(table)

    os.makedirs("results/ablations", exist_ok=True)
    with open(f"results/ablations/{args.name}.md", "w") as f:
        f.write(table + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, p in zip(axes.flat, PANELS):
        for run, rows in logs.items():
            steps = [int(r["step"]) for r in rows]
            ax.plot(steps, moving_average([float(r[p]) for r in rows], WINDOW), label=run)
        ax.set_title(f"{p} ({WINDOW}-step average)")
        ax.set_xlabel("step")
    axes[0, 0].legend()
    fig.suptitle(f"ablation: {args.name} (training rollouts, not the held-out set)")
    fig.tight_layout()
    fig.savefig(f"results/ablations/{args.name}.png", dpi=120)
    print(f"saved results/ablations/{args.name}.md and .png")


if __name__ == "__main__":
    main()
