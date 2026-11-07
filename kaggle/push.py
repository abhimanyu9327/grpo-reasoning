# Push one training run as its own Kaggle kernel, so several runs can be queued at once.
# Usage: python kaggle/push.py <run_name> [train.py args...]
#   python kaggle/push.py g6_g4 --group_size 4
# It copies kaggle/run.py with RUN_NAME and TRAIN_ARGS filled in, plus a metadata file with
# its own kernel id (abhimanyu9327/grpo-<run_name>), into a temp folder and pushes that.
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

run_name, extra = sys.argv[1], sys.argv[2:]
slug = "grpo-" + run_name.replace("_", "-")

with open(os.path.join(HERE, "kernel-metadata.json")) as f:
    meta = json.load(f)
meta["id"] = f"abhimanyu9327/{slug}"
meta["title"] = slug

lines = []
with open(os.path.join(HERE, "run.py")) as f:
    for line in f:
        if line.startswith("RUN_NAME = "):
            line = f"RUN_NAME = {run_name!r}\n"
        elif line.startswith("TRAIN_ARGS = "):
            line = f"TRAIN_ARGS = {['--run_name', run_name] + extra!r}\n"
        lines.append(line)

folder = tempfile.mkdtemp()
with open(os.path.join(folder, "kernel-metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)
with open(os.path.join(folder, "run.py"), "w") as f:
    f.writelines(lines)
print(f"pushing {meta['id']} with TRAIN_ARGS = {['--run_name', run_name] + extra}")
subprocess.run(["kaggle", "kernels", "push", "-p", folder], check=True)
