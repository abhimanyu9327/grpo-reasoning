# Kaggle kernel: clone the public repo, install the pinned dependencies, run train.py.
# Kaggle only keeps /kaggle/working after the run, so results are written there.
# No tokens here: the repo is public and the model downloads without login.
import glob
import os
import shutil
import subprocess
import time

REPO = "https://github.com/abhimanyu9327/grpo-reasoning.git"
RUN_NAME = "grpo_d2"
TRAIN_ARGS = ["--run_name", RUN_NAME]   # everything else from TrainConfig: 200 steps, ckpt every 50
# Tiny T4 timing test that was run first:
# RUN_NAME = "t4_test"
# TRAIN_ARGS = ["--run_name", RUN_NAME, "--num_steps", "10", "--samples_every", "5", "--save_every", "10"]

CODE_DIR = "/tmp/grpo-reasoning"      # not under /kaggle/working, so the code is not saved as output
OUT_DIR = "/kaggle/working/results"


def run(cmd, cwd=None, env=None):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


run(["nvidia-smi"])
run(["git", "clone", "--depth", "1", REPO, CODE_DIR])

# train.py writes to results/<run_name>/. Point that folder at /kaggle/working.
os.makedirs(OUT_DIR, exist_ok=True)
shutil.rmtree(os.path.join(CODE_DIR, "results"), ignore_errors=True)
os.symlink(OUT_DIR, os.path.join(CODE_DIR, "results"))

# Resume: if an attached dataset has results/<RUN_NAME>/ from an earlier session, copy it in.
# train.py then continues from the newest checkpoint in it.
found = glob.glob(f"/kaggle/input/**/{RUN_NAME}/checkpoints", recursive=True)
if found:
    prev_run = os.path.dirname(found[0])
    print("resuming from", prev_run, flush=True)
    shutil.copytree(prev_run, os.path.join(OUT_DIR, RUN_NAME), dirs_exist_ok=True)

# uv.lock pins every package version. The venv lives in /tmp so it is not saved as output.
env = {**os.environ, "UV_PROJECT_ENVIRONMENT": "/tmp/venv", "PYTHONUNBUFFERED": "1"}
run(["pip", "install", "-q", "uv==0.12.9"])
run(["uv", "sync", "--frozen", "--no-dev"], cwd=CODE_DIR, env=env)

start = time.time()
run(["uv", "run", "--frozen", "--no-dev", "python", "-m", "grpo.train", *TRAIN_ARGS], cwd=CODE_DIR, env=env)
print(f"training wall time: {time.time() - start:.0f} s", flush=True)
