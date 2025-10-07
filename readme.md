## NAI Project — RL-driven Job Orchestrator

P16 - Aayush Mishra (2022011) & Sarthak Sharma (2022456)

An experimental job orchestration system that can execute workloads either via a simple FIFO policy or a Reinforcement Learning (RL) policy trained to improve throughput and reduce wait times. Jobs can be simulated or executed with `stress-ng`, and all runs emit structured metrics for analysis and visualization.

### Key Features

- **Pluggable policies**: `fifo` and `rl` (train or load a saved policy)
- **Action-masked RL environment** with sliding window over the queue
- **Deterministic, structured metrics** saved as `.jsonl` and `.csv`
- **TensorBoard** logging during RL training
- **Windows-friendly** via WSL; Linux/macOS also supported

---

## Quickstart

### 1) Prerequisites

- Python 3.10+
- pip, venv (or conda)
- Optional: `stress-ng` for real execution (else use `--simulate`)
  - Linux/WSL: `sudo apt update && sudo apt install -y stress-ng`

On Windows, use WSL for `stress-ng` or run in simulation mode.

### 2) Install Python dependencies

Create and activate a virtual environment, then install packages:

```bash
python -m venv .venv
source .venv/bin/activate    # PowerShell: .venv\Scripts\Activate.ps1  |  CMD: .venv\Scripts\activate.bat
pip install --upgrade pip
pip install torch gymnasium numpy tensorboard
```

> Note: Install a CUDA-enabled PyTorch build if desired (see the official PyTorch selector). CPU-only works fine for this project.

### 3) Run a FIFO simulation

```bash
python orchestrator/orchestrator.py \
  --policy fifo \
  --generate-jobs 10 \
  --simulate \
  --output-dir artifacts
```

This creates metrics under `artifacts/metrics_*.jsonl` and `artifacts/metrics_*.csv`.

### 4) Train the RL scheduler (simulation)

```bash
python orchestrator/orchestrator.py \
  --policy rl \
  --rl-train \
  --rl-episodes 200 \
  --rl-window-size 7 \
  --rl-simulate \
  --generate-jobs 50 \
  --simulate \
  --output-dir artifacts
```

The trained model is saved to `models/job_policy.pth`. Training logs go to `runs/`.

View training curves in TensorBoard:

```bash
tensorboard --logdir runs
```

### 5) Use a saved RL model (inference)

```bash
python orchestrator/orchestrator.py \
  --policy rl \
  --rl-model-path models/job_policy.pth \
  --rl-window-size 7 \
  --generate-jobs 50 \
  --simulate \
  --output-dir artifacts
```

---

## Project Architecture

High-level design, flowcharts, class diagrams, and the RL sequence diagram are documented in `ARCHITECTURE.md`. Highlights:

- `jobs/producer.py` creates `Job` instances and auto-registers them in `Job.registry`.
- `orchestrator/orchestrator.py` selects a policy (`fifo` or `rl`), orders jobs, then executes them.
- `scheduler/rl_scheduler.py` contains the `JobEnv`, `PolicyNetwork`, training (`train_rl`) and testing (`test_rl`).
- `executor/executor.py` executes jobs (simulation or `stress-ng`), returns metrics, and writes `.jsonl`/`.csv`.

Artifacts:

- Metrics: `artifacts/metrics_*.jsonl` and `artifacts/metrics_*.csv`
- RL runs: `runs/` (for TensorBoard)
- Saved models: `models/job_policy.pth`

---

## CLI Reference (orchestrator)

```bash
python orchestrator/orchestrator.py [OPTIONS]
```

- `--policy {fifo,rl}`: Select scheduling policy (default: `fifo`).
- `--generate-jobs INT`: Number of jobs to create (default: 10).
- `--n-min INT` / `--n-max INT`: Range for `n` (CPU workers for `stress-ng`).
- `--p-min INT` / `--p-max INT`: Range for `p` (seconds to run).
- `--simulate`: Execute via sleep-based simulation instead of `stress-ng`.
- `--output-dir PATH`: Where to write metrics files (default: `artifacts`).

RL-specific:

- `--rl-train`: Train an RL policy instead of loading a checkpoint.
- `--rl-episodes INT`: Number of training episodes (default: 200).
- `--rl-window-size INT`: Sliding window size for the environment (default: 7).
- `--rl-model-path PATH`: Path to a saved policy (`models/job_policy.pth`). Required if not training.
- `--rl-simulate`: Use simulation during RL training/testing.
- `--rl-gamma FLOAT`: Discount factor (default: 0.99).
- `--rl-lr FLOAT`: Learning rate (default: 1e-3).

---

## How It Works

- **Job generation**: `JobProducer` samples `n` and `p` within provided ranges and instantiates `Job` dataclasses. Each `Job` is stored in `Job.registry`.
- **Policy selection**:
  - `fifo`: Sorts by `(creation_time, job_id)`.
  - `rl`: Uses `JobEnv` to pick actions with a policy network. In training, actions are sampled; in inference, argmax over masked probabilities is used.
- **Execution**: `executor.execute_job` records `queue_wait_s`, `duration_s`, timestamps, and success status. In simulation, `time.sleep(p)` emulates run time. With `stress-ng`, the command used is equivalent to `stress-ng --cpu {n} --timeout {p}s --metrics-brief`.
- **Logging**: All job metrics are collected and written once per run as `.jsonl` and `.csv`. RL training also logs to TensorBoard.

---

## Repository Structure

```
NAI_Project/
  ARCHITECTURE.md
  orchestrator/
    orchestrator.py
  scheduler/
    rl_scheduler.py
  executor/
    executor.py
    docker.yaml
  jobs/
    job.py
    producer.py
  artifacts/            # output metrics (.jsonl, .csv)
  models/               # saved RL policies
  runs/                 # TensorBoard logs
```

---

## Stress-ng (optional)

Install on Debian/Ubuntu/WSL:

```bash
sudo apt update && sudo apt install -y stress-ng
```

If `stress-ng` is absent or `--simulate` is passed, jobs are executed via a sleep-based simulator. You can mix and match: train RL in simulation and later test with `stress-ng` by omitting `--simulate`.

---

## Troubleshooting

- **PyTorch not found**: Ensure the venv is activated and run `pip install torch` (use the official installer for CUDA builds).
- **gymnasium missing**: `pip install gymnasium`.
- **TensorBoard not found**: `pip install tensorboard`.
- **Permission issues on Windows**: Prefer running inside WSL; otherwise stick with `--simulate`.
- **`--policy rl` without `--rl-train` or `--rl-model-path`**: You must provide `--rl-model-path` when not training.

---

