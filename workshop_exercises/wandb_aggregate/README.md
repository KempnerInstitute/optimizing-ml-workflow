# W&B Sweep with Cross-Run Live Loss Averaging via Redis

Compute **cross-run averages** of training and validation losses in real time during W&B hyperparameter sweeps. Workers publish per-epoch losses to a shared Redis server; aggregators compute averages and publish them back so all workers log synchronized `Train Loss Average` and `Validation Loss Average` metrics to W&B.

## Features
- Live cross-run averaging for early insights into hyperparameter performance
- Cluster-friendly: Redis as Slurm job, aggregators as CPU jobs, workers as W&B agents
- Minimal code changes: `LiveAverager` class handles Redis pub/sub
- Flexible strategies: timeout, quorum, or full-count triggers

## Files
- **`wandb_sweep.py`** - Training script with W&B and live averaging
- **`live_avg.py`** - Client library for publishing/subscribing to averages
- **`aggregator.py`** - Computes and publishes cross-run averages
- **`sweep_config.yaml`** - W&B sweep definition
- **`config.yaml`** — Training hyperparameters
- **`redis_server.slrm`** - Launches Redis and creates `redis_connection.env`
- **`sweep_aggregator.slrm`** - Runs aggregator (train or validation)
- **`sweep_submission.slrm`** - Launches W&B agent workers

## Prerequisites
- Conda environment from `environment.yml` (includes `wandb`, `redis`, `torch`, etc.)
- Slurm cluster with Apptainer/Singularity, or `redis-server` via pip
- W&B account configured

**Note:** Slurm scripts auto-activate the conda environment. Update `CONDA_ENV` path in scripts to match your environment.

## Workflow

**Run from this folder:**
```bash
cd /path/to/workshop_exercises/wandb_sweep
```

**Overview:** The workflow has 5 steps: (1) Launch Redis server, (2) Create W&B sweep, (3) Start aggregators, (4) Submit worker agents, (5) Monitor results. Steps 1-2 can run in either order, but Redis must be ready before starting steps 3-4.

### Step 0 (Optional): Pre-download CIFAR-10 dataset

To avoid race conditions when multiple workers try to download CIFAR-10 simultaneously, download the dataset once before launching workers:

```bash
# Activate your conda environment
conda activate /path/to/your/conda_env

# Download CIFAR-10 dataset
python -c "
import torchvision
torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=True)
print('Training data downloaded')
torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=True)
print('Test data downloaded')
"
```

### Step 1: Launch Redis

```bash
sbatch redis_server.slrm
```

The script auto-detects the assigned node, selects an available port, and writes connection info to `redis_connection.env`. Containers cached in `.singularity_cache/` or `.apptainer_cache/`.

**Verify:**
```bash
squeue -u $USER | grep redis
cat redis_connection.env
tail redis-*.out  # Look for "Ready to accept connections tcp"
```

**Optional:** Set custom port/password before submitting:
```bash
export REDIS_PORT=26379
export REDIS_PASSWORD=your-secret
sbatch redis_server.slrm
```

### Step 2: Create W&B sweep

Activate your conda environment and create the sweep:
```bash
conda activate /path/to/your/conda_env  # e.g., /n/holylfs06/LABS/kempner_dev/Lab/mmsh/conda_envs/sweep_ave
wandb login  # If not already logged in
wandb sweep sweep_config.yaml
```

Copy the returned sweep ID (e.g., `abcd1234`).

### Step 3: Start aggregators

**Training loss:**
```bash
export SWEEP_ID=abcd1234
export LIVE_AVG_NS=loss
sbatch sweep_aggregator.slrm
```

**Validation loss (optional):**
```bash
export SWEEP_ID=abcd1234
export LIVE_AVG_NS=val
sbatch sweep_aggregator.slrm
```

Both can run concurrently. Auto-sources `redis_connection.env`.

**Important:** After starting the aggregators, unset `LIVE_AVG_NS` to avoid polluting the worker environment:
```bash
unset LIVE_AVG_NS
```

### Step 4: Submit workers

Edit `sweep_submission.slrm` lines 20-22:
```bash
export ENTITY=your_entity
export PROJECT=your_project
export SWEEP_ID=abcd1234
```

Submit:
```bash
sbatch sweep_submission.slrm
```

Default: 12 workers. Adjust `#SBATCH --array=1-12` if needed.

**Important:** If you change the array size, also update `EXPECTED_RUNS` in `sweep_aggregator.slrm` to match (e.g., array `1-12` = 12 runs).

### Step 5: Monitor

- **W&B UI:** Check for `Train Loss Average` and `Validation Loss Average` on each run
- **Aggregator logs:** `tail -f slurm-agg-*.out`
- **Worker logs:** `tail -f img-*.out`

## Architecture

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          W&B Sweep Orchestration                            │
│                   (Coordinates hyperparameter assignments)                  │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │   Multiple Workers      │
                    │        W&B Agents       │
                    └────────────┬────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
        ▼                        ▼                        ▼
   ┌─────────┐            ┌─────────┐            ┌─────────┐
   │Worker 1 │            │Worker 2 │    ...     │Worker N │
   │         │            │         │            │         │
   │wandb_   │            │wandb_   │            │wandb_   │
   │sweep.py │            │sweep.py │            │sweep.py │
   └────┬────┘            └────┬────┘            └────┬────┘
        │                      │                      │
        │ ┌────────────────────┘                      │
        │ │                                           │
        │ │  Each worker (LiveAverager):              │
        │ │  1. Trains model with assigned params     │
        │ │  2. Computes loss per epoch               │
        ▼ ▼  3. Publishes & subscribes via Redis      ▼
   ┌─────────────────────────────────────────────────────┐
   │              Redis Pub/Sub Server                   │
   │         (Slurm job on shared partition)             │
   │                                                     │
   │  Channels:                                          │
   │  • loss:REPORTS:{SWEEP_ID}    (workers → agg)       │
   │  • loss:AVGS:{SWEEP_ID}       (agg → workers)       │
   │  • val:REPORTS:{SWEEP_ID}     (workers → agg)       │
   │  • val:AVGS:{SWEEP_ID}        (agg → workers)       │
   │                                                     │
   │  State (Redis hashes):                              │
   │  • loss:STATE:{SWEEP_ID}:{STEP} (sum, count, ...)   │
   │  • val:STATE:{SWEEP_ID}:{STEP}  (sum, count, ...)   │
   └───────┬──────────────────────────────┬──────────────┘
           │                              │
           │ Subscribe to reports         │ Subscribe to reports
           ▼                              ▼
   ┌──────────────────┐          ┌──────────────────┐
   │  Aggregator      │          │  Aggregator      │
   │  (Train Loss)    │          │  (Val Loss)      │
   │                  │          │                  │
   │  aggregator.py   │          │  aggregator.py   │
   │  LIVE_AVG_NS=    │          │  LIVE_AVG_NS=    │
   │    "loss"        │          │    "val"         │
   └────────┬─────────┘          └────────┬─────────┘
            │                             │
            │ Compute averages            │ Compute averages
            │ Publish to AVGS channels    │ Publish to AVGS channels
            │                             │
            └─────────────┬───────────────┘
                          │
          ┌───────────────┴────────────────┐
          │  Workers (LiveAverager)        │
          │  Subscribe to AVGS channels    │
          └───────────────┬────────────────┘
                          │
                          ▼
                ┌───────────────────┐
                │  Weights & Biases │
                │    (W&B Cloud)    │
                │                   │
                │  Logged Metrics:  │
                │  • Train Loss     │
                │  • Train Loss     │
                │    Average        │
                │  • Validation     │
                │    Loss           │
                │  • Validation     │
                │    Loss Average   │
                │  • Validation     │
                │    Accuracy       │
                └───────────────────┘
```

### Component Interaction

**Workers** publish losses to Redis channels after each epoch:
- Train: `loss:REPORTS:{SWEEP_ID}`
- Validation: `val:REPORTS:{SWEEP_ID}`

**Aggregators** subscribe to reports, maintain state in Redis hashes, and emit averages when triggered (full count, timeout, or quorum).

**Workers** subscribe to average channels and log `Train Loss Average`/`Validation Loss Average` to W&B.

### Message Flow (Per Epoch)

1. **Worker Training**: Each worker completes an epoch and computes train/validation loss
2. **Publish to Redis**: `LiveAverager.report()` publishes `{"run_id", "step", "loss"}` to `loss:REPORTS:{SWEEP_ID}`
3. **Aggregator Receives**: Aggregator increments sum and count in `loss:STATE:{SWEEP_ID}:{STEP}`
4. **Trigger Check**: Aggregator checks if ready to emit (full count, timeout, or quorum met)
5. **Publish Average**: Aggregator computes `avg = sum/count` and publishes `{"step", "avg", "n", "complete"}` to `loss:AVGS:{SWEEP_ID}`
6. **Workers Subscribe**: All workers' background threads receive the average
7. **Log to W&B**: Workers log `{"Train Loss Average": avg}` to their respective W&B runs
8. **Same for Validation**: Parallel flow for validation loss with `val:*` channels

This enables **real-time cross-run insights**: you can see how the average across all hyperparameter combinations performs as training progresses!

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SWEEP_ID` | *required* | W&B sweep ID |
| `REDIS_HOST/PORT` | *auto* | From `redis_connection.env` |
| `REDIS_PASSWORD` | none | Optional auth |
| `EXPECTED_RUNS` | 12 | Workers per step |
| `STEP_TIMEOUT` | 10.0 | Seconds before partial average |
| `QUORUM` | 0 | Min count for early emit (0=disabled) |
| `LIVE_AVG_NS` | loss | Namespace (use `val` for validation) |

## Troubleshooting

**Can't connect to Redis:** Check Redis job is running, node is reachable, password matches

**Missing averaged metrics:** Verify `SWEEP_ID` matches, aggregator is running, check logs

**Import errors:** Ensure conda environment has required packages

**Workers fail on `SWEEP_ID`/`REDIS_HOST`:** Script only creates `LiveAverager` if both set (optional for standalone runs)

## Cleanup

```bash
scancel <jobid>  # Cancel Redis, aggregator, and worker jobs
rm redis_connection.env  # Optional
```
