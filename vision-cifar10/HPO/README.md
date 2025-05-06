#  Hyperparameter Optimization Workflows

This directory contains two scalable and modular workflows for performing hyperparameter optimization (HPO) on vision models trained on the CIFAR-10 dataset. Both approaches are designed for use in high-performance computing environments and support reproducible experimentation.

## Available Workflows

###  `distributed-array-sweep/` — **SLURM Array-Based Parameter Sweep**

This workflow performs a grid search over a defined set of hyperparameters using SLURM job arrays. Each SLURM task runs a training job with a unique combination of parameters from a pre-generated list. It is ideal for running large-scale sweeps over a cluster or GPU node pool.

**Key Features:**
- Parameters are defined and expanded using `generate_param_comb.py`
- Each parameter set is executed as an independent SLURM array job
- Training logs and results are saved per-job
- Includes tools for:
  - Requeuing failed jobs
  - Parsing logs and summarizing metrics (e.g., validation loss) into CSV format


---

###  `wandb-sweep/` — **Automated Sweep with Weights & Biases**

This workflow leverages [Weights & Biases Sweeps](https://docs.wandb.ai/guides/sweeps) to automate the hyperparameter search using `wandb.agent`. It handles sampling, scheduling, and tracking of all experiments, making it easy to visualize metrics and identify optimal configurations.

**Key Features:**
- Define search space and sweep method (`random`, `grid`, `bayes`, etc.) via `sweep_config.yaml`
- Automatically schedules and runs experiments via `wandb.agent`
- Real-time visualization and logging through the WandB dashboard




