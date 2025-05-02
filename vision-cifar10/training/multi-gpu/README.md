# Multi-GPU Job Setup

This example demonstrates how to run 
a PyTorch DDP (DistributedDataParallel) training script on multiple GPUs across multiple nodes using SLURM. 


## SLURM Configuration

Before submitting the job, ensure you update the following fields
in the SLURM script with appropriate values:

```bash
#SBATCH --partition=<your-partition-name>
#SBATCH --account=<your-account-name>
```

You can modify the number of nodes, GPUs per node, CPUs, memory, and time limit according to your cluster's capabilities and your experiment's needs.

---

## Data Path

The `--data_path` argument specifies where the CIFAR-10 dataset is 
located. If not provided, it will be downloaded to the default location. You can update the following line to point to your dataset:

```bash
export DATA_PATH="/path/to/your/cifar10"
```

---

## Conda Environment

You must specify the **absolute path** to your conda environment:

```bash
export CONDA_ENV="/absolute/path/to/your/conda/env"
conda activate $CONDA_ENV
```

Make sure the environment includes all dependencies for PyTorch DDP and your script.

---

## Distributed Training Configuration

The script automatically calculates DDP environment variables:

- `GPUS_ON_NODE`: Number of GPUs per node
- `NNODES`: Total number of nodes
- `WORLD_SIZE`: Total number of GPUs across nodes
- `MASTER_ADDR` and `MASTER_PORT`: Used for rendezvous during multi-node communication

These are used to launch the training via `torchrun`.

---

## Program Arguments

You can customize training by editing the `CMD` variable. Here's an example configuration:

```bash
export CMD="torchrun \
  --nproc_per_node \$GPUS_ON_NODE \
  --nnodes \$NNODES \
  --rdzv_id=\$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint \$MASTER_ADDR:\$MASTER_PORT \
  ./torch_vision_ddp_cifar10.py \
  --data_path \$DATA_PATH \
  --metrics_csv results.csv \
  --sample_ratio 1.0 \
  --batch_size 512 \
  --num_epochs 10 \
  --learning_rate 0.0005 \
  --model_name resnet50 \
  --mixed_precision auto \
  --pin_memory True \
  --tensorboard_csv ./tensorboard.csv \
  --use_checkpoint --checkpoint_path best_model.pth \
  --use_snapshot --snapshot_path snapshot.pth \
  --use_wandb --wandb_mode online --wandb_project optimize-ml-workflow"
```

Feel free to adjust these arguments for your own experiments.

---

## Job Submission

To submit the job, run:

```bash
sbatch torch_vision_ddp_cifar10.slrm
```

The script also records execution time and reports job statistics using `seff` and `sacct`.
