# Single-GPU Job Using Singularity

This guide describes how to run a PyTorch training script on a single GPU using a Singularity container within a SLURM job. The script trains a ResNet model on the CIFAR-10 dataset and includes logging, checkpointing, and Weights & Biases integration.

---

## SLURM Job Configuration

Make sure to update the SLURM script with your actual partition and account values:

```bash
#SBATCH --partition=<your-partition-name>
#SBATCH --account=<your-account-name>
```

---

## Environment Variables

### Dataset Path
If the dataset path is not provided, the script will attempt to download 
CIFAR-10 automatically. It is optional the path to the CIFAR-10 dataset:

```bash
export DATA_PATH="/path/to/cifar10"
```

### Singularity Container
Set the path to your `.sif` container file:

```bash
export CONTAINER_FILE="/path/to/pytorch_container.sif"
```

### Weights & Biases
Ensure your API key is loaded:

```bash
source ~/.ssh/wandb_key.sh
```

---

## Training Command

You can modify the training command by editing the `CMD` variable. The current configuration runs `torch_vision_cifar10.py` with the following options:

```bash
export CMD="python torch_vision_cifar10.py \
  --data_path $DATA_PATH \
  --metrics_csv results.csv \
  --sample_ratio 1.0 \
  --batch_size 512 \
  --num_epochs 10 \
  --learning_rate 0.0005 \
  --model_name resnet50 \
  --mixed_precision auto \
  --pin_memory True \
  --use_tensorboard \
  --use_checkpoint --checkpoint_path best_model.pth \
  --use_snapshot --snapshot_path snapshot.pth \
  --use_wandb --wandb_mode online --wandb_project optimize-ml-workflow"
```

---

## Singularity Bind Paths

The script binds necessary system and SLURM paths for full compatibility:

```bash
export SINGULARITY_BIND="/etc/nsswitch.conf,...,/usr/lib64/pmix,"
```

Modify this list if your cluster setup differs.

---

## Running the Job

Use the following command to submit your job:

```bash
sbatch torch_vision_cifar10.singularity.slrm
```

The training script will be executed inside the container using:

```bash
srun -l singularity run --nv $CONTAINER_FILE $CMD
```

---

## Monitoring

Execution time is measured and printed in `HH:MM:SS` format. Additionally, job statistics are summarized using:

```bash
seff $SLURM_JOB_ID
sacct -j $SLURM_JOB_ID --format=...
```

---

## Output

Log and error files will be generated as:

```bash
img-<NodeName>.img.<JobID>.out
img-<NodeName>.img.<JobID>.err
```

---
