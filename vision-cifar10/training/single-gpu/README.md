
## Single-GPU Job Setup

This example demonstrates how to run a PyTorch training script on a **single GPU** using SLURM. The associated SLURM script is configured to request one GPU.

###  SLURM Configuration

Before submitting the job, make sure to fill in your SLURM partition and account details:

```bash
#SBATCH --partition=<your-partition-name>
#SBATCH --account=<your-account-name>
```

### Data Path

The --data_path argument is optional. If it is not provided, the CIFAR-10 dataset will be automatically downloaded to the default location.

### Conda Environment
You must specify the absolute path to your conda environment for activation:

```bash
CONDA_ENV=<absolute-path-to-conda-env>
```

### Program Arguments

You can modify the training configuration by changing the CMD variable. Below is an example:

```bash
export CMD="python torch_vision_cifar10.py \
  --data_path \$DATA_PATH \
  --metrics_csv results.csv \
  --sample_ratio 1.0 \
  --batch_size 512 \
  --num_epochs 10 \
  --learning_rate 0.0005 \
  --model_name resnet50 \
  --mixed_precision auto \
  --pin_memory False \
  --use_tensorboard \
  --use_checkpoint --checkpoint_path best_model.pth \
  --use_snapshot --snapshot_path snapshot.pth \
  --use_wandb --wandb_mode online --wandb_project optimize-ml-workflow"

```
Feel free to adjust the arguments as needed to match your experimental setup.

### Job Submission

```bash
sbatch torch_vision_cifar10.slrm
```
