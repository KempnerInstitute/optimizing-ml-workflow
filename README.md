#  Optimizing ML Workflow

This repository provides practical and scalable deep learning training pipelines using PyTorch — both for **single-GPU** and **multi-GPU** setups using **DistributedDataParallel (DDP)**. It includes reproducible experiments with CIFAR-10 and ImageNet datasets.

---

##  Key Features

-  Modular, extensible training pipelines
-  Single-GPU and Multi-GPU (DDP) support
-  Mixed precision training 
-  Early stopping
-  Auto checkpointing and resume
-  Metrics logging (CSV + optional Weights & Biases)
-  Optional LR schedulers: StepLR, CosineAnnealingLR

---

##  Directory Structure

```bash
optimizing-ml-workflow/
├── vision-cifar10/
│   ├── single-gpu/         # Single-GPU CIFAR10 training
│   └── multi-gpu-ddp/      # Multi-GPU DDP CIFAR10 training
├── imagenet/               # ImageNet training examples (DDP-ready)
├── utils/                  # Shared utilities
└── README.md               # This file
```

---

##  Getting Started

###  Install Dependencies
We will create a conda environment and install the required packages such as pytorch, torch-vision, wandb, etc. We wil use some of the existing packages that aravailable in the module environment. 

Lets load the Python, cuda, and cudnn modules. 
```bash
module load python/3.12.8-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/9.5.1.17_cuda12-fasrc01

```
Use `mamba` or `conda` to create the environment to install the required packages such as pytorch, torch-vision, wandb, etc.,
```bash
mamba env create --prefix=<absolute-path-for-conda-env" -f environment.yml

```
---

##  Single-GPU Training

**Path:** `vision-cifar10/single-gpu/`

```bash
cd vision-cifar10/single-gpu
python train.py --epochs 100 --batch-size 128 --lr 0.1
```

Supports:

- `--resume`: Resume from checkpoint
- `--early-stopping-patience N`
- `--use-wandb` (optional)
- Outputs logs + CSV + checkpoints

---

##  Multi-GPU Training (Distributed Data Parallel)

**Path:** `vision-cifar10/multi-gpu-ddp/`

Launch using PyTorch’s `torchrun` (or `python -m torch.distributed.launch`):

```bash
torchrun --nproc_per_node=4 train_distributed_cifar10.py \
  --batch_size 128 --num_epochs 90 --model_name resnet50 \
  --mixed_precision auto --scheduler cosine --resume
```

Arguments include:

- `--model_name resnet18|resnet50|...`
- `--mixed_precision fp16|bf16|auto|none`
- `--num_classes`, `--finetune`, `--scheduler`
- `--metrics_csv`: Save results to file

---

##  ImageNet Training (DDP)

**Path:** `imagenet/`

Launch similarly using:

```bash
torchrun --nproc_per_node=8 train_imagenet.py \
  --train_ds /path/to/train \
  --val_ds /path/to/val \
  --model_name resnet50 --mixed_precision auto --batch_size 256
```

---

## Experiment Logging

Enable Weights & Biases integration:

```bash
pip install wandb
wandb login
python train.py --use-wandb
```

---

## More Workflow Examples

- For performance benchmarking of Resnet and AlexNet models refer: https://github.com/KempnerInstitute/scalable-vision-workflows

- Workflows related to LLMs such as Llama3 and GPT with NeMo framework are available in here: https://github.com/KempnerInstitute/nvidia-nemo-workflows

