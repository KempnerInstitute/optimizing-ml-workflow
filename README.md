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

##  Getting Started

###  Clone the repository

```bash
git clone https://github.com/KempnerInstitute/optimizing-ml-workflow.git
cd optimizing-ml-workflow

```

###  Create Conda Environment

Lets load the Python, cuda, and cudnn modules. This way we don't have to install the required to install the conda manager and the cuda libraries. 
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

## Experiment Logging

Enable Weights & Biases integration:

```bash
pip install wandb
wandb login
```

## Directory Structure 

```
.
├── singularity_build                  # Contains Singularity examples for building container images
├── utils
│   └── imagenet1k_download            # Scripts/utilities to download and organize the ImageNet1K dataset
├── vision-cifar10                     # All workflows specific to the CIFAR-10 dataset
│   ├── HPO                            # Hyperparameter optimization workflows
│   │   ├── distributed-array-sweep    # Slurm-based parameter sweep using job arrays
│   │   └── wandb-sweep                # W&B agent-based hyperparameter sweep
│   ├── inference                      # Inference pipelines for CIFAR-10 models
│   │   ├── checkpoint_model           # Inference from the saved PyTorch checkpoint models
│   │   ├── data                       # Example input images or batches for inference
│   │   └── onnx_model                 # Exported models in ONNX format for portable inference
│   └── training                       # Training workflows for CIFAR-10
│       ├── multi-gpu                 # Training using DistributedDataParallel (DDP)
│       ├── single-gpu                # Standard single-GPU training scripts
│       └── singularity-single-gpu    # Single-GPU training using Singularity containers
└── vision-imagenet1k                 # All workflows specific to the ImageNet-1K dataset
    ├── inference                     # Inference pipelines for ImageNet1K models
    │   ├── checkpoint_model          # Trained model checkpoints (PyTorch)
    │   ├── data                      # Input samples for inference
    │   ├── onnx_model                # Portable inference with the exported ONNX models
    │   └── pretrained_model          # Inference from the pretrained image models
    └── training                      # Training workflows for ImageNet1K
```


---

## More Workflow Examples

- For performance benchmarking of Resnet and AlexNet models refer: https://github.com/KempnerInstitute/scalable-vision-workflows

- Workflows related to LLMs such as Llama3 and GPT with NeMo framework are available in here: https://github.com/KempnerInstitute/nvidia-nemo-workflows

