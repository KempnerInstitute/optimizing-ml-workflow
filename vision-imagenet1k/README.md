
# ImageNet-1k Dataset Setup & Job Submission 

## Dataset: ImageNet-1k

This example uses the [ImageNet-1k dataset](https://image-net.org/), which consists of **1.2 million training images** and **50,000 validation images**, distributed across **1,000 object categories**. This dataset is widely used for benchmarking image classification models.

---

## Downloading the Dataset

To access the ImageNet dataset, you must first create an account on the [ImageNet website](https://image-net.org/download-images).

For detailed guidance on downloading and setting up the dataset, refer to the  
[ResNet Benchmarking Repository](https://github.com/KempnerInstitute/scalable-vision-workflows/tree/main/imagenet1k_resnet50).

Once you have an account and have logged in:

Start an interactive Slurm session to download the dataset:

```bash
salloc -p <partition-name> -A <account-name> -c 4 --mem=64GB -t 08:00:00
```

Then, on the compute node, run the following commands to download the training and validation archives:

```bash
wget --user=your_username --password=your_password https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
wget --user=your_username --password=your_password https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
```

---

## Extracting the Data

Use the provided shell script to extract the tar files:

```bash
../utils/extract_ILSVRC.sh
```

This will organize the training and validation images into the appropriate folder structure expected by PyTorch.

---

## Submitting a Job

Before submitting your training job, ensure you:

- Set the correct `DATA_PATH` environment variable.
- Modify the Slurm partition and account in the job script.
- Provide the full path to your Conda environment as `CONDA_ENV`.

Refer to the CIFAR-10 examples in this repository for specific job script usage patterns.
