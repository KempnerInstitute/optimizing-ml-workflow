
#! /bin/bash

export CONDA_ENV="/n/netscratch/kempner_dev/Everyone/workshop/ml-opt-workshop/shared_env/pytorch_image"

module load cuda
module load cudnn
module load python
conda activate $CONDA_ENV

python infer_cifar10.py --image ./YellowLabradorLooking_new.jpg --model resnet50 --checkpoint ./best_model.pth 

echo "Job Completed"
