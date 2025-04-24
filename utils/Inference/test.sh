#! /bin/bash


DATADIR="/n/holylfs06/LABS/kempner_shared/Lab/data/imagenet_1k"
TRAIN_DS="${DATADIR}/train"
VAL_DS="${DATADIR}/val"

CONTAINER_PATH=/n/holylabs/LABS/kempner_dev/Lab/containers/pytorch_2.1.2-cuda12.1-cudnn8-runtime.sif


export CMD="python ./infer.py" 
echo $CMD 
singularity run --nv $CONTAINER_PATH $CMD

echo "Job Completed"

