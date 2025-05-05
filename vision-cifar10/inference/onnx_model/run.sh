export DATA_PATH="/n/netscratch/kempner_dev/Everyone/workshop/ml-opt-workshop/data/cifar10"
export CONDA_ENV="/n/netscratch/kempner_dev/Everyone/workshop/ml-opt-workshop/shared_env/pytorch_image"

module load cuda
module load cudnn
module load python
conda activate $CONDA_ENV

python onnx_inference.py --model ./final_model.onnx --image ./YellowLabradorLooking_new.jpg --labels labels.txt --thread 4 
