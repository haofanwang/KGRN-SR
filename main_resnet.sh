#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J main_resnet_tda
#SBATCH -o ./logs_tda_data/main_resnet_tda.%J.out
#SBATCH -e ./logs_tda_data/main_resnet_tda.%J.err
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12

#run the application:
module load torch/0.4.0
source activate vlnce

python main_resnet.py --gpuid=1 --epochs=100

python main_resnet.py --gpuid=1 --resume_training --resume_model=./trained_models/scratch_main_resnet_tda.model --test