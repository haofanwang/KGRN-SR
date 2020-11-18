#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J role_accuracy
#SBATCH -o ./logs_tda_data/role_accuracy.%J.out
#SBATCH -e ./logs_tda_data/role_accuracy.%J.err
#SBATCH --time=00:20:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12

#run the application:
module load torch/0.4.0
source activate vlnce

python role_accuracy.py --gpuid=1