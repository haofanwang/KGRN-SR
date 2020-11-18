#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J main_ggnn_r2v_w_tda
#SBATCH -o ./logs_tda_data/main_ggnn_r2v_w_tda.%J.out
#SBATCH -e ./logs_tda_data/main_ggnn_r2v_w_tda.%J.err
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12

#run the application:
module load torch/0.4.0
source activate vlnce

python main_ggnn_r2v_w_tda.py --gpuid=1 --epochs=100

python main_ggnn_r2v_w_tda.py --gpuid=1 --test --resume_training --resume_model=./trained_models/train_full_main_ggnn_r2v_w.model