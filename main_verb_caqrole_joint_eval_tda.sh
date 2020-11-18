#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J main_verb_ggnnrole_joint_eval_tda
#SBATCH -o ./logs_tda_data/main_verb_ggnnrole_joint_eval_tda.%J.out
#SBATCH -e ./logs_tda_data/main_verb_ggnnrole_joint_eval_tda.%J.err
#SBATCH --time=00:30:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12

#run the application:
module load torch/0.4.0
source activate vlnce

# python main_verb_caqrole_joint_eval_tda.py --gpuid=1 --evaluate --caq_model=./trained_models/train_full_caq.model

# python main_verb_caqrole_joint_eval_tda.py --gpuid=1 --test --caq_model=./trained_models/train_full_caq.model
