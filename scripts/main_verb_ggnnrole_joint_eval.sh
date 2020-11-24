#!/bin/bash

python main_verb_ggnnrole_joint_eval.py --gpuid=1 --evaluate --ggnn_model=./trained_models/train_full_main_ggnn_r2r.model

python main_verb_ggnnrole_joint_eval.py --gpuid=1 --evaluate --ggnn_model=./trained_models/train_full_main_ggnn_r2v.model

python main_verb_ggnnrole_joint_eval.py --gpuid=1 --evaluate --ggnn_model=./trained_models/train_full_main_ggnn_baseline.model


python main_verb_ggnnrole_joint_eval.py --gpuid=1 --test --ggnn_model=./trained_models/train_full_main_ggnn_r2r.model

python main_verb_ggnnrole_joint_eval.py --gpuid=1 --test --ggnn_model=./trained_models/train_full_main_ggnn_r2v.model

python main_verb_ggnnrole_joint_eval.py --gpuid=1 --test --ggnn_model=./trained_models/train_full_main_ggnn_baseline.model
