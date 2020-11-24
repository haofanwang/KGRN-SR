#!/bin/bash

python main_verb_tdarole_joint_eval.py --gpuid=1 --evaluate --tda_model=./trained_models/train_full_tda.model

python main_verb_tdarole_joint_eval.py --gpuid=1 --test --tda_model=./trained_models/train_full_tda.model
