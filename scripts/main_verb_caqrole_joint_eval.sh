#!/bin/bash

python main_verb_caqrole_joint_eval.py --gpuid=1 --evaluate --caq_model=./trained_models/train_full_caq.model

python main_verb_caqrole_joint_eval.py --gpuid=1 --test --caq_model=./trained_models/train_full_caq.model
