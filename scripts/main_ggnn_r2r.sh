#!/bin/bash

python main_ggnn_r2r.py --gpuid=1 --epochs=100

python main_ggnn_r2r.py --gpuid=1 --test --resume_training --resume_model=./trained_models/train_full_main_ggnn_r2r.model