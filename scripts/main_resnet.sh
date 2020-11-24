#!/bin/bash

python main_resnet.py --gpuid=1 --epochs=100

python main_resnet.py --gpuid=1 --resume_training --resume_model=./trained_models/scratch_main_resnet.model --test