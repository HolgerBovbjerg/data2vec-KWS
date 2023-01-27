#!/bin/bash

# Baselines
python train.py --conf KWT_configs/kwt1_base_mean_config.yaml --id 2
python train.py --conf KWT_configs/kwt2_base_mean_config.yaml --id 2
python train.py --conf KWT_configs/kwt3_base_mean_config.yaml --id 2

# Data2Vec pretraining
python train_data2vec.py --conf data2vec/data2vec_configs/kwt1_data2vec_config.yaml --id 2
python train_data2vec.py --conf data2vec/data2vec_configs/kwt2_data2vec_config.yaml --id 2
python train_data2vec.py --conf data2vec/data2vec_configs/kwt3_data2vec_config.yaml --id 2

# Finetuning
python train.py --conf KWT_configs/kwt1_finetune_mean_config.yaml --ckpt runs/kwt1_data2vec2/best_encoder.pth --id 2
python train.py --conf KWT_configs/kwt2_finetune_mean_config.yaml --ckpt runs/kwt2_data2vec2/best_encoder.pth --id 2
python train.py --conf KWT_configs/kwt3_finetune_mean_config.yaml --ckpt runs/kwt3_data2vec2/best_encoder.pth --id 2
