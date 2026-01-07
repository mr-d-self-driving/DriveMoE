#!/bin/bash

#SBATCH --job-name=eval-pretrain
#SBATCH --output=logs/eval/%A.out
#SBATCH --error=logs/eval/%A.err
#SBATCH --time=5:59:59
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G


CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 torchrun \
    script/run.py \
    --config-name=open_loop \
    --config-path=../config/eval/DrivePi0\