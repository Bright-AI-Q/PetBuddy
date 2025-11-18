#!/bin/bash
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --ntasks-per-node=1
#SBATCH -t 16:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=64G
#SBATCH -J petbuddy    # jobs name
#SBATCH -o ./slurm_outs/slurm_%j.out   # file to write logs, prints, etc

python ../tools/experiments/optuna_train.py
