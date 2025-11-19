#!/bin/bash
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --ntasks-per-node=1
#SBATCH -t 16:00:00
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem-per-gpu=48G
#SBATCH -J petbuddy    # jobs name
#SBATCH -o ./slurm_outs/slurm_%j.out   # file to write logs, prints, etc

if [ $SLURM_GPUS_ON_NODE -gt 1 ]; then
  echo "Running process with $SLURM_GPUS_ON_NODE GPUs"
  torchrun --standalone --nproc_per_node="gpu" ../tools/experiments/optuna_train.py
else
  echo "Running on single GPU"
  python ../tools/experiments/optuna_train.py
fi
