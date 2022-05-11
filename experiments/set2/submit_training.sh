#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --array=[0-143]


python3 /home/cpsc490_ac2788/CPSC490/experiments/set2/pepita-mirror.py ${SLURM_ARRAY_TASK_ID}

