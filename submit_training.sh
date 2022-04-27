#!/bin/bash

#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=gpu
#SBATCH --gpus=1

python3 figures/report3/fig4/script.py

