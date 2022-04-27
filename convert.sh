#!/bin/bash

jupyter nbconvert --to script pepita-mirror.ipynb
mkdir figures/report3/fig4 -p
mv pepita-mirror.py figures/report3/fig4/script.py
sbatch submit_training.sh