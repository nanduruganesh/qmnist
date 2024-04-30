#!/bin/bash -l
module purge

module load anaconda

conda deactivate
# conda activate /home/$(whoami)/.conda/envs/qmnist
conda activate qmnist

python mnist.py --epochs=50 --noise=0 --model_name=ClassicalNN --mult-noise-by=0.5 --save_to=runs/hybrid2 --classical_layers=2