#!/bin/bash -l

#SBATCH -A collabrobogroup
#SBATCH --array=0-9
#SBATCH -t 12:00:00
#SBATCH -p standard
#SBATCH -N 1
#SBATCH --cpus-per-task=10
#SBATCH --output=logs/%A-noise-%a.log
#SBATCH -J classical_nn_runs

module purge

module load anaconda

conda deactivate
conda activate /home/$(whoami)/.conda/envs/qmnist

mkdir -p logs
python mnist.py --epochs=50 --noise=${SLURM_ARRAY_TASK_ID} --model_name=ClassicalNN --mult-noise-by=0.5 --save_to=runs/hybrid2