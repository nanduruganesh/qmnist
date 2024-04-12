#!/bin/bash -l

#SBATCH -A collabrobogroup
#SBATCH --array=0-9
#SBATCH -t 72:00:00
#SBATCH -p standard
#SBATCH -N 1
#SBATCH --cpus-per-task=10
#SBATCH --output=logs/noise-%a.log
#SBATCH -J official_runs

module purge

module load anaconda

conda deactivate
conda activate /project/CollabRoboGroup/bae9wk/tq

python examples/mnist/mnist.py --epochs=1 --noise=${SLURM_ARRAY_TASK_ID} --mult-noise-by=0.5 --save_to=runs/test