#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=rtx6000
#SBATCH --qos=normal
#SBATCH --job-name=train_clear_boundary
#SBATCH --output=%j_%x.out
#SBATCH --error=%j_%x.err
#SBATCH --time=12:00:00

module load cuda-12.4
source /scratch/ssd004/scratch/yuchongz/python_venvs/clear_boundary_venv/bin/activate

python -m main