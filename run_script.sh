#!/bin/bash
#SBATCH --job-name=peak_counts
#SBATCH --account=fc_cosmoml
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --partition=savio2_gpu
#SBATCH --qos=savio_normal

module load tensorflow/1.12.0-py36-pip-gpu
srun python "$@"
