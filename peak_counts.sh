#!/bin/bash
#SBATCH --job-name=peak_counts
#SBATCH --account=fc_cosmoml
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --partition=savio2_gpu
#SBATCH --qos=savio_normal

module load python
module load tensorflow
srun python peak_counts.py
