#!/bin/bash
#SBATCH --job-name=vae-training      # Job name
#SBATCH --output=vae-training.out    # Standard output and error log
#SBATCH --ntasks=1                   # Number of tasks (processes)
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --gres=gpu:1                 # Number of GPUs (per node)
#SBATCH --time=24:00:00              # Time limit hrs:min:sec
#SBATCH --mem=16G                    # Total memory limit
#SBATCH --partition=gpu              # Partition (queue) name

# Load modules or environment
module load anaconda/2021.05  # Adjust to your cluster's module system
source activate vae-env       # Activate the Conda environment

# Run the training script
python train_vae.py
