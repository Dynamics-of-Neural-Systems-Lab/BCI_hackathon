#!/bin/bash
#SBATCH -N 1
#SBATCH -q a100
#SBATCH -c 10
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=256G
#SBATCH --job-name=train_job
#SBATCH --output=train_output_%j.log

# Load necessary modules (if applicable)

source ~/miniconda3/etc/profile.d/conda.sh

conda activate ALVI


# Navigate to your project directory
cd /msc/home/alopez22/BCI_hackathon

# Run your Python script
python train.py
