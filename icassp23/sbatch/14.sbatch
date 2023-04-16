#!/bin/bash

#BATCH --job-name=14_merge_h5file.py
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH -A aej@v100
#SBATCH --output=14_merge_h5file_%j.out

module purge
module load pytorch-gpu/py3/1.12.1

#python /gpfsdswork/projects/rech/rwb/ufg99no/perceptual_neural_physical/icassp23/14_merge_h5file.py

