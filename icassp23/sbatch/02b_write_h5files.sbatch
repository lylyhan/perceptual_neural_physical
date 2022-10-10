#!/bin/bash

#BATCH --job-name=01
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --mem=62GB
#SBATCH --gres=gpu:1
#SBATCH --output=./01s01_thread-9_%j.out

module purge
module load cuda/9.0.176
module load cudnn/9.0v7.3.0.29
module load ffmpeg/intel/3.2.2

python /home/vl1019/perceptual_neural_physical/icassp23/02b_write_h5files.py
