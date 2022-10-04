#!/bin/bash

#BATCH --job-name=01
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --mem=8GB
#SBATCH --gres=gpu:1
#SBATCH --output=01_thread-4_%j.out

module purge
module load cuda/11.6.2
module load ffmpeg/4.2.4

python /Users/user/perceptual_neural_physical/icassp23/01_generate_audio.py /scratch/vl1019/icassp23_data 40000 50000
