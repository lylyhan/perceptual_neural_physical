#!/bin/bash

#BATCH --job-name=01_generate_audio.py
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --mem=8GB
#SBATCH --gres=gpu:1
#SBATCH --output=01_thread-7_%j.out

module purge
module load cuda/11.6.2
module load ffmpeg/4.2.4

singularity exec --overlay /scratch/vl1019/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif /bin/bash

python /home/vl1019/perceptual_neural_physical/icassp23/01_generate_audio.py /scratch/vl1019/icassp23_data 70000 80000
