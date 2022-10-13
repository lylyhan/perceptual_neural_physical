#!/bin/bash

#BATCH --job-name=jupyter
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=16GB
#SBATCH --output=./jupyter_%j.out

module purge

singularity exec --overlay /scratch/vl1019/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif /bin/bash -c "source /scratch/vl1019/env.sh; jupyter notebook --port=8080"
