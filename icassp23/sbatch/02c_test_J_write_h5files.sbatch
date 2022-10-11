#!/bin/bash

#BATCH --job-name=01
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=4GB
#SBATCH --output=./02c_%j.out

module purge

singularity exec --overlay /scratch/vl1019/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif /bin/bash -c "source /scratch/vl1019/env.sh; python test J /home/vl1019/perceptual_neural_physical/icassp23/02c_write_h5files.py"
