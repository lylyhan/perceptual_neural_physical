#!/bin/bash

#BATCH --job-name=02_compute_pnp_jacobian.py
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6:00:00
#SBATCH --mem=32GB
#SBATCH --output=02_thread-420_%j.out

module purge

singularity exec --overlay /scratch/vl1019/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif /bin/bash -c "source /scratch/vl1019/env.sh; python /home/han/perceptual_neural_physical/icassp23/02_compute_pnp_jacobian.py /scratch/vl1019/icassp23_data 84000 84200"

