#!/bin/bash

#BATCH --job-name=02
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=16GB
#SBATCH --output=02_thread-027_%j.out

module purge

python /scratch/vl1019/icassp23_data 5400 5600
