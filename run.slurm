#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 32 
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
#SBATCH -t 12:00:00
#SBATCH -A tb901114
#SBATCH -J test_multigpu

module load Apptainer
./run.sh
