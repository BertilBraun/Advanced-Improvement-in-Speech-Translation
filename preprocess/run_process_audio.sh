#!/bin/bash

#SBATCH --job-name=PST_process_audio       # job name
#SBATCH --partition=gpu_4	               # check gpu queue for the resource allocation.
#SBATCH --time=01:00:00                    # wall-clock time limit  
#SBATCH --mem=150000                       # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --cpus-per-task=1                  # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --gres=gpu:1
#SBATCH --output=../../ASR/logs/output_%j.txt
#SBATCH --error=../../ASR/logs/error_%j.txt

# call ../setup.sh
source ../setup.sh

# mpirun python process_audio.py
python process_audio.py
