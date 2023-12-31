#!/bin/bash

#SBATCH --job-name=process_audio           # job name
#SBATCH --partition=gpu_4	               # check gpu queue for the resource allocation.
#SBATCH --time=06:00:00                    # wall-clock time limit  
#SBATCH --mem=150000                       # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --cpus-per-task=1                  # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --gres=gpu:1

# call ../setup.sh
source ../setup.sh

# mpirun python process_audio.py
python process_audio.py >> process_audio_$SLURM_JOB_ID.txt 2>&1
