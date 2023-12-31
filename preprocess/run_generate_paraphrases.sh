#!/bin/bash

#SBATCH --job-name=generate_paraphrases     # job name
#SBATCH --partition=gpu_4                  # mby GPU queue for the resource allocation.
#SBATCH --time=24:00:00                    # wall-clock time limit  
#SBATCH --mem=100000                       # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --cpus-per-task=1                  # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --gres=gpu:1

# call ../setup.sh
source ../setup.sh

# mpirun python generate_paraphrases.py
python generate_paraphrases.py >> paraphrases_logs_$SLURM_JOB_ID.txt 2>&1
