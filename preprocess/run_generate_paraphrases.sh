#!/bin/bash

#SBATCH --job-name=PST_generate_paraphrases # job name
#SBATCH --partition=gpu_4                  # mby GPU queue for the resource allocation.
#SBATCH --time=1:00:00                     # wall-clock time limit  
#SBATCH --mem=100000                       # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --cpus-per-task=1                  # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --mail-user=ubppd@student.kit.edu  # notification email address
#SBATCH --gres=gpu:1
#SBATCH --output=~/PST/MT/logs/output_%j.txt
#SBATCH --error=~/PST/MT/logs/error_%j.txt

# call ../setup.sh
source ../setup.sh

# mpirun python ${PYDIR}/generate_paraphrases.py
python ${PYDIR}/generate_paraphrases.py
