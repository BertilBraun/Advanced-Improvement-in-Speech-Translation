#!/bin/bash

#SBATCH --job-name=PST_generate_paraphrases # job name
#SBATCH --partition=gpu_4                  # mby GPU queue for the resource allocation.
#SBATCH --time=1:00:00                     # wall-clock time limit  
#SBATCH --mem=100000                       # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --cpus-per-task=1                  # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --gres=gpu:1
#SBATCH --output=/pfs/work7/workspace/scratch/uxude-MT/logs/output_%j.txt
#SBATCH --error=/pfs/work7/workspace/scratch/uxude-MT/logs/error_%j.txt

# call ../setup.sh
source ../setup.sh

# mpirun python generate_paraphrases.py
python generate_paraphrases.py
