#!/bin/bash

#SBATCH --job-name=prepare_all_datasets      # job name
#SBATCH --partition=single                   # mby GPU queue for the resource allocation.
#SBATCH --time=16:00:00                      # wall-clock time limit
#SBATCH --mem=100000                         # memory per node
#SBATCH --nodes=1                            # number of nodes to be used
#SBATCH --cpus-per-task=1                    # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                  # maximum count of tasks per node
#SBATCH --mail-type=ALL                      # Notify user by email when certain event types occur.
#SBATCH --output=prepare_all_datasets_%j.txt
#SBATCH --error=prepare_all_datasets_%j.txt

cd ../../
source setup.sh

python -m src.train.prepare_all_datasets
