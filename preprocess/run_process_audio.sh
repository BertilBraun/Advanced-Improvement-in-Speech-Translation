#!/bin/bash

#SBATCH --job-name=PST_process_audio       # job name
#SBATCH --partition=gpu_4	               # check gpu queue for the resource allocation.
#SBATCH --time=01:00:00                    # wall-clock time limit  
#SBATCH --mem=200000                       # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --cpus-per-task=1                  # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --mail-user=ubppd@student.kit.edu  # notification email address
#SBATCH --gres=gpu:4
#SBATCH --output=/home/kit/stud/ubppd/asr/logs/output.txt
#SBATCH --error=/home/kit/stud/ubppd/asr/logs/error.txt

# call ../setup.sh
source ../setup.sh

# mpirun python ${PYDIR}/process_audio.py
python ${PYDIR}/process_audio.py
