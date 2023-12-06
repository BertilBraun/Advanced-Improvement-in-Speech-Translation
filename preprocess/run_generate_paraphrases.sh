#!/bin/bash

# TODO Can this be run on the GPU?

#SBATCH --job-name=PST_generate_paraphrases # job name
#SBATCH --partition=gpu_4                  # TODO mby GPU queue for the resource allocation.
#SBATCH --time=1:00:00                     # TODO wall-clock time limit  
#SBATCH --mem=100000                       # TODO memory per node
#SBATCH --nodes=4                          # number of nodes to be used
#SBATCH --cpus-per-task=40                 # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --mail-user=uxude@student.kit.edu  # notification email address

# call ~/setup.sh
source ~/setup.sh

# mpirun python ${PYDIR}/generate_paraphrases.py
python ${PYDIR}/generate_paraphrases.py
