#!/bin/bash

# TODO must be completly on the GPU

#SBATCH --job-name=PST_process_audio       # job name
#SBATCH --partition=multiple_gpu           # TODO check gpu queue for the resource allocation.
#SBATCH --time=30:00                       # TODO wall-clock time limit  
#SBATCH --mem=40000                        # TODO memory per node
#SBATCH --nodes=4                          # number of nodes to be used
#SBATCH --cpus-per-task=40                 # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --mail-user=uxude@student.kit.edu  # notification email address

export OMP_NUM_THREADS=40

module purge                                    # Unload all currently loaded modules.
module load compiler/gnu/10.2                   # Load required modules.  
module load devel/python/3.8.6_gnu_10.2
module load mpi/openmpi/4.1
module load devel/cuda/10.2

source ~/env/bin/activate  # Activate your virtual environment.

mpirun python ~/preprocess/process_audio.py
