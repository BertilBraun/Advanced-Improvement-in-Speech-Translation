#!/bin/bash

#SBATCH --job-name=PST                     # job name
#SBATCH --partition=multiple               # queue for the resource allocation.
#SBATCH --time=30:00                       # wall-clock time limit  
#SBATCH --mem=40000                        # memory per node
#SBATCH --nodes=4                          # number of nodes to be used
#SBATCH --cpus-per-task=40                 # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --mail-user=uxude@student.kit.edu  # notification email address

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export VENVDIR=~/.venvs/scalableai             # Export path to your virtual environment.
export PYDIR=~/   			                   # Export path to directory containing Python script.

module purge                                    # Unload all currently loaded modules.
module load compiler/gnu/10.2                   # Load required modules.  
module load devel/python/3.8.6_gnu_10.2
module load mpi/openmpi/4.1
module load devel/cuda/10.2

source ${VENVDIR}/bin/activate  # Activate your virtual environment.

mpirun python ${PYDIR}<path to your python script>
