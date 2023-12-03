#!/bin/bash

# TODO must be completly on the GPU

#SBATCH --job-name=PST_process_audio       # job name
#SBATCH --partition=gpu_4	           # check gpu queue for the resource allocation.
#SBATCH --time=01:00:00                       # wall-clock time limit  
#SBATCH --mem=200000                        # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --cpus-per-task=1                 # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --mail-user=ubppd@student.kit.edu  # notification email address
#SBATCH --gres=gpu:4
#SBATCH --output=/home/kit/stud/ubppd/asr/logs/output.txt
#SBATCH --error=/home/kit/stud/ubppd/asr/logs/error.txt


export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
# export VENVDIR=~/env                       # Export path to your virtual environment.
export PYDIR=.    			                   # Export path to directory containing Python script.

module purge                                    # Unload all currently loaded modules.
module load compiler/gnu/10.2                   # Load required modules.  
module load devel/python/3.8.6_gnu_10.2
# module load mpi/openmpi/4.1
# module load devel/cuda/10.2

# source ${VENVDIR}/bin/activate  # Activate your virtual environment.

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home//kit/stud/ubppd/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home//kit/stud/ubppd/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home//kit/stud/ubppd/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home//kit/stud/ubppd/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
	
conda activate nmt

# mpirun python ${PYDIR}/process_audio.py
python ${PYDIR}/process_audio.py
