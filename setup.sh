
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYDIR=.    			                   # Export path to directory containing Python script.

module purge                                    # Unload all currently loaded modules.
module load compiler/gnu/10.2                   # Load required modules.  
# module load devel/python/3.8.6_gnu_10.2
# module load mpi/openmpi/4.1
# module load devel/cuda/10.2

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

pip install --editable ~/fairseq/

export PYTHONPATH="~/fairseq/:$PYTHONPATH"
export PATH="~/fairseq/:$PATH"
