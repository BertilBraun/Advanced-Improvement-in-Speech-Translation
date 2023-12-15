
export OMP_NUM_THREADS=1 # ${SLURM_CPUS_PER_TASK}

module purge                                    # Unload all currently loaded modules.
module load compiler/gnu/10.2                   # Load required modules.  
# module load devel/python/3.8.6_gnu_10.2
# module load mpi/openmpi/4.1
# module load devel/cuda/10.2

# >>> conda initialize >>>
__conda_setup="$('~/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "~/miniconda3/etc/profile.d/conda.sh" ]; then
        . "~/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="~/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
	
conda activate nmt

# if fairseq is not cloned into ~/fairseq, install it
if [ ! -d "~/fairseq" ]; then
    git clone https://github.com/facebookresearch/fairseq.git ~/fairseq
    pip install --editable ~/fairseq/
fi

export PYTHONPATH="~/fairseq/:$PYTHONPATH"
export PATH="~/fairseq/:$PATH"
