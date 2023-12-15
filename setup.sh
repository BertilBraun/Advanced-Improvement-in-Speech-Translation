
export OMP_NUM_THREADS=1 # ${SLURM_CPUS_PER_TASK}

module purge                                    # Unload all currently loaded modules.
module load compiler/gnu/10.2                   # Load required modules.  
# module load devel/python/3.8.6_gnu_10.2
# module load mpi/openmpi/4.1
# module load devel/cuda/10.2

# conda initialize
source ~/miniconda/etc/profile.d/conda.sh
conda activate nmt

# if fairseq is not cloned into ~/fairseq, install it
if [ ! -d "~/fairseq" ]; then
    git clone https://github.com/facebookresearch/fairseq.git ~/fairseq
    pip install --editable ~/fairseq/
fi

export PYTHONPATH="~/fairseq/:$PYTHONPATH"
export PATH="~/fairseq/:$PATH"
