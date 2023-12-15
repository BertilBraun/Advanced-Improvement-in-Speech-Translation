
export OMP_NUM_THREADS=1 # ${SLURM_CPUS_PER_TASK}

module purge                                    # Unload all currently loaded modules.
module load compiler/gnu/10.2                   # Load required modules.  
# module load devel/python/3.8.6_gnu_10.2
# module load mpi/openmpi/4.1
# module load devel/cuda/10.2

# conda initialize
source ~/miniconda3/bin/activate nmt

if [ -d "~/fairseq" ]; then
    echo "Fairseq directory exists. Checking if installed..."

    # Check if fairseq is installed
    if pip list | grep -q fairseq; then
        echo "Fairseq is already installed. Skipping installation."
    else
        echo "Fairseq directory exists but not installed. Installing..."
        pip install --editable ~/fairseq/
    fi
else
    echo "Fairseq directory does not exist. Cloning and installing..."
    git clone https://github.com/facebookresearch/fairseq.git ~/fairseq
    pip install --editable ~/fairseq/
fi


export PYTHONPATH="~/fairseq/:$PYTHONPATH"
export PATH="~/fairseq/:$PATH"
