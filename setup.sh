#!/bin/bash

export OMP_NUM_THREADS=1 # ${SLURM_CPUS_PER_TASK}

# module purge                                    # Unload all currently loaded modules.
# module load compiler/gnu/10.2                   # Load required modules.  
# module load devel/python/3.8.6_gnu_10.2
# module load mpi/openmpi/4.1
# module load devel/cuda/11.8


# Function to check if extension to the workspace is needed
check_and_extend_workspace() {
    workspace=$1
    days=$2

    # Get the remaining time for the workspace
    remaining_time=$(ws_list | grep -A 1 "id: $workspace" | grep 'remaining time' | awk '{print $4}')

    # Check if remaining time is less than specified days
    if [ "$remaining_time" != "" ] && [ "$remaining_time" -lt "$days" ]; then
        echo "Extending workspace $workspace for $days days."
        ws_extend "$workspace" "$days"
    else
        echo "No extension needed for workspace $workspace."
    fi
}

# Check and extend workspaces
# check_and_extend_workspace ASR 30
# check_and_extend_workspace MT 30



# conda initialize
source ~/miniconda3/bin/activate nmt



if [ -d ~/fairseq ]; then
    echo "Fairseq directory exists. Checking if installed..."

    # Check if fairseq is installed
    if pip list | grep fairseq; then
        echo "Fairseq is already installed. Skipping installation."
    else
        echo "Fairseq directory exists but not installed. Installing..."
        pip install --editable ~/fairseq/
    fi
else
    echo "Fairseq directory does not exist. Cloning and installing..."
    git clone https://github.com/facebookresearch/fairseq.git ~/fairseq
    
    # in fairseq/models/speech_to_text/s2t_transformer.py replace "args.input_feat_per_channel * args.input_channels," with "768, # args.input_feat_per_channel * args.input_channels"
    sed -i 's/args.input_feat_per_channel \* args.input_channels,/768, # args.input_feat_per_channel \* args.input_channels/' ~/fairseq/fairseq/models/speech_to_text/s2t_transformer.py

    pip install --editable ~/fairseq/
fi


export PYTHONPATH=~/fairseq/:$PYTHONPATH
export PATH=~/fairseq/:$PATH

echo "Setup complete. Starting script execution..."