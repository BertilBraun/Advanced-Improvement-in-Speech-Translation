#!/bin/bash

# Set log level -- used by custom logger in our code
export OUR_LOG_LEVEL="INFO"  # ["INFO", "DEBUG", ...]
export VERBOSE=False

# Function to check if extension to the workspace is needed
check_and_extend_workspace() {
    workspace=$1
    days=$2

    # Get the remaining time for the workspace
    remaining_time=$(ws_list | grep -A 1 "id: $workspace" | grep 'remaining time' | awk '{print $4}')

    # Check if remaining time is less than specified days
    if [ "$remaining_time" != "" ] && [ "$remaining_time" -lt "$days" ]; then
        echo "Extending workspace $workspace for 60 days."
        ws_extend "$workspace" 60
    else
        echo "No extension needed for workspace $workspace."
    fi
}

# Check and extend workspaces
check_and_extend_workspace ASR 30
check_and_extend_workspace MT 30


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
    # sed -i 's/args.input_feat_per_channel \* args.input_channels,/768, # args.input_feat_per_channel \* args.input_channels/' ~/fairseq/fairseq/models/speech_to_text/s2t_transformer.py

    pip install --editable ~/fairseq/
fi

export PYTHONPATH=~/fairseq/:$PYTHONPATH
export PATH=~/fairseq/:$PATH

export NUMEXPR_MAX_THREADS=8


# ----------------- ASR -----------------
export ASR_WORKSPACE=/pfs/work7/workspace/scratch/uxude-ASR

export ASR_TRAIN_WORKSPACE=$ASR_WORKSPACE/train/finetune_asr_covost

export ASR_DATA_DIR=$ASR_WORKSPACE/dataset/covost
export ASR_MODEL_DIR=$ASR_TRAIN_WORKSPACE/models
# --------------------------------------


# ----------------- MT -----------------
export MT_WORKSPACE=/pfs/work7/workspace/scratch/uxude-MT

export MT_TRAIN_WORKSPACE=$MT_WORKSPACE/train/finetune_mt_covost

export MT_DATA_DIR=$MT_WORKSPACE/dataset/covost_paraphrased/spm
export MT_BINARY_DATA_DIR=$MT_TRAIN_WORKSPACE/binarized_dataset
export MT_MODEL_DIR=$MT_TRAIN_WORKSPACE/models
# --------------------------------------


# ----------------- Punctuation---------
export PUNCTUATION_TRAIN_WORKSPACE=$MT_WORKSPACE/train/train_punctuation_covost

export PUNCTUATION_DATA_DIR=$MT_WORKSPACE/dataset/covost_punctuation_enhancement/spm
export PUNCTUATION_BINARY_DATA_DIR=$PUNCTUATION_TRAIN_WORKSPACE/binarized_dataset
export PUNCTUATION_MODEL_DIR=$PUNCTUATION_TRAIN_WORKSPACE/models
# --------------------------------------



echo "Setup complete. Starting script execution..."
