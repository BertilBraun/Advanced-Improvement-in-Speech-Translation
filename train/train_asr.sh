#!/bin/bash

#SBATCH --job-name=PST_train_asr           # job name
#SBATCH --partition=gpu_8                  # TODO mby GPU queue for the resource allocation.
#SBATCH --time=6:00:00                     # TODO wall-clock time limit  
#SBATCH --mem=200000                       # TODO memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --cpus-per-task=1                  # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --mail-user=uxude@student.kit.edu  # notification email address
#SBATCH --gres=gpu:8
#SBATCH --output=~/PST/ASR/logs/train_output.txt
#SBATCH --error=~/PST/ASR/logs/train_error.txt


# call ~/setup.sh
source ~/Advanced-Improvement-in-Speech-Translation/setup.sh

# Define the model types in an array
MODEL_TYPES=("wav2vec" "mel")

for model in "${MODEL_TYPES[@]}"; do
    # Set directories based on the model type
    DATA_DIR=~/PST/ASR/$model
    MODEL_DIR=~/PST/ASR/models/$model

    # Create folders if they don't exist
    mkdir -p $DATA_DIR
    mkdir -p $MODEL_DIR

    # Check if DATA_DIR is empty, print error and return
    if [ -z "$(ls -A $DATA_DIR)" ]; then
        echo "Error: DATA_DIR ($DATA_DIR) is empty. Run preprocess/run_process_audio.sh first."
        exit 1
    fi

    # Train the model in parallel
    fairseq-train $DATA_DIR --save-dir $MODEL_DIR \
        --train-subset train-clean-100 --valid-subset dev-clean \
        --num-workers 4 --max-tokens 40000 --max-update 300000 \
        --task speech_to_text --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --report-accuracy \
        --arch s2t_transformer_s --share-decoder-input-output-embed \
        --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
        --clip-norm 10.0 --seed 1 --update-freq 8

    # Log the completion of training for the current model
    echo "Training completed for $model."
done
