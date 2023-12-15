#!/bin/bash


#SBATCH --job-name=PST_train_asr_w2v           # job name
#SBATCH --partition=gpu_4                  # mby GPU queue for the resource allocation.
#SBATCH --time=00:45:00                     # wall-clock time limit  
#SBATCH --mem=20000                       # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --cpus-per-task=1                  # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --gres=gpu:2
#SBATCH --output=/pfs/work7/workspace/scratch/uxude-ASR/logs/nvidia_output_%j.txt
#SBATCH --error=/pfs/work7/workspace/scratch/uxude-ASR/logs/nvidia_error_%j.txt


# call ../setup.sh
source ../setup.sh

# Start nvidia-smi in the background and redirect its output to a file
GPU_MONITOR_INTERVAL=60 # Interval in seconds
GPU_LOG_FILE=$(ws_find ASR)/logs/gpu_usage.log
while true; do
    nvidia-smi >> "$GPU_LOG_FILE"
    sleep $GPU_MONITOR_INTERVAL
done &


# Define the model types in an array
MODEL_TYPES=("wav2vec")

for model in "${MODEL_TYPES[@]}"; do
    # Set directories based on the model type
    DATA_DIR=$(ws_find ASR)/$model
    MODEL_DIR=$(ws_find ASR)/models/$model

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
        --num-workers 4 --max-tokens 40000 --max-update 30 \
        --task speech_to_text --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --report-accuracy \
        --arch s2t_transformer_s --share-decoder-input-output-embed \
        --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
        --clip-norm 10.0 --seed 1 --update-freq 8 \
        --keep-last-epochs 2 --save-interval-updates 100 --keep-best-checkpoints 1

    # Log the completion of training for the current model
    echo "Training completed for $model."
done


