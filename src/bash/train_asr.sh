#!/bin/bash

TRAIN_SUBSET=$1
VAL_SUBSET=$2
DATA_DIR=$3
MODEL_DIR=$4
TRAIN_TIME_IN_HOURS=$5

mkdir -p $DATA_DIR
mkdir -p $MODEL_DIR

# Check if DATA_DIR is empty, print error and return
if [ -z "$(ls -A $DATA_DIR)" ]; then
    echo "Error: DATA_DIR ($DATA_DIR) is empty. Run preprocess/run_process_audio.sh first."
    exit 1
fi

# Train the model
echo "Training the model..."

fairseq-train $DATA_DIR --save-dir $MODEL_DIR \
    --train-subset $TRAIN_SUBSET --valid-subset $VAL_SUBSET \
    --num-workers 4 --max-tokens 40000 --max-epoch 100 \
    --task speech_to_text --criterion label_smoothed_cross_entropy_with_ctc \
    --arch s2t_transformer_m --share-decoder-input-output-embed \
    --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
    --keep-last-epochs 1 --save-interval-updates 50000 --keep-best-checkpoints 1 \
    --stop-time-hours $TRAIN_TIME_IN_HOURS 


echo "Training complete."
