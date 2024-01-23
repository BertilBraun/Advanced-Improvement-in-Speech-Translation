#!/bin/bash

BINARY_DATA_DIR=$1
TRAIN_SPM_PREF=$2
VALID_SPM_PREF=$3
MODEL_DIR=$5
TRAIN_TIME_IN_HOURS=$6

mkdir -p $BINARY_DATA_DIR
mkdir -p $MODEL_DIR

# Binarize the data for training
if [ -z "$(ls -A $BINARY_DATA_DIR)" ]; then
  echo "Binarizing the data..."
  fairseq-preprocess \
    --source-lang en --target-lang de \
    --trainpref $TRAIN_SPM_PREF \
    --validpref $VALID_SPM_PREF \
    --destdir $BINARY_DATA_DIR
    
  echo "Binarization complete."
else
  echo "Binarized data already exists. Skipping binarization."
fi


# Train the model in parallel
echo "Training the model..."

fairseq-train \
    $BINARY_DATA_DIR --save-dir $MODEL_DIR \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr 5e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --keep-last-epochs 1 --save-interval-updates 50000 --keep-best-checkpoints 1 \
    --max-tokens 4096 --max-epoch 100 --stop-time-hours $TRAIN_TIME_IN_HOURS \
    --fp16

echo "Training complete."
