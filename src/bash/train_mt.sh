#!/bin/bash

BINARY_DATA_DIR=$1
TRAIN_SPM_PREF=$2
VALID_SPM_PREF=$3
MODEL_DIR=$4
TRAIN_TIME_IN_HOURS=$5

mkdir -p $BINARY_DATA_DIR
mkdir -p $MODEL_DIR

rm $BINARY_DATA_DIR/preprocess.log

# Binarize the data for training
if [ -z "$(ls -A $BINARY_DATA_DIR)" ]; then
  echo "Binarizing the data..."
  echo "Binarized data will be stored in $BINARY_DATA_DIR"
  echo "Training data: $TRAIN_SPM_PREF"
  echo "Validation data: $VALID_SPM_PREF"

  fairseq-preprocess \
    --source-lang en --target-lang de \
    --trainpref $TRAIN_SPM_PREF \
    --validpref $VALID_SPM_PREF \
    --destdir $BINARY_DATA_DIR \
    --workers 8
    
  echo "Binarization complete."
else
  echo "Binarized data already exists. Skipping binarization."
fi


# Train the model in parallel
echo "Training the model..."
echo "Model will be stored in $MODEL_DIR"
echo "Training time: $TRAIN_TIME_IN_HOURS hours"

fairseq-train \
    $BINARY_DATA_DIR --save-dir $MODEL_DIR \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 --max-epoch 500 --stop-time-hours $TRAIN_TIME_IN_HOURS \
    --fp16

echo "Training complete."
