#!/bin/bash

TEST_SUBSET=$1
DATA_DIR=$2
CHECKPOINT_FOLDER=$3
PRED_OUTPUT_DIR=$4

PRED_LOG=$PRED_OUTPUT_DIR/transcribe_asr_pred.log

CHECKPOINT_PATH=$CHECKPOINT_FOLDER/avg_last_5_checkpoint.pt

mkdir -p $PRED_OUTPUT_DIR

echo "Starting transcription..."
echo "Average checkpoints..."
echo "Checkpoints folder: $CHECKPOINT_FOLDER"
echo "Checkpoint path: $CHECKPOINT_PATH"

python ~/fairseq/scripts/average_checkpoints.py \
  --inputs $CHECKPOINT_FOLDER --num-epoch-checkpoints 5 \
  --output $CHECKPOINT_PATH

echo "Checkpoints averaged"
echo "Generating transcriptions..."
echo "Test subset: $TEST_SUBSET"
echo "Data directory: $DATA_DIR"
echo "Prediction output directory: $PRED_OUTPUT_DIR"

fairseq-generate $DATA_DIR \
    --config-yaml config.yaml --gen-subset $TEST_SUBSET \
    --task speech_to_text \
    --path $CHECKPOINT_PATH \
    --max-tokens 50000 --beam 5 --scoring wer > $PRED_LOG

echo "Transcription done"

source src/bash/extract_from_prediction.sh  \
    $PRED_LOG \
    $PRED_OUTPUT_DIR/hyp.txt \
    $PRED_OUTPUT_DIR/ref.txt
