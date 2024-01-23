#!/bin/bash

TEST_SUBSET=$1
DATA_DIR=$2
CHECKPOINT_PATH=$3
PRED_OUTPUT_DIR=$4

PRED_LOG=$PRED_OUTPUT_DIR/transcribe_asr_pred.log

mkdir -p $PRED_OUTPUT_DIR

echo "Starting transcription..."

fairseq-generate $DATA_DIR \
    --config-yaml config.yaml --gen-subset $TEST_SUBSET \
    --task speech_to_text \
    --path $CHECKPOINT_PATH \
    --max-tokens 50000 --beam 5 --scoring wer > $PRED_LOG

echo "Transcription done"

source extract_from_prediction.sh  \
    $PRED_OUTPUT_DIR \
    $PRED_OUTPUT_DIR/hyp.txt \
    $PRED_OUTPUT_DIR/ref.txt
