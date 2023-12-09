#!/bin/bash

source ../setup.sh

model="wav2vec" # "wav2vec" or "mel"

DATA_DIR=~/PST/ASR/$model
MODEL_DIR=~/PST/ASR/models/$model


PRED_OUTPUT_DIR="~/PST/ASR/predictions/$model"
PRED_LOG="$PRED_OUTPUT_DIR/en_s2t.pred.log"

mkdir -p $PRED_OUTPUT_DIR

echo "Starting prediction for $model"

fairseq-generate "$DATA_DIR" \
    --config-yaml config.yaml --gen-subset test-clean \
    --task speech_to_text \
    --path "$MODEL_DIR/checkpoint_best.pt" \
    --max-tokens 50000 --beam 5 --scoring wer > $PRED_LOG


grep ^H $PRED_LOG | sed 's/^H-//g' | cut -f 3 | sed 's/ ##//g' > $PRED_OUTPUT_DIR/hyp.txt
grep ^T $PRED_LOG | sed 's/^T-//g' | cut -f 2 | sed 's/ ##//g' > $PRED_OUTPUT_DIR/ref.txt

head $PRED_OUTPUT_DIR/hyp.txt
head $PRED_OUTPUT_DIR/ref.txt

tail -n 1 $PRED_LOG