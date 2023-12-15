#!/bin/bash

source ../setup.sh

BINARY_DATA_DIR=~/MT/binarized_dataset
MODEL_DIR=~/MT/models

PRED_OUTPUT_DIR=~/MT/predictions
PRED_LOG=$PRED_OUTPUT_DIR/en-de.pred.log

mkdir -p $PRED_OUTPUT_DIR

echo "Starting translation..."

fairseq-generate $BINARY_DATA_DIR/iwslt14.de-en \
      --task translation \
      --source-lang en \
      --target-lang de \
      --path $MODEL_DIR/checkpoint_best.pt \
      --batch-size 256 \
      --beam 4 \
      --remove-bpe=sentencepiece > $PRED_LOG

echo "Translations done"

grep ^H $PRED_LOG | sed 's/^H-//g' | cut -f 3 | sed 's/ ##//g' > $PRED_OUTPUT_DIR/hyp.txt
grep ^T $PRED_LOG | sed 's/^T-//g' | cut -f 2 | sed 's/ ##//g' > $PRED_OUTPUT_DIR/ref.txt

echo "Translations files written"
echo "Sample predictions:"

head $PRED_OUTPUT_DIR/hyp.txt

echo "Sample references:"

head $PRED_OUTPUT_DIR/ref.txt

echo "WER:"
tail -n 1 $PRED_LOG

echo "BLEU:"
cat $PRED_OUTPUT_DIR/hyp.txt | sacrebleu $PRED_OUTPUT_DIR/ref.txt