#!/bin/bash

SRC_DICT=$1
TGT_DICT=$2
TEST_PREF=$3
BINARY_DATA_DIR=$4
CHECKPOINT_PATH=$5
PRED_OUTPUT_DIR=$6

PRED_LOG=$PRED_OUTPUT_DIR/translate_mt_pred.log

mkdir -p $PRED_OUTPUT_DIR
mkdir -p $BINARY_DATA_DIR

echo "Starting translation..."

fairseq-preprocess \
      --source-lang en --target-lang de \
      --srcdict $SRC_DICT \
      --tgtdict $TGT_DICT \
      --testpref $TEST_PREF \
      --destdir $BINARY_DATA_DIR \
      --thresholdtgt 0 --thresholdsrc 0

fairseq-generate $BINARY_DATA_DIR \
      --task translation \
      --source-lang en \
      --target-lang de \
      --path $CHECKPOINT_PATH \
      --batch-size 256 \
      --beam 4 \
      --remove-bpe=sentencepiece > $PRED_LOG

echo "Translations done"

source extract_from_prediction.sh  \
    $PRED_OUTPUT_DIR \
    $PRED_OUTPUT_DIR/hyp.txt \
    $PRED_OUTPUT_DIR/ref.txt
