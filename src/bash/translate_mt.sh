#!/bin/bash

SRC_DICT=$1
TGT_DICT=$2
TEST_PREF=$3
BINARY_DATA_DIR=$4
CHECKPOINT_FOLDER=$5
PRED_OUTPUT_DIR=$6

PRED_LOG=$PRED_OUTPUT_DIR/translate_mt_pred.log

CHECKPOINT_PATH=$CHECKPOINT_FOLDER/avg_last_5_checkpoint.pt

mkdir -p $PRED_OUTPUT_DIR
mkdir -p $BINARY_DATA_DIR

echo "Starting translation..."
echo "Preprocessing data..."

fairseq-preprocess \
      --source-lang en --target-lang de \
      --srcdict $SRC_DICT \
      --tgtdict $TGT_DICT \
      --testpref $TEST_PREF \
      --destdir $BINARY_DATA_DIR \
      --thresholdtgt 0 --thresholdsrc 0

echo "Preprocessing done"
echo "Average checkpoints..."

python ~/fairseq/scripts/average_checkpoints.py \
  --inputs $CHECKPOINT_FOLDER --num-epoch-checkpoints 5 \
  --output $CHECKPOINT_PATH

echo "Checkpoints averaged"
echo "Generating translations..."

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
