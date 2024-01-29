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
echo "Source dictionary: $SRC_DICT"
echo "Target dictionary: $TGT_DICT"
echo "Test prefix: $TEST_PREF"
echo "Binary data directory: $BINARY_DATA_DIR"

fairseq-preprocess \
      --source-lang en --target-lang de \
      --srcdict $SRC_DICT \
      --tgtdict $TGT_DICT \
      --testpref $TEST_PREF \
      --destdir $BINARY_DATA_DIR \
      --thresholdtgt 0 --thresholdsrc 0

echo "Preprocessing done"
echo "Average checkpoints..."
echo "Checkpoints folder: $CHECKPOINT_FOLDER"
echo "Checkpoint path: $CHECKPOINT_PATH"

python ~/fairseq/scripts/average_checkpoints.py \
  --inputs $CHECKPOINT_FOLDER --num-epoch-checkpoints 5 \
  --output $CHECKPOINT_PATH

echo "Checkpoints averaged"
echo "Generating translations..."
echo "Binary data directory: $BINARY_DATA_DIR"
echo "Prediction output directory: $PRED_OUTPUT_DIR"

fairseq-generate $BINARY_DATA_DIR \
      --task translation \
      --source-lang en \
      --target-lang de \
      --path $CHECKPOINT_PATH \
      --batch-size 256 \
      --beam 4 \
      --remove-bpe=sentencepiece > $PRED_LOG

echo "Translations done"

source src/bash/extract_from_prediction.sh  \
    $PRED_LOG \
    $PRED_OUTPUT_DIR/hyp.txt \
    $PRED_OUTPUT_DIR/ref.txt
