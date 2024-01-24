#!/bin/bash

INPUT_FILE=$1
HYP_OUTPUT_FILE=$2
REF_OUTPUT_FILE=$3

grep ^D $INPUT_FILE | sed 's/^D-//g' | cut -f 3 | sed 's/ ##//g' > $HYP_OUTPUT_FILE
grep ^T $INPUT_FILE | sed 's/^T-//g' | cut -f 2 | sed 's/ ##//g' > $REF_OUTPUT_FILE

echo "Prediction files written for $HYP_OUTPUT_FILE and $REF_OUTPUT_FILE"

echo "Sample predictions:"
echo "Sample: $(head -1 $HYP_OUTPUT_FILE)"
echo "Reference: $(head -1 $REF_OUTPUT_FILE)"
echo "Sample: $(head -2 $HYP_OUTPUT_FILE | tail -1)"
echo "Reference: $(head -2 $REF_OUTPUT_FILE | tail -1)"

# compute WER and BLEU of HYP_OUTPUT_FILE and REF_OUTPUT_FILE
echo "WER:"
tail -n 1 $INPUT_FILE

echo "BLEU:"
sacrebleu -tok none -s 'none' $REF_OUTPUT_FILE < $HYP_OUTPUT_FILE