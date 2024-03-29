#!/bin/bash

INPUT_FILE=$1
HYP_OUTPUT_FILE=$2
REF_OUTPUT_FILE=$3
NUM_SAMPLES_PER_PREDICTION=$4
SRC_OUTPUT_FILE=$5

SAMPLE_HYP_OUTPUT_FILE=$HYP_OUTPUT_FILE.sampled

# if the number of samples per prediction is not provided, set it to 1
if [ -z "$NUM_SAMPLES_PER_PREDICTION" ]
then
    NUM_SAMPLES_PER_PREDICTION=1
fi

grep ^D $INPUT_FILE | sed 's/^D-//g' | cut -f 3 | sed 's/ ##//g' > $HYP_OUTPUT_FILE
grep ^T $INPUT_FILE | sed 's/^T-//g' | cut -f 2 | sed 's/ ##//g' > $REF_OUTPUT_FILE

# if SRC_OUTPUT_FILE is provided, write the source sentences to it
if [ -n "$SRC_OUTPUT_FILE" ]
then
    grep ^S $INPUT_FILE | sed 's/^S-//g' | cut -f 2 | sed 's/ ##//g' > $SRC_OUTPUT_FILE
fi

echo "Prediction files written for $HYP_OUTPUT_FILE and $REF_OUTPUT_FILE and $SRC_OUTPUT_FILE"

# write the first, the num_samples_per_prediction-th, the 2*num_samples_per_prediction-th, etc. predictions to one file named HYP_OUTPUT_FILE + ".sampled"
# Reset SAMPLE_HYP_OUTPUT_FILE
> $SAMPLE_HYP_OUTPUT_FILE

# Process in blocks of NUM_SAMPLES_PER_PREDICTION
TOTAL_LINES=$(wc -l < $HYP_OUTPUT_FILE)
for ((i=1; i<=TOTAL_LINES; i+=$NUM_SAMPLES_PER_PREDICTION)); do
    # Assuming the first prediction in each block is the 'best'
    head -n $i $HYP_OUTPUT_FILE | tail -1 >> $SAMPLE_HYP_OUTPUT_FILE
done
echo "Sampled predictions written to $SAMPLE_HYP_OUTPUT_FILE"

echo "Sample predictions:"
for i in {1..20}
do
    echo "Sample: $(head -$i $SAMPLE_HYP_OUTPUT_FILE | tail -1)"
    echo "Reference: $(head -$i $REF_OUTPUT_FILE | tail -1)"
done


# compute WER and BLEU of HYP_OUTPUT_FILE and REF_OUTPUT_FILE
echo "WER:"
tail -n 1 $INPUT_FILE

echo "BLEU:"
sacrebleu -tok none -s 'none' $REF_OUTPUT_FILE < $SAMPLE_HYP_OUTPUT_FILE
