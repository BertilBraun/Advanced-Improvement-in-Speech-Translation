#!/bin/bash

#SBATCH --job-name=eval_asr                # job name
#SBATCH --partition=dev_gpu_4              # mby GPU queue for the resource allocation.
#SBATCH --time=00:30:00                    # wall-clock time limit  
#SBATCH --mem=40000                        # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --cpus-per-task=1                  # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --gres=gpu:1
#SBATCH --output=eval_asr_logs_%j.txt
#SBATCH --error=eval_asr_logs_%j.txt

source ../setup.sh

model="wav2vec" # "wav2vec" or "mel"

DATA_DIR="$COVOST/$model"
MODEL_DIR="/pfs/work7/workspace/scratch/ubppd-ASR/res/asr"

TABLES="/pfs/work7/workspace/scratch/ubppd-ASR/covost/corpus-16.1/asr_input_data_tables"

PRED_OUTPUT_DIR=~/ASR/predictions/covost/$model
PRED_LOG=$PRED_OUTPUT_DIR/en_s2t.pred.log

mkdir -p $PRED_OUTPUT_DIR

echo "Starting prediction for $model"

fairseq-generate "$COVOST" \
    --config-yaml /home/kit/stud/ubppd/fairseq/.circleci/config.yaml \
    --gen-subset wav2vec \
    --task speech_to_text \
    --path "$MODEL_DIR/checkpoint_best_after_120h_train.pt" \
    --max-tokens 50000 --beam 5 --scoring wer > $PRED_LOG

echo "Prediction done for $model"

grep ^H $PRED_LOG | sed 's/^H-//g' | cut -f 3 | sed 's/ ##//g' > $PRED_OUTPUT_DIR/hyp.txt

echo "Prediction files written for $model"
echo "Sample predictions:"

head $PRED_OUTPUT_DIR/hyp.txt

echo "Now go ahead and implement WER"
