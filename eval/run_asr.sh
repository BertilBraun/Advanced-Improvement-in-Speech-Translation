#!/bin/bash

#SBATCH --job-name=PST_eval_asr            # job name
#SBATCH --partition=gpu_4                  # mby GPU queue for the resource allocation.
#SBATCH --time=00:45:00                    # wall-clock time limit  
#SBATCH --mem=40000                        # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --cpus-per-task=1                  # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --gres=gpu:1
#SBATCH --output=../../ASR/logs/eval_asr_%j.txt
#SBATCH --error=../../ASR/logs/eval_asr_%j.txt

source ../setup.sh

model="wav2vec" # "wav2vec" or "mel"

DATA_DIR=~/ASR/$model
MODEL_DIR=~/ASR/models/$model


PRED_OUTPUT_DIR=~/ASR/predictions/$model
PRED_LOG=$PRED_OUTPUT_DIR/en_s2t.pred.log

mkdir -p $PRED_OUTPUT_DIR

echo "Starting prediction for $model"

fairseq-generate "$DATA_DIR" \
    --config-yaml config.yaml --gen-subset test-clean \
    --task speech_to_text \
    --path "$MODEL_DIR/checkpoint_best.pt" \
    --max-tokens 50000 --beam 5 --scoring wer > $PRED_LOG

echo "Prediction done for $model"

grep ^H $PRED_LOG | sed 's/^H-//g' | cut -f 3 | sed 's/ ##//g' > $PRED_OUTPUT_DIR/hyp.txt
grep ^T $PRED_LOG | sed 's/^T-//g' | cut -f 2 | sed 's/ ##//g' > $PRED_OUTPUT_DIR/ref.txt

echo "Prediction files written for $model"
echo "Sample predictions:"

head $PRED_OUTPUT_DIR/hyp.txt

echo "Sample references:"

head $PRED_OUTPUT_DIR/ref.txt

echo "WER:"
tail -n 1 $PRED_LOG
