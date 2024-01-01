#!/bin/bash

#SBATCH --job-name=PST_eval_mt             # job name
#SBATCH --partition=gpu_4                  # mby GPU queue for the resource allocation.
#SBATCH --time=00:30:00                    # wall-clock time limit  
#SBATCH --mem=40000                        # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --cpus-per-task=1                  # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --gres=gpu:1
#SBATCH --output=eval_mt_logs_%j.txt
#SBATCH --error=eval_mt_logs_%j.txt

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