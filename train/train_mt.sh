#!/bin/bash

#SBATCH --job-name=PST_train_asr           # job name
#SBATCH --partition=gpu_8                  # mby GPU queue for the resource allocation.
#SBATCH --time=2:00:00                     # wall-clock time limit  
#SBATCH --mem=200000                       # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --cpus-per-task=1                  # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --gres=gpu:8
#SBATCH --output=../../MT/logs/train_mt_%j.txt
#SBATCH --error=../../MT/logs/train_mt_%j.txt

# call ../setup.sh
source ../setup.sh

DATA_DIR=~/MT/dataset
PARAPHRASED_DATA_DIR=~/MT/output
BINARY_DATA_DIR=~/MT/binarized_dataset
MODEL_DIR=~/MT/models

# create folders if they don't exist
mkdir -p $DATA_DIR
mkdir -p $PARAPHRASED_DATA_DIR
mkdir -p $BINARY_DATA_DIR
mkdir -p $MODEL_DIR

# if PARAPHRASED_DATA_DIR is empty, print error and return
if [ -z "$(ls -A $PARAPHRASED_DATA_DIR)" ]; then
  echo "Error: PARAPHRASED_DATA_DIR is empty. Run preprocess/run_generate_paraphrases.sh first."
  exit 1
fi

# Binarize the data for training
fairseq-preprocess \
    --source-lang en --target-lang de \
    --trainpref "$PARAPHRASED_DATA_DIR/spm.train.de-en" \ 
    --validpref "$DATA_DIR/spm.dev.de-en" \
    --testpref "$DATA_DIR/spm.tst.de-en" \
    --destdir "$BINARY_DATA_DIR/iwslt14.de-en" \
    --thresholdtgt 0 --thresholdsrc 0 \
    --workers 8

# Train the model in parallel
fairseq-train \
    "$BINARY_DATA_DIR/iwslt14.de-en" --save-dir $MODEL_DIR \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --keep-last-epochs 2 --save-interval-updates 100 --keep-best-checkpoints 1 \
    --max-tokens 4096 \
    --max-epoch 20 \
    --fp16
