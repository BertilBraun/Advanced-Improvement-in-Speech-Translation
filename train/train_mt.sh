#!/bin/bash

#SBATCH --job-name=train_mt                # job name
#SBATCH --partition=gpu_4                  # mby GPU queue for the resource allocation.
#SBATCH --time=12:00:00                    # wall-clock time limit  
#SBATCH --mem=200000                       # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --cpus-per-task=1                  # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --gres=gpu:1
#SBATCH --output=train_mt_logs_%j.txt
#SBATCH --error=train_mt_logs_%j.txt

# call ../setup.sh
source ../setup.sh

DATA_DIR=~/MT/dataset
PARAPHRASED_DATA_DIR=/pfs/work7/workspace/scratch/uxude-MT/output
BINARY_DATA_DIR=/pfs/work7/workspace/scratch/uxude-MT/binarized_dataset
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

# Already processed, don't repeat work :) python preprocess_mt_dataset.py

# Binarize the data for training
if [ -z "$(ls -A $BINARY_DATA_DIR)" ]; then
  echo "Binarizing the data..."
  fairseq-preprocess \
    --source-lang en --target-lang de \
    --trainpref $PARAPHRASED_DATA_DIR/spm.train_complete.de-en \
    --validpref $DATA_DIR/spm.dev.de-en \
    --testpref $DATA_DIR/spm.tst.de-en \
    --destdir $BINARY_DATA_DIR \
    --thresholdtgt 0 --thresholdsrc 0
  echo "Binarization complete."
else
  echo "Binarized data already exists. Skipping binarization."
fi


# Train the model in parallel
echo "Training the model..."

fairseq-train \
    $BINARY_DATA_DIR --save-dir $MODEL_DIR \
    --arch transformer_vaswani_wmt_en_de_big \
    --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --keep-last-epochs 1 --save-interval-updates 50000 --keep-best-checkpoints 1 \
    --max-tokens 1024 \
    --max-epoch 500 \
    --fp16

echo "Training complete."