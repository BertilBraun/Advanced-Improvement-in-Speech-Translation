#!/bin/bash

#SBATCH --job-name=finetune_mt_covost      # job name
#SBATCH --partition=gpu_4                  # mby GPU queue for the resource allocation.
#SBATCH --time=06:30:00                    # wall-clock time limit
#SBATCH --mem=200000                       # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --cpus-per-task=1                  # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --gres=gpu:1
#SBATCH --output=finetune_mt_covost_%j.txt
#SBATCH --error=finetune_mt_covost_%j.txt

cd ../../
source setup.sh

MT_WORKSPACE=/pfs/work7/workspace/scratch/uxude-MT

TRAIN_WORKSPACE=$MT_WORKSPACE/train/finetune_mt_covost

DATA_DIR=$MT_WORKSPACE/dataset/covost/spm
BINARY_DATA_DIR=$TRAIN_WORKSPACE/binarized_dataset
MODEL_DIR=$TRAIN_WORKSPACE/models

TRAIN_TIME_IN_HOURS=6


python -m src.train.prepare_all_datasets
# check exit code of python call, if crash, exit
if [ $? -ne 0 ]; then
    exit 1
fi

echo "Finetuning the MT model..."

source src/bash/train_mt.sh \
        $BINARY_DATA_DIR \
        $DATA_DIR/train \
        $DATA_DIR/dev \
        $MODEL_DIR \
        $TRAIN_TIME_IN_HOURS

echo "Finetuning complete."
echo "----------------------------------------------------------"
echo "Translating the test set..."

source src/bash/translate_mt.sh \
        $BINARY_DATA_DIR/dict.en.txt \
        $BINARY_DATA_DIR/dict.de.txt \
        $DATA_DIR/test \
        $BINARY_DATA_DIR \
        $MODEL_DIR \
        ~/predictions/finetune_mt_covost
