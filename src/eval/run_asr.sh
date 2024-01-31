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

cd ../../
source setup.sh

ASR_WORKSPACE=/pfs/work7/workspace/scratch/uxude-ASR

TRAIN_WORKSPACE=$ASR_WORKSPACE/train/finetune_asr_covost

DATA_DIR=$ASR_WORKSPACE/dataset/covost
MODEL_DIR=$TRAIN_WORKSPACE/models

TEST_SUBSET=test

python -m src.train.prepare_all_datasets
# check exit code of python call, if crash, exit
if [ $? -ne 0 ]; then
    exit 1
fi

echo "Transcribing the test set..."

source src/bash/transcribe_asr.sh \
        $TEST_SUBSET \
        $DATA_DIR \
        $MODEL_DIR \
        ~/predictions/finetune_asr_covost
