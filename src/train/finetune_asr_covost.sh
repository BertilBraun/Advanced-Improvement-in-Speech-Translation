#!/bin/bash

#SBATCH --job-name=finetune_asr_covost     # job name
#SBATCH --partition=gpu_4                  # mby GPU queue for the resource allocation.
#SBATCH --time=16:30:00                    # wall-clock time limit
#SBATCH --mem=100000                       # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --cpus-per-task=1                  # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --gres=gpu:1
#SBATCH --output=finetune_asr_covost_%j.txt
#SBATCH --error=finetune_asr_covost_%j.txt

cd ../../
source setup.sh

TRAIN_SUBSET=train
VAL_SUBSET=dev
TEST_SUBSET=test

TRAIN_TIME_IN_HOURS=46

python -m src.train.prepare_all_datasets
# check exit code of python call, if crash, exit
if [ $? -ne 0 ]; then
    exit 1
fi

echo "Finetuning the ASR model..."

source src/bash/train_asr.sh \
        $TRAIN_SUBSET \
        $VAL_SUBSET \
        $ASR_DATA_DIR \
        $ASR_MODEL_DIR \
        $TRAIN_TIME_IN_HOURS

echo "Finetuning complete."
echo "----------------------------------------------------------"
echo "Transcribing the test set..."

source src/bash/transcribe_asr.sh \
        $TEST_SUBSET \
        $ASR_DATA_DIR \
        $ASR_MODEL_DIR \
        ~/predictions/finetune_asr_covost
