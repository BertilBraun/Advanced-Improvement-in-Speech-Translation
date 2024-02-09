#!/bin/bash

#SBATCH --job-name=train_punctuation_covost  # job name
#SBATCH --partition=gpu_4                    # mby GPU queue for the resource allocation.
#SBATCH --time=04:00:00                      # wall-clock time limit
#SBATCH --mem=100000                         # memory per node
#SBATCH --nodes=1                            # number of nodes to be used
#SBATCH --cpus-per-task=1                    # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                  # maximum count of tasks per node
#SBATCH --mail-type=ALL                      # Notify user by email when certain event types occur.
#SBATCH --gres=gpu:1
#SBATCH --output=train_punctuation_covost_%j.txt
#SBATCH --error=train_punctuation_covost_%j.txt

cd ../../
source setup.sh

TRAIN_TIME_IN_HOURS=6


python -m src.train.prepare_all_datasets
# check exit code of python call, if crash, exit
if [ $? -ne 0 ]; then
    exit 1
fi

echo "Training the Punctuation model..."

source src/bash/train_mt.sh \
        $PUNCTUATION_BINARY_DATA_DIR \
        $PUNCTUATION_DATA_DIR/train \
        $PUNCTUATION_DATA_DIR/dev \
        $PUNCTUATION_MODEL_DIR \
        $TRAIN_TIME_IN_HOURS

echo "Training complete."
echo "----------------------------------------------------------"
echo "Translating the test set..."

source src/bash/translate_mt.sh \
        $PUNCTUATION_BINARY_DATA_DIR/dict.en.txt \
        $PUNCTUATION_BINARY_DATA_DIR/dict.de.txt \
        $PUNCTUATION_DATA_DIR/test \
        $PUNCTUATION_BINARY_DATA_DIR \
        $PUNCTUATION_MODEL_DIR \
        ~/predictions/train_punctuation_covost
