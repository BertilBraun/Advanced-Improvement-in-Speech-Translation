#!/bin/bash

#SBATCH --job-name=generate_paraphrases    # job name
#SBATCH --partition=gpu_4                  # mby GPU queue for the resource allocation.
#SBATCH --time=24:30:00                    # wall-clock time limit
#SBATCH --mem=150000                       # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --cpus-per-task=1                  # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --gres=gpu:1
#SBATCH --output=generate_paraphrases_%j.txt
#SBATCH --error=generate_paraphrases_%j.txt

cd ../..
source setup.sh

export TOTAL_PROCESSING_TIME_IN_HOURS=24

python -m src.preprocess.generate_paraphrases

# if python script fails, exit with error
if [ $? -ne 0 ]; then
    exit 1
fi

# remove MT_PARAPHRASED_TRAIN_WORKSPACE and create a new one
rm -rf $MT_PARAPHRASED_TRAIN_WORKSPACE
mkdir -p $MT_PARAPHRASED_TRAIN_WORKSPACE

# remove MT_PARAPHRASED_DATA_DIR and create a new one
rm -rf $MT_PARAPHRASED_DATA_DIR
mkdir -p $MT_PARAPHRASED_DATA_DIR

cd src/train
sbatch finetune_mt_paraphrased_covost.sh