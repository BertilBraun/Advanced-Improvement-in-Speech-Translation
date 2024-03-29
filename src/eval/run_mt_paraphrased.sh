#!/bin/bash

#SBATCH --job-name=eval_mt_paraphrased     # job name
#SBATCH --partition=dev_gpu_4              # mby GPU queue for the resource allocation.
#SBATCH --time=00:30:00                    # wall-clock time limit  
#SBATCH --mem=40000                        # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --cpus-per-task=1                  # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --gres=gpu:1
#SBATCH --output=eval_mt_paraphrased_logs_%j.txt
#SBATCH --error=eval_mt_paraphrased_logs_%j.txt

cd ../../
source setup.sh

python -m src.train.prepare_all_datasets
# check exit code of python call, if crash, exit
if [ $? -ne 0 ]; then
    exit 1
fi

echo "Translating the test set..."

source src/bash/translate_mt.sh \
        $MT_PARAPHRASED_BINARY_DATA_DIR/dict.en.txt \
        $MT_PARAPHRASED_BINARY_DATA_DIR/dict.de.txt \
        $MT_PARAPHRASED_DATA_DIR/test \
        $MT_PARAPHRASED_BINARY_DATA_DIR \
        $MT_PARAPHRASED_MODEL_DIR \
        ~/predictions/finetune_mt_covost
