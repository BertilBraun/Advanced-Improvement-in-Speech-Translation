#!/bin/bash

#SBATCH --job-name=covost_preprocess       # job name
#SBATCH --partition=dev_gpu_4              # mby GPU queue for the resource allocation.
#SBATCH --time=00:30:00                    # wall-clock time limit  
#SBATCH --mem=100000                       # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --cpus-per-task=1                  # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --gres=gpu:1
#SBATCH --output=covost/train_output_sbatch_%j.txt
#SBATCH --error=covost/train_error_sbatch_%j.txt


echo "Starting the script execution."

source ../setup.sh
echo "Setup script sourced successfully."

# Set log level
export LOGLEVEL="INFO"  # ["INFO", "DEBUG", ...]

cd "$HOME/Advanced-Improvement-in-Speech-Translation" || { echo "Could not change directory. Exiting script ..."; exit 1;}

#python -m preprocess.process_audio_covost_wav2vec
python -m eval.prepare_covost_for_eval

echo "Script execution finished."
