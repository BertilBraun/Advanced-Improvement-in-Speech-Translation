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
#SBATCH --output=train_output_sbatch_%j.txt
#SBATCH --error=logs/train_error_sbatch_%j.txt


echo "Starting the script execution."

source ../setup.sh
echo "Setup script sourced successfully."

# Set log level
export LOGLEVEL="INFO"  # ["INFO", "DEBUG", ...]

export ASR="/pfs/work7/workspace/scratch/ubppd-ASR"
export COVOST="$ASR/covost"
export COVOST_CORPUS="$COVOST/corpus-16.1"

export COVOST_DATA="$COVOST/data"
echo "COVOST_DATA environment variable set to $COVOST_DATA."

# Define the file you want to check
FILE_TO_CHECK="$COVOST_DATA/translations/covost_v2.en_de.test.tsv"
echo "File to check for existence: $FILE_TO_CHECK."

# Check if the file exists
if [ ! -f "$FILE_TO_CHECK" ]; then
    echo "File $FILE_TO_CHECK not found. Running Python script to generate TSV files."
    
    # Run the Python script with the file
    python "$COVOST/get_covost_splits.py" \
      --version 2 --src-lang en --tgt-lang de \
      --root "$COVOST_DATA/translations" \
      --cv-tsv "$COVOST_CORPUS/en/validated.tsv"

    echo "Python script to get the covost splits are completed. TSV files should be generated in the $COVOST_DATA directory."
else
    echo "File $FILE_TO_CHECK already exists."
fi

cd "$HOME/Advanced-Improvement-in-Speech-Translation" || { echo "Could not change directory. Exiting script ..."; exit 1;}

echo "Running python script: process_audio_covost.py"
python -m preprocess.process_audio_covost


echo "Script execution finished."
