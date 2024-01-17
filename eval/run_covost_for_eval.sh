#!/bin/bash

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

cd "$HOME/Advanced-Improvement-in-Speech-Translation" || { echo "Could not change directory. Exiting script ..."; exit 1;}

#python -m preprocess.process_audio_covost_wav2vec
python -m eval.prepare_covost_for_eval

echo "Script execution finished."
