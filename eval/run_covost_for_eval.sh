#!/bin/bash

echo "Starting the script execution."

source ../setup.sh
echo "Setup script sourced successfully."

# Set log level
export LOGLEVEL="INFO"  # ["INFO", "DEBUG", ...]

cd "$HOME/Advanced-Improvement-in-Speech-Translation" || { echo "Could not change directory. Exiting script ..."; exit 1;}

#python -m preprocess.process_audio_covost_wav2vec
python -m eval.prepare_covost_for_eval

echo "Script execution finished."
